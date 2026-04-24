import Foundation
import SwiftUI

@MainActor
final class AppState: ObservableObject {
    @Published var selectedSection: SidebarSection = .library
    @Published var summary = MemorySummary(documents: 0, chunks: 0, regions: 0, links: 0, mapBytes: 0)
    @Published var snapshot: VisualizationSnapshot?
    @Published var importResult: ImportResult?
    @Published var modelConfig = ModelConfig(
        mode: .local,
        endpoint: "http://localhost:11434",
        apiKeyName: nil,
        chatModel: nil,
        embeddingModel: "embeddinggemma:300m",
        health: ModelHealth(status: "disconnected", message: "Using local Ollama embeddings", checkedAt: nil)
    )
    @Published var apiKey = ""
    @Published var inspector = InspectorState()
    @Published var session = SessionState(id: "session-1", current: nil, history: [], visited: [])
    @Published var searchText = ""
    @Published var grepText = ""
    @Published var semanticText = ""
    @Published var statusMessage = "Load files to begin building memory."
    @Published var isBusy = false
    @Published var operationProgress: OperationProgress?

    let storePath: String
    private let keychainService = "ai-memory-app"
    private let apiKeyAccount = "api-key"
    private var progressTask: Task<Void, Never>?

    init() {
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let root = support.appendingPathComponent("AIMemory", isDirectory: true)
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        self.storePath = root.path
        self.apiKey = KeychainStore.load(service: keychainService, account: apiKeyAccount) ?? ""
        loadInitialState()
    }

    func loadInitialState() {
        do {
            modelConfig = try RustBridge.loadModelConfig(storePath: storePath)
            summary = try RustBridge.getSummary(storePath: storePath)
            snapshot = try? RustBridge.getSnapshot(storePath: storePath)
        } catch {
            statusMessage = error.localizedDescription
        }
    }

    func importFiles(urls: [URL]) {
        let storePath = self.storePath
        let paths = urls.map(\.path)
        runTask(
            message: "Importing files into memory…",
            operation: {
                let result = try RustBridge.ingestPaths(storePath: storePath, paths: paths)
                let snapshot = try RustBridge.getSnapshot(storePath: storePath)
                return (result, snapshot)
            },
            apply: { [weak self] result, snapshot in
                guard let self else { return }
                self.importResult = result
                self.summary = result.summary
                self.snapshot = snapshot
                self.statusMessage = "Imported \(result.importedCount), reused \(result.reusedEmbeddingCount), embedded \(result.embeddedCount)."
            }
        )
    }

    func rebuild() {
        let storePath = self.storePath
        runTask(
            message: "Rebuilding memory index…",
            operation: {
                let result = try RustBridge.rebuildMemory(storePath: storePath)
                let snapshot = try RustBridge.getSnapshot(storePath: storePath)
                return (result, snapshot)
            },
            apply: { [weak self] result, snapshot in
                guard let self else { return }
                self.importResult = result
                self.summary = result.summary
                self.snapshot = snapshot
                self.statusMessage = "Memory rebuilt: reused \(result.reusedEmbeddingCount), embedded \(result.embeddedCount)."
            }
        )
    }

    func refreshSnapshot() {
        let storePath = self.storePath
        runTask(
            message: "Refreshing map…",
            operation: {
                let summary = try RustBridge.getSummary(storePath: storePath)
                let snapshot = try RustBridge.getSnapshot(storePath: storePath)
                return (summary, snapshot)
            },
            apply: { [weak self] summary, snapshot in
                guard let self else { return }
                self.summary = summary
                self.snapshot = snapshot
            }
        )
    }

    func saveModelConfig() {
        let baseConfig = self.modelConfig
        let storePath = self.storePath
        let apiKey = self.apiKey
        let keychainService = self.keychainService
        let apiKeyAccount = self.apiKeyAccount
        runTask(
            message: "Saving model configuration…",
            operation: {
                var config = baseConfig
                if config.mode == .api, !apiKey.isEmpty {
                    try KeychainStore.save(service: keychainService, account: apiKeyAccount, value: apiKey)
                    config.apiKeyName = apiKeyAccount
                }
                return try RustBridge.saveModelConfig(storePath: storePath, config: config)
            },
            apply: { [weak self] savedConfig in
                guard let self else { return }
                self.modelConfig = savedConfig
                self.statusMessage = "Model settings saved."
            }
        )
    }

    func testModelConnection() {
        let request = ModelConnectionTestRequest(
            mode: self.modelConfig.mode,
            endpoint: self.modelConfig.endpoint,
            apiKey: self.modelConfig.mode == .api ? self.apiKey : nil,
            chatModel: self.modelConfig.chatModel,
            embeddingModel: self.modelConfig.embeddingModel
        )
        let storePath = self.storePath
        let baseConfig = self.modelConfig
        runTask(
            message: "Testing model connection…",
            operation: {
                var config = baseConfig
                let health = try RustBridge.testModelConnection(request)
                config.health = health
                _ = try RustBridge.saveModelConfig(storePath: storePath, config: config)
                return health
            },
            apply: { [weak self] health in
                guard let self else { return }
                self.modelConfig.health = health
                self.statusMessage = health.message
            }
        )
    }

    func selectNode(_ node: GraphNode) {
        inspector.selectedNode = node
        inspector.breadcrumbs.append(node.label)
        inspector.passages = []
        session.current = node.nodeRef
        let storePath = self.storePath
        runTask(
            message: "Loading node details…",
            operation: {
                let opened = try RustBridge.surfOpen(storePath: storePath, node: node.nodeRef)
                let neighbors = try RustBridge.surfNeighbors(storePath: storePath, node: node.nodeRef, maxResults: 8)
                return (opened, neighbors)
            },
            apply: { [weak self] opened, neighbors in
                guard let self else { return }
                self.inspector.links = opened.links
                self.inspector.neighbors = neighbors
                self.inspector.passages = opened.passages
                self.inspector.excerpt = opened.excerpt
                self.inspector.sourceAnchor = opened.sourceAnchor
                self.inspector.expansion = nil
            }
        )
    }

    func expandSelectedChunk(mode: ExpandMode) {
        guard let node = inspector.selectedNode, case .chunk(let id) = node.nodeRef else { return }
        let storePath = self.storePath
        runTask(
            message: "Expanding source context…",
            operation: {
                try RustBridge.surfExpand(storePath: storePath, chunkId: id, mode: mode, window: 420)
            },
            apply: { [weak self] expansion in
                guard let self else { return }
                self.inspector.expansion = expansion
                self.inspector.excerpt = expansion.excerpt
                self.inspector.sourceAnchor = expansion.sourceAnchor
                self.statusMessage = "Expanded \(mode.rawValue.lowercased()) context."
            }
        )
    }

    func performSearch() {
        guard !searchText.isEmpty else { return }
        let query = self.searchText
        let storePath = self.storePath
        runTask(
            message: "Routing query through memory…",
            operation: {
                try RustBridge.runQuery(
                    storePath: storePath,
                    request: .init(text: query, filters: [:], maxRegions: 3, maxChunks: 8)
                )
            },
            apply: { [weak self] result in
                guard let self else { return }
                self.inspector.queryResult = result
                self.statusMessage = result.routed.rationale
            }
        )
    }

    func grepSelectedDocument() {
        guard let node = inspector.selectedNode else { return }
        guard case .document(let id) = node.nodeRef else { return }
        let needle = grepText.isEmpty ? node.label : grepText
        let storePath = self.storePath
        runTask(
            message: "Grepping document…",
            operation: {
                try RustBridge.grepDocument(storePath: storePath, documentId: id, needle: needle)
            },
            apply: { [weak self] hits in
                self?.inspector.grepHits = hits
            }
        )
    }

    func semanticSearchSelectedDocument() {
        guard let node = inspector.selectedNode else { return }
        guard case .document(let id) = node.nodeRef else { return }
        let query = semanticText.isEmpty ? node.label : semanticText
        let storePath = self.storePath
        runTask(
            message: "Running semantic passage search…",
            operation: {
                try RustBridge.semanticDocumentSearch(storePath: storePath, documentId: id, query: query)
            },
            apply: { [weak self] hits in
                self?.inspector.grepHits = hits
            }
        )
    }

    func followLink(_ link: LinkRecord) {
        let storePath = self.storePath
        let session = self.session
        runTask(
            message: "Navigating memory link…",
            operation: {
                try RustBridge.stepNavigation(storePath: storePath, session: session, linkId: link.id)
            },
            apply: { [weak self] result in
                guard let self else { return }
                self.session = result.session
                self.inspector.links = result.links
                self.inspector.excerpt = result.currentExcerpt
                self.statusMessage = "Stepped through \(link.label)."
            }
        )
    }

    func backtrack() {
        let storePath = self.storePath
        let session = self.session
        runTask(
            message: "Backtracking…",
            operation: {
                try RustBridge.backtrackNavigation(storePath: storePath, session: session)
            },
            apply: { [weak self] result in
                guard let self else { return }
                self.session = result.session
                self.inspector.links = result.links
                self.inspector.excerpt = result.currentExcerpt
                self.statusMessage = "Moved back in navigation history."
            }
        )
    }

    private func runTask<Result: Sendable>(
        message: String,
        operation: @escaping @Sendable () throws -> Result,
        apply: @escaping @MainActor (Result) -> Void
    ) {
        isBusy = true
        statusMessage = message
        operationProgress = nil
        startProgressPolling()
        Task.detached(priority: .userInitiated) {
            do {
                let result = try operation()
                await MainActor.run {
                    apply(result)
                    self.isBusy = false
                    self.stopProgressPolling()
                }
            } catch {
                await MainActor.run {
                    self.statusMessage = error.localizedDescription
                    self.isBusy = false
                    self.stopProgressPolling()
                }
            }
        }
    }

    private func startProgressPolling() {
        progressTask?.cancel()
        let progressURL = URL(fileURLWithPath: storePath).appendingPathComponent("progress.json")
        progressTask = Task { [weak self] in
            let decoder = JSONDecoder()
            while !Task.isCancelled {
                if let data = try? Data(contentsOf: progressURL),
                   let progress = try? decoder.decode(OperationProgress.self, from: data) {
                    await MainActor.run {
                        self?.operationProgress = progress
                        self?.statusMessage = progress.message
                    }
                }
                try? await Task.sleep(nanoseconds: 300_000_000)
            }
        }
    }

    private func stopProgressPolling() {
        progressTask?.cancel()
        progressTask = nil
    }
}
