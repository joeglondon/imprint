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
        endpoint: "http://localhost:8080/v1",
        apiKeyName: nil,
        chatModel: "mlx-community/LFM2.5-1.2B-Instruct-8bit",
        plannerModel: "LiquidAI/LFM2.5-350M-MLX-8bit",
        responseModel: "mlx-community/LFM2.5-1.2B-Instruct-8bit",
        plannerEndpoint: "http://127.0.0.1:8082/v1",
        plannerAdapterPath: nil,
        responseAdapterPath: nil,
        sharedCortexAdapterPath: nil,
        activeAdapterHash: nil,
        adapterActivationPolicy: "automatic",
        runtimePreset: .mlx,
        cortexEnabled: true,
        cortexRounds: 3,
        criticModel: "LiquidAI/LFM2.5-350M-MLX-8bit",
        criticEndpoint: "http://127.0.0.1:8082/v1",
        compilerModel: "mlx-community/LFM2.5-1.2B-Instruct-8bit",
        embeddingModel: "embeddinggemma:300m",
        embeddingEndpoint: "http://localhost:11434",
        embeddingRuntimePreset: .ollama,
        health: ModelHealth(status: "disconnected", message: "Using local OpenAI-compatible models", checkedAt: nil)
    )
    @Published var apiKey = ""
    @Published var inspector = InspectorState()
    @Published var session = SessionState(id: "session-1", current: nil, history: [], visited: [])
    @Published var searchText = ""
    @Published var grepText = ""
    @Published var semanticText = ""
    @Published var chatSessions: [ChatSession] = []
    @Published var selectedChatSession: ChatSession?
    @Published var chatMessages: [ChatMessage] = []
    @Published var chatInput = ""
    @Published var chatContextTraces: [ChatContextTrace] = []
    @Published var derivedMemories: [DerivedMemory] = []
    @Published var workspaceConnections = WorkspaceConnection.defaults
    @Published var cortexAdapterState: CortexAdapterState?
    @Published var cortexAdapterJobs: [CortexAdapterJob] = []
    @Published var cortexRouteProbe: CortexRouteProbeResult?
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
            if let adapterSnapshot = try? RustBridge.loadCortexAdapterSnapshot(storePath: storePath) {
                applyAdapterSnapshot(adapterSnapshot)
            }
            loadChatState()
        } catch {
            statusMessage = error.localizedDescription
        }
    }

    func loadChatState() {
        do {
            chatSessions = try RustBridge.listChatSessions(storePath: storePath)
            if selectedChatSession == nil {
                selectedChatSession = chatSessions.first
            }
            if let selectedChatSession {
                chatMessages = try RustBridge.listChatMessages(storePath: storePath, sessionId: selectedChatSession.id)
                chatContextTraces = try RustBridge.listChatContextTraces(storePath: storePath, sessionId: selectedChatSession.id)
                derivedMemories = try RustBridge.listDerivedMemories(storePath: storePath, sessionId: selectedChatSession.id)
            }
        } catch {
            statusMessage = error.localizedDescription
        }
    }

    func startChat() {
        let storePath = self.storePath
        runTask(
            message: "Creating chat session…",
            operation: {
                let session = try RustBridge.createChatSession(storePath: storePath, title: "Memory chat")
                let sessions = try RustBridge.listChatSessions(storePath: storePath)
                return (session, sessions)
            },
            apply: { [weak self] session, sessions in
                guard let self else { return }
                self.selectedSection = .chat
                self.selectedChatSession = session
                self.chatSessions = sessions
                self.chatMessages = []
                self.chatContextTraces = []
                self.derivedMemories = []
                self.statusMessage = "Chat memory is active."
            }
        )
    }

    func selectChatSession(_ session: ChatSession) {
        selectedChatSession = session
        let storePath = self.storePath
        runTask(
            message: "Loading chat…",
            operation: {
                let messages = try RustBridge.listChatMessages(storePath: storePath, sessionId: session.id)
                let traces = try RustBridge.listChatContextTraces(storePath: storePath, sessionId: session.id)
                let derived = try RustBridge.listDerivedMemories(storePath: storePath, sessionId: session.id)
                return (messages, traces, derived)
            },
            apply: { [weak self] messages, traces, derived in
                guard let self else { return }
                self.chatMessages = messages
                self.chatContextTraces = traces
                self.derivedMemories = derived
                self.selectedSection = .chat
            }
        )
    }

    func sendChatTurn() {
        let trimmed = chatInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        let storePath = self.storePath
        let existingSession = self.selectedChatSession
        chatInput = ""
        runTask(
            message: "Routing chat through hot memory…",
            operation: {
                let session: ChatSession
                if let existingSession {
                    session = existingSession
                } else {
                    session = try RustBridge.createChatSession(storePath: storePath, title: "Memory chat")
                }
                let result = try RustBridge.sendChatTurn(
                    storePath: storePath,
                    request: ChatTurnRequest(sessionId: session.id, message: trimmed)
                )
                let messages = try RustBridge.listChatMessages(storePath: storePath, sessionId: session.id)
                let traces = try RustBridge.listChatContextTraces(storePath: storePath, sessionId: session.id)
                let derived = try RustBridge.listDerivedMemories(storePath: storePath, sessionId: session.id)
                let sessions = try RustBridge.listChatSessions(storePath: storePath)
                return (result, messages, traces, derived, sessions)
            },
            apply: { [weak self] result, messages, traces, derived, sessions in
                guard let self else { return }
                self.selectedSection = .chat
                self.selectedChatSession = result.session
                self.chatSessions = sessions
                self.chatMessages = messages
                self.chatContextTraces = traces
                self.derivedMemories = derived
                self.statusMessage = "Used \(result.contextTrace.snippets.count) memory snippets."
            }
        )
    }

    func connectWorkspace(_ kind: WorkspaceConnectionKind) {
        updateWorkspaceConnection(kind, status: .connecting)
        statusMessage = "\(kind.rawValue) is optional. Connect it when you want imprint to index that workspace."

        Task { [weak self] in
            try? await Task.sleep(nanoseconds: 700_000_000)
            await MainActor.run {
                self?.updateWorkspaceConnection(kind, status: .disconnected)
                self?.statusMessage = "\(kind.rawValue) connection is ready to configure when the connector is available."
            }
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
                self.summary = snapshot.summary
                self.snapshot = snapshot
                self.cortexAdapterState = result.adapterState
                self.refreshAdapterSnapshotQuietly()
                if let adapterState = result.adapterState {
                    self.statusMessage = "Imported \(result.importedCount), reused \(result.reusedEmbeddingCount), embedded \(result.embeddedCount). Adapter \(adapterState.freshness)."
                } else {
                    self.statusMessage = "Imported \(result.importedCount), reused \(result.reusedEmbeddingCount), embedded \(result.embeddedCount)."
                }
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
                self.summary = snapshot.summary
                self.snapshot = snapshot
                self.cortexAdapterState = result.adapterState
                self.refreshAdapterSnapshotQuietly()
                if let adapterState = result.adapterState {
                    self.statusMessage = "Memory rebuilt: reused \(result.reusedEmbeddingCount), embedded \(result.embeddedCount). Adapter \(adapterState.freshness)."
                } else {
                    self.statusMessage = "Memory rebuilt: reused \(result.reusedEmbeddingCount), embedded \(result.embeddedCount)."
                }
            }
        )
    }

    func compileMemoryBrain() {
        let storePath = self.storePath
        runTask(
            message: "Compiling cortex memory…",
            operation: {
                let result = try RustBridge.compileMemoryBrain(storePath: storePath)
                let snapshot = try RustBridge.loadCortexAdapterSnapshot(storePath: storePath)
                return (result, snapshot)
            },
            apply: { [weak self] result, snapshot in
                guard let self else { return }
                self.applyAdapterSnapshot(snapshot)
                self.cortexAdapterState = snapshot.adapterState ?? result.adapterState
                if let adapterState = self.cortexAdapterState {
                    self.statusMessage = "Compiled \(result.artifactsWritten) artifacts. Adapter \(adapterState.freshness)."
                } else {
                    self.statusMessage = "Compiled \(result.artifactsWritten) artifacts."
                }
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
                let adapterSnapshot = try RustBridge.loadCortexAdapterSnapshot(storePath: storePath)
                return (summary, snapshot, adapterSnapshot)
            },
            apply: { [weak self] summary, snapshot, adapterSnapshot in
                guard let self else { return }
                self.summary = summary
                self.snapshot = snapshot
                self.applyAdapterSnapshot(adapterSnapshot)
            }
        )
    }

    func retryLatestCortexAdapterJob() {
        guard let job = cortexAdapterJobs.first(where: { $0.status == "failed" || $0.status == "cancelled" || $0.status == "eval_failed" }) else {
            statusMessage = "No retryable cortex adapter job."
            return
        }
        let storePath = self.storePath
        runTask(
            message: "Queueing cortex adapter retry…",
            operation: {
                let retry = try RustBridge.retryCortexAdapterJob(storePath: storePath, jobId: job.id)
                let snapshot = try RustBridge.loadCortexAdapterSnapshot(storePath: storePath)
                return (retry, snapshot)
            },
            apply: { [weak self] retry, snapshot in
                guard let self else { return }
                self.applyAdapterSnapshot(snapshot)
                self.statusMessage = "Queued cortex adapter retry \(retry.id)."
            }
        )
    }

    func trainCortexAdapterNow() {
        let storePath = self.storePath
        runTask(
            message: "Queueing cortex adapter training...",
            operation: {
                let job = try RustBridge.trainCortexAdapterNow(storePath: storePath)
                let snapshot = try RustBridge.loadCortexAdapterSnapshot(storePath: storePath)
                return (job, snapshot)
            },
            apply: { [weak self] job, snapshot in
                guard let self else { return }
                self.applyAdapterSnapshot(snapshot)
                self.statusMessage = "Queued cortex adapter training \(job.id)."
            }
        )
    }

    func activateLastTrainedCortexAdapter() {
        let storePath = self.storePath
        runTask(
            message: "Activating trained cortex adapter...",
            operation: {
                let state = try RustBridge.activateLastTrainedCortexAdapter(storePath: storePath)
                let config = try RustBridge.loadModelConfig(storePath: storePath)
                let snapshot = try RustBridge.loadCortexAdapterSnapshot(storePath: storePath)
                return (state, config, snapshot)
            },
            apply: { [weak self] state, config, snapshot in
                guard let self else { return }
                self.modelConfig = config
                self.applyAdapterSnapshot(snapshot)
                self.cortexAdapterState = state
                self.statusMessage = "Activated cortex adapter."
            }
        )
    }

    func disableCortexAdapter() {
        let storePath = self.storePath
        runTask(
            message: "Disabling cortex adapter...",
            operation: {
                try RustBridge.disableCortexAdapter(storePath: storePath)
            },
            apply: { [weak self] config in
                guard let self else { return }
                self.modelConfig = config
                self.statusMessage = "Cortex adapter disabled for model calls."
            }
        )
    }

    func compareBaseVsAdaptedRouting() {
        let query = searchText.isEmpty ? "Which source family should answer this memory question?" : searchText
        let storePath = self.storePath
        runTask(
            message: "Probing adapted route selection...",
            operation: {
                try RustBridge.probeCortexAdapterRoute(storePath: storePath, query: query)
            },
            apply: { [weak self] result in
                guard let self else { return }
                self.cortexRouteProbe = result
                self.statusMessage = result.matched ? "Adapter route probe matched \(result.expectedSourceFamily)." : "Adapter route probe needs review."
            }
        )
    }

    private func refreshAdapterSnapshotQuietly() {
        if let adapterSnapshot = try? RustBridge.loadCortexAdapterSnapshot(storePath: storePath) {
            applyAdapterSnapshot(adapterSnapshot)
        }
    }

    private func applyAdapterSnapshot(_ snapshot: CortexAdapterSnapshot) {
        cortexAdapterState = snapshot.adapterState
        cortexAdapterJobs = snapshot.recentJobs
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
            plannerModel: self.modelConfig.plannerModel,
            responseModel: self.modelConfig.responseModel,
            plannerEndpoint: self.modelConfig.plannerEndpoint,
            plannerAdapterPath: self.modelConfig.plannerAdapterPath,
            responseAdapterPath: self.modelConfig.responseAdapterPath,
            sharedCortexAdapterPath: self.modelConfig.sharedCortexAdapterPath,
            activeAdapterHash: self.modelConfig.activeAdapterHash,
            adapterActivationPolicy: self.modelConfig.adapterActivationPolicy,
            runtimePreset: self.modelConfig.runtimePreset,
            cortexEnabled: self.modelConfig.cortexEnabled,
            cortexRounds: self.modelConfig.cortexRounds,
            criticModel: self.modelConfig.criticModel,
            criticEndpoint: self.modelConfig.criticEndpoint,
            compilerModel: self.modelConfig.compilerModel,
            embeddingModel: self.modelConfig.embeddingModel,
            embeddingEndpoint: self.modelConfig.embeddingEndpoint,
            embeddingRuntimePreset: self.modelConfig.embeddingRuntimePreset
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
            decoder.keyDecodingStrategy = .convertFromSnakeCase
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
        operationProgress = nil
    }

    private func updateWorkspaceConnection(_ kind: WorkspaceConnectionKind, status: WorkspaceConnectionStatus) {
        guard let index = workspaceConnections.firstIndex(where: { $0.kind == kind }) else { return }
        workspaceConnections[index].status = status
    }
}
