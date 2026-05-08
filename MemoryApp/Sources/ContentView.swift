import AppKit
import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @EnvironmentObject private var appState: AppState
    @Environment(\.memoryTheme) private var theme

    var body: some View {
        VStack(spacing: 0) {
            GraphiteTitleBar(pickFiles: pickFiles)
            DividerLine()
            HStack(spacing: 0) {
                GraphiteSidebar()
                    .frame(width: 208)
                DividerLine(axis: .vertical)
                CenterWorkspace()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                DividerLine(axis: .vertical)
                GraphiteInspector()
                    .frame(width: 320)
            }
        }
        .background(theme.surfaces.surface1)
        .foregroundStyle(theme.ink.primary)
        .environment(\.memoryTheme, ThemeCatalog.graphite)
        .background(WindowChromeConfigurator())
        .ignoresSafeArea(.container, edges: .top)
        .onDrop(of: [UTType.fileURL.identifier], isTargeted: nil, perform: handleDrop)
    }

    private func pickFiles() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = true
        if panel.runModal() == .OK {
            appState.importFiles(urls: panel.urls)
        }
    }

    private func handleDrop(_ providers: [NSItemProvider]) -> Bool {
        let collector = DropURLCollector()
        let group = DispatchGroup()
        for provider in providers where provider.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) {
            group.enter()
            provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { item, _ in
                defer { group.leave() }
                if let data = item as? Data,
                   let raw = String(data: data, encoding: .utf8),
                   let url = URL(string: raw) {
                    collector.append(url)
                } else if let url = item as? URL {
                    collector.append(url)
                }
            }
        }
        group.notify(queue: .main) {
            let urls = collector.values()
            if !urls.isEmpty {
                appState.importFiles(urls: urls)
            }
        }
        return true
    }
}

private final class DropURLCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: [URL] = []

    func append(_ url: URL) {
        lock.lock()
        stored.append(url)
        lock.unlock()
    }

    func values() -> [URL] {
        lock.lock()
        defer { lock.unlock() }
        return stored
    }
}

private struct GraphiteTitleBar: View {
    @EnvironmentObject private var appState: AppState
    @Environment(\.memoryTheme) private var theme
    let pickFiles: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            HStack(spacing: 10) {
                Text("imprint")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(theme.ink.secondary)
                Text("— \(theme.name.lowercased())")
                    .font(.system(size: 12))
                    .foregroundStyle(theme.ink.quaternary)
            }

            Spacer()

            HStack(spacing: 8) {
                Text("\(appState.summary.documents) docs · \(appState.summary.chunks) chunks")
                    .font(.system(size: 10.5, design: .monospaced))
                    .foregroundStyle(theme.ink.tertiary)
                    .lineLimit(1)
                DividerLine(axis: .vertical)
                    .frame(height: 14)
                    .padding(.horizontal, 2)
                TopBarActivityStatus(
                    isBusy: appState.isBusy,
                    progress: appState.operationProgress,
                    message: appState.statusMessage
                )
                IconButton("Import Files", systemImage: "square.and.arrow.down", action: pickFiles)
                    .disabled(appState.isBusy)
                IconButton("Rebuild", systemImage: "arrow.triangle.2.circlepath", action: appState.rebuild)
                    .disabled(appState.isBusy)
                IconButton("Compile Cortex", systemImage: "brain.head.profile", action: appState.compileMemoryBrain)
                    .disabled(appState.isBusy)
                IconButton("Refresh", systemImage: "arrow.clockwise", action: appState.refreshSnapshot)
                    .disabled(appState.isBusy)
            }
        }
        .frame(height: 44)
        .padding(.leading, 116)
        .padding(.trailing, 14)
        .background(theme.surfaces.chrome)
    }
}

private struct WindowChromeConfigurator: NSViewRepresentable {
    func makeNSView(context: Context) -> NSView {
        let view = NSView(frame: .zero)
        DispatchQueue.main.async {
            configure(window: view.window)
        }
        return view
    }

    func updateNSView(_ view: NSView, context: Context) {
        DispatchQueue.main.async {
            configure(window: view.window)
        }
    }

    private func configure(window: NSWindow?) {
        guard let window else { return }
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true
        window.styleMask.insert(.fullSizeContentView)
        window.isMovableByWindowBackground = true
    }
}

private struct GraphiteSidebar: View {
    @EnvironmentObject private var appState: AppState
    @Environment(\.memoryTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            VStack(alignment: .leading, spacing: 6) {
                SectionLabel("Store") {
                    Text("~/Library")
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundStyle(theme.ink.quaternary)
                }
                VStack(spacing: 2) {
                    ForEach(SidebarSection.allCases) { section in
                        SidebarSectionRow(
                            section: section,
                            count: count(for: section),
                            selected: appState.selectedSection == section
                        ) {
                            appState.selectedSection = section
                        }
                    }
                }
            }

            VStack(alignment: .leading, spacing: 6) {
                SectionLabel("Regions") {
                    Image(systemName: "plus")
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(theme.ink.quaternary)
                }
                VStack(spacing: 3) {
                    if let entries = appState.cortexIndex?.regions, !entries.isEmpty {
                        ForEach(Array(entries.prefix(8).enumerated()), id: \.element.id) { index, entry in
                            CortexRegionRow(entry: entry, count: regionCount(entry.regionId), tone: tone(for: index))
                        }
                    } else {
                        EmptyHint("Import files to build regions.")
                    }
                }
            }

            VStack(alignment: .leading, spacing: 8) {
                SectionLabel("Summary")
                SummaryLine(title: "Documents", value: appState.summary.documents)
                SummaryLine(title: "Chunks", value: appState.summary.chunks)
                SummaryLine(title: "Regions", value: appState.summary.regions)
                SummaryLine(title: "Links", value: appState.summary.links)
            }

            Spacer(minLength: 12)

            VStack(alignment: .leading, spacing: 8) {
                SectionLabel("Presence")
                HStack(spacing: 8) {
                    ActorBadge(kind: .human)
                    Text("on \(currentRegionLabel)")
                        .foregroundStyle(theme.ink.tertiary)
                }
                HStack(spacing: 8) {
                    ActorBadge(kind: .ai)
                    Text(currentAgentRegionLabel)
                        .foregroundStyle(theme.ink.tertiary)
                        .lineLimit(1)
                }
            }
            .font(.system(size: 11.5))
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 14)
        .background(theme.surfaces.surface2)
    }

    private var currentRegionLabel: String {
        appState.inspector.selectedNode?.label ?? "library"
    }

    private var currentAgentRegionLabel: String {
        if appState.isBusy, let label = appState.operationProgress?.activeNodeLabel, !label.isEmpty {
            return "on \(label)"
        }
        return appState.inspector.queryResult == nil ? "waiting" : "citing"
    }

    private func count(for section: SidebarSection) -> Int? {
        switch section {
        case .library: return appState.summary.documents
        case .chat: return appState.chatSessions.count
        case .sourceRecall: return appState.summary.chunks
        case .cortex: return appState.summary.regions
        case .surf: return appState.session.history.count
        case .importQueue: return appState.librarySnapshot?.importQueue.items.count
        case .connections:
            return appState.workspaceConnections.filter { $0.status == .connected }.count
        case .settings: return nil
        }
    }

    private func regionCount(_ regionID: String) -> Int {
        appState.snapshot?.nodes.filter { $0.regionId == regionID }.count ?? 0
    }

    private func tone(for index: Int) -> ThemeTone {
        [.human, .ai, .c, .d][index % 4]
    }
}

private struct CenterWorkspace: View {
    @EnvironmentObject private var appState: AppState
    @Environment(\.memoryTheme) private var theme

    var body: some View {
        if appState.selectedSection == .chat {
            ChatWorkspace()
        } else if appState.selectedSection == .library {
            LibraryWorkspace()
        } else if appState.selectedSection == .cortex {
            CortexWorkspace()
        } else if appState.selectedSection == .sourceRecall {
            mapWorkspace(title: "Source Recall", footer: "spatial recall map · click chunks and documents to inspect exact anchors")
        } else if appState.selectedSection == .surf {
            SurfWorkspace()
        } else if appState.selectedSection == .importQueue {
            ImportQueueWorkspace()
        } else if appState.selectedSection == .settings || appState.selectedSection == .connections {
            SettingsWorkspace()
        } else {
            mapWorkspace(title: "Semantic Map", footer: "2D map · click nodes to inspect")
        }
    }

    private func mapWorkspace(title: String, footer: String) -> some View {
        VStack(spacing: 0) {
            HStack(spacing: 10) {
                Text(title)
                    .font(.system(size: 12.5, weight: .medium))
                    .foregroundStyle(theme.ink.secondary)
                HStack(spacing: 8) {
                    Image(systemName: "magnifyingglass")
                        .font(.system(size: 13))
                        .foregroundStyle(theme.ink.tertiary)
                    TextField("How is rate limiting implemented across the gateway?", text: $appState.searchText)
                        .textFieldStyle(.plain)
                        .font(.system(size: 12.5))
                        .foregroundStyle(theme.ink.secondary)
                    Text("return to route")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(theme.ink.quaternary)
                }
                .padding(.horizontal, 10)
                .frame(height: 28)
                .background(theme.surfaces.surface2, in: RoundedRectangle(cornerRadius: 6, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 6, style: .continuous).stroke(theme.surfaces.rule, lineWidth: 0.5))

                GraphiteButton("Route", style: .primary, action: appState.performSearch)
                    .keyboardShortcut(.return, modifiers: [])
                    .disabled(appState.isBusy || appState.searchText.isEmpty)
                GraphiteButton("Backtrack", systemImage: "chevron.left", style: .ghost, action: appState.backtrack)
                    .disabled(appState.isBusy)
            }
            .padding(.horizontal, 16)
            .frame(height: 48)
            .background(theme.surfaces.surface1)

            DividerLine()

            SemanticCloudView(
                snapshot: appState.snapshot,
                selectedNode: appState.inspector.selectedNode,
                agentPresence: GraphAgentPresence.from(progress: appState.isBusy ? appState.operationProgress : nil)
            ) { node in
                appState.selectNode(node)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            DividerLine()

            HStack(spacing: 14) {
                Text(footer)
                    .font(.system(size: 11, design: .monospaced))
                Spacer()
                Text(appState.modelConfig.embeddingModel ?? "embedding model unset")
                    .font(.system(size: 11, design: .monospaced))
                DividerLine(axis: .vertical)
                    .frame(height: 12)
                HStack(spacing: 5) {
                    Circle()
                        .fill(statusColor)
                        .frame(width: 6, height: 6)
                    Text(appState.modelConfig.health?.status ?? "local")
                }
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(statusColor)
            }
            .foregroundStyle(theme.ink.tertiary)
            .padding(.horizontal, 14)
            .frame(height: 30)
            .background(theme.surfaces.surface2)
        }
    }

    private var statusColor: Color {
        appState.modelConfig.health?.status == "connected" ? theme.accents.success : theme.ink.tertiary
    }
}

private struct LibraryWorkspace: View {
    @EnvironmentObject private var appState: AppState
    @Environment(\.memoryTheme) private var theme

    private var documentNodes: [GraphNode] {
        (appState.snapshot?.nodes ?? [])
            .filter { $0.kind == .document }
            .sorted { $0.label.localizedCaseInsensitiveCompare($1.label) == .orderedAscending }
    }

    var body: some View {
        VStack(spacing: 0) {
            WorkspaceHeader(title: "Document Library", detail: "\(documentNodes.count) documents · \(appState.librarySnapshot?.sourceArtifacts.count ?? 0) source artifacts") {
                GraphiteButton("Refresh", systemImage: "arrow.clockwise", style: .ghost, action: appState.refreshSnapshot)
            }
            DividerLine()
            HStack(spacing: 0) {
                VStack(spacing: 0) {
                    LibraryTableHeader()
                    ScrollView {
                        LazyVStack(spacing: 0) {
                            if documentNodes.isEmpty {
                                EmptyTableState("Import local documents to browse source type, trust, freshness, collections, and original locations.")
                                    .padding(18)
                            } else {
                                ForEach(documentNodes) { node in
                                    LibraryDocumentRow(
                                        node: node,
                                        artifact: artifact(for: node),
                                        selected: appState.inspector.selectedNode?.id == node.id
                                    ) {
                                        appState.selectNode(node)
                                    }
                                }
                            }
                        }
                    }
                }
                .frame(minWidth: 520)

                DividerLine(axis: .vertical)

                VStack(alignment: .leading, spacing: 12) {
                    SectionLabel("Source Types")
                    if let filters = appState.librarySnapshot?.sourceTypeFilters, !filters.isEmpty {
                        ForEach(filters.prefix(8)) { filter in
                            SummaryLine(title: filter.sourceType, value: filter.count)
                        }
                    } else {
                        EmptyHint("No source type summary yet.")
                    }
                    InspectorDivider()
                    SectionLabel("Collections")
                    if let collections = appState.librarySnapshot?.collections, !collections.isEmpty {
                        ForEach(collections.prefix(8)) { collection in
                            VStack(alignment: .leading, spacing: 3) {
                                Text(collection.name)
                                    .font(.system(size: 12, weight: .medium))
                                Text(collection.description ?? collection.id)
                                    .font(.system(size: 10.5))
                                    .foregroundStyle(theme.ink.tertiary)
                                    .lineLimit(2)
                            }
                        }
                    } else {
                        EmptyHint("Collections and saved views will appear here.")
                    }
                }
                .padding(14)
                .frame(minWidth: 260, idealWidth: 260, maxWidth: 260, maxHeight: .infinity, alignment: .topLeading)
                .background(theme.surfaces.surface2)
            }
        }
    }

    private func artifact(for node: GraphNode) -> SourceArtifact? {
        guard let artifacts = appState.librarySnapshot?.sourceArtifacts else { return nil }
        if let exact = artifacts.first(where: { node.detail.contains($0.originalPath) || node.detail.contains($0.id) }) {
            return exact
        }
        return artifacts.first { artifact in
            let last = (artifact.originalPath as NSString).lastPathComponent
            return !last.isEmpty && node.label.localizedCaseInsensitiveContains(last)
        }
    }
}

private struct CortexWorkspace: View {
    @EnvironmentObject private var appState: AppState
    @Environment(\.memoryTheme) private var theme

    var body: some View {
        VStack(spacing: 0) {
            WorkspaceHeader(title: "Cortex", detail: cortexDetail) {
                GraphiteButton("Compile", systemImage: "brain.head.profile", style: .ghost, action: appState.compileMemoryBrain)
                GraphiteButton("Train", systemImage: "play.fill", style: .ghost, action: appState.trainCortexAdapterNow)
                GraphiteButton("Probe", systemImage: "arrow.left.arrow.right", style: .ghost, action: appState.compareBaseVsAdaptedRouting)
            }
            DividerLine()
            HStack(spacing: 0) {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 8) {
                        if let regions = appState.cortexIndex?.regions, !regions.isEmpty {
                            ForEach(regions) { region in
                                CortexRegionCard(region: region)
                            }
                        } else {
                            EmptyTableState("Compile the cortex to inspect region sketches, source-family hints, route examples, and source refs.")
                                .padding(18)
                        }
                    }
                    .padding(14)
                }

                DividerLine(axis: .vertical)

                ScrollView {
                    VStack(alignment: .leading, spacing: 12) {
                        AdapterStateSummary(
                            state: appState.cortexAdapterState,
                            jobs: appState.cortexAdapterJobs,
                            retryAction: appState.retryLatestCortexAdapterJob
                        )
                        if let index = appState.cortexIndex {
                            InspectorDivider()
                            DetailRow(label: "Corpus", value: shortHash(index.corpusHash))
                            DetailRow(label: "Schema", value: "\(index.schemaVersion)")
                            DetailRow(label: "Compiler", value: index.compiler)
                            DetailRow(label: "Created", value: timeLabel(index.createdAt))
                            DetailRow(label: "Sources", value: "\(index.sourceRefs.count)")
                            DetailRow(label: "Artifacts", value: "\(index.artifactIds.count)")
                        }
                        if let probe = appState.cortexRouteProbe {
                            InspectorDivider()
                            SectionLabel("Route Probe")
                            DetailRow(label: "Expected", value: probe.expectedSourceFamily)
                            DetailRow(label: "Matched", value: probe.matched ? "yes" : "no")
                            if let family = probe.modelSourceFamily {
                                DetailRow(label: "Model", value: family)
                            }
                        }
                    }
                    .padding(14)
                }
                .frame(width: 320)
                .background(theme.surfaces.surface2)
            }
        }
    }

    private var cortexDetail: String {
        guard let index = appState.cortexIndex else { return "No persisted cortex index" }
        return "\(index.regions.count) regions · corpus \(shortHash(index.corpusHash))"
    }
}

private struct SurfWorkspace: View {
    @EnvironmentObject private var appState: AppState
    @Environment(\.memoryTheme) private var theme

    var body: some View {
        HStack(spacing: 0) {
            VStack(spacing: 0) {
                WorkspaceHeader(title: "Surf Session", detail: "\(appState.session.history.count) steps · \(appState.session.visited.count) visited") {
                    GraphiteButton("Backtrack", systemImage: "chevron.left", style: .ghost, action: appState.backtrack)
                }
                DividerLine()
                SemanticCloudView(
                    snapshot: appState.snapshot,
                    selectedNode: appState.inspector.selectedNode,
                    agentPresence: GraphAgentPresence.from(progress: appState.isBusy ? appState.operationProgress : nil)
                ) { node in
                    appState.selectNode(node)
                }
            }

            DividerLine(axis: .vertical)

            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    SectionLabel("History")
                    if appState.session.history.isEmpty {
                        EmptyHint("Open nodes and follow links to build a reversible surf trail.")
                    } else {
                        ForEach(Array(appState.session.history.enumerated()), id: \.offset) { index, node in
                            DetailRow(label: "#\(index + 1)", value: node.id)
                        }
                    }
                    InspectorDivider()
                    SectionLabel("Neighborhood")
                    if appState.inspector.neighbors.isEmpty {
                        EmptyHint("Select a node to load semantic, citation, and same-source neighbors.")
                    } else {
                        ForEach(appState.inspector.neighbors.prefix(8)) { neighbor in
                            SurfNeighborRow(neighbor: neighbor)
                        }
                    }
                    InspectorDivider()
                    SectionLabel("Saved Trails")
                    if let trails = appState.librarySnapshot?.savedTrails, !trails.isEmpty {
                        ForEach(trails.prefix(6)) { trail in
                            DetailRow(label: trail.name, value: "\(trail.steps.count) steps")
                        }
                    } else {
                        EmptyHint("Saved trails will appear after cite/save workflows run.")
                    }
                }
                .padding(14)
            }
            .frame(width: 340)
            .background(theme.surfaces.surface2)
        }
    }
}

private struct ImportQueueWorkspace: View {
    @EnvironmentObject private var appState: AppState
    @Environment(\.memoryTheme) private var theme

    var body: some View {
        VStack(spacing: 0) {
            WorkspaceHeader(title: "Import Queue", detail: queueDetail) {
                GraphiteButton("Refresh", systemImage: "arrow.clockwise", style: .ghost, action: appState.refreshSnapshot)
            }
            DividerLine()
            ScrollView {
                LazyVStack(spacing: 8) {
                    if let queue = appState.librarySnapshot?.importQueue, !queue.items.isEmpty {
                        ForEach(queue.items) { item in
                            ImportQueueItemRow(item: item)
                        }
                    } else {
                        EmptyTableState("Queued, running, succeeded, failed, and cancelled imports will appear here with per-file progress.")
                            .padding(18)
                    }
                }
                .padding(14)
            }
            DividerLine()
            HStack(spacing: 16) {
                QueueMetric("Pending", appState.librarySnapshot?.importQueue.pending ?? 0)
                QueueMetric("Running", appState.librarySnapshot?.importQueue.running ?? 0)
                QueueMetric("Succeeded", appState.librarySnapshot?.importQueue.succeeded ?? 0)
                QueueMetric("Failed", appState.librarySnapshot?.importQueue.failed ?? 0)
                QueueMetric("Cancelled", appState.librarySnapshot?.importQueue.cancelled ?? 0)
                Spacer()
            }
            .padding(.horizontal, 14)
            .frame(height: 36)
            .background(theme.surfaces.surface2)
        }
    }

    private var queueDetail: String {
        guard let queue = appState.librarySnapshot?.importQueue else { return "No queue snapshot" }
        return "\(queue.items.count) files · \(queue.progressCompleted)/\(queue.progressTotal)"
    }
}

private struct SettingsWorkspace: View {
    @EnvironmentObject private var appState: AppState
    @Environment(\.memoryTheme) private var theme

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                WorkspaceHeader(title: "Settings", detail: "local-first storage, model endpoints, adapter policy, privacy") {
                    GraphiteButton("Save", systemImage: "checkmark", style: .primary, action: appState.saveModelConfig)
                    GraphiteButton("Test", systemImage: "bolt", style: .ghost, action: appState.testModelConnection)
                }
                SettingsBand(title: "Local Storage") {
                    DetailRow(label: "Store", value: appState.storePath)
                    DetailRow(label: "Mode", value: "local managed copies plus original path reconciliation")
                    DetailRow(label: "Artifacts", value: "\(appState.librarySnapshot?.sourceArtifacts.count ?? 0)")
                }
                SettingsBand(title: "Model Endpoints") {
                    LabeledField("Chat Endpoint", text: $appState.modelConfig.endpoint)
                    LabeledField("Planner Endpoint", text: stringBinding($appState.modelConfig.plannerEndpoint))
                    LabeledField("Embedding Endpoint", text: stringBinding($appState.modelConfig.embeddingEndpoint))
                    LabeledField("Response Model", text: stringBinding($appState.modelConfig.responseModel))
                    LabeledField("Compiler Model", text: stringBinding($appState.modelConfig.compilerModel))
                }
                SettingsBand(title: "Adapter Policy") {
                    Picker("Adapter Policy", selection: $appState.modelConfig.adapterActivationPolicy) {
                        Text("Automatic").tag("automatic")
                        Text("Manual").tag("manual")
                        Text("Disabled").tag("disabled")
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    Toggle("Cortex routing", isOn: $appState.modelConfig.cortexEnabled)
                        .toggleStyle(.switch)
                    Toggle("Latent recursion research", isOn: $appState.modelConfig.latentRecursiveEnabled)
                        .toggleStyle(.switch)
                }
                SettingsBand(title: "Privacy") {
                    PolicyLine(systemImage: "externaldrive", text: "Imports are stored in the local application support library.")
                    PolicyLine(systemImage: "lock", text: "Adapters and training manifests stay local unless a model endpoint is configured outside the machine.")
                    PolicyLine(systemImage: "eye.slash", text: "Exact claims still route back to source anchors before citation.")
                }
            }
            .padding(16)
        }
    }
}

private struct ChatWorkspace: View {
    @EnvironmentObject private var appState: AppState
    @Environment(\.memoryTheme) private var theme

    var body: some View {
        HStack(spacing: 0) {
            VStack(spacing: 0) {
                HStack(spacing: 10) {
                    Image(systemName: "flame")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(theme.accents.ai)
                    Text(appState.selectedChatSession?.title ?? "Memory chat")
                        .font(.system(size: 12.5, weight: .medium))
                        .foregroundStyle(theme.ink.primary)
                    Spacer()
                    Text("\(appState.chatMessages.count) turns")
                        .font(.system(size: 10.5, design: .monospaced))
                        .foregroundStyle(theme.ink.quaternary)
                    GraphiteButton("New", systemImage: "plus", style: .ghost, action: appState.startChat)
                }
                .padding(.horizontal, 14)
                .frame(height: 42)
                .background(theme.surfaces.surface1)

                DividerLine()

                SemanticCloudView(
                    snapshot: appState.snapshot,
                    selectedNode: appState.inspector.selectedNode,
                    traceHighlight: GraphTraceHighlight.from(trace: appState.chatContextTraces.first),
                    agentPresence: GraphAgentPresence.from(progress: appState.isBusy ? appState.operationProgress : nil)
                ) { node in
                    appState.selectNode(node)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)

                DividerLine()

                ContextStrip(trace: appState.chatContextTraces.first)
                    .frame(height: 112)
            }
            .frame(minWidth: 360)

            DividerLine(axis: .vertical)

            VStack(spacing: 0) {
                ChatTranscriptView()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                DividerLine()
                ChatComposer()
            }
            .frame(width: 390)
            .background(theme.surfaces.surface1)
        }
    }
}

private struct ChatTranscriptView: View {
    @EnvironmentObject private var appState: AppState
    @Environment(\.memoryTheme) private var theme

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 10) {
                    if appState.chatMessages.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            SectionLabel("Local Agent")
                            EmptyHint("Start a chat to write transcript memory, derive summaries, and keep recent turns hot for retrieval.")
                        }
                        .padding(16)
                    } else {
                        ForEach(appState.chatMessages) { message in
                            ChatBubble(message: message)
                                .id(message.id)
                        }
                    }
                }
                .padding(14)
            }
            .onChange(of: appState.chatMessages.count) {
                if let last = appState.chatMessages.last {
                    proxy.scrollTo(last.id, anchor: .bottom)
                }
            }
        }
    }
}

private struct ChatBubble: View {
    @Environment(\.memoryTheme) private var theme
    let message: ChatMessage

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 7) {
                Tag(message.role == .user ? "You" : "Agent", tone: message.role == .user ? .human : .ai)
                Text("\(message.tokenEstimate) tok")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(theme.ink.quaternary)
                Spacer()
            }
            Text(message.content)
                .font(.system(size: 12.5))
                .lineSpacing(3)
                .foregroundStyle(theme.ink.primary)
                .textSelection(.enabled)
            if let anchor = message.sourceAnchor {
                Text(anchor.path)
                    .font(.system(size: 9.5, design: .monospaced))
                    .foregroundStyle(theme.ink.quaternary)
                    .lineLimit(1)
            }
        }
        .padding(10)
        .background(message.role == .user ? theme.tags.human : theme.surfaces.surface2, in: RoundedRectangle(cornerRadius: 6, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 6, style: .continuous).stroke(theme.surfaces.rule, lineWidth: 0.5))
    }
}

private struct ChatComposer: View {
    @EnvironmentObject private var appState: AppState
    @Environment(\.memoryTheme) private var theme

    var body: some View {
        VStack(spacing: 8) {
            HStack(spacing: 8) {
                TextField("Ask imprint; every turn becomes memory", text: $appState.chatInput)
                    .textFieldStyle(.plain)
                    .font(.system(size: 12.5))
                    .padding(.horizontal, 10)
                    .frame(height: 34)
                    .background(theme.surfaces.surface2, in: RoundedRectangle(cornerRadius: 6, style: .continuous))
                    .overlay(RoundedRectangle(cornerRadius: 6, style: .continuous).stroke(theme.surfaces.rule, lineWidth: 0.5))
                    .onSubmit(appState.sendChatTurn)
                GraphiteButton("Send", systemImage: "paperplane.fill", style: .primary, action: appState.sendChatTurn)
                    .disabled(appState.isBusy || appState.chatInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
            HStack(spacing: 8) {
                Text(appState.modelConfig.runtimePreset.rawValue)
                DividerLine(axis: .vertical)
                    .frame(height: 12)
                Text(appState.modelConfig.responseModel ?? appState.modelConfig.chatModel ?? "response model unset")
                    .lineLimit(1)
                Spacer()
            }
            .font(.system(size: 10.5, design: .monospaced))
            .foregroundStyle(theme.ink.quaternary)
        }
        .padding(12)
        .background(theme.surfaces.surface1)
    }
}

private struct ContextStrip: View {
    @Environment(\.memoryTheme) private var theme
    let trace: ChatContextTrace?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionLabel("Context Used") {
                Text("\(trace?.snippets.count ?? 0)")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(theme.ink.quaternary)
            }
            if let trace, !trace.snippets.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(trace.snippets.prefix(6)) { snippet in
                            VStack(alignment: .leading, spacing: 4) {
                                HStack {
                                    Tag(snippet.sourceKind, tone: snippet.hotness > 0.7 ? .ai : .ink)
                                    Spacer()
                                    Text(String(format: "%.2f", snippet.hotness))
                                        .font(.system(size: 10, design: .monospaced))
                                        .foregroundStyle(theme.ink.quaternary)
                                }
                                Text(snippet.excerpt)
                                    .font(.system(size: 10.5))
                                    .lineLimit(3)
                                    .foregroundStyle(theme.ink.secondary)
                            }
                            .padding(8)
                            .frame(width: 210, height: 70, alignment: .topLeading)
                            .background(theme.surfaces.surface2, in: RoundedRectangle(cornerRadius: 5, style: .continuous))
                            .overlay(RoundedRectangle(cornerRadius: 5, style: .continuous).stroke(theme.surfaces.rule, lineWidth: 0.5))
                        }
                    }
                }
            } else {
                EmptyHint("The next response will show the exact transcript and source snippets used.")
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(theme.surfaces.surface2)
    }
}

private struct GraphiteInspector: View {
    @EnvironmentObject private var appState: AppState
    @Environment(\.memoryTheme) private var theme

    var body: some View {
        VStack(spacing: 0) {
            switch appState.selectedSection {
            case .library:
                LibraryInspector()
            case .chat:
                ChatInspector()
            case .sourceRecall:
                SourceRecallInspector()
            case .cortex:
                CortexInspector()
            case .surf:
                SurfInspector()
            case .importQueue:
                ImportQueueInspector()
            case .connections:
                ConnectionsInspector()
            case .settings:
                ModelInspector()
            }
        }
        .background(theme.surfaces.surface1)
    }
}

private struct SourceRecallInspector: View {
    @EnvironmentObject private var appState: AppState
    @Environment(\.memoryTheme) private var theme

    var body: some View {
        ScrollView {
            VStack(spacing: 0) {
                InspectorBlock {
                    SectionLabel("Selection") {
                        if let selected = appState.inspector.selectedNode {
                            Tag(selected.kind.rawValue, tone: tone(for: selected))
                        }
                    }
                    if let selected = appState.inspector.selectedNode {
                        Text(selected.label)
                            .font(.system(size: 19, weight: .medium))
                            .lineLimit(2)
                            .foregroundStyle(theme.ink.primary)
                        Text(selectionMeta(selected))
                            .font(.system(size: 10.5, design: .monospaced))
                            .foregroundStyle(theme.ink.quaternary)
                            .padding(.top, 1)
                        Text(selected.detail)
                            .font(.system(size: 12.5))
                            .lineSpacing(3)
                            .foregroundStyle(theme.ink.secondary)
                            .padding(.top, 8)
                    } else {
                        EmptyHint("Select a region, chunk, or document in the cloud to inspect it.")
                    }
                }

                InspectorDivider()

                InspectorBlock {
                    SectionLabel("Attention") {
                        if !appState.inspector.attentionMarks.isEmpty {
                            Text("\(activeAttentionCount)")
                                .font(.system(size: 10, design: .monospaced))
                                .foregroundStyle(theme.ink.quaternary)
                        }
                    }
                    if appState.inspector.selectedNode == nil {
                        EmptyHint("Select a node to pin, promote, suppress, or set hotness.")
                    } else {
                        HStack(spacing: 6) {
                            GraphiteButton("Pin", systemImage: "pin", style: .ghost) { appState.applySelectionAttention(.pin) }
                            GraphiteButton("Promote", systemImage: "arrow.up", style: .ghost) { appState.applySelectionAttention(.promote) }
                            GraphiteButton("Suppress", systemImage: "eye.slash", style: .ghost) { appState.applySelectionAttention(.suppress) }
                        }
                        HStack(spacing: 6) {
                            GraphiteButton("Hot", systemImage: "flame", style: .ghost) { appState.applySelectionAttention(.hot) }
                            GraphiteButton("Warm", systemImage: "thermometer.medium", style: .ghost) { appState.applySelectionAttention(.warm) }
                            GraphiteButton("Cold", systemImage: "snowflake", style: .ghost) { appState.applySelectionAttention(.cold) }
                        }
                        .padding(.top, 4)
                        .disabled(appState.isBusy)
                        Text(selectedAttentionTargetLine)
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(theme.ink.quaternary)
                            .lineLimit(2)
                            .padding(.top, 4)
                        if appState.inspector.attentionMarks.isEmpty {
                            EmptyHint("No attention marks on this target yet.")
                                .padding(.top, 4)
                        } else {
                            VStack(spacing: 5) {
                                ForEach(appState.inspector.attentionMarks.prefix(6)) { mark in
                                    AttentionMarkRow(mark: mark) {
                                        appState.revertAttentionMark(mark)
                                    }
                                }
                            }
                            .padding(.top, 6)
                        }
                    }
                }

                InspectorDivider()

                InspectorBlock {
                    SectionLabel("Excerpt")
                    if let excerpt = appState.inspector.excerpt, !excerpt.isEmpty {
                        HighlightedText(excerpt)
                            .textSelection(.enabled)
                    } else {
                        EmptyHint("No excerpt loaded yet.")
                    }
                    if let anchor = appState.inspector.sourceAnchor {
                        Text(anchorLine(anchor))
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(theme.ink.quaternary)
                            .lineLimit(3)
                            .padding(.top, 6)
                    }
                    if let target = appState.inspector.openTarget {
                        Text(openTargetLine(target))
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(theme.ink.quaternary)
                            .lineLimit(3)
                    }
                    if appState.inspector.sourceAnchor != nil || appState.inspector.openTarget != nil {
                        provenanceInspector(anchor: appState.inspector.sourceAnchor, target: appState.inspector.openTarget)
                            .padding(.top, 6)
                    }
                    if appState.inspector.openTarget != nil {
                        GraphiteButton("Open Source", systemImage: "arrow.up.right.square", style: .ghost, action: appState.openSelectedSource)
                            .padding(.top, 4)
                    }
                    HStack(spacing: 8) {
                        TextField("Grep or search selection", text: $appState.grepText)
                            .textFieldStyle(.plain)
                            .font(.system(size: 12))
                            .padding(.horizontal, 8)
                            .frame(height: 26)
                            .background(theme.surfaces.surface2, in: RoundedRectangle(cornerRadius: 5, style: .continuous))
                            .overlay(RoundedRectangle(cornerRadius: 5, style: .continuous).stroke(theme.surfaces.rule, lineWidth: 0.5))
                        GraphiteButton("Grep", style: .ghost, action: appState.grepSelectedDocument)
                        GraphiteButton("Semantic", style: .ghost, action: appState.semanticSearchSelectedDocument)
                    }
                    .padding(.top, 8)
                    if let selected = appState.inspector.selectedNode, selected.kind == .chunk {
                        HStack(spacing: 8) {
                            GraphiteButton("Window", style: .ghost) { appState.expandSelectedChunk(mode: .window) }
                            GraphiteButton("Page", style: .ghost) { appState.expandSelectedChunk(mode: .page) }
                            GraphiteButton("Section", style: .ghost) { appState.expandSelectedChunk(mode: .section) }
                            GraphiteButton("Document", style: .ghost) { appState.expandSelectedChunk(mode: .document) }
                        }
                        .padding(.top, 6)
                    }
                }

                InspectorDivider()

                InspectorBlock {
                    SectionLabel("Chunks") {
                        if !appState.inspector.passages.isEmpty {
                            Text("\(appState.inspector.passages.count)")
                                .font(.system(size: 10, design: .monospaced))
                                .foregroundStyle(theme.ink.quaternary)
                        }
                    }
                    if appState.inspector.passages.isEmpty {
                        EmptyHint("No chunks loaded for this selection.")
                    } else {
                        VStack(spacing: 6) {
                            ForEach(appState.inspector.passages.prefix(8)) { passage in
                                PassageRow(passage: passage)
                            }
                        }
                    }
                }

                InspectorDivider()

                InspectorBlock {
                    SectionLabel("Trails") {
                        Text("last \(min(appState.inspector.breadcrumbs.count, 10))")
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(theme.ink.quaternary)
                    }
                    TrailView(labels: Array(appState.inspector.breadcrumbs.suffix(5)), color: theme.accents.human)
                    if let query = appState.inspector.queryResult {
                        Text(query.routed.rationale)
                            .font(.system(size: 11.5))
                            .foregroundStyle(theme.ink.secondary)
                            .lineLimit(4)
                            .padding(.top, 8)
                    }
                }

                InspectorDivider()

                InspectorBlock {
                    SectionLabel("Ranking")
                    if let query = appState.inspector.queryResult {
                        VStack(alignment: .leading, spacing: 6) {
                            ForEach(query.routed.routePlan.candidates.prefix(3)) { candidate in
                                VStack(alignment: .leading, spacing: 2) {
                                    HStack {
                                        Text(candidate.label)
                                            .font(.system(size: 11.5, weight: .medium))
                                            .foregroundStyle(theme.ink.primary)
                                            .lineLimit(1)
                                        Spacer()
                                        Text(String(format: "%.2f", candidate.score))
                                            .font(.system(size: 10, design: .monospaced))
                                            .foregroundStyle(theme.ink.quaternary)
                                    }
                                    Text(candidate.reason)
                                        .font(.system(size: 10.5))
                                        .foregroundStyle(theme.ink.tertiary)
                                        .lineLimit(3)
                                }
                            }
                            ForEach(query.hits.prefix(3)) { hit in
                                DetailRow(label: "Hit", value: "\(hit.chunkId) · score \(String(format: "%.2f", hit.score))")
                            }
                        }
                    } else {
                        EmptyHint("Run a route to see why regions and source hits ranked.")
                    }
                }

                InspectorDivider()

                InspectorBlock {
                    SectionLabel("Nearby")
                    if appState.inspector.neighbors.isEmpty {
                        EmptyHint("No neighbors loaded.")
                    } else {
                        VStack(spacing: 4) {
                            ForEach(appState.inspector.neighbors.prefix(8)) { neighbor in
                                VStack(alignment: .leading, spacing: 3) {
                                    HStack {
                                        Text(neighbor.label)
                                            .font(.system(size: 11.5, weight: .medium))
                                            .foregroundStyle(theme.ink.primary)
                                            .lineLimit(1)
                                        Spacer()
                                        Text(String(format: "%.2f", neighbor.score))
                                            .font(.system(size: 10, design: .monospaced))
                                            .foregroundStyle(theme.ink.quaternary)
                                    }
                                    Text(neighbor.excerpt)
                                        .font(.system(size: 10.5))
                                        .foregroundStyle(theme.ink.tertiary)
                                        .lineLimit(2)
                                }
                                .padding(8)
                                .background(theme.surfaces.surface2, in: RoundedRectangle(cornerRadius: 5, style: .continuous))
                            }
                        }
                    }
                }

                InspectorDivider()

                InspectorBlock {
                    SectionLabel("Links from here")
                    if appState.inspector.links.isEmpty {
                        EmptyHint("No links loaded.")
                    } else {
                        VStack(spacing: 4) {
                            ForEach(appState.inspector.links.prefix(8)) { link in
                                LinkRow(link: link)
                            }
                        }
                    }
                }

                InspectorDivider()

                InspectorBlock {
                    SectionLabel("Passage Hits")
                    if appState.inspector.grepHits.isEmpty {
                        EmptyHint("Run grep or semantic search to inspect precise passages.")
                    } else {
                        VStack(alignment: .leading, spacing: 8) {
                            ForEach(appState.inspector.grepHits.prefix(6)) { hit in
                                VStack(alignment: .leading, spacing: 3) {
                                    Text(hit.excerpt)
                                        .font(.system(size: 11.5))
                                        .foregroundStyle(theme.ink.primary)
                                        .lineLimit(4)
                                    Text("\(hit.start)-\(hit.end)")
                                        .font(.system(size: 10, design: .monospaced))
                                        .foregroundStyle(theme.ink.quaternary)
                                }
                                .padding(8)
                                .background(theme.surfaces.surface2, in: RoundedRectangle(cornerRadius: 5, style: .continuous))
                            }
                        }
                    }
                }
            }
        }
    }

    private func selectionMeta(_ selected: GraphNode) -> String {
        "\(selected.id) · score \(String(format: "%.2f", selected.score))"
    }

    private func tone(for node: GraphNode) -> ThemeTone {
        switch node.kind {
        case .region: return .ai
        case .document: return .human
        case .chunk: return .d
        }
    }

    private var activeAttentionCount: Int {
        appState.inspector.attentionMarks.filter { $0.revertedAt == nil }.count
    }

    private var selectedAttentionTargetLine: String {
        guard let node = appState.inspector.selectedNode else { return "" }
        switch node.nodeRef {
        case .document(let id):
            return "document · \(id)"
        case .chunk(let id):
            return "chunk · \(id)"
        case .region(let id):
            return "region · \(id)"
        }
    }

    private func anchorLine(_ anchor: SourceAnchor) -> String {
        var parts = [anchor.path]
        if let page = anchor.page {
            parts.append("p.\(page)")
        }
        if let section = anchor.section {
            parts.append(section)
        }
        if !anchor.sectionHierarchy.isEmpty {
            parts.append(anchor.sectionHierarchy.joined(separator: " > "))
        }
        if let paragraph = anchor.paragraphIndex {
            parts.append("para \(paragraph)")
        }
        if let byteStart = anchor.byteStart, let byteEnd = anchor.byteEnd {
            parts.append("bytes \(byteStart)-\(byteEnd)")
        }
        parts.append("\(anchor.start)-\(anchor.end)")
        return parts.joined(separator: " · ")
    }

    private func openTargetLine(_ target: SourceOpenTarget) -> String {
        var parts = [target.kind.rawValue, target.locationHint]
        if let path = target.path {
            parts.append(path)
        } else if let url = target.browserUrl {
            parts.append(url)
        }
        if let selection = target.pdfSelection {
            parts.append("pdf text \(selection.textStart)-\(selection.textEnd)")
            if selection.boundingBox != nil {
                parts.append("bbox")
            }
        }
        if let email = target.emailLocation {
            parts.append("thread \(email.threadId)")
        }
        parts.append(target.sourceTrust.label)
        return parts.joined(separator: " · ")
    }

    private func shortHash(_ hash: String) -> String {
        String(hash.prefix(12))
    }

    @ViewBuilder
    private func provenanceInspector(anchor: SourceAnchor?, target: SourceOpenTarget?) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            SectionLabel("Provenance")
            if let target {
                DetailRow(label: "Trust", value: "\(target.sourceTrust.label) · \(String(format: "%.2f", target.sourceTrust.score))")
                if let freshness = target.sourceFreshness {
                    DetailRow(label: "Freshness", value: freshnessLine(freshness))
                    if let warning = freshness.staleWarning {
                        DetailRow(label: "Warning", value: warning)
                    }
                    if freshness.refreshNeeded, let url = freshness.recaptureUrl {
                        DetailRow(label: "Refresh", value: url)
                    }
                }
                if target.isDerived {
                    DetailRow(label: "Truth", value: "Derived artifact, expand originals before citing")
                }
                if let caveat = target.caveat {
                    DetailRow(label: "Caveat", value: caveat)
                }
            }
            if let anchor {
                DetailRow(label: "Hash", value: shortHash(anchor.contentHash))
                DetailRow(label: "Parser", value: "\(anchor.parserVersion)")
                if let artifact = anchor.sourceArtifactId {
                    DetailRow(label: "Artifact", value: artifact)
                }
                if let rendered = anchor.renderedPage {
                    DetailRow(label: "Rendered", value: "page \(rendered.label ?? "\(rendered.page)")")
                }
                if let email = anchor.emailLocation {
                    DetailRow(label: "Email", value: email.subject ?? email.threadId)
                }
            }
        }
    }

    private func freshnessLine(_ freshness: SourceFreshnessPolicy) -> String {
        var parts = [freshness.status.rawValue]
        if let retrievedAt = freshness.retrievedAt {
            parts.append("retrieved \(freshnessTimeLabel(retrievedAt))")
        }
        if let expiresAt = freshness.freshnessExpiresAt {
            parts.append("expires \(freshnessTimeLabel(expiresAt))")
        }
        return parts.joined(separator: " · ")
    }

    private func freshnessTimeLabel(_ millis: UInt64) -> String {
        let date = Date(timeIntervalSince1970: TimeInterval(millis) / 1000)
        return date.formatted(date: .abbreviated, time: .shortened)
    }
}

private struct LibraryInspector: View {
    @EnvironmentObject private var appState: AppState
    @Environment(\.memoryTheme) private var theme

    var body: some View {
        ScrollView {
            VStack(spacing: 0) {
                InspectorBlock {
                    SectionLabel("Library")
                    SummaryLine(title: "Documents", value: appState.summary.documents)
                    SummaryLine(title: "Chunks", value: appState.summary.chunks)
                    SummaryLine(title: "Regions", value: appState.summary.regions)
                    SummaryLine(title: "Links", value: appState.summary.links)
                    SummaryLine(title: "Cortex Bytes", value: appState.summary.mapBytes)
                }
                if let importResult = appState.importResult {
                    InspectorDivider()
                    InspectorBlock {
                        SectionLabel("Recent Import")
                        Text("\(importResult.importedCount) new, \(importResult.replacedCount) replaced, \(importResult.skippedCount) skipped")
                            .font(.system(size: 12))
                            .foregroundStyle(theme.ink.secondary)
                        Text("\(importResult.reusedEmbeddingCount) reused, \(importResult.embeddedCount) embedded")
                            .font(.system(size: 12))
                            .foregroundStyle(theme.ink.secondary)
                        ForEach((importResult.importedPaths + importResult.replacedPaths).prefix(5), id: \.self) { path in
                            Text(path)
                                .font(.system(size: 11))
                                .foregroundStyle(theme.ink.tertiary)
                                .lineLimit(2)
                        }
                    }
                }
                InspectorDivider()
                InspectorBlock {
                    SectionLabel("Navigation Trail")
                    if appState.inspector.breadcrumbs.isEmpty {
                        EmptyHint("No trail yet.")
                    } else {
                        TrailView(labels: Array(appState.inspector.breadcrumbs.suffix(6)), color: theme.accents.human)
                    }
                }
            }
        }
    }
}

private struct CortexInspector: View {
    @EnvironmentObject private var appState: AppState
    @Environment(\.memoryTheme) private var theme

    var body: some View {
        ScrollView {
            VStack(spacing: 0) {
                InspectorBlock {
                    SectionLabel("Cortex Index") {
                        Tag(appState.cortexAdapterState?.activationStatus ?? "inactive", tone: .ai)
                    }
                    if let index = appState.cortexIndex {
                        DetailRow(label: "Corpus", value: shortHash(index.corpusHash))
                        DetailRow(label: "Schema", value: "\(index.schemaVersion)")
                        DetailRow(label: "Regions", value: "\(index.regions.count)")
                        DetailRow(label: "Sources", value: "\(index.sourceRefs.count)")
                        DetailRow(label: "Artifacts", value: "\(index.artifactIds.count)")
                    } else {
                        EmptyHint("Compile the cortex to inspect semantic-address sketches.")
                    }
                }
                InspectorDivider()
                InspectorBlock {
                    AdapterStateSummary(
                        state: appState.cortexAdapterState,
                        jobs: appState.cortexAdapterJobs,
                        retryAction: appState.retryLatestCortexAdapterJob
                    )
                }
                InspectorDivider()
                InspectorBlock {
                    SectionLabel("Route Examples")
                    if let examples = appState.cortexIndex?.regions.flatMap(\.routeExamples), !examples.isEmpty {
                        ForEach(examples.prefix(8), id: \.self) { example in
                            Text(example)
                                .font(.system(size: 10.5, design: .monospaced))
                                .foregroundStyle(theme.ink.secondary)
                                .lineLimit(2)
                        }
                    } else {
                        EmptyHint("No route examples in the current cortex index.")
                    }
                }
            }
        }
    }
}

private struct SurfInspector: View {
    @EnvironmentObject private var appState: AppState

    var body: some View {
        ScrollView {
            VStack(spacing: 0) {
                InspectorBlock {
                    SectionLabel("Session")
                    DetailRow(label: "Current", value: appState.session.current?.id ?? "none")
                    SummaryLine(title: "History", value: appState.session.history.count)
                    SummaryLine(title: "Visited", value: appState.session.visited.count)
                    GraphiteButton("Backtrack", systemImage: "chevron.left", style: .ghost, action: appState.backtrack)
                }
                InspectorDivider()
                InspectorBlock {
                    SectionLabel("Expand Source")
                    if let selected = appState.inspector.selectedNode, selected.kind == .chunk {
                        HStack(spacing: 8) {
                            GraphiteButton("Window", style: .ghost) { appState.expandSelectedChunk(mode: .window) }
                            GraphiteButton("Page", style: .ghost) { appState.expandSelectedChunk(mode: .page) }
                        }
                        HStack(spacing: 8) {
                            GraphiteButton("Section", style: .ghost) { appState.expandSelectedChunk(mode: .section) }
                            GraphiteButton("Document", style: .ghost) { appState.expandSelectedChunk(mode: .document) }
                        }
                    } else {
                        EmptyHint("Select a chunk in the map to expand exact source context.")
                    }
                }
                InspectorDivider()
                InspectorBlock {
                    SectionLabel("Saved Trails")
                    if let trails = appState.librarySnapshot?.savedTrails, !trails.isEmpty {
                        ForEach(trails.prefix(8)) { trail in
                            DetailRow(label: trail.name, value: "\(trail.steps.count) steps")
                        }
                    } else {
                        EmptyHint("Saved surf trails are not present yet.")
                    }
                }
            }
        }
    }
}

private struct ImportQueueInspector: View {
    @EnvironmentObject private var appState: AppState

    var body: some View {
        ScrollView {
            VStack(spacing: 0) {
                InspectorBlock {
                    SectionLabel("Queue")
                    SummaryLine(title: "Pending", value: appState.librarySnapshot?.importQueue.pending ?? 0)
                    SummaryLine(title: "Running", value: appState.librarySnapshot?.importQueue.running ?? 0)
                    SummaryLine(title: "Succeeded", value: appState.librarySnapshot?.importQueue.succeeded ?? 0)
                    SummaryLine(title: "Failed", value: appState.librarySnapshot?.importQueue.failed ?? 0)
                    SummaryLine(title: "Cancelled", value: appState.librarySnapshot?.importQueue.cancelled ?? 0)
                }
                InspectorDivider()
                InspectorBlock {
                    SectionLabel("Watch Roots")
                    if let roots = appState.librarySnapshot?.watchRoots, !roots.isEmpty {
                        ForEach(roots.prefix(8)) { root in
                            DetailRow(label: root.recursive ? "Recursive" : "Folder", value: root.path)
                        }
                    } else {
                        EmptyHint("No watched folders configured.")
                    }
                }
                InspectorDivider()
                InspectorBlock {
                    SectionLabel("Updates")
                    if let updates = appState.librarySnapshot?.updates, !updates.isEmpty {
                        ForEach(updates.prefix(8)) { update in
                            DetailRow(label: update.kind.rawValue, value: update.candidatePath ?? update.originalPath)
                        }
                    } else {
                        EmptyHint("No changed, moved, deleted, duplicate, or replaced files detected.")
                    }
                }
            }
        }
    }
}

private struct ChatInspector: View {
    @EnvironmentObject private var appState: AppState
    @Environment(\.memoryTheme) private var theme

    var body: some View {
        ScrollView {
            VStack(spacing: 0) {
                InspectorBlock {
                    SectionLabel("Sessions") {
                        IconButton("New Chat", systemImage: "plus", action: appState.startChat)
                    }
                    if appState.chatSessions.isEmpty {
                        EmptyHint("No chat sessions yet.")
                    } else {
                        VStack(spacing: 4) {
                            ForEach(appState.chatSessions.prefix(8)) { session in
                                Button {
                                    appState.selectChatSession(session)
                                } label: {
                                    HStack(spacing: 8) {
                                        Circle()
                                            .fill(session.id == appState.selectedChatSession?.id ? theme.accents.ai : theme.ink.quaternary)
                                            .frame(width: 7, height: 7)
                                        VStack(alignment: .leading, spacing: 2) {
                                            Text(session.title)
                                                .font(.system(size: 12, weight: .medium))
                                                .lineLimit(1)
                                            Text(String(format: "hot %.2f", session.hotness))
                                                .font(.system(size: 9.5, design: .monospaced))
                                                .foregroundStyle(theme.ink.quaternary)
                                        }
                                        Spacer()
                                    }
                                    .foregroundStyle(theme.ink.secondary)
                                    .padding(8)
                                    .background(theme.surfaces.surface2, in: RoundedRectangle(cornerRadius: 5, style: .continuous))
                                }
                                .buttonStyle(.plain)
                            }
                        }
                    }
                }

                InspectorDivider()

                InspectorBlock {
                    SectionLabel("Derived Memories") {
                        Text("\(appState.derivedMemories.count)")
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(theme.ink.quaternary)
                    }
                    if appState.derivedMemories.isEmpty {
                        EmptyHint("Summaries, facts, decisions, and tasks will appear here after chat turns.")
                    } else {
                        VStack(alignment: .leading, spacing: 6) {
                            ForEach(appState.derivedMemories.prefix(8)) { memory in
                                VStack(alignment: .leading, spacing: 5) {
                                    HStack {
                                        Tag(memory.kind.rawValue, tone: tone(for: memory.kind))
                                        Spacer()
                                        Text(String(format: "%.2f", memory.confidence))
                                            .font(.system(size: 10, design: .monospaced))
                                            .foregroundStyle(theme.ink.quaternary)
                                    }
                                    Text(memory.text)
                                        .font(.system(size: 11.5))
                                        .lineLimit(4)
                                        .foregroundStyle(theme.ink.secondary)
                                }
                                .padding(8)
                                .background(theme.surfaces.surface2, in: RoundedRectangle(cornerRadius: 5, style: .continuous))
                            }
                        }
                    }
                }

                InspectorDivider()

                InspectorBlock {
                    SectionLabel("Tool Trace")
                    if let trace = appState.chatContextTraces.first {
                        VStack(alignment: .leading, spacing: 5) {
                            ForEach(trace.toolTrace, id: \.self) { line in
                                Text(line)
                                    .font(.system(size: 11, design: .monospaced))
                                    .foregroundStyle(theme.ink.tertiary)
                            }
                        }
                    } else {
                        EmptyHint("No routing trace yet.")
                    }
                }
            }
        }
    }

    private func tone(for kind: DerivedMemoryKind) -> ThemeTone {
        switch kind {
        case .summary: return .ai
        case .decision: return .d
        case .task: return .c
        case .fact: return .human
        }
    }
}

private struct ConnectionsInspector: View {
    @EnvironmentObject private var appState: AppState
    @Environment(\.memoryTheme) private var theme

    var body: some View {
        ScrollView {
            VStack(spacing: 0) {
                InspectorBlock {
                    SectionLabel("Workspace Connections") {
                        Tag("Optional", tone: .d)
                    }
                    Text("imprint works with local files and chat memory without workspace access. Add Slack, email, or calendar only when that source should become part of the library.")
                        .font(.system(size: 12.5))
                        .lineSpacing(3)
                        .foregroundStyle(theme.ink.secondary)
                }

                InspectorDivider()

                InspectorBlock {
                    VStack(spacing: 8) {
                        ForEach(appState.workspaceConnections) { connection in
                            WorkspaceConnectionRow(connection: connection) {
                                appState.connectWorkspace(connection.kind)
                            }
                        }
                    }
                }

                InspectorDivider()

                InspectorBlock {
                    SectionLabel("Access Policy")
                    VStack(alignment: .leading, spacing: 6) {
                        PolicyLine(systemImage: "checkmark.circle", text: "No email or calendar permission is needed to import files, browse memory, search, or chat.")
                        PolicyLine(systemImage: "lock", text: "Disconnected workspace sources stay out of retrieval and indexing.")
                        PolicyLine(systemImage: "arrow.clockwise", text: "Connections can be added later without rebuilding the rest of the library.")
                    }
                }
            }
        }
    }
}

private struct WorkspaceConnectionRow: View {
    @Environment(\.memoryTheme) private var theme
    let connection: WorkspaceConnection
    let action: () -> Void

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: icon)
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(iconColor)
                .frame(width: 24, height: 24)
                .background(theme.surfaces.surface3, in: RoundedRectangle(cornerRadius: 5, style: .continuous))

            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(connection.kind.rawValue)
                        .font(.system(size: 12.5, weight: .medium))
                        .foregroundStyle(theme.ink.primary)
                    Tag(connection.statusLabel, tone: tagTone)
                }
                Text(detail)
                    .font(.system(size: 11))
                    .foregroundStyle(theme.ink.tertiary)
                    .lineLimit(2)
            }

            Spacer()

            GraphiteButton(connection.actionTitle, systemImage: buttonIcon, style: .ghost, action: action)
                .disabled(connection.status == .connecting)
        }
        .padding(9)
        .background(theme.surfaces.surface2, in: RoundedRectangle(cornerRadius: 6, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 6, style: .continuous).stroke(theme.surfaces.rule, lineWidth: 0.5))
    }

    private var icon: String {
        switch connection.kind {
        case .slack: return "bubble.left.and.bubble.right"
        case .email: return "envelope"
        case .calendar: return "calendar"
        }
    }

    private var buttonIcon: String {
        connection.status == .connected ? "slider.horizontal.3" : "plus"
    }

    private var iconColor: Color {
        connection.status == .connected ? theme.accents.success : theme.ink.tertiary
    }

    private var tagTone: ThemeTone {
        switch connection.status {
        case .connected: return .d
        case .connecting: return .ai
        case .disconnected: return .ink
        }
    }

    private var detail: String {
        switch connection.kind {
        case .slack:
            return "Search and cite channels only after a workspace is connected."
        case .email:
            return "Mail remains private unless this source is explicitly connected."
        case .calendar:
            return "Events are optional context, not a requirement for memory."
        }
    }
}

private struct PolicyLine: View {
    @Environment(\.memoryTheme) private var theme
    let systemImage: String
    let text: String

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: systemImage)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(theme.accents.success)
                .frame(width: 14)
            Text(text)
                .font(.system(size: 11.5))
                .lineSpacing(2)
                .foregroundStyle(theme.ink.secondary)
        }
    }
}

private struct AdapterStateSummary: View {
    @Environment(\.memoryTheme) private var theme
    let state: CortexAdapterState?
    let jobs: [CortexAdapterJob]
    let retryAction: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Cortex Adapter")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(theme.ink.tertiary)
                Spacer()
                Tag(state?.freshness ?? "not compiled", tone: tone)
            }

            if let state {
                DetailRow(label: "Data", value: state.dataFreshness)
                DetailRow(label: "Training", value: state.trainingStatus)
                DetailRow(label: "Activation", value: state.activationStatus)
                if let baseModel = state.baseModel {
                    DetailRow(label: "Base", value: baseModel)
                }
                DetailRow(label: "Source Hash", value: shortHash(state.currentSourceDatasetHash))
                if let trainedSourceDatasetHash = state.trainedSourceDatasetHash {
                    DetailRow(label: "Trained Hash", value: shortHash(trainedSourceDatasetHash))
                }
                if let preparedDatasetHash = state.preparedDatasetHash {
                    DetailRow(label: "Prepared Hash", value: shortHash(preparedDatasetHash))
                }
                if let evalScore = state.evalScore {
                    DetailRow(label: "Eval", value: String(format: "%.3f", evalScore))
                }
                if let lastSuccessfulTrainingAt = state.lastSuccessfulTrainingAt {
                    DetailRow(label: "Last Train", value: timeLabel(lastSuccessfulTrainingAt))
                }
                if let activatedAt = state.activatedAt {
                    DetailRow(label: "Activated", value: timeLabel(activatedAt))
                }
                if let reason = state.reason {
                    Text(reason)
                        .font(.system(size: 11.5))
                        .lineSpacing(2)
                        .foregroundStyle(theme.ink.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if let failureReason = state.failureReason {
                    Text(failureReason)
                        .font(.system(size: 11.5))
                        .lineSpacing(2)
                        .foregroundStyle(theme.accents.d)
                        .fixedSize(horizontal: false, vertical: true)
                }
            } else {
                Text("No persisted adapter freshness for this library.")
                    .font(.system(size: 11.5))
                    .foregroundStyle(theme.ink.secondary)
            }

            if let job = primaryJob {
                Divider().overlay(theme.surfaces.rule)
                DetailRow(label: "Job", value: job.status)
                DetailRow(label: "Job Hash", value: shortHash(job.sourceDatasetHash))
                if let startedAt = job.startedAt {
                    DetailRow(label: "Started", value: timeLabel(startedAt))
                } else {
                    DetailRow(label: "Queued", value: timeLabel(job.createdAt))
                }
                if let finishedAt = job.finishedAt {
                    DetailRow(label: "Finished", value: timeLabel(finishedAt))
                }
                if let logPath = job.logPath {
                    DetailRow(label: "Log", value: (logPath as NSString).lastPathComponent)
                }
                if let failureReason = job.failureReason {
                    Text(failureReason)
                        .font(.system(size: 11.5))
                        .lineSpacing(2)
                        .foregroundStyle(theme.accents.d)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if isRetryable(job) {
                    GraphiteButton("Retry", systemImage: "arrow.clockwise", style: .ghost, action: retryAction)
                }
            }
        }
        .padding(.top, 6)
    }

    private var tone: ThemeTone {
        switch state?.freshness {
        case "fresh": return .ai
        case "stale": return .d
        case "missing": return .c
        default: return .ink
        }
    }

    private func shortHash(_ hash: String) -> String {
        String(hash.prefix(12))
    }

    private var primaryJob: CortexAdapterJob? {
        jobs.first { $0.status == "training" }
            ?? jobs.first { $0.status == "queued" }
            ?? jobs.first { isRetryable($0) }
            ?? jobs.first
    }

    private func isRetryable(_ job: CortexAdapterJob) -> Bool {
        job.status == "failed" || job.status == "cancelled" || job.status == "eval_failed"
    }

    private func timeLabel(_ millis: UInt64) -> String {
        let date = Date(timeIntervalSince1970: TimeInterval(millis) / 1000)
        return date.formatted(date: .abbreviated, time: .shortened)
    }
}

private struct DetailRow: View {
    @Environment(\.memoryTheme) private var theme
    let label: String
    let value: String

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(label)
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(theme.ink.quaternary)
                .frame(width: 86, alignment: .leading)
            Text(value)
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(theme.ink.secondary)
                .lineLimit(1)
                .truncationMode(.middle)
        }
    }
}

private struct AttentionMarkRow: View {
    @Environment(\.memoryTheme) private var theme
    let mark: AttentionMark
    let revert: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack(spacing: 6) {
                Tag(mark.action.rawValue, tone: tone)
                Text(mark.revertedAt == nil ? "active" : "reverted")
                    .font(.system(size: 9.5, design: .monospaced))
                    .foregroundStyle(mark.revertedAt == nil ? theme.accents.success : theme.ink.quaternary)
                Spacer()
                IconButton("Revert Mark", systemImage: "arrow.uturn.backward") {
                    revert()
                }
                .disabled(mark.revertedAt != nil)
            }
            Text(mark.reason)
                .font(.system(size: 10.5))
                .foregroundStyle(theme.ink.secondary)
                .lineLimit(2)
            Text("\(mark.actor) · \(mark.targetKind.rawValue) · \(mark.targetId) · \(timeLabel(mark.createdAt))")
                .font(.system(size: 9.5, design: .monospaced))
                .foregroundStyle(theme.ink.quaternary)
                .lineLimit(1)
                .truncationMode(.middle)
        }
        .padding(8)
        .background(theme.surfaces.surface2, in: RoundedRectangle(cornerRadius: 5, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 5, style: .continuous).stroke(theme.surfaces.rule, lineWidth: 0.5))
    }

    private var tone: ThemeTone {
        switch mark.action {
        case .pin, .hot, .active:
            return .ai
        case .promote, .warm:
            return .human
        case .suppress, .cold, .decay:
            return .d
        }
    }

    private func timeLabel(_ millis: UInt64) -> String {
        let date = Date(timeIntervalSince1970: TimeInterval(millis) / 1000)
        return date.formatted(date: .abbreviated, time: .shortened)
    }
}

private struct ModelInspector: View {
    @EnvironmentObject private var appState: AppState
    @Environment(\.memoryTheme) private var theme

    var body: some View {
        ScrollView {
            VStack(spacing: 0) {
                InspectorBlock {
                    SectionLabel("Model") {
                        Tag(appState.modelConfig.mode.rawValue, tone: .ink)
                    }
                    Picker("Mode", selection: $appState.modelConfig.mode) {
                        ForEach(ModelConnectionMode.allCases) { mode in
                            Text(mode.rawValue).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()

                    Picker("Runtime", selection: $appState.modelConfig.runtimePreset) {
                        ForEach(ModelRuntimePreset.allCases) { preset in
                            Text(preset.rawValue).tag(preset)
                        }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()

                    LabeledField("Endpoint", text: $appState.modelConfig.endpoint)
                    LabeledField("Chat Model", text: stringBinding($appState.modelConfig.chatModel))
                    LabeledField("Planner Model", text: stringBinding($appState.modelConfig.plannerModel))
                    LabeledField("Planner Endpoint", text: stringBinding($appState.modelConfig.plannerEndpoint))
                    LabeledField("Response Model", text: stringBinding($appState.modelConfig.responseModel))
                    LabeledField("Planner Adapter", text: stringBinding($appState.modelConfig.plannerAdapterPath))
                    LabeledField("Response Adapter", text: stringBinding($appState.modelConfig.responseAdapterPath))
                    LabeledField("Shared Adapter", text: stringBinding($appState.modelConfig.sharedCortexAdapterPath))
                    LabeledField("Active Adapter Hash", text: stringBinding($appState.modelConfig.activeAdapterHash))
                    Picker("Adapter Policy", selection: $appState.modelConfig.adapterActivationPolicy) {
                        Text("Automatic").tag("automatic")
                        Text("Manual").tag("manual")
                        Text("Disabled").tag("disabled")
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    Toggle("Recursive Cortex", isOn: $appState.modelConfig.cortexEnabled)
                        .toggleStyle(.switch)
                    Toggle("Latent RecursiveMAS Research", isOn: $appState.modelConfig.latentRecursiveEnabled)
                        .toggleStyle(.switch)
                    Stepper("Cortex Rounds \(appState.modelConfig.cortexRounds)", value: $appState.modelConfig.cortexRounds, in: 1...4)
                    LabeledField("Critic Model", text: stringBinding($appState.modelConfig.criticModel))
                    LabeledField("Critic Endpoint", text: stringBinding($appState.modelConfig.criticEndpoint))
                    LabeledField("Compiler Model", text: stringBinding($appState.modelConfig.compilerModel))
                    LabeledField("Embedding Model", text: stringBinding($appState.modelConfig.embeddingModel))
                    LabeledField("Embedding Endpoint", text: stringBinding($appState.modelConfig.embeddingEndpoint))
                    Picker("Embedding Runtime", selection: optionalRuntimeBinding($appState.modelConfig.embeddingRuntimePreset, fallback: appState.modelConfig.runtimePreset)) {
                        ForEach(ModelRuntimePreset.allCases) { preset in
                            Text(preset.rawValue).tag(preset)
                        }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    if appState.modelConfig.mode == .api {
                        VStack(alignment: .leading, spacing: 5) {
                            Text("API Key")
                                .font(.system(size: 10, design: .monospaced))
                                .foregroundStyle(theme.ink.tertiary)
                            SecureField("API Key", text: $appState.apiKey)
                                .textFieldStyle(.plain)
                                .font(.system(size: 12))
                                .padding(.horizontal, 8)
                                .frame(height: 28)
                                .background(theme.surfaces.surface2, in: RoundedRectangle(cornerRadius: 5, style: .continuous))
                                .overlay(RoundedRectangle(cornerRadius: 5, style: .continuous).stroke(theme.surfaces.rule, lineWidth: 0.5))
                        }
                    }

                    HStack {
                        GraphiteButton("Save", style: .ghost, action: appState.saveModelConfig)
                        GraphiteButton("Test Connection", style: .primary, action: appState.testModelConnection)
                    }

                    HStack {
                        GraphiteButton("Train Now", systemImage: "play.fill", style: .ghost, action: appState.trainCortexAdapterNow)
                        GraphiteButton("Activate", systemImage: "checkmark.seal", style: .ghost, action: appState.activateLastTrainedCortexAdapter)
                    }
                    HStack {
                        GraphiteButton("Disable Adapter", systemImage: "pause.fill", style: .ghost, action: appState.disableCortexAdapter)
                        GraphiteButton("Compare Routing", systemImage: "arrow.left.arrow.right", style: .ghost, action: appState.compareBaseVsAdaptedRouting)
                    }

                    if let health = appState.modelConfig.health {
                        Label(health.message, systemImage: health.status == "connected" ? "checkmark.circle.fill" : "bolt.horizontal.circle")
                            .font(.system(size: 11.5))
                            .foregroundStyle(health.status == "connected" ? theme.accents.success : theme.ink.tertiary)
                            .padding(.top, 4)
                    }

                    AdapterStateSummary(
                        state: appState.cortexAdapterState,
                        jobs: appState.cortexAdapterJobs,
                        retryAction: appState.retryLatestCortexAdapterJob
                    )
                    if let probe = appState.cortexRouteProbe {
                        Divider().overlay(theme.surfaces.rule)
                        DetailRow(label: "Probe", value: probe.matched ? "matched" : "review")
                        DetailRow(label: "Expected", value: probe.expectedSourceFamily)
                        if let modelSourceFamily = probe.modelSourceFamily {
                            DetailRow(label: "Adapted", value: modelSourceFamily)
                        }
                        if let warning = probe.warning {
                            Text(warning)
                                .font(.system(size: 11.5))
                                .foregroundStyle(theme.ink.secondary)
                        }
                    }
                }

                InspectorDivider()

                InspectorBlock {
                    SectionLabel("Connection")
                    Text("Keep model setup lightweight here, then spend most of your time exploring the cloud.")
                        .font(.system(size: 12.5))
                        .lineSpacing(3)
                        .foregroundStyle(theme.ink.secondary)
                }
            }
        }
    }
}

private struct WorkspaceHeader<Actions: View>: View {
    @Environment(\.memoryTheme) private var theme
    let title: String
    let detail: String
    let actions: Actions

    init(title: String, detail: String, @ViewBuilder actions: () -> Actions) {
        self.title = title
        self.detail = detail
        self.actions = actions()
    }

    var body: some View {
        HStack(spacing: 10) {
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(theme.ink.primary)
                Text(detail)
                    .font(.system(size: 10.5, design: .monospaced))
                    .foregroundStyle(theme.ink.quaternary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            Spacer()
            actions
        }
        .padding(.horizontal, 14)
        .frame(height: 50)
        .background(theme.surfaces.surface1)
    }
}

private struct LibraryTableHeader: View {
    @Environment(\.memoryTheme) private var theme

    var body: some View {
        HStack(spacing: 8) {
            Text("Document").frame(minWidth: 180, maxWidth: .infinity, alignment: .leading)
            Text("Type").frame(width: 90, alignment: .leading)
            Text("Imported").frame(width: 100, alignment: .leading)
            Text("Trust").frame(width: 96, alignment: .leading)
            Text("Freshness").frame(width: 90, alignment: .leading)
        }
        .font(.system(size: 10, design: .monospaced))
        .foregroundStyle(theme.ink.tertiary)
        .padding(.horizontal, 12)
        .frame(height: 30)
        .background(theme.surfaces.surface2)
    }
}

private struct LibraryDocumentRow: View {
    @Environment(\.memoryTheme) private var theme
    let node: GraphNode
    let artifact: SourceArtifact?
    let selected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(node.label)
                        .font(.system(size: 12.5, weight: selected ? .medium : .regular))
                        .foregroundStyle(theme.ink.primary)
                        .lineLimit(1)
                    Text(sourcePath)
                        .font(.system(size: 10.5, design: .monospaced))
                        .foregroundStyle(theme.ink.quaternary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                .frame(minWidth: 180, maxWidth: .infinity, alignment: .leading)
                Text(artifact?.sourceType ?? "document")
                    .frame(width: 90, alignment: .leading)
                Text(artifact.map { timeLabel($0.importedAt) } ?? "unknown")
                    .frame(width: 100, alignment: .leading)
                Text(artifact?.trust.label ?? "Unscored")
                    .frame(width: 96, alignment: .leading)
                Text(freshnessLabel)
                    .frame(width: 90, alignment: .leading)
            }
            .font(.system(size: 11))
            .foregroundStyle(theme.ink.secondary)
            .padding(.horizontal, 12)
            .frame(height: 44)
            .background(selected ? theme.surfaces.selection : theme.surfaces.surface1)
            .overlay(alignment: .bottom) {
                Rectangle().fill(theme.surfaces.rule).frame(height: 0.5)
            }
        }
        .buttonStyle(.plain)
    }

    private var sourcePath: String {
        artifact?.currentPath ?? artifact?.originalPath ?? node.detail
    }

    private var freshnessLabel: String {
        switch artifact?.trust.kind {
        case .webFinding:
            return "verify"
        case .compilerArtifact, .generatedSummary:
            return "derived"
        case .none:
            return "unknown"
        default:
            return "local"
        }
    }
}

private struct CortexRegionCard: View {
    @Environment(\.memoryTheme) private var theme
    let region: CortexRegionSketch

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack {
                Text(region.label)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(theme.ink.primary)
                Spacer()
                Tag("\(region.sourceRefs.count) refs", tone: .ai)
            }
            Text(region.summary)
                .font(.system(size: 12))
                .foregroundStyle(theme.ink.secondary)
                .lineLimit(3)
            if !region.routeExamples.isEmpty {
                VStack(alignment: .leading, spacing: 3) {
                    ForEach(region.routeExamples.prefix(3), id: \.self) { example in
                        Text("route: \(example)")
                            .font(.system(size: 10.5, design: .monospaced))
                            .foregroundStyle(theme.ink.tertiary)
                            .lineLimit(1)
                    }
                }
            }
            if !region.sourceRefs.isEmpty {
                Text(region.sourceRefs.prefix(3).joined(separator: " · "))
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(theme.ink.quaternary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
        }
        .padding(10)
        .background(theme.surfaces.surface2, in: RoundedRectangle(cornerRadius: 6, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 6, style: .continuous).stroke(theme.surfaces.rule, lineWidth: 0.5))
    }
}

private struct SurfNeighborRow: View {
    @Environment(\.memoryTheme) private var theme
    let neighbor: SurfNeighbor

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(neighbor.label)
                    .font(.system(size: 11.5, weight: .medium))
                    .foregroundStyle(theme.ink.primary)
                    .lineLimit(1)
                Spacer()
                Text(String(format: "%.2f", neighbor.score))
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(theme.ink.quaternary)
            }
            Text(neighbor.excerpt)
                .font(.system(size: 10.5))
                .foregroundStyle(theme.ink.tertiary)
                .lineLimit(2)
        }
        .padding(8)
        .background(theme.surfaces.surface1, in: RoundedRectangle(cornerRadius: 5, style: .continuous))
    }
}

private struct ImportQueueItemRow: View {
    @Environment(\.memoryTheme) private var theme
    let item: ImportQueueItem

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack(spacing: 8) {
                Tag(item.status.rawValue, tone: tone)
                Text((item.path as NSString).lastPathComponent)
                    .font(.system(size: 12.5, weight: .medium))
                    .foregroundStyle(theme.ink.primary)
                    .lineLimit(1)
                Spacer()
                Text("\(item.progressCompleted)/\(item.progressTotal)")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(theme.ink.quaternary)
            }
            GeometryReader { proxy in
                ZStack(alignment: .leading) {
                    Capsule().fill(theme.surfaces.rule.opacity(0.6))
                    Capsule()
                        .fill(progressColor)
                        .frame(width: proxy.size.width * progressFraction)
                }
            }
            .frame(height: 4)
            Text(item.error ?? item.path)
                .font(.system(size: 10.5, design: .monospaced))
                .foregroundStyle(item.error == nil ? theme.ink.quaternary : theme.accents.d)
                .lineLimit(2)
                .truncationMode(.middle)
        }
        .padding(10)
        .background(theme.surfaces.surface2, in: RoundedRectangle(cornerRadius: 6, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 6, style: .continuous).stroke(theme.surfaces.rule, lineWidth: 0.5))
    }

    private var progressFraction: CGFloat {
        guard item.progressTotal > 0 else { return item.status == .succeeded ? 1 : 0 }
        return CGFloat(min(Double(item.progressCompleted) / Double(item.progressTotal), 1))
    }

    private var progressColor: Color {
        item.status == .failed ? theme.accents.d : theme.accents.ai
    }

    private var tone: ThemeTone {
        switch item.status {
        case .pending: return .ink
        case .running: return .ai
        case .succeeded: return .d
        case .failed: return .c
        case .cancelled: return .ink
        }
    }
}

private struct QueueMetric: View {
    @Environment(\.memoryTheme) private var theme
    let label: String
    let value: Int

    init(_ label: String, _ value: Int) {
        self.label = label
        self.value = value
    }

    var body: some View {
        HStack(spacing: 5) {
            Text(label)
            Text("\(value)")
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(theme.ink.tertiary)
        }
        .font(.system(size: 11.5))
        .foregroundStyle(theme.ink.secondary)
    }
}

private struct SettingsBand<Content: View>: View {
    @Environment(\.memoryTheme) private var theme
    let title: String
    let content: Content

    init(title: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionLabel(title)
            content
        }
        .padding(12)
        .background(theme.surfaces.surface2, in: RoundedRectangle(cornerRadius: 6, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 6, style: .continuous).stroke(theme.surfaces.rule, lineWidth: 0.5))
    }
}

private struct EmptyTableState: View {
    @Environment(\.memoryTheme) private var theme
    let message: String

    init(_ message: String) {
        self.message = message
    }

    var body: some View {
        Text(message)
            .font(.system(size: 12.5))
            .lineSpacing(3)
            .foregroundStyle(theme.ink.tertiary)
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}

private func shortHash(_ hash: String) -> String {
    String(hash.prefix(12))
}

private func timeLabel(_ millis: UInt64) -> String {
    let date = Date(timeIntervalSince1970: TimeInterval(millis) / 1000)
    return date.formatted(date: .abbreviated, time: .shortened)
}

extension CortexRegionSketch: Identifiable {
    var id: String { regionId }
}

private struct TopBarActivityStatus: View {
    @Environment(\.memoryTheme) private var theme
    let isBusy: Bool
    let progress: OperationProgress?
    let message: String

    var body: some View {
        let fraction = min(max((progress?.percent ?? 0) / 100, 0), 1)
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 7) {
                statusDot
                Text(message)
                    .font(.system(size: 10.5, weight: isBusy ? .medium : .regular, design: .monospaced))
                    .foregroundStyle(isBusy ? theme.ink.secondary : theme.ink.tertiary)
                    .lineLimit(1)
                    .truncationMode(.tail)
                if let progress {
                    Text("· \(Int(progress.percent.rounded()))% · \(progress.phase.capitalized)")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(theme.ink.quaternary)
                        .lineLimit(1)
                        .monospacedDigit()
                } else if isBusy {
                    Text("· working")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(theme.ink.quaternary)
                        .lineLimit(1)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            GeometryReader { proxy in
                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(theme.surfaces.rule.opacity(0.55))
                    Capsule()
                        .fill(isBusy ? theme.accents.ai : theme.ink.quaternary.opacity(0.55))
                        .frame(width: max(10, proxy.size.width * CGFloat(fraction)))
                        .opacity(progress == nil && !isBusy ? 0 : 1)
                }
            }
            .frame(height: 3)
            .overlay(alignment: .trailing) {
                if let progress {
                    Text("\(progress.completed)/\(progress.total)")
                        .font(.system(size: 8.5, design: .monospaced))
                        .foregroundStyle(theme.ink.quaternary)
                        .padding(.leading, 6)
                        .background(theme.surfaces.surface2)
                        .offset(y: 8)
                        .monospacedDigit()
                }
            }
        }
        .padding(.horizontal, 9)
        .padding(.vertical, 5)
        .frame(width: 360, height: 30, alignment: .center)
        .background(theme.surfaces.surface2, in: RoundedRectangle(cornerRadius: 6, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 6, style: .continuous).stroke(theme.surfaces.rule, lineWidth: 0.5))
        .help(message)
    }

    private var statusDot: some View {
        Circle()
            .fill(isBusy ? theme.accents.ai : theme.ink.quaternary)
            .frame(width: 7, height: 7)
            .overlay {
                if isBusy {
                    Circle()
                        .stroke(theme.accents.ai.opacity(0.35), lineWidth: 3)
                }
            }
    }
}

private enum ActorKind {
    case human
    case ai
}

private struct ActorBadge: View {
    @Environment(\.memoryTheme) private var theme
    let kind: ActorKind
    var label: String?

    var body: some View {
        let color = kind == .human ? theme.accents.human : theme.accents.ai
        HStack(spacing: 6) {
            Circle()
                .fill(color)
                .frame(width: 8, height: 8)
                .background(Circle().fill(theme.surfaces.surface1).frame(width: 12, height: 12))
                .overlay(Circle().stroke(color.opacity(0.35), lineWidth: 3))
            Text(label ?? (kind == .human ? "you" : "Agent"))
        }
        .font(.system(size: 10.5, design: .monospaced))
        .foregroundStyle(theme.ink.secondary)
    }
}

private enum ThemeTone {
    case ink
    case human
    case ai
    case c
    case d
}

private struct Tag: View {
    @Environment(\.memoryTheme) private var theme
    let text: String
    let tone: ThemeTone

    init(_ text: String, tone: ThemeTone = .ink) {
        self.text = text
        self.tone = tone
    }

    var body: some View {
        Text(text.uppercased())
            .font(.system(size: 10, weight: .medium, design: .monospaced))
            .foregroundStyle(foreground)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(background, in: RoundedRectangle(cornerRadius: 4, style: .continuous))
    }

    private var background: Color {
        switch tone {
        case .ink: return theme.tags.ink
        case .human: return theme.tags.human
        case .ai: return theme.tags.ai
        case .c: return theme.tags.c
        case .d: return theme.tags.d
        }
    }

    private var foreground: Color {
        switch tone {
        case .ink: return theme.ink.secondary
        case .human: return theme.accents.humanInk
        case .ai: return theme.accents.aiInk
        case .c: return theme.accents.cInk
        case .d: return theme.accents.dInk
        }
    }
}

private struct GraphiteButton: View {
    @Environment(\.memoryTheme) private var theme
    enum ButtonFlavor { case primary, ghost }

    let title: String
    var systemImage: String?
    let style: ButtonFlavor
    let action: () -> Void

    init(_ title: String, systemImage: String? = nil, style: ButtonFlavor, action: @escaping () -> Void) {
        self.title = title
        self.systemImage = systemImage
        self.style = style
        self.action = action
    }

    var body: some View {
        Button(action: action) {
            HStack(spacing: 5) {
                if let systemImage {
                    Image(systemName: systemImage)
                        .font(.system(size: 10.5, weight: .medium))
                }
                Text(title)
            }
            .font(.system(size: 12, weight: style == .primary ? .medium : .regular))
            .frame(height: 28)
            .padding(.horizontal, style == .primary ? 12 : 10)
            .foregroundStyle(style == .primary ? theme.buttons.foreground : theme.ink.secondary)
            .background(style == .primary ? theme.buttons.background : .clear, in: RoundedRectangle(cornerRadius: 6, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 6, style: .continuous).stroke(style == .primary ? .clear : theme.surfaces.rule, lineWidth: 0.5))
        }
        .buttonStyle(.plain)
    }
}

private struct IconButton: View {
    @Environment(\.memoryTheme) private var theme
    let title: String
    let systemImage: String
    let action: () -> Void

    init(_ title: String, systemImage: String, action: @escaping () -> Void) {
        self.title = title
        self.systemImage = systemImage
        self.action = action
    }

    var body: some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(theme.ink.secondary)
                .frame(width: 24, height: 24)
                .background(theme.surfaces.surface2, in: RoundedRectangle(cornerRadius: 5, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 5, style: .continuous).stroke(theme.surfaces.rule, lineWidth: 0.5))
        }
        .buttonStyle(.plain)
        .help(title)
    }
}

private struct SidebarSectionRow: View {
    @Environment(\.memoryTheme) private var theme
    let section: SidebarSection
    let count: Int?
    let selected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                Image(systemName: icon)
                    .font(.system(size: 13))
                    .frame(width: 14)
                    .foregroundStyle(selected ? theme.ink.primary : theme.ink.tertiary)
                Text(section.rawValue)
                    .font(.system(size: 12.5, weight: selected ? .medium : .regular))
                Spacer()
                if let count {
                    Text("\(count)")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(theme.ink.quaternary)
                }
            }
            .foregroundStyle(selected ? theme.ink.primary : theme.ink.secondary)
            .padding(.horizontal, 8)
            .frame(height: 26)
            .background(selected ? theme.surfaces.selection : .clear, in: RoundedRectangle(cornerRadius: 5, style: .continuous))
        }
        .buttonStyle(.plain)
    }

    private var icon: String {
        switch section {
        case .library: return "tray.full"
        case .chat: return "message"
        case .sourceRecall: return "scope"
        case .cortex: return "brain.head.profile"
        case .surf: return "point.topleft.down.curvedto.point.bottomright.up"
        case .importQueue: return "tray.and.arrow.down"
        case .connections: return "link.badge.plus"
        case .settings: return "gearshape"
        }
    }
}

private struct CortexRegionRow: View {
    @Environment(\.memoryTheme) private var theme
    let entry: CortexRegionSketch
    let count: Int
    let tone: ThemeTone

    var body: some View {
        HStack(spacing: 7) {
            Circle()
                .fill(toneColor)
                .frame(width: 8, height: 8)
            Text(entry.label)
                .font(.system(size: 12))
                .lineLimit(1)
                .truncationMode(.tail)
            Spacer()
            Text("\(count)")
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(theme.ink.quaternary)
        }
        .foregroundStyle(theme.ink.secondary)
        .padding(.horizontal, 4)
        .frame(height: 22)
        .help(entry.summary)
    }

    private var toneColor: Color {
        switch tone {
        case .ink: return theme.ink.tertiary
        case .human: return theme.accents.human
        case .ai: return theme.accents.ai
        case .c: return theme.accents.c
        case .d: return theme.accents.d
        }
    }
}

private struct SectionLabel<Right: View>: View {
    @Environment(\.memoryTheme) private var theme
    let text: String
    let right: Right

    init(_ text: String, @ViewBuilder right: () -> Right) {
        self.text = text
        self.right = right()
    }

    var body: some View {
        HStack {
            Text(text.uppercased())
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(theme.ink.tertiary)
            Spacer()
            right
        }
    }
}

private extension SectionLabel where Right == EmptyView {
    init(_ text: String) {
        self.text = text
        self.right = EmptyView()
    }
}

private struct SummaryLine: View {
    @Environment(\.memoryTheme) private var theme
    let title: String
    let value: Int

    var body: some View {
        HStack {
            Text(title)
            Spacer()
            Text("\(value)")
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(theme.ink.tertiary)
        }
        .font(.system(size: 12))
        .foregroundStyle(theme.ink.secondary)
    }
}

private struct EmptyHint: View {
    @Environment(\.memoryTheme) private var theme
    let message: String

    init(_ message: String) {
        self.message = message
    }

    var body: some View {
        Text(message)
            .font(.system(size: 12))
            .foregroundStyle(theme.ink.tertiary)
            .lineSpacing(3)
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}

private struct InspectorBlock<Content: View>: View {
    let content: Content

    init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            content
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }
}

private struct InspectorDivider: View {
    var body: some View {
        DividerLine()
    }
}

private struct DividerLine: View {
    @Environment(\.memoryTheme) private var theme
    enum Axis { case horizontal, vertical }
    var axis: Axis = .horizontal

    var body: some View {
        Rectangle()
            .fill(theme.surfaces.rule)
            .frame(
                width: axis == .vertical ? 0.5 : nil,
                height: axis == .horizontal ? 0.5 : nil
            )
    }
}

private struct HighlightedText: View {
    @Environment(\.memoryTheme) private var theme
    let text: String

    init(_ text: String) {
        self.text = text
    }

    var body: some View {
        Text(text)
            .font(.system(size: 13))
            .lineSpacing(4)
            .foregroundStyle(theme.ink.primary)
            .padding(8)
            .background(theme.accents.highlight.opacity(0.32), in: RoundedRectangle(cornerRadius: 5, style: .continuous))
    }
}

private struct PassageRow: View {
    @Environment(\.memoryTheme) private var theme
    let passage: SurfPassage

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 7) {
                Tag(roleTitle, tone: tone)
                Text(passage.label)
                    .font(.system(size: 11.5, weight: .medium))
                    .foregroundStyle(theme.ink.primary)
                    .lineLimit(1)
                Spacer()
                if passage.role == .representativeChunk {
                    Text(String(format: "%.2f", passage.score))
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(theme.ink.quaternary)
                }
            }
            Text(sourceLine)
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(theme.ink.quaternary)
                .lineLimit(2)
            Text(passage.excerpt)
                .font(.system(size: 12))
                .lineSpacing(3)
                .foregroundStyle(theme.ink.secondary)
                .textSelection(.enabled)
        }
        .padding(8)
        .background(theme.surfaces.surface2, in: RoundedRectangle(cornerRadius: 5, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 5, style: .continuous).stroke(theme.surfaces.rule.opacity(0.7), lineWidth: 0.5))
    }

    private var roleTitle: String {
        switch passage.role {
        case .selectedChunk: return "Selected"
        case .representativeChunk: return "Related"
        case .documentChunk: return "Chunk"
        }
    }

    private var tone: ThemeTone {
        switch passage.role {
        case .selectedChunk: return .d
        case .representativeChunk: return .ai
        case .documentChunk: return .human
        }
    }

    private var sourceLine: String {
        var parts = [String]()
        if let anchor = passage.sourceAnchor {
            parts.append(anchor.path)
            if let page = anchor.page {
                parts.append("p.\(page)")
            }
            if let section = anchor.section {
                parts.append(section)
            }
        }
        if let target = passage.openTarget {
            parts.append(target.locationHint)
            parts.append(target.sourceTrust.label)
            if target.isDerived {
                parts.append("derived")
            }
        }
        parts.append("\(passage.start)-\(passage.end)")
        return parts.joined(separator: " · ")
    }
}

private struct TrailView: View {
    @Environment(\.memoryTheme) private var theme
    let labels: [String]
    let color: Color

    var body: some View {
        if labels.isEmpty {
            EmptyHint("No trail yet.")
        } else {
            VStack(alignment: .leading, spacing: 0) {
                ForEach(Array(labels.enumerated()), id: \.offset) { index, label in
                    HStack(alignment: .top, spacing: 8) {
                        VStack(spacing: 2) {
                            Circle()
                                .fill(index == labels.count - 1 ? color : .clear)
                                .overlay(Circle().stroke(color, lineWidth: 1.5))
                                .frame(width: 7, height: 7)
                            if index != labels.count - 1 {
                                Rectangle()
                                    .fill(color.opacity(0.4))
                                    .frame(width: 1, height: 14)
                            }
                        }
                        .padding(.top, 4)
                        Text(label)
                            .font(.system(size: 11.5, weight: index == labels.count - 1 ? .medium : .regular))
                            .foregroundStyle(index == labels.count - 1 ? theme.ink.primary : theme.ink.secondary)
                            .lineLimit(1)
                    }
                    .padding(.bottom, index == labels.count - 1 ? 0 : 4)
                }
            }
        }
    }
}

private struct LinkRow: View {
    @EnvironmentObject private var appState: AppState
    @Environment(\.memoryTheme) private var theme
    let link: LinkRecord

    var body: some View {
        Button {
            appState.followLink(link)
        } label: {
            HStack(spacing: 8) {
                Tag(shortType, tone: tone)
                    .frame(minWidth: 54)
                Text(link.label)
                    .font(.system(size: 12))
                    .lineLimit(1)
                    .foregroundStyle(theme.ink.primary)
                Spacer()
                Text(String(format: "%.2f", link.score))
                    .font(.system(size: 10.5, design: .monospaced))
                    .foregroundStyle(theme.ink.tertiary)
            }
            .padding(.horizontal, 8)
            .frame(height: 30)
            .background(theme.surfaces.surface2, in: RoundedRectangle(cornerRadius: 5, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 5, style: .continuous).stroke(theme.surfaces.rule, lineWidth: 0.5))
        }
        .buttonStyle(.plain)
    }

    private var shortType: String {
        switch link.linkType {
        case .semanticNeighbor: return "semantic"
        case .sameDocument: return "same-doc"
        case .citationReference: return "citation"
        case .entityOverlap: return "entity"
        case .regionMembership: return "region"
        case .explicit: return "explicit"
        }
    }

    private var tone: ThemeTone {
        switch link.linkType {
        case .semanticNeighbor: return .ai
        case .sameDocument: return .ink
        case .citationReference: return .c
        case .entityOverlap: return .d
        case .regionMembership: return .human
        case .explicit: return .human
        }
    }
}

private struct LabeledField: View {
    @Environment(\.memoryTheme) private var theme
    let label: String
    @Binding var text: String

    init(_ label: String, text: Binding<String>) {
        self.label = label
        self._text = text
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(label)
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(theme.ink.tertiary)
            TextField(label, text: $text)
                .textFieldStyle(.plain)
                .font(.system(size: 12))
                .padding(.horizontal, 8)
                .frame(height: 28)
                .background(theme.surfaces.surface2, in: RoundedRectangle(cornerRadius: 5, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 5, style: .continuous).stroke(theme.surfaces.rule, lineWidth: 0.5))
        }
    }
}

private func stringBinding(_ source: Binding<String?>) -> Binding<String> {
    Binding<String>(
        get: { source.wrappedValue ?? "" },
        set: { source.wrappedValue = $0.isEmpty ? nil : $0 }
    )
}

private func optionalRuntimeBinding(_ source: Binding<ModelRuntimePreset?>, fallback: ModelRuntimePreset) -> Binding<ModelRuntimePreset> {
    Binding<ModelRuntimePreset>(
        get: { source.wrappedValue ?? fallback },
        set: { source.wrappedValue = $0 }
    )
}
