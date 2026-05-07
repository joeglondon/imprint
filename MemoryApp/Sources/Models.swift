import Foundation

struct ResponseEnvelope<T: Decodable>: Decodable {
    let ok: Bool
    let data: T?
    let error: String?
}

enum SidebarSection: String, CaseIterable, Identifiable {
    case library = "Library"
    case chat = "Chat"
    case connections = "Connections"
    case model = "Model"
    case map = "Map"

    var id: String { rawValue }
}

enum WorkspaceConnectionKind: String, Codable, CaseIterable, Identifiable {
    case slack = "Slack"
    case email = "Email"
    case calendar = "Calendar"

    var id: String { rawValue }
}

enum WorkspaceConnectionStatus: String, Codable, Equatable {
    case disconnected = "Disconnected"
    case connecting = "Connecting"
    case connected = "Connected"
}

struct WorkspaceConnection: Codable, Equatable, Identifiable {
    var kind: WorkspaceConnectionKind
    var status: WorkspaceConnectionStatus
    var isRequired: Bool

    var id: String { kind.id }

    static let defaults: [WorkspaceConnection] = WorkspaceConnectionKind.allCases.map {
        WorkspaceConnection(kind: $0, status: .disconnected, isRequired: false)
    }

    var statusLabel: String {
        switch status {
        case .connected:
            return "Connected"
        case .connecting:
            return "Connecting"
        case .disconnected:
            return isRequired ? "Required" : "Optional"
        }
    }

    var actionTitle: String {
        switch status {
        case .connected:
            return "Manage"
        case .connecting:
            return "Connecting"
        case .disconnected:
            return "Connect"
        }
    }
}

enum ModelConnectionMode: String, Codable, CaseIterable, Identifiable {
    case local = "Local"
    case api = "Api"

    var id: String { rawValue }
}

enum ModelRuntimePreset: String, Codable, CaseIterable, Identifiable {
    case ollama = "Ollama"
    case mlx = "Mlx"
    case llamaCpp = "LlamaCpp"
    case customOpenAi = "CustomOpenAi"

    var id: String { rawValue }
}

struct ModelHealth: Codable, Equatable {
    var status: String
    var message: String
    var checkedAt: UInt64?
}

struct ModelConfig: Codable, Equatable {
    var mode: ModelConnectionMode
    var endpoint: String
    var apiKeyName: String?
    var chatModel: String?
    var plannerModel: String?
    var responseModel: String?
    var plannerEndpoint: String?
    var plannerAdapterPath: String?
    var responseAdapterPath: String?
    var sharedCortexAdapterPath: String?
    var activeAdapterHash: String?
    var adapterActivationPolicy: String
    var runtimePreset: ModelRuntimePreset
    var cortexEnabled: Bool
    var latentRecursiveEnabled: Bool
    var cortexRounds: Int
    var criticModel: String?
    var criticEndpoint: String?
    var compilerModel: String?
    var embeddingModel: String?
    var embeddingEndpoint: String?
    var embeddingRuntimePreset: ModelRuntimePreset?
    var health: ModelHealth?
}

struct ModelConnectionTestRequest: Codable {
    var mode: ModelConnectionMode
    var endpoint: String
    var apiKey: String?
    var chatModel: String?
    var plannerModel: String?
    var responseModel: String?
    var plannerEndpoint: String?
    var plannerAdapterPath: String?
    var responseAdapterPath: String?
    var sharedCortexAdapterPath: String?
    var activeAdapterHash: String?
    var adapterActivationPolicy: String
    var runtimePreset: ModelRuntimePreset
    var cortexEnabled: Bool
    var latentRecursiveEnabled: Bool
    var cortexRounds: Int
    var criticModel: String?
    var criticEndpoint: String?
    var compilerModel: String?
    var embeddingModel: String?
    var embeddingEndpoint: String?
    var embeddingRuntimePreset: ModelRuntimePreset?
}

struct MemorySummary: Codable, Equatable {
    var documents: Int
    var chunks: Int
    var regions: Int
    var links: Int
    var mapBytes: Int
}

struct SourceAnchor: Codable, Equatable, Identifiable {
    var id: String
    var documentId: String
    var chunkId: String?
    var path: String
    var contentHash: String
    var start: Int
    var end: Int
    var page: Int?
    var section: String?
    var parserVersion: UInt32
}

enum SourceStorageMode: String, Codable, Equatable {
    case referenceInPlace = "ReferenceInPlace"
    case managedCopy = "ManagedCopy"
    case referenceWithManagedCopy = "ReferenceWithManagedCopy"
    case external = "External"
    case generated = "Generated"
}

struct SourceArtifact: Codable, Equatable, Identifiable {
    var id: String
    var sourceType: String
    var storageMode: SourceStorageMode
    var originalPath: String
    var currentPath: String?
    var managedPath: String?
    var fileHash: String
    var parserVersion: UInt32
    var importedAt: UInt64
    var provenance: ProvenanceRecord
}

enum SourceOpenTargetKind: String, Codable, Equatable {
    case textOffset = "TextOffset"
    case markdownHeading = "MarkdownHeading"
    case pdfPage = "PdfPage"
    case url = "Url"
    case emailThread = "EmailThread"
    case generated = "Generated"
}

struct SourceOpenTarget: Codable, Equatable {
    var kind: SourceOpenTargetKind
    var uri: String
    var path: String?
    var originalPath: String?
    var managedPath: String?
    var sourceArtifactId: String?
    var textStart: Int
    var textEnd: Int
    var markdownHeading: String?
    var pdfPage: Int?
    var browserUrl: String?
    var locationHint: String
}

struct ImportResult: Codable, Equatable {
    var summary: MemorySummary
    var importedPaths: [String]
    var replacedPaths: [String]
    var skippedPaths: [ImportSkip]
    var importedCount: Int
    var replacedCount: Int
    var skippedCount: Int
    var embeddedCount: Int
    var reusedEmbeddingCount: Int
    var adapterState: CortexAdapterState?
}

struct ImportSkip: Codable, Equatable, Identifiable {
    var path: String
    var reason: String

    var id: String { "\(path):\(reason)" }
}

struct OperationProgress: Codable, Equatable {
    var phase: String
    var completed: Int
    var total: Int
    var percent: Double
    var message: String
    var activeNodeIds: [String]?
    var activeNodeLabel: String?
}

enum ChatRole: String, Codable, Equatable {
    case system = "System"
    case user = "User"
    case assistant = "Assistant"
    case tool = "Tool"
}

struct ChatSession: Codable, Equatable, Identifiable {
    var id: String
    var title: String
    var createdAt: UInt64
    var updatedAt: UInt64
    var hotness: Float
}

struct ChatMessage: Codable, Equatable, Identifiable {
    var id: String
    var sessionId: String
    var role: ChatRole
    var content: String
    var createdAt: UInt64
    var tokenEstimate: Int
    var sourceAnchor: SourceAnchor?
}

enum AttentionState: String, Codable, Equatable {
    case hot = "Hot"
    case warm = "Warm"
    case cold = "Cold"
}

struct TranscriptChunk: Codable, Equatable, Identifiable {
    var id: String
    var sessionId: String
    var messageId: String
    var ordinal: Int
    var text: String
    var embedding: [Float]
    var embeddingProvider: String
    var embeddingModel: String
    var embeddingEndpoint: String
    var sourceAnchor: SourceAnchor
    var attentionState: AttentionState
    var hotness: Float
    var createdAt: UInt64
}

enum DerivedMemoryKind: String, Codable, Equatable {
    case summary = "Summary"
    case decision = "Decision"
    case task = "Task"
    case fact = "Fact"
}

struct ProvenanceRecord: Codable, Equatable {
    var actor: String
    var reason: String
    var createdAt: UInt64
    var sourceRefs: [String]
}

struct DerivedMemory: Codable, Equatable, Identifiable {
    var id: String
    var sessionId: String?
    var kind: DerivedMemoryKind
    var text: String
    var sourceMessageIds: [String]
    var actor: String
    var confidence: Float
    var createdAt: UInt64
    var provenance: ProvenanceRecord
}

struct DerivedMemoryWrite: Codable, Equatable {
    var sessionId: String?
    var kind: DerivedMemoryKind
    var text: String
    var sourceMessageIds: [String]
    var actor: String
    var confidence: Float
}

struct WebFindingWrite: Codable, Equatable {
    var sessionId: String?
    var query: String
    var url: String
    var title: String
    var summary: String
    var retrievedAt: UInt64
    var confidence: Float
    var actor: String
}

struct WebFinding: Codable, Equatable, Identifiable {
    var id: String
    var sessionId: String?
    var query: String
    var url: String
    var title: String
    var summary: String
    var retrievedAt: UInt64
    var confidence: Float
    var actor: String
    var createdAt: UInt64
    var provenance: ProvenanceRecord
}

struct AgentLinkWrite: Codable, Equatable {
    var sourceId: String
    var targetId: String
    var label: String
    var actor: String
}

struct AgentLinkMemory: Codable, Equatable, Identifiable {
    var id: String
    var sourceId: String
    var targetId: String
    var label: String
    var actor: String
    var createdAt: UInt64
    var provenance: ProvenanceRecord
}

enum AttentionTargetKind: String, Codable, Equatable {
    case chatSession = "ChatSession"
    case chatMessage = "ChatMessage"
    case transcriptChunk = "TranscriptChunk"
    case derivedMemory = "DerivedMemory"
    case webFinding = "WebFinding"
    case document = "Document"
    case chunk = "Chunk"
    case region = "Region"
    case link = "Link"
}

enum AttentionAction: String, Codable, Equatable {
    case active = "Active"
    case hot = "Hot"
    case warm = "Warm"
    case cold = "Cold"
    case promote = "Promote"
    case decay = "Decay"
    case pin = "Pin"
    case suppress = "Suppress"
}

struct AttentionMark: Codable, Equatable, Identifiable {
    var id: String
    var targetId: String
    var targetKind: AttentionTargetKind
    var action: AttentionAction
    var reason: String
    var actor: String
    var createdAt: UInt64
    var revertedAt: UInt64?
}

struct AttentionMarkWrite: Codable, Equatable {
    var targetId: String
    var targetKind: AttentionTargetKind
    var action: AttentionAction
    var reason: String
    var actor: String
}

struct ChatContextSnippet: Codable, Equatable, Identifiable {
    var id: String
    var sourceKind: String
    var sourceId: String
    var excerpt: String
    var score: Float
    var hotness: Float
    var sourceAnchor: SourceAnchor?
}

struct ChatContextTrace: Codable, Equatable, Identifiable {
    var id: String
    var sessionId: String
    var userMessageId: String
    var snippets: [ChatContextSnippet]
    var toolTrace: [String]
    var cortexTrace: CortexTrace?
    var createdAt: UInt64
}

struct CortexCritique: Codable, Equatable {
    var sufficient: Bool
    var gap: String
    var note: String
}

struct CortexRoundTrace: Codable, Equatable {
    var round: Int
    var actions: [String]
    var snippetsBefore: Int
    var snippetsAfter: Int
    var critique: CortexCritique
}

struct CortexTrace: Codable, Equatable {
    var enabled: Bool
    var rounds: [CortexRoundTrace]
    var finalNote: String
}

struct BrainCompileResult: Codable, Equatable {
    var artifactsWritten: Int
    var artifactIds: [String]
    var trainingRecordsWritten: Int
    var trainingRecordsPath: String
    var exportFiles: [String]
    var adapterState: CortexAdapterState?
    var cortexIndex: CortexIndex?
}

struct CortexAdapterState: Codable, Equatable {
    var freshness: String
    var status: String
    var reason: String?
    var dataFreshness: String
    var trainingStatus: String
    var activationStatus: String
    var baseModel: String?
    var adapterPath: String?
    var manifestPath: String?
    var sourceDatasetHash: String?
    var currentSourceDatasetHash: String
    var trainedSourceDatasetHash: String?
    var activeAdapterHash: String?
    var preparedDatasetHash: String?
    var evalScore: Double?
    var failureReason: String?
    var trainRecords: Int?
    var validRecords: Int?
    var testRecords: Int?
    var iters: Int?
    var lastSuccessfulTrainingAt: UInt64?
    var activatedAt: UInt64?
    var checkedAt: UInt64
}

struct CortexAdapterSnapshot: Codable, Equatable {
    var adapterState: CortexAdapterState?
    var recentJobs: [CortexAdapterJob]
}

struct CortexRouteProbeResult: Codable, Equatable {
    var query: String
    var expectedSourceFamily: String
    var modelSourceFamily: String?
    var matched: Bool
    var usedAdapterPath: String?
    var usedAdapterHash: String?
    var warning: String?
    var rawResponse: String?
}

struct CortexAdapterJob: Codable, Equatable, Identifiable {
    var id: String
    var status: String
    var sourceDatasetHash: String
    var preparedDatasetHash: String?
    var baseModel: String?
    var adapterOutputPath: String
    var manifestPath: String?
    var trainRecords: Int?
    var validRecords: Int?
    var testRecords: Int?
    var iters: Int?
    var command: [String]
    var logPath: String?
    var failureReason: String?
    var payload: [String: String]
    var createdAt: UInt64
    var updatedAt: UInt64
    var startedAt: UInt64?
    var finishedAt: UInt64?
}

struct ChatTurnRequest: Codable, Equatable {
    var sessionId: String
    var message: String
}

struct ChatTurnResult: Codable, Equatable {
    var session: ChatSession
    var messages: [ChatMessage]
    var contextTrace: ChatContextTrace
    var derivedMemories: [DerivedMemory]
}

struct MapEntry: Codable, Equatable, Identifiable {
    var regionId: String
    var label: String
    var summary: String
    var filters: [String: String]

    var id: String { regionId }
}

struct MemoryMap: Codable, Equatable {
    var budgetBytes: Int
    var serialized: String
    var entries: [MapEntry]
}

struct CortexRegionSketch: Codable, Equatable {
    var regionId: String
    var label: String
    var summary: String
    var sourceRefs: [String]
    var artifactIds: [String]
    var routeExamples: [String]
}

struct CortexIndex: Codable, Equatable {
    var id: String
    var schemaVersion: UInt32
    var corpusHash: String
    var createdAt: UInt64
    var compiler: String
    var sourceRefs: [String]
    var artifactIds: [String]
    var regions: [CortexRegionSketch]
    var compatibilityMap: MemoryMap
}

enum NodeRef: Codable, Equatable, Hashable {
    case document(String)
    case chunk(String)
    case region(String)

    private enum CodingKeys: String, CodingKey {
        case document = "Document"
        case chunk = "Chunk"
        case region = "Region"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        if let value = try container.decodeIfPresent(String.self, forKey: .document) {
            self = .document(value)
        } else if let value = try container.decodeIfPresent(String.self, forKey: .chunk) {
            self = .chunk(value)
        } else if let value = try container.decodeIfPresent(String.self, forKey: .region) {
            self = .region(value)
        } else {
            throw DecodingError.dataCorrupted(.init(codingPath: container.codingPath, debugDescription: "Unknown NodeRef"))
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .document(let value):
            try container.encode(value, forKey: .document)
        case .chunk(let value):
            try container.encode(value, forKey: .chunk)
        case .region(let value):
            try container.encode(value, forKey: .region)
        }
    }

    var id: String {
        switch self {
        case .document(let value): return "document:\(value)"
        case .chunk(let value): return "chunk:\(value)"
        case .region(let value): return "region:\(value)"
        }
    }
}

enum LinkType: String, Codable {
    case semanticNeighbor = "SemanticNeighbor"
    case sameDocument = "SameDocument"
    case citationReference = "CitationReference"
    case entityOverlap = "EntityOverlap"
    case regionMembership = "RegionMembership"
}

struct LinkRecord: Codable, Equatable, Identifiable {
    var id: String
    var source: NodeRef
    var target: NodeRef
    var linkType: LinkType
    var score: Float
    var label: String
}

enum GraphNodeKind: String, Codable {
    case region = "Region"
    case chunk = "Chunk"
    case document = "Document"
}

struct GraphPosition: Codable, Equatable {
    var x: Float
    var y: Float
    var z: Float
}

struct GraphNode: Codable, Equatable, Identifiable {
    var id: String
    var nodeRef: NodeRef
    var kind: GraphNodeKind
    var label: String
    var detail: String
    var score: Float
    var position: GraphPosition
    var regionId: String?
}

struct GraphEdge: Codable, Equatable, Identifiable {
    var id: String
    var source: String
    var target: String
    var label: String
    var weight: Float
}

struct VisualizationSnapshot: Codable, Equatable {
    var map: MemoryMap
    var summary: MemorySummary
    var nodes: [GraphNode]
    var edges: [GraphEdge]
}

struct ExtractHit: Codable, Equatable, Identifiable {
    var hitId: String
    var node: NodeRef
    var score: Float
    var excerpt: String
    var start: Int
    var end: Int
    var metadata: [String: String]
    var sourceAnchor: SourceAnchor?

    var id: String { hitId }
}

struct ChunkHit: Codable, Equatable, Identifiable {
    var hitId: String
    var chunkId: String
    var documentId: String
    var regionId: String
    var score: Float
    var excerpt: String
    var start: Int
    var end: Int
    var sourceAnchor: SourceAnchor?

    var id: String { hitId }
}

struct RoutedQuery: Codable, Equatable {
    var query: String
    var regionIds: [String]
    var filters: [String: String]
    var rationale: String
    var routePlan: RoutePlan
}

struct RoutePlan: Codable, Equatable {
    var candidates: [RouteCandidate]
    var nextSteps: [String]
}

struct RouteCandidate: Codable, Equatable, Identifiable {
    var regionId: String
    var label: String
    var score: Float
    var matchedTerms: [String]
    var reason: String

    var id: String { regionId }
}

struct QueryRequest: Codable, Equatable {
    var text: String
    var filters: [String: String]
    var maxRegions: Int
    var maxChunks: Int
}

struct QueryResult: Codable, Equatable {
    var routed: RoutedQuery
    var hits: [ChunkHit]
}

struct SessionState: Codable, Equatable {
    var id: String
    var current: NodeRef?
    var history: [NodeRef]
    var visited: [NodeRef]
}

struct NavigationResult: Codable, Equatable {
    var session: SessionState
    var links: [LinkRecord]
    var currentExcerpt: String?
}

struct DocumentExcerpt: Codable, Equatable {
    var documentId: String
    var title: String
    var chunkId: String
    var excerpt: String
    var start: Int
    var end: Int
    var sourceAnchor: SourceAnchor?
}

enum ExpandMode: String, Codable, CaseIterable, Identifiable {
    case window = "Window"
    case page = "Page"
    case section = "Section"
    case document = "Document"

    var id: String { rawValue }
}

struct SurfOpenResult: Codable, Equatable {
    var node: NodeRef
    var label: String
    var excerpt: String
    var sourceAnchor: SourceAnchor?
    var openTarget: SourceOpenTarget?
    var passages: [SurfPassage]
    var links: [LinkRecord]
}

enum SurfPassageRole: String, Codable {
    case selectedChunk = "SelectedChunk"
    case representativeChunk = "RepresentativeChunk"
    case documentChunk = "DocumentChunk"
}

struct SurfPassage: Codable, Equatable, Identifiable {
    var node: NodeRef
    var label: String
    var excerpt: String
    var start: Int
    var end: Int
    var score: Float
    var sourceAnchor: SourceAnchor?
    var openTarget: SourceOpenTarget?
    var role: SurfPassageRole

    var id: String { "\(node.id):\(role.rawValue):\(start):\(end)" }
}

struct SurfNeighbor: Codable, Equatable, Identifiable {
    var node: NodeRef
    var score: Float
    var label: String
    var excerpt: String
    var sourceAnchor: SourceAnchor?
    var openTarget: SourceOpenTarget?
    var linkType: LinkType?

    var id: String { node.id }
}

struct SurfExpansion: Codable, Equatable {
    var chunkId: String
    var mode: ExpandMode
    var excerpt: String
    var start: Int
    var end: Int
    var sourceAnchor: SourceAnchor?
    var openTarget: SourceOpenTarget?
}

struct InspectorState {
    var selectedNode: GraphNode?
    var links: [LinkRecord] = []
    var neighbors: [SurfNeighbor] = []
    var passages: [SurfPassage] = []
    var excerpt: String?
    var sourceAnchor: SourceAnchor?
    var openTarget: SourceOpenTarget?
    var expansion: SurfExpansion?
    var grepHits: [ExtractHit] = []
    var queryResult: QueryResult?
    var breadcrumbs: [String] = []
}
