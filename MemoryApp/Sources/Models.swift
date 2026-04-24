import Foundation

struct ResponseEnvelope<T: Decodable>: Decodable {
    let ok: Bool
    let data: T?
    let error: String?
}

enum SidebarSection: String, CaseIterable, Identifiable {
    case library = "Library"
    case model = "Model"
    case map = "Map"

    var id: String { rawValue }
}

enum ModelConnectionMode: String, Codable, CaseIterable, Identifiable {
    case local = "Local"
    case api = "Api"

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
    var embeddingModel: String?
    var health: ModelHealth?
}

struct ModelConnectionTestRequest: Codable {
    var mode: ModelConnectionMode
    var endpoint: String
    var apiKey: String?
    var chatModel: String?
    var embeddingModel: String?
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
    var role: SurfPassageRole

    var id: String { "\(node.id):\(role.rawValue):\(start):\(end)" }
}

struct SurfNeighbor: Codable, Equatable, Identifiable {
    var node: NodeRef
    var score: Float
    var label: String
    var excerpt: String
    var sourceAnchor: SourceAnchor?
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
}

struct InspectorState {
    var selectedNode: GraphNode?
    var links: [LinkRecord] = []
    var neighbors: [SurfNeighbor] = []
    var passages: [SurfPassage] = []
    var excerpt: String?
    var sourceAnchor: SourceAnchor?
    var expansion: SurfExpansion?
    var grepHits: [ExtractHit] = []
    var queryResult: QueryResult?
    var breadcrumbs: [String] = []
}
