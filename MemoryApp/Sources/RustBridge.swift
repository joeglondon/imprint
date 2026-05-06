import Foundation
import AIMemoryFFI

enum RustBridgeError: Error, LocalizedError {
    case invalidResponse
    case rust(String)

    var errorDescription: String? {
        switch self {
        case .invalidResponse:
            return "The Rust bridge returned an invalid response."
        case .rust(let message):
            return message
        }
    }
}

enum RustBridge {
    static func ingestPaths(storePath: String, paths: [String]) throws -> ImportResult {
        try decode(ai_memory_ingest_paths(storePath, jsonString(paths)))
    }

    static func rebuildMemory(storePath: String) throws -> ImportResult {
        try decode(ai_memory_rebuild_memory(storePath))
    }

    static func getSummary(storePath: String) throws -> MemorySummary {
        try decode(ai_memory_get_memory_summary(storePath))
    }

    static func getSnapshot(storePath: String) throws -> VisualizationSnapshot {
        try decode(ai_memory_get_visualization_snapshot(storePath))
    }

    static func runQuery(storePath: String, request: QueryRequest) throws -> QueryResult {
        try decode(ai_memory_run_query(storePath, jsonString(request)))
    }

    static func grepRegion(storePath: String, regionId: String, needle: String) throws -> [ExtractHit] {
        try decode(ai_memory_grep_region(storePath, regionId, needle))
    }

    static func openDocumentExcerpt(storePath: String, chunkId: String) throws -> DocumentExcerpt {
        try decode(ai_memory_open_document_excerpt(storePath, chunkId))
    }

    static func grepDocument(storePath: String, documentId: String, needle: String) throws -> [ExtractHit] {
        try decode(ai_memory_grep_document(storePath, documentId, needle))
    }

    static func semanticDocumentSearch(storePath: String, documentId: String, query: String) throws -> [ExtractHit] {
        try decode(ai_memory_semantic_document_search(storePath, documentId, query))
    }

    static func surfOpen(storePath: String, node: NodeRef) throws -> SurfOpenResult {
        try decode(ai_memory_surf_open(storePath, jsonString(node)))
    }

    static func surfNeighbors(storePath: String, node: NodeRef, maxResults: Int) throws -> [SurfNeighbor] {
        try decode(ai_memory_surf_neighbors(storePath, jsonString(node), UInt(maxResults)))
    }

    static func surfExpand(storePath: String, chunkId: String, mode: ExpandMode, window: Int) throws -> SurfExpansion {
        try decode(ai_memory_surf_expand(storePath, chunkId, jsonString(mode), UInt(window)))
    }

    static func surfJumpToAnchor(storePath: String, anchorId: String, window: Int) throws -> SurfExpansion {
        try decode(ai_memory_surf_jump_to_anchor(storePath, anchorId, UInt(window)))
    }

    static func listLinks(storePath: String, node: NodeRef) throws -> [LinkRecord] {
        try decode(ai_memory_list_links(storePath, jsonString(node)))
    }

    static func stepNavigation(storePath: String, session: SessionState, linkId: String) throws -> NavigationResult {
        try decode(ai_memory_step_navigation(storePath, jsonString(session), linkId))
    }

    static func backtrackNavigation(storePath: String, session: SessionState) throws -> NavigationResult {
        try decode(ai_memory_backtrack_navigation(storePath, jsonString(session)))
    }

    static func saveModelConfig(storePath: String, config: ModelConfig) throws -> ModelConfig {
        try decode(ai_memory_save_model_config(storePath, jsonString(config)))
    }

    static func loadModelConfig(storePath: String) throws -> ModelConfig {
        try decode(ai_memory_load_model_config(storePath))
    }

    static func testModelConnection(_ request: ModelConnectionTestRequest) throws -> ModelHealth {
        try decode(ai_memory_test_model_connection(jsonString(request)))
    }

    static func createChatSession(storePath: String, title: String) throws -> ChatSession {
        try decode(ai_memory_create_chat_session(storePath, title))
    }

    static func listChatSessions(storePath: String) throws -> [ChatSession] {
        try decode(ai_memory_list_chat_sessions(storePath))
    }

    static func listChatMessages(storePath: String, sessionId: String) throws -> [ChatMessage] {
        try decode(ai_memory_list_chat_messages(storePath, sessionId))
    }

    static func sendChatTurn(storePath: String, request: ChatTurnRequest) throws -> ChatTurnResult {
        try decode(ai_memory_send_chat_turn(storePath, jsonString(request)))
    }

    static func listChatContextTraces(storePath: String, sessionId: String) throws -> [ChatContextTrace] {
        try decode(ai_memory_list_chat_context_traces(storePath, sessionId))
    }

    static func listDerivedMemories(storePath: String, sessionId: String?) throws -> [DerivedMemory] {
        try decode(ai_memory_list_derived_memories(storePath, sessionId ?? ""))
    }

    static func writeDerivedMemory(storePath: String, write: DerivedMemoryWrite) throws -> DerivedMemory {
        try decode(ai_memory_write_derived_memory(storePath, jsonString(write)))
    }

    static func writeWebFinding(storePath: String, write: WebFindingWrite) throws -> WebFinding {
        try decode(ai_memory_write_web_finding(storePath, jsonString(write)))
    }

    static func writeAgentLink(storePath: String, write: AgentLinkWrite) throws -> AgentLinkMemory {
        try decode(ai_memory_write_agent_link(storePath, jsonString(write)))
    }

    static func applyAttentionMark(storePath: String, write: AttentionMarkWrite) throws -> AttentionMark {
        try decode(ai_memory_apply_attention_mark(storePath, jsonString(write)))
    }

    static func compileMemoryBrain(storePath: String) throws -> BrainCompileResult {
        try decode(ai_memory_compile_memory_brain(storePath))
    }

    static func loadCortexAdapterSnapshot(storePath: String) throws -> CortexAdapterSnapshot {
        try decode(ai_memory_load_cortex_adapter_snapshot(storePath))
    }

    private static func decode<T: Decodable>(_ ptr: UnsafeMutablePointer<CChar>?) throws -> T {
        guard let ptr else {
            throw RustBridgeError.invalidResponse
        }
        defer { ai_memory_free_string(ptr) }
        let raw = String(cString: ptr)
        let data = Data(raw.utf8)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let envelope = try decoder.decode(ResponseEnvelope<T>.self, from: data)
        if envelope.ok, let payload = envelope.data {
            return payload
        }
        throw RustBridgeError.rust(envelope.error ?? "Unknown Rust error")
    }

    private static func jsonString<T: Encodable>(_ value: T) -> String {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let data = try! encoder.encode(value)
        return String(decoding: data, as: UTF8.self)
    }
}
