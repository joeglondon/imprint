use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub type DocumentId = String;
pub type ChunkId = String;
pub type RegionId = String;
pub type LinkId = String;
pub type HitId = String;
pub type SessionId = String;
pub type AnchorId = String;
pub type ChatSessionId = String;
pub type ChatMessageId = String;
pub type TranscriptChunkId = String;
pub type DerivedMemoryId = String;
pub type WebFindingId = String;
pub type AttentionMarkId = String;
pub type AuditEventId = String;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Document {
    pub id: DocumentId,
    pub title: String,
    pub text: String,
    pub metadata: BTreeMap<String, String>,
    #[serde(default)]
    pub source_anchor: Option<SourceAnchor>,
    #[serde(default)]
    pub content_hash: Option<String>,
    #[serde(default)]
    pub parser_version: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceAnchor {
    pub id: AnchorId,
    pub document_id: DocumentId,
    #[serde(default)]
    pub chunk_id: Option<ChunkId>,
    pub path: String,
    pub content_hash: String,
    pub start: usize,
    pub end: usize,
    #[serde(default)]
    pub page: Option<usize>,
    #[serde(default)]
    pub section: Option<String>,
    pub parser_version: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Chunk {
    pub id: ChunkId,
    pub document_id: DocumentId,
    pub region_id: RegionId,
    pub ordinal: usize,
    pub start: usize,
    pub end: usize,
    pub text: String,
    pub metadata: BTreeMap<String, String>,
    #[serde(default)]
    pub source_anchor: Option<SourceAnchor>,
    pub embedding: Vec<f32>,
    #[serde(default)]
    pub embedding_text_hash: Option<String>,
    #[serde(default)]
    pub embedding_provider: Option<String>,
    #[serde(default)]
    pub embedding_model: Option<String>,
    #[serde(default)]
    pub embedding_endpoint: Option<String>,
    #[serde(default)]
    pub embedding_dimension: Option<usize>,
    #[serde(default)]
    pub chunking_version: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Region {
    pub id: RegionId,
    pub label: String,
    pub summary: String,
    pub filters: BTreeMap<String, String>,
    pub chunk_ids: Vec<ChunkId>,
    pub centroid: Vec<f32>,
    pub neighbors: Vec<RegionId>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MapEntry {
    pub region_id: RegionId,
    pub label: String,
    pub summary: String,
    pub filters: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MemoryMap {
    pub budget_bytes: usize,
    pub serialized: String,
    pub entries: Vec<MapEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum LinkType {
    SemanticNeighbor,
    SameDocument,
    CitationReference,
    EntityOverlap,
    RegionMembership,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Link {
    pub id: LinkId,
    pub source: NodeRef,
    pub target: NodeRef,
    pub link_type: LinkType,
    pub score: f32,
    pub label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum NodeRef {
    Document(DocumentId),
    Chunk(ChunkId),
    Region(RegionId),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QueryRequest {
    pub text: String,
    pub filters: BTreeMap<String, String>,
    pub max_regions: usize,
    pub max_chunks: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RoutedQuery {
    pub query: String,
    pub region_ids: Vec<RegionId>,
    pub filters: BTreeMap<String, String>,
    pub rationale: String,
    #[serde(default)]
    pub route_plan: RoutePlan,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct RoutePlan {
    pub candidates: Vec<RouteCandidate>,
    pub next_steps: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RouteCandidate {
    pub region_id: RegionId,
    pub label: String,
    pub score: f32,
    pub matched_terms: Vec<String>,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QueryResult {
    pub routed: RoutedQuery,
    pub hits: Vec<ChunkHit>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChunkHit {
    pub hit_id: HitId,
    pub chunk_id: ChunkId,
    pub document_id: DocumentId,
    pub region_id: RegionId,
    pub score: f32,
    pub excerpt: String,
    pub start: usize,
    pub end: usize,
    #[serde(default)]
    pub source_anchor: Option<SourceAnchor>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExtractHit {
    pub hit_id: HitId,
    pub node: NodeRef,
    pub score: f32,
    pub excerpt: String,
    pub start: usize,
    pub end: usize,
    pub metadata: BTreeMap<String, String>,
    #[serde(default)]
    pub source_anchor: Option<SourceAnchor>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionState {
    pub id: SessionId,
    pub current: Option<NodeRef>,
    pub history: Vec<NodeRef>,
    pub visited: Vec<NodeRef>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PersistedMemory {
    pub documents: Vec<Document>,
    pub chunks: Vec<Chunk>,
    pub regions: Vec<Region>,
    pub links: Vec<Link>,
    pub memory_map: Option<MemoryMap>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QueryTrace {
    pub query: String,
    pub chosen_regions: Vec<RegionId>,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatSession {
    pub id: ChatSessionId,
    pub title: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub hotness: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatMessage {
    pub id: ChatMessageId,
    pub session_id: ChatSessionId,
    pub role: ChatRole,
    pub content: String,
    pub created_at: u64,
    pub token_estimate: usize,
    #[serde(default)]
    pub source_anchor: Option<SourceAnchor>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AttentionState {
    Hot,
    Warm,
    Cold,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TranscriptChunk {
    pub id: TranscriptChunkId,
    pub session_id: ChatSessionId,
    pub message_id: ChatMessageId,
    pub ordinal: usize,
    pub text: String,
    pub embedding: Vec<f32>,
    pub embedding_provider: String,
    pub embedding_model: String,
    pub embedding_endpoint: String,
    pub source_anchor: SourceAnchor,
    pub attention_state: AttentionState,
    pub hotness: f32,
    pub created_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DerivedMemoryKind {
    Summary,
    Decision,
    Task,
    Fact,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProvenanceRecord {
    pub actor: String,
    pub reason: String,
    pub created_at: u64,
    pub source_refs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DerivedMemory {
    pub id: DerivedMemoryId,
    #[serde(default)]
    pub session_id: Option<ChatSessionId>,
    pub kind: DerivedMemoryKind,
    pub text: String,
    pub source_message_ids: Vec<ChatMessageId>,
    pub actor: String,
    pub confidence: f32,
    pub created_at: u64,
    pub provenance: ProvenanceRecord,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DerivedMemoryWrite {
    #[serde(default)]
    pub session_id: Option<ChatSessionId>,
    pub kind: DerivedMemoryKind,
    pub text: String,
    pub source_message_ids: Vec<ChatMessageId>,
    pub actor: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WebFinding {
    pub id: WebFindingId,
    #[serde(default)]
    pub session_id: Option<ChatSessionId>,
    pub query: String,
    pub url: String,
    pub title: String,
    pub summary: String,
    pub retrieved_at: u64,
    pub confidence: f32,
    pub actor: String,
    pub created_at: u64,
    pub provenance: ProvenanceRecord,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WebFindingWrite {
    #[serde(default)]
    pub session_id: Option<ChatSessionId>,
    pub query: String,
    pub url: String,
    pub title: String,
    pub summary: String,
    pub retrieved_at: u64,
    pub confidence: f32,
    pub actor: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentLinkMemory {
    pub id: String,
    pub source_id: String,
    pub target_id: String,
    pub label: String,
    pub actor: String,
    pub created_at: u64,
    pub provenance: ProvenanceRecord,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentLinkWrite {
    pub source_id: String,
    pub target_id: String,
    pub label: String,
    pub actor: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AttentionTargetKind {
    ChatSession,
    ChatMessage,
    TranscriptChunk,
    DerivedMemory,
    WebFinding,
    Document,
    Chunk,
    Region,
    Link,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AttentionAction {
    Active,
    Promote,
    Decay,
    Pin,
    Suppress,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AttentionMark {
    pub id: AttentionMarkId,
    pub target_id: String,
    pub target_kind: AttentionTargetKind,
    pub action: AttentionAction,
    pub reason: String,
    pub actor: String,
    pub created_at: u64,
    #[serde(default)]
    pub reverted_at: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AttentionMarkWrite {
    pub target_id: String,
    pub target_kind: AttentionTargetKind,
    pub action: AttentionAction,
    pub reason: String,
    pub actor: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AuditEvent {
    pub id: AuditEventId,
    #[serde(default)]
    pub session_id: Option<ChatSessionId>,
    pub event_type: String,
    pub target_id: String,
    pub actor: String,
    pub payload_json: String,
    pub created_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatContextSnippet {
    pub id: String,
    pub source_kind: String,
    pub source_id: String,
    pub excerpt: String,
    pub score: f32,
    pub hotness: f32,
    #[serde(default)]
    pub source_anchor: Option<SourceAnchor>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CortexCritique {
    pub sufficient: bool,
    pub gap: String,
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CortexRoundTrace {
    pub round: usize,
    pub actions: Vec<String>,
    pub snippets_before: usize,
    pub snippets_after: usize,
    pub critique: CortexCritique,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CortexTrace {
    pub enabled: bool,
    pub rounds: Vec<CortexRoundTrace>,
    pub final_note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatContextTrace {
    pub id: String,
    pub session_id: ChatSessionId,
    pub user_message_id: ChatMessageId,
    pub snippets: Vec<ChatContextSnippet>,
    pub tool_trace: Vec<String>,
    #[serde(default)]
    pub cortex_trace: Option<CortexTrace>,
    pub created_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BrainCompileResult {
    pub artifacts_written: usize,
    pub artifact_ids: Vec<String>,
    pub training_records_written: usize,
    pub training_records_path: String,
    #[serde(default)]
    pub export_files: Vec<String>,
    #[serde(default)]
    pub adapter_state: Option<CortexAdapterState>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CortexAdapterState {
    pub freshness: String,
    pub status: String,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub base_model: Option<String>,
    #[serde(default)]
    pub adapter_path: Option<String>,
    #[serde(default)]
    pub manifest_path: Option<String>,
    #[serde(default)]
    pub source_dataset_hash: Option<String>,
    pub current_source_dataset_hash: String,
    #[serde(default)]
    pub prepared_dataset_hash: Option<String>,
    #[serde(default)]
    pub train_records: Option<usize>,
    #[serde(default)]
    pub valid_records: Option<usize>,
    #[serde(default)]
    pub test_records: Option<usize>,
    #[serde(default)]
    pub iters: Option<usize>,
    pub checked_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum BrainArtifactKind {
    LibraryMap,
    RegionCard,
    EntityCard,
    ProjectCard,
    PreferenceCard,
    RoutingRule,
    ToolPattern,
    CritiquePattern,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BrainArtifact {
    pub id: String,
    pub kind: BrainArtifactKind,
    pub title: String,
    pub body: String,
    pub source_refs: Vec<String>,
    pub content_hash: String,
    pub provenance: ProvenanceRecord,
    pub confidence: u8,
    pub created_at: u64,
    pub updated_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatTurnRequest {
    pub session_id: ChatSessionId,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatTurnResult {
    pub session: ChatSession,
    pub messages: Vec<ChatMessage>,
    pub context_trace: ChatContextTrace,
    pub derived_memories: Vec<DerivedMemory>,
}
