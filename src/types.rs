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
pub type MemoryAccessId = String;
pub type AuditEventId = String;
pub type CortexIndexId = String;
pub type CortexAdapterJobId = String;
pub type SourceArtifactId = String;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SourceStorageMode {
    ReferenceInPlace,
    ManagedCopy,
    ReferenceWithManagedCopy,
    External,
    Generated,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SourceArtifact {
    pub id: SourceArtifactId,
    pub source_type: String,
    #[serde(default)]
    pub trust: SourceTrustPolicy,
    pub storage_mode: SourceStorageMode,
    pub original_path: String,
    #[serde(default)]
    pub current_path: Option<String>,
    #[serde(default)]
    pub managed_path: Option<String>,
    pub file_hash: String,
    pub parser_version: u32,
    pub imported_at: u64,
    pub provenance: ProvenanceRecord,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SourceTrustKind {
    LocalSource,
    UserAuthoredNote,
    ImportedDocument,
    WebFinding,
    GeneratedSummary,
    CompilerArtifact,
    ChatTranscript,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SourceTrustPolicy {
    pub kind: SourceTrustKind,
    pub score: f32,
    pub label: String,
    pub caveat: String,
}

impl Default for SourceTrustPolicy {
    fn default() -> Self {
        Self {
            kind: SourceTrustKind::Unknown,
            score: 0.5,
            label: "Unknown source".into(),
            caveat: "Verify against an original source anchor before making exact claims.".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SourceOpenTargetKind {
    TextOffset,
    MarkdownHeading,
    PdfPage,
    Url,
    EmailThread,
    Generated,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RenderedPageMetadata {
    pub page: usize,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub width: Option<f32>,
    #[serde(default)]
    pub height: Option<f32>,
    #[serde(default)]
    pub unit: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    #[serde(default)]
    pub page: Option<usize>,
    #[serde(default)]
    pub unit: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EmailThreadLocation {
    pub thread_id: String,
    #[serde(default)]
    pub message_id: Option<String>,
    #[serde(default)]
    pub mailbox: Option<String>,
    #[serde(default)]
    pub subject: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PdfTextSelection {
    pub page: usize,
    pub text_start: usize,
    pub text_end: usize,
    #[serde(default)]
    pub selected_text: Option<String>,
    #[serde(default)]
    pub bounding_box: Option<BoundingBox>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SourceOpenTarget {
    pub kind: SourceOpenTargetKind,
    pub uri: String,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub original_path: Option<String>,
    #[serde(default)]
    pub managed_path: Option<String>,
    #[serde(default)]
    pub source_artifact_id: Option<SourceArtifactId>,
    pub text_start: usize,
    pub text_end: usize,
    #[serde(default)]
    pub markdown_heading: Option<String>,
    #[serde(default)]
    pub pdf_page: Option<usize>,
    #[serde(default)]
    pub pdf_selection: Option<PdfTextSelection>,
    #[serde(default)]
    pub browser_url: Option<String>,
    #[serde(default)]
    pub email_location: Option<EmailThreadLocation>,
    #[serde(default)]
    pub source_trust: SourceTrustPolicy,
    #[serde(default)]
    pub is_derived: bool,
    #[serde(default)]
    pub caveat: Option<String>,
    pub location_hint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SourceAnchor {
    pub id: AnchorId,
    pub document_id: DocumentId,
    #[serde(default)]
    pub chunk_id: Option<ChunkId>,
    #[serde(default)]
    pub source_artifact_id: Option<SourceArtifactId>,
    pub path: String,
    pub content_hash: String,
    pub start: usize,
    pub end: usize,
    #[serde(default)]
    pub byte_start: Option<usize>,
    #[serde(default)]
    pub byte_end: Option<usize>,
    #[serde(default)]
    pub char_start: Option<usize>,
    #[serde(default)]
    pub char_end: Option<usize>,
    #[serde(default)]
    pub page: Option<usize>,
    #[serde(default)]
    pub rendered_page: Option<RenderedPageMetadata>,
    #[serde(default)]
    pub pdf_selection: Option<PdfTextSelection>,
    #[serde(default)]
    pub email_location: Option<EmailThreadLocation>,
    #[serde(default)]
    pub section: Option<String>,
    #[serde(default)]
    pub section_hierarchy: Vec<String>,
    #[serde(default)]
    pub paragraph_index: Option<usize>,
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
pub struct CortexRegionSketch {
    pub region_id: RegionId,
    pub label: String,
    pub summary: String,
    pub source_refs: Vec<String>,
    pub artifact_ids: Vec<String>,
    pub route_examples: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CortexIndex {
    pub id: CortexIndexId,
    pub schema_version: u32,
    pub corpus_hash: String,
    pub created_at: u64,
    pub compiler: String,
    pub source_refs: Vec<String>,
    pub artifact_ids: Vec<String>,
    pub regions: Vec<CortexRegionSketch>,
    pub compatibility_map: MemoryMap,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum LinkType {
    SemanticNeighbor,
    SameDocument,
    CitationReference,
    EntityOverlap,
    RegionMembership,
    Explicit,
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LinkEvidence {
    pub reason: String,
    pub source_refs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LinkInspection {
    pub link: Link,
    pub why_linked: String,
    pub evidence: Vec<LinkEvidence>,
    pub confidence: String,
    pub provenance: ProvenanceRecord,
    pub attention_marks: Vec<AttentionMark>,
    pub suppressed: bool,
    pub promoted: bool,
    pub pinned: bool,
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
    Hot,
    Warm,
    Cold,
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MemoryAccessKind {
    QueryHit,
    Open,
    Expand,
    JumpToAnchor,
    Cite,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryAccess {
    pub id: MemoryAccessId,
    pub target_id: String,
    pub target_kind: AttentionTargetKind,
    pub access_kind: MemoryAccessKind,
    pub reason: String,
    pub actor: String,
    pub accessed_at: u64,
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BrainCompileResult {
    pub artifacts_written: usize,
    pub artifact_ids: Vec<String>,
    pub training_records_written: usize,
    pub training_records_path: String,
    #[serde(default)]
    pub export_files: Vec<String>,
    #[serde(default)]
    pub adapter_state: Option<CortexAdapterState>,
    #[serde(default)]
    pub cortex_index: Option<CortexIndex>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CortexAdapterState {
    /// Compatibility summary: missing/fresh/stale/unknown.
    pub freshness: String,
    /// Compatibility summary: the raw manifest status when available.
    pub status: String,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default = "default_adapter_data_freshness")]
    pub data_freshness: String,
    #[serde(default = "default_adapter_training_status")]
    pub training_status: String,
    #[serde(default = "default_adapter_activation_status")]
    pub activation_status: String,
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
    pub trained_source_dataset_hash: Option<String>,
    #[serde(default)]
    pub active_adapter_hash: Option<String>,
    #[serde(default)]
    pub prepared_dataset_hash: Option<String>,
    #[serde(default)]
    pub eval_score: Option<f64>,
    #[serde(default)]
    pub failure_reason: Option<String>,
    #[serde(default)]
    pub train_records: Option<usize>,
    #[serde(default)]
    pub valid_records: Option<usize>,
    #[serde(default)]
    pub test_records: Option<usize>,
    #[serde(default)]
    pub iters: Option<usize>,
    #[serde(default)]
    pub last_successful_training_at: Option<u64>,
    #[serde(default)]
    pub activated_at: Option<u64>,
    pub checked_at: u64,
}

fn default_adapter_data_freshness() -> String {
    "unknown".into()
}

fn default_adapter_training_status() -> String {
    "missing".into()
}

fn default_adapter_activation_status() -> String {
    "inactive".into()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CortexAdapterSnapshot {
    #[serde(default)]
    pub adapter_state: Option<CortexAdapterState>,
    #[serde(default)]
    pub recent_jobs: Vec<CortexAdapterJob>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CortexAdapterJob {
    pub id: CortexAdapterJobId,
    pub status: String,
    pub source_dataset_hash: String,
    #[serde(default)]
    pub prepared_dataset_hash: Option<String>,
    #[serde(default)]
    pub base_model: Option<String>,
    pub adapter_output_path: String,
    #[serde(default)]
    pub manifest_path: Option<String>,
    #[serde(default)]
    pub train_records: Option<usize>,
    #[serde(default)]
    pub valid_records: Option<usize>,
    #[serde(default)]
    pub test_records: Option<usize>,
    #[serde(default)]
    pub iters: Option<usize>,
    #[serde(default)]
    pub command: Vec<String>,
    #[serde(default)]
    pub log_path: Option<String>,
    #[serde(default)]
    pub failure_reason: Option<String>,
    #[serde(default)]
    pub payload: BTreeMap<String, String>,
    pub created_at: u64,
    pub updated_at: u64,
    #[serde(default)]
    pub started_at: Option<u64>,
    #[serde(default)]
    pub finished_at: Option<u64>,
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
