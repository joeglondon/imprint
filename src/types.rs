use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub type DocumentId = String;
pub type ChunkId = String;
pub type RegionId = String;
pub type LinkId = String;
pub type HitId = String;
pub type SessionId = String;
pub type AnchorId = String;

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
