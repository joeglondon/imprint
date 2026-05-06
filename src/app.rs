use crate::extract::{Extractor, MemoryExtractor};
use crate::graph::GraphBuilder;
use crate::index::{
    cosine_similarity, Embedder, HashEmbedder, Indexer, OllamaEmbedder, OpenAiCompatibleEmbedder,
    RegionIndexer, RuntimeEmbedder,
};
use crate::ingest::{
    format_region_label, meaningful_terms_for_chunks, EmbeddingReuseStats, ImportSkip, Ingester,
    PARSER_VERSION,
};
use crate::map::MapBuilder;
use crate::navigation::{MemoryNavigator, Navigator};
use crate::query::MemoryQueryEngine;
use crate::store::{FileMemoryStore, MemoryStore};
use crate::surf::{
    self, ExpandMode, SurfAction, SurfExpansion, SurfNeighbor, SurfOpenResult, SurfStepResult,
};
use crate::types::*;
use anyhow::{anyhow, Context};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_WINDOW: usize = 90;
const DEFAULT_LOCAL_ENDPOINT: &str = "http://localhost:11434";
const DEFAULT_LOCAL_EMBEDDING_MODEL: &str = "embeddinggemma:300m";
const HASH_EMBEDDING_MODEL: &str = "hash";
const VISUAL_SCALE: f32 = 18.0;
const MAX_VISUAL_CHUNK_NEIGHBORS: usize = 2;
const DEFAULT_PLANNER_MODEL: &str = "LiquidAI/LFM2.5-350M-MLX-8bit";
const DEFAULT_RESPONSE_MODEL: &str = "mlx-community/LFM2.5-1.2B-Instruct-8bit";
const DEFAULT_PLANNER_STEPS: usize = 6;
const MAX_RESPONSE_CONTEXT_SNIPPETS: usize = 8;
const MAX_CONTEXT_SNIPPET_CHARS: usize = 2200;
const WEB_SEARCH_MAX_RESULTS: usize = 4;
const WEB_SEARCH_MIN_LOCAL_SCORE: f32 = 0.45;
const MEMORY_ACCESS_HORIZON_MILLIS: u64 = 30 * 24 * 60 * 60 * 1000;
#[cfg(not(test))]
const MAX_WEB_BODY_CHARS: usize = 12_000;
const MAX_WEB_SUMMARY_CHARS: usize = 16_000;

fn default_runtime_preset() -> ModelRuntimePreset {
    ModelRuntimePreset::Mlx
}

fn default_cortex_enabled() -> bool {
    true
}

fn default_cortex_rounds() -> usize {
    3
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelConnectionMode {
    Local,
    Api,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelRuntimePreset {
    Ollama,
    Mlx,
    LlamaCpp,
    CustomOpenAi,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelHealth {
    pub status: String,
    pub message: String,
    pub checked_at: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelConfig {
    pub mode: ModelConnectionMode,
    pub endpoint: String,
    pub api_key_name: Option<String>,
    pub chat_model: Option<String>,
    #[serde(default)]
    pub planner_model: Option<String>,
    #[serde(default)]
    pub response_model: Option<String>,
    #[serde(default)]
    pub planner_endpoint: Option<String>,
    #[serde(default = "default_runtime_preset")]
    pub runtime_preset: ModelRuntimePreset,
    #[serde(default = "default_cortex_enabled")]
    pub cortex_enabled: bool,
    #[serde(default = "default_cortex_rounds")]
    pub cortex_rounds: usize,
    #[serde(default)]
    pub critic_model: Option<String>,
    #[serde(default)]
    pub critic_endpoint: Option<String>,
    #[serde(default)]
    pub compiler_model: Option<String>,
    pub embedding_model: Option<String>,
    #[serde(default)]
    pub embedding_endpoint: Option<String>,
    #[serde(default)]
    pub embedding_runtime_preset: Option<ModelRuntimePreset>,
    pub health: Option<ModelHealth>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelConnectionTestRequest {
    pub mode: ModelConnectionMode,
    pub endpoint: String,
    pub api_key: Option<String>,
    pub chat_model: Option<String>,
    #[serde(default)]
    pub planner_model: Option<String>,
    #[serde(default)]
    pub response_model: Option<String>,
    #[serde(default)]
    pub planner_endpoint: Option<String>,
    #[serde(default = "default_runtime_preset")]
    pub runtime_preset: ModelRuntimePreset,
    #[serde(default = "default_cortex_enabled")]
    pub cortex_enabled: bool,
    #[serde(default = "default_cortex_rounds")]
    pub cortex_rounds: usize,
    #[serde(default)]
    pub critic_model: Option<String>,
    #[serde(default)]
    pub critic_endpoint: Option<String>,
    #[serde(default)]
    pub compiler_model: Option<String>,
    pub embedding_model: Option<String>,
    #[serde(default)]
    pub embedding_endpoint: Option<String>,
    #[serde(default)]
    pub embedding_runtime_preset: Option<ModelRuntimePreset>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemorySummary {
    pub documents: usize,
    pub chunks: usize,
    pub regions: usize,
    pub links: usize,
    pub map_bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImportResult {
    pub summary: MemorySummary,
    pub imported_paths: Vec<String>,
    pub replaced_paths: Vec<String>,
    pub skipped_paths: Vec<ImportSkip>,
    pub imported_count: usize,
    pub replaced_count: usize,
    pub skipped_count: usize,
    pub embedded_count: usize,
    pub reused_embedding_count: usize,
    #[serde(default)]
    pub adapter_state: Option<CortexAdapterState>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum GraphNodeKind {
    Region,
    Chunk,
    Document,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct GraphPosition {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GraphNode {
    pub id: String,
    pub node_ref: NodeRef,
    pub kind: GraphNodeKind,
    pub label: String,
    pub detail: String,
    pub score: f32,
    pub position: GraphPosition,
    pub region_id: Option<RegionId>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GraphEdge {
    pub id: String,
    pub source: String,
    pub target: String,
    pub label: String,
    pub weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VisualizationSnapshot {
    pub map: MemoryMap,
    pub summary: MemorySummary,
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NavigationResult {
    pub session: SessionState,
    pub links: Vec<Link>,
    pub current_excerpt: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DocumentExcerpt {
    pub document_id: DocumentId,
    pub title: String,
    pub chunk_id: ChunkId,
    pub excerpt: String,
    pub start: usize,
    pub end: usize,
    #[serde(default)]
    pub source_anchor: Option<SourceAnchor>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OperationProgress {
    pub phase: String,
    pub completed: usize,
    pub total: usize,
    pub percent: f32,
    pub message: String,
    #[serde(default)]
    pub active_node_ids: Vec<String>,
    #[serde(default)]
    pub active_node_label: Option<String>,
}

pub fn ingest_paths(store_root: &Path, paths: &[PathBuf]) -> anyhow::Result<ImportResult> {
    write_progress(store_root, "extract", 0, 1, "Reading selected files")?;
    let store = FileMemoryStore::new(store_root);
    let existing = store.load()?;
    let config = load_model_config(store_root)?;
    let ingester = Ingester::new(embedder_for_config(&config)?);
    let batch = ingester.extract_documents(paths)?;
    let existing_source_documents = existing
        .documents
        .iter()
        .filter(|document| !is_compiler_generated_document(document))
        .cloned()
        .collect::<Vec<_>>();
    let existing_ids = existing_source_documents
        .iter()
        .map(|document| document.id.clone())
        .collect::<HashSet<_>>();
    let incoming_ids = batch
        .documents
        .iter()
        .map(|document| document.id.clone())
        .collect::<HashSet<_>>();
    let replaced_paths = batch
        .documents
        .iter()
        .filter(|document| existing_ids.contains(&document.id))
        .filter_map(|document| document.metadata.get("path").cloned())
        .collect::<Vec<_>>();
    let imported_paths = batch
        .documents
        .iter()
        .filter(|document| !existing_ids.contains(&document.id))
        .filter_map(|document| document.metadata.get("path").cloned())
        .collect::<Vec<_>>();

    let mut documents = existing_source_documents
        .into_iter()
        .filter(|document| !incoming_ids.contains(&document.id))
        .collect::<Vec<_>>();
    documents.extend(batch.documents);
    let (memory, stats) = rebuild_from_documents(store_root, documents, &existing.chunks, &config)?;
    write_progress(store_root, "save", 99, 100, "Saving memory store")?;
    store.save(&memory)?;
    write_progress(
        store_root,
        "cortex",
        99,
        100,
        "Refreshing cortex adapter data",
    )?;
    let adapter_state = compile_memory_brain(store_root)?.adapter_state;
    write_progress(
        store_root,
        "complete",
        100,
        100,
        "Memory updated and cortex refreshed",
    )?;
    Ok(ImportResult {
        summary: summarize(&memory),
        imported_count: imported_paths.len(),
        replaced_count: replaced_paths.len(),
        skipped_count: batch.skipped_paths.len(),
        embedded_count: stats.embedded_count,
        reused_embedding_count: stats.reused_embedding_count,
        adapter_state,
        imported_paths,
        replaced_paths,
        skipped_paths: batch.skipped_paths,
    })
}

pub fn rebuild_memory(store_root: &Path) -> anyhow::Result<ImportResult> {
    write_progress(store_root, "load", 0, 1, "Loading existing memory")?;
    let store = FileMemoryStore::new(store_root);
    let memory = store.load()?;
    if memory.documents.is_empty() {
        return Err(anyhow!("store is empty; ingest files first"));
    }
    let docs = memory
        .documents
        .into_iter()
        .filter(|document| !is_compiler_generated_document(document))
        .collect::<Vec<_>>();
    let reusable_chunks = memory.chunks;
    let config = load_model_config(store_root)?;
    let (rebuilt, stats) = rebuild_from_documents(store_root, docs, &reusable_chunks, &config)?;
    write_progress(store_root, "save", 99, 100, "Saving rebuilt memory")?;
    store.save(&rebuilt)?;
    write_progress(
        store_root,
        "cortex",
        99,
        100,
        "Refreshing cortex adapter data",
    )?;
    let adapter_state = compile_memory_brain(store_root)?.adapter_state;
    write_progress(
        store_root,
        "complete",
        100,
        100,
        "Memory rebuilt and cortex refreshed",
    )?;
    Ok(ImportResult {
        summary: summarize(&rebuilt),
        imported_paths: Vec::new(),
        replaced_paths: Vec::new(),
        skipped_paths: Vec::new(),
        imported_count: 0,
        replaced_count: 0,
        skipped_count: 0,
        embedded_count: stats.embedded_count,
        reused_embedding_count: stats.reused_embedding_count,
        adapter_state,
    })
}

pub fn get_memory_summary(store_root: &Path) -> anyhow::Result<MemorySummary> {
    let _ = sync_chat_documents(store_root);
    let _ = sync_web_findings(store_root);
    let store = FileMemoryStore::new(store_root);
    let memory = store.load()?;
    Ok(summarize(&memory))
}

pub fn get_visualization_snapshot(store_root: &Path) -> anyhow::Result<VisualizationSnapshot> {
    let _ = sync_chat_documents(store_root);
    let _ = sync_web_findings(store_root);
    let store = FileMemoryStore::new(store_root);
    let memory = store.load()?;
    let display_regions = display_regions(&memory);
    let display_region_by_id = display_regions
        .iter()
        .map(|region| (region.id.clone(), region.clone()))
        .collect::<HashMap<_, _>>();
    let map = MapBuilder::default().build(&display_regions);
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let chunk_positions = vector_positions(&memory.chunks);
    let document_chunk_counts =
        memory
            .chunks
            .iter()
            .fold(HashMap::<String, usize>::new(), |mut counts, chunk| {
                *counts.entry(chunk.document_id.clone()).or_default() += 1;
                counts
            });
    let mut region_positions = HashMap::<String, GraphPosition>::new();
    let mut document_positions = HashMap::<String, GraphPosition>::new();

    for region in &memory.regions {
        let positions = region
            .chunk_ids
            .iter()
            .filter_map(|chunk_id| chunk_positions.get(chunk_id))
            .cloned()
            .collect::<Vec<_>>();
        let position = centroid_position(&positions).unwrap_or_default();
        region_positions.insert(region.id.clone(), position.clone());
        let display_region = display_region_by_id.get(&region.id).unwrap_or(region);
        nodes.push(GraphNode {
            id: format!("region:{}", region.id),
            node_ref: NodeRef::Region(region.id.clone()),
            kind: GraphNodeKind::Region,
            label: display_region.label.clone(),
            detail: format!(
                "{} chunks · {}",
                region.chunk_ids.len(),
                display_region.summary
            ),
            score: 1.0,
            position: position.clone(),
            region_id: Some(region.id.clone()),
        });
    }

    for document in &memory.documents {
        let positions = memory
            .chunks
            .iter()
            .filter(|chunk| chunk.document_id == document.id)
            .filter_map(|chunk| chunk_positions.get(&chunk.id))
            .cloned()
            .collect::<Vec<_>>();
        if let Some(position) = centroid_position(&positions) {
            document_positions.insert(
                document.id.clone(),
                GraphPosition {
                    y: position.y + 2.2,
                    ..position
                },
            );
        }
    }

    for chunk in &memory.chunks {
        let position = chunk_positions.get(&chunk.id).cloned().unwrap_or_default();
        nodes.push(GraphNode {
            id: format!("chunk:{}", chunk.id),
            node_ref: NodeRef::Chunk(chunk.id.clone()),
            kind: GraphNodeKind::Chunk,
            label: chunk_label(&chunk.text, 32),
            detail: chunk
                .metadata
                .get("document_title")
                .cloned()
                .unwrap_or_else(|| chunk.document_id.clone()),
            score: 0.6,
            position,
            region_id: Some(chunk.region_id.clone()),
        });
    }

    for document in &memory.documents {
        let position = document_positions
            .get(&document.id)
            .cloned()
            .unwrap_or(GraphPosition {
                x: 0.0,
                y: 4.0,
                z: 0.0,
            });
        let region_id = memory
            .chunks
            .iter()
            .find(|chunk| chunk.document_id == document.id)
            .map(|chunk| chunk.region_id.clone());
        nodes.push(GraphNode {
            id: format!("document:{}", document.id),
            node_ref: NodeRef::Document(document.id.clone()),
            kind: GraphNodeKind::Document,
            label: document.title.clone(),
            detail: format!(
                "{} chunks · {}",
                document_chunk_counts
                    .get(&document.id)
                    .copied()
                    .unwrap_or(0),
                truncate(&document.text, 80)
            ),
            score: 0.9,
            position,
            region_id,
        });
    }

    add_visual_similarity_edges(&memory.chunks, &mut edges);
    for link in memory.links.iter().filter(|link| {
        matches!(
            link.link_type,
            LinkType::SemanticNeighbor | LinkType::EntityOverlap | LinkType::CitationReference
        )
    }) {
        edges.push(GraphEdge {
            id: link.id.clone(),
            source: node_key(&link.source),
            target: node_key(&link.target),
            label: link.label.clone(),
            weight: link.score.max(0.45),
        });
    }
    for chunk in &memory.chunks {
        edges.push(GraphEdge {
            id: format!("chunk-doc:{}", chunk.id),
            source: format!("chunk:{}", chunk.id),
            target: format!("document:{}", chunk.document_id),
            label: "source text".into(),
            weight: 0.24,
        });
    }

    Ok(VisualizationSnapshot {
        map,
        summary: summarize(&memory),
        nodes,
        edges,
    })
}

pub fn run_query(store_root: &Path, request: QueryRequest) -> anyhow::Result<QueryResult> {
    sync_chat_documents(store_root)?;
    sync_web_findings(store_root)?;
    let store = FileMemoryStore::new(store_root);
    let memory = load_ready_memory(store_root)?;
    let attention_marks = store.list_attention_marks(None).unwrap_or_default();
    let now = now_millis();
    let memory_accesses = store
        .list_memory_accesses(None, Some(now.saturating_sub(MEMORY_ACCESS_HORIZON_MILLIS)))
        .unwrap_or_default();
    let config = load_model_config(store_root)?;
    let embedder = embedder_for_config(&config)?;
    let ann = RegionIndexer.rebuild(&memory.chunks, &memory.regions);
    let result = MemoryQueryEngine.execute_with_signals(
        &embedder,
        &memory,
        &ann,
        request,
        &attention_marks,
        &memory_accesses,
        now,
    )?;
    record_query_hit_accesses(&store, &result);
    Ok(result)
}

pub fn grep_region(
    store_root: &Path,
    region_id: &str,
    needle: &str,
) -> anyhow::Result<Vec<ExtractHit>> {
    let memory = load_ready_memory(store_root)?;
    Ok(MemoryExtractor.grep_region(&memory, region_id, needle, DEFAULT_WINDOW))
}

pub fn open_document_excerpt(store_root: &Path, chunk_id: &str) -> anyhow::Result<DocumentExcerpt> {
    let memory = load_ready_memory(store_root)?;
    let chunk = memory
        .chunks
        .iter()
        .find(|chunk| chunk.id == chunk_id)
        .context("chunk not found")?;
    let document = memory
        .documents
        .iter()
        .find(|document| document.id == chunk.document_id)
        .context("document not found")?;
    let excerpt = DocumentExcerpt {
        document_id: document.id.clone(),
        title: document.title.clone(),
        chunk_id: chunk.id.clone(),
        excerpt: excerpt_window(&document.text, chunk.start, chunk.end, 140),
        start: chunk.start,
        end: chunk.end,
        source_anchor: chunk.source_anchor.clone(),
    };
    let store = FileMemoryStore::new(store_root);
    record_memory_access(
        &store,
        AttentionTargetKind::Chunk,
        chunk.id.clone(),
        MemoryAccessKind::Open,
        "Opened exact document excerpt for source recall.",
        "memory-runtime",
    );
    Ok(excerpt)
}

pub fn grep_document(
    store_root: &Path,
    document_id: &str,
    needle: &str,
) -> anyhow::Result<Vec<ExtractHit>> {
    let memory = load_ready_memory(store_root)?;
    Ok(MemoryExtractor.grep_document(&memory, document_id, needle, DEFAULT_WINDOW))
}

pub fn semantic_document_search(
    store_root: &Path,
    document_id: &str,
    query: &str,
) -> anyhow::Result<Vec<ExtractHit>> {
    let memory = load_ready_memory(store_root)?;
    let config = load_model_config(store_root)?;
    let embedder = embedder_for_config(&config)?;
    MemoryExtractor.semantic_document(&embedder, &memory, document_id, query, DEFAULT_WINDOW)
}

pub fn surf_open(store_root: &Path, node: &NodeRef) -> anyhow::Result<SurfOpenResult> {
    let memory = load_ready_memory(store_root)?;
    let opened = surf::open(&memory, node).context("node not found")?;
    let store = FileMemoryStore::new(store_root);
    let (target_kind, target_id) = access_target_from_node_ref(node);
    record_memory_access(
        &store,
        target_kind,
        target_id,
        MemoryAccessKind::Open,
        "Opened memory node for agent surf/source recall.",
        "memory-runtime",
    );
    Ok(opened)
}

pub fn surf_neighbors(
    store_root: &Path,
    node: &NodeRef,
    max_results: usize,
) -> anyhow::Result<Vec<SurfNeighbor>> {
    let memory = load_ready_memory(store_root)?;
    Ok(surf::neighbors(&memory, node, max_results))
}

pub fn surf_expand(
    store_root: &Path,
    chunk_id: &str,
    mode: ExpandMode,
    window: usize,
) -> anyhow::Result<SurfExpansion> {
    let memory = load_ready_memory(store_root)?;
    let expanded =
        surf::expand_chunk(&memory, chunk_id, mode, window).context("chunk not found")?;
    let store = FileMemoryStore::new(store_root);
    record_memory_access(
        &store,
        AttentionTargetKind::Chunk,
        expanded.chunk_id.clone(),
        MemoryAccessKind::Expand,
        "Expanded source context around chunk.",
        "memory-runtime",
    );
    Ok(expanded)
}

pub fn surf_jump_to_anchor(
    store_root: &Path,
    anchor_id: &str,
    window: usize,
) -> anyhow::Result<SurfExpansion> {
    let memory = load_ready_memory(store_root)?;
    let expanded = surf::jump_to_anchor(&memory, anchor_id, window).context("anchor not found")?;
    let store = FileMemoryStore::new(store_root);
    record_memory_access(
        &store,
        AttentionTargetKind::Chunk,
        expanded.chunk_id.clone(),
        MemoryAccessKind::JumpToAnchor,
        "Jumped from source anchor to exact context.",
        "memory-runtime",
    );
    Ok(expanded)
}

pub fn surf_session_step(
    store_root: &Path,
    session: SessionState,
    action: SurfAction,
) -> anyhow::Result<SurfStepResult> {
    let memory = load_ready_memory(store_root)?;
    surf::session_step(&memory, session, action).context("navigation action failed")
}

pub fn list_links(store_root: &Path, node: &NodeRef) -> anyhow::Result<Vec<Link>> {
    let memory = load_ready_memory(store_root)?;
    Ok(MemoryNavigator.list_links(&memory, node))
}

pub fn step_navigation(
    store_root: &Path,
    session: SessionState,
    link_id: &str,
) -> anyhow::Result<NavigationResult> {
    let memory = load_ready_memory(store_root)?;
    let mut next_session = session;
    MemoryNavigator
        .step(&memory, &mut next_session, link_id)
        .context("link not found")?;
    let current_excerpt = current_excerpt(&memory, next_session.current.as_ref());
    let links = next_session
        .current
        .as_ref()
        .map(|node| MemoryNavigator.list_links(&memory, node))
        .unwrap_or_default();
    Ok(NavigationResult {
        session: next_session,
        links,
        current_excerpt,
    })
}

pub fn backtrack_navigation(
    store_root: &Path,
    session: SessionState,
) -> anyhow::Result<NavigationResult> {
    let memory = load_ready_memory(store_root)?;
    let mut next_session = session;
    MemoryNavigator
        .backtrack(&mut next_session)
        .context("no history")?;
    let current_excerpt = current_excerpt(&memory, next_session.current.as_ref());
    let links = next_session
        .current
        .as_ref()
        .map(|node| MemoryNavigator.list_links(&memory, node))
        .unwrap_or_default();
    Ok(NavigationResult {
        session: next_session,
        links,
        current_excerpt,
    })
}

pub fn create_chat_session(
    store_root: &Path,
    title: Option<String>,
) -> anyhow::Result<ChatSession> {
    let store = FileMemoryStore::new(store_root);
    let created_at = now_millis();
    let session = ChatSession {
        id: unique_id("chat"),
        title: title
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "Untitled chat".into()),
        created_at,
        updated_at: created_at,
        hotness: 1.0,
    };
    store.insert_chat_session(&session)?;
    store.insert_audit_event(&AuditEvent {
        id: unique_id("audit"),
        session_id: Some(session.id.clone()),
        event_type: "chat_session.create".into(),
        target_id: session.id.clone(),
        actor: "system".into(),
        payload_json: serde_json::to_string(&session)?,
        created_at,
    })?;
    Ok(session)
}

pub fn list_chat_sessions(store_root: &Path) -> anyhow::Result<Vec<ChatSession>> {
    FileMemoryStore::new(store_root).list_chat_sessions()
}

pub fn list_chat_messages(store_root: &Path, session_id: &str) -> anyhow::Result<Vec<ChatMessage>> {
    FileMemoryStore::new(store_root).list_chat_messages(session_id)
}

pub fn list_transcript_chunks(
    store_root: &Path,
    session_id: &str,
) -> anyhow::Result<Vec<TranscriptChunk>> {
    FileMemoryStore::new(store_root).list_transcript_chunks(Some(session_id))
}

pub fn list_derived_memories(
    store_root: &Path,
    session_id: Option<String>,
) -> anyhow::Result<Vec<DerivedMemory>> {
    FileMemoryStore::new(store_root).list_derived_memories(session_id.as_deref())
}

pub fn list_attention_marks(
    store_root: &Path,
    target_id: Option<String>,
) -> anyhow::Result<Vec<AttentionMark>> {
    FileMemoryStore::new(store_root).list_attention_marks(target_id.as_deref())
}

pub fn list_audit_events(
    store_root: &Path,
    session_id: Option<String>,
) -> anyhow::Result<Vec<AuditEvent>> {
    FileMemoryStore::new(store_root).list_audit_events(session_id.as_deref())
}

pub fn list_chat_context_traces(
    store_root: &Path,
    session_id: &str,
) -> anyhow::Result<Vec<ChatContextTrace>> {
    FileMemoryStore::new(store_root).list_chat_context_traces(session_id)
}

pub fn send_chat_turn(
    store_root: &Path,
    request: ChatTurnRequest,
) -> anyhow::Result<ChatTurnResult> {
    let store = FileMemoryStore::new(store_root);
    let mut session = store
        .load_chat_session(&request.session_id)?
        .context("chat session not found")?;
    let config = load_model_config(store_root)?;
    let embedder = embedder_for_config(&config)?;

    let user_message = append_chat_message(
        &store,
        &mut session,
        ChatRole::User,
        request.message.trim().to_string(),
    )?;
    refresh_chat_session_document(store_root, &store, &session.id, &config)?;
    let context_trace = build_chat_context_trace(
        store_root,
        &store,
        &embedder,
        &config,
        &session.id,
        &user_message,
    )?;
    store.insert_chat_context_trace(&context_trace)?;

    let assistant_content = match complete_chat_response(
        store_root,
        &config,
        &user_message.content,
        &context_trace,
    ) {
        Ok(content) => content,
        Err(error) => {
            let _ = store.insert_audit_event(&AuditEvent {
                id: unique_id("audit"),
                session_id: Some(session.id.clone()),
                event_type: "chat.response.error".into(),
                target_id: user_message.id.clone(),
                actor: "assistant".into(),
                payload_json: serde_json::json!({ "error": error.to_string() }).to_string(),
                created_at: now_millis(),
            });
            format!(
                "I saved this turn and retrieved {} memory snippet(s), but the local response model is not reachable yet: {}",
                context_trace.snippets.len(),
                error
            )
        }
    };
    let assistant_message =
        append_chat_message(&store, &mut session, ChatRole::Assistant, assistant_content)?;
    refresh_chat_session_document(store_root, &store, &session.id, &config)?;

    let derived =
        extract_turn_memories(store_root, &session.id, &user_message, &assistant_message)?;
    let _ = apply_attention_mark(
        store_root,
        AttentionMarkWrite {
            target_id: session.id.clone(),
            target_kind: AttentionTargetKind::ChatSession,
            action: AttentionAction::Active,
            reason: "Most recent chat turn is active context.".into(),
            actor: "assistant".into(),
        },
    )?;

    Ok(ChatTurnResult {
        session,
        messages: vec![user_message, assistant_message],
        context_trace,
        derived_memories: derived,
    })
}

pub fn write_derived_memory(
    store_root: &Path,
    write: DerivedMemoryWrite,
) -> anyhow::Result<DerivedMemory> {
    let store = FileMemoryStore::new(store_root);
    let created_at = now_millis();
    let source_refs = write
        .source_message_ids
        .iter()
        .map(|id| format!("imprint://chat-message/{id}"))
        .collect::<Vec<_>>();
    let memory = DerivedMemory {
        id: unique_id("derived"),
        session_id: write.session_id.clone(),
        kind: write.kind,
        text: write.text,
        source_message_ids: write.source_message_ids,
        actor: write.actor,
        confidence: write.confidence,
        created_at,
        provenance: ProvenanceRecord {
            actor: "assistant".into(),
            reason: "Derived from chat transcript.".into(),
            created_at,
            source_refs,
        },
    };
    store.insert_derived_memory(&memory)?;
    audit_write(
        &store,
        memory.session_id.clone(),
        "derived_memory.write",
        &memory.id,
        &memory.actor,
        &memory,
    )?;
    Ok(memory)
}

pub fn write_web_finding(store_root: &Path, write: WebFindingWrite) -> anyhow::Result<WebFinding> {
    let store = FileMemoryStore::new(store_root);
    let config = load_model_config(store_root)?;
    let created_at = now_millis();
    let finding = WebFinding {
        id: unique_id("web"),
        session_id: write.session_id.clone(),
        query: write.query,
        url: write.url.clone(),
        title: write.title,
        summary: write.summary,
        retrieved_at: write.retrieved_at,
        confidence: write.confidence,
        actor: write.actor,
        created_at,
        provenance: ProvenanceRecord {
            actor: "assistant".into(),
            reason: "Web finding supplied by user or agent.".into(),
            created_at,
            source_refs: vec![write.url],
        },
    };
    store.insert_web_finding(&finding)?;
    audit_write(
        &store,
        finding.session_id.clone(),
        "web_finding.write",
        &finding.id,
        &finding.actor,
        &finding,
    )?;
    sync_web_findings_with_config(store_root, &store, &config)?;
    Ok(finding)
}

pub fn compile_memory_brain(store_root: &Path) -> anyhow::Result<BrainCompileResult> {
    let store = FileMemoryStore::new(store_root);
    let config = load_model_config(store_root)?;
    let memory = store.load()?;
    let compile_memory = memory_for_brain_compile(store_root, &memory, &config)?;
    let created_at = now_millis();
    let artifacts = crate::compiler::build_brain_artifacts(&compile_memory, created_at);

    let mut artifact_ids = Vec::new();
    for artifact in &artifacts {
        store.insert_brain_artifact(artifact)?;
        store.insert_derived_memory(&brain_artifact_derived_memory(artifact))?;
        audit_write(
            &store,
            None,
            "brain.compile",
            &artifact.id,
            "memory-compiler",
            artifact,
        )?;
        artifact_ids.push(artifact.id.clone());
    }
    artifact_ids.sort();
    let training_records_path = store_root.join("brain-training.jsonl");
    let training_records = artifacts
        .iter()
        .map(|artifact| {
            serde_json::json!({
                "id": artifact.id,
                "task": "memory_routing",
                "input": artifact.body,
                "target": "Use this compact artifact to choose memory regions, then expand source anchors before answering.",
                "provenance": artifact.provenance,
            })
            .to_string()
        })
        .collect::<Vec<_>>();
    std::fs::create_dir_all(store_root)?;
    std::fs::write(&training_records_path, training_records.join("\n"))?;
    let export_files = crate::training::write_training_exports(store_root, &artifacts)?;
    let source_dataset_hash = crate::training::training_source_hash(&store_root.join("training"))?;
    let initial_adapter_state = crate::training::read_cortex_adapter_state(
        store_root,
        source_dataset_hash.clone(),
        created_at,
    )?;
    if initial_adapter_state.freshness != "fresh" {
        let compiler_model = config
            .compiler_model
            .as_deref()
            .or(config.response_model.as_deref())
            .unwrap_or(DEFAULT_RESPONSE_MODEL);
        crate::training::prepare_cortex_adapter_dataset(
            store_root,
            compiler_model,
            &source_dataset_hash,
        )?;
    }
    let adapter_state =
        crate::training::read_cortex_adapter_state(store_root, source_dataset_hash, created_at)?;
    store.save_cortex_adapter_state(&adapter_state)?;
    sync_brain_artifacts_with_config(store_root, &store, &config)?;
    sync_derived_memories_with_config(store_root, &store, &config)?;
    Ok(BrainCompileResult {
        artifacts_written: artifact_ids.len(),
        artifact_ids,
        training_records_written: training_records.len(),
        training_records_path: training_records_path.display().to_string(),
        export_files,
        adapter_state: Some(adapter_state),
    })
}

pub fn load_cortex_adapter_snapshot(store_root: &Path) -> anyhow::Result<CortexAdapterSnapshot> {
    let store = FileMemoryStore::new(store_root);
    Ok(CortexAdapterSnapshot {
        adapter_state: store.load_cortex_adapter_state()?,
    })
}

fn memory_for_brain_compile(
    store_root: &Path,
    memory: &PersistedMemory,
    config: &ModelConfig,
) -> anyhow::Result<PersistedMemory> {
    let source_documents = memory
        .documents
        .iter()
        .filter(|document| !is_compiler_generated_document(document))
        .cloned()
        .collect::<Vec<_>>();
    if source_documents.len() == memory.documents.len() {
        return Ok(memory.clone());
    }
    let (source_memory, _) =
        rebuild_from_documents(store_root, source_documents, &memory.chunks, config)?;
    Ok(source_memory)
}

fn is_compiler_generated_document(document: &Document) -> bool {
    matches!(
        document.metadata.get("source_type").map(String::as_str),
        Some("brain_artifact")
    ) || (document.metadata.get("source_type").map(String::as_str) == Some("derived_memory")
        && document.metadata.get("actor").map(String::as_str) == Some("memory-compiler"))
}

pub fn write_agent_link(
    store_root: &Path,
    write: AgentLinkWrite,
) -> anyhow::Result<AgentLinkMemory> {
    let store = FileMemoryStore::new(store_root);
    let created_at = now_millis();
    let link = AgentLinkMemory {
        id: unique_id("agent-link"),
        source_id: write.source_id,
        target_id: write.target_id,
        label: write.label,
        actor: write.actor,
        created_at,
        provenance: ProvenanceRecord {
            actor: "assistant".into(),
            reason: "Agent-created memory link.".into(),
            created_at,
            source_refs: Vec::new(),
        },
    };
    store.insert_agent_link(&link)?;
    audit_write(
        &store,
        None,
        "agent_link.write",
        &link.id,
        &link.actor,
        &link,
    )?;
    Ok(link)
}

pub fn apply_attention_mark(
    store_root: &Path,
    write: AttentionMarkWrite,
) -> anyhow::Result<AttentionMark> {
    let store = FileMemoryStore::new(store_root);
    let created_at = now_millis();
    let audit_session_id =
        attention_target_session_id(&store, &write.target_kind, &write.target_id)?;
    let mark = AttentionMark {
        id: unique_id("attention"),
        target_id: write.target_id,
        target_kind: write.target_kind,
        action: write.action,
        reason: write.reason,
        actor: write.actor,
        created_at,
        reverted_at: None,
    };
    store.insert_attention_mark(&mark)?;
    audit_write(
        &store,
        audit_session_id,
        "attention.mark",
        &mark.target_id,
        &mark.actor,
        &mark,
    )?;
    Ok(mark)
}

pub fn save_model_config(store_root: &Path, config: &ModelConfig) -> anyhow::Result<ModelConfig> {
    std::fs::create_dir_all(store_root)?;
    let path = model_config_path(store_root);
    let raw = serde_json::to_string_pretty(config)?;
    std::fs::write(path, raw)?;
    Ok(config.clone())
}

pub fn load_model_config(store_root: &Path) -> anyhow::Result<ModelConfig> {
    let path = model_config_path(store_root);
    if !path.exists() {
        return Ok(ModelConfig {
            mode: ModelConnectionMode::Local,
            endpoint: "http://localhost:8080/v1".into(),
            api_key_name: None,
            chat_model: Some(DEFAULT_RESPONSE_MODEL.into()),
            planner_model: Some(DEFAULT_PLANNER_MODEL.into()),
            response_model: Some(DEFAULT_RESPONSE_MODEL.into()),
            planner_endpoint: None,
            runtime_preset: ModelRuntimePreset::Mlx,
            cortex_enabled: default_cortex_enabled(),
            cortex_rounds: default_cortex_rounds(),
            critic_model: Some(DEFAULT_PLANNER_MODEL.into()),
            critic_endpoint: None,
            compiler_model: Some(DEFAULT_RESPONSE_MODEL.into()),
            embedding_model: Some(DEFAULT_LOCAL_EMBEDDING_MODEL.into()),
            embedding_endpoint: Some(DEFAULT_LOCAL_ENDPOINT.into()),
            embedding_runtime_preset: Some(ModelRuntimePreset::Ollama),
            health: Some(ModelHealth {
                status: "disconnected".into(),
                message: "Using local MLX-compatible chat with local embeddings".into(),
                checked_at: None,
            }),
        });
    }
    let raw = std::fs::read_to_string(path)?;
    let mut config: ModelConfig = serde_json::from_str(&raw)?;
    if config.mode == ModelConnectionMode::Local {
        if config.endpoint.trim().is_empty() {
            config.endpoint = DEFAULT_LOCAL_ENDPOINT.into();
        }
        if config
            .embedding_model
            .as_deref()
            .map(str::trim)
            .unwrap_or_default()
            .is_empty()
        {
            config.embedding_model = Some(DEFAULT_LOCAL_EMBEDDING_MODEL.into());
        }
    }
    if config
        .embedding_endpoint
        .as_deref()
        .map(str::trim)
        .unwrap_or_default()
        .is_empty()
    {
        config.embedding_endpoint = Some(config.endpoint.clone());
    }
    if config.embedding_runtime_preset.is_none() {
        config.embedding_runtime_preset = Some(config.runtime_preset.clone());
    }
    if config
        .planner_model
        .as_deref()
        .map(str::trim)
        .unwrap_or_default()
        .is_empty()
    {
        config.planner_model = Some(DEFAULT_PLANNER_MODEL.into());
    }
    if config
        .response_model
        .as_deref()
        .map(str::trim)
        .unwrap_or_default()
        .is_empty()
    {
        config.response_model = config
            .chat_model
            .clone()
            .or_else(|| Some(DEFAULT_RESPONSE_MODEL.into()));
    }
    if config
        .planner_endpoint
        .as_deref()
        .map(str::trim)
        .unwrap_or_default()
        .is_empty()
    {
        config.planner_endpoint = Some(config.endpoint.clone());
    }
    config.cortex_rounds = config.cortex_rounds.clamp(1, 4);
    if config
        .critic_model
        .as_deref()
        .map(str::trim)
        .unwrap_or_default()
        .is_empty()
    {
        config.critic_model = config.planner_model.clone();
    }
    if config
        .critic_endpoint
        .as_deref()
        .map(str::trim)
        .unwrap_or_default()
        .is_empty()
    {
        config.critic_endpoint = config.planner_endpoint.clone();
    }
    if config
        .compiler_model
        .as_deref()
        .map(str::trim)
        .unwrap_or_default()
        .is_empty()
    {
        config.compiler_model = config.response_model.clone();
    }
    Ok(config)
}

pub fn test_model_connection(request: ModelConnectionTestRequest) -> anyhow::Result<ModelHealth> {
    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(8))
        .build()?;
    let endpoint = request
        .embedding_endpoint
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or(&request.endpoint)
        .trim_end_matches('/');
    let checked_at = Some(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    );

    let response = match request.mode {
        ModelConnectionMode::Local => {
            let model = request
                .embedding_model
                .as_deref()
                .filter(|value| !value.trim().is_empty())
                .unwrap_or(DEFAULT_LOCAL_EMBEDDING_MODEL);
            if model == HASH_EMBEDDING_MODEL {
                return Ok(ModelHealth {
                    status: "connected".into(),
                    message: "Using offline hash embedder".into(),
                    checked_at,
                });
            }
            let embedder = match request
                .embedding_runtime_preset
                .unwrap_or(request.runtime_preset)
            {
                ModelRuntimePreset::Ollama => {
                    RuntimeEmbedder::Ollama(OllamaEmbedder::new(endpoint, model)?)
                }
                ModelRuntimePreset::Mlx
                | ModelRuntimePreset::LlamaCpp
                | ModelRuntimePreset::CustomOpenAi => {
                    RuntimeEmbedder::OpenAi(OpenAiCompatibleEmbedder::new(endpoint, model)?)
                }
            };
            embedder.embed("AI memory embedding health check")?;
            return Ok(ModelHealth {
                status: "connected".into(),
                message: format!("Local embedding model {model} responded"),
                checked_at,
            });
        }
        ModelConnectionMode::Api => client
            .get(format!("{endpoint}/v1/models"))
            .bearer_auth(request.api_key.unwrap_or_default())
            .send(),
    }?;
    response.error_for_status()?;
    Ok(ModelHealth {
        status: "connected".into(),
        message: "Model endpoint responded".into(),
        checked_at,
    })
}

fn rebuild_from_documents(
    store_root: &Path,
    documents: Vec<Document>,
    reusable_chunks: &[Chunk],
    config: &ModelConfig,
) -> anyhow::Result<(PersistedMemory, EmbeddingReuseStats)> {
    let ingester = Ingester::new(embedder_for_config(config)?);
    let (mut memory, stats) = ingester.ingest_documents_reusing_embeddings(
        documents,
        reusable_chunks,
        |completed, total| {
            let _ = write_progress(
                store_root,
                "embed",
                completed,
                total.max(1),
                &format!("Embedding/reusing chunks {completed}/{}", total.max(1)),
            );
        },
    )?;
    write_progress(store_root, "graph", 96, 100, "Building graph links")?;
    GraphBuilder.build(&mut memory);
    write_progress(store_root, "map", 98, 100, "Building memory map")?;
    memory.memory_map = Some(MapBuilder::default().build(&memory.regions));
    Ok((memory, stats))
}

fn write_progress(
    store_root: &Path,
    phase: &str,
    completed: usize,
    total: usize,
    message: &str,
) -> anyhow::Result<()> {
    write_progress_with_presence(store_root, phase, completed, total, message, &[], None)
}

fn write_progress_with_presence(
    store_root: &Path,
    phase: &str,
    completed: usize,
    total: usize,
    message: &str,
    active_node_ids: &[String],
    active_node_label: Option<&str>,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(store_root)?;
    let total = total.max(1);
    let percent = ((completed as f32 / total as f32) * 100.0).clamp(0.0, 100.0);
    let payload = OperationProgress {
        phase: phase.into(),
        completed,
        total,
        percent,
        message: message.into(),
        active_node_ids: active_node_ids.to_vec(),
        active_node_label: active_node_label.map(str::to_string),
    };
    let raw = serde_json::to_string(&payload)?;
    std::fs::write(progress_path(store_root), raw)?;
    Ok(())
}

fn progress_path(store_root: &Path) -> PathBuf {
    store_root.join("progress.json")
}

fn embedder_for_config(config: &ModelConfig) -> anyhow::Result<RuntimeEmbedder> {
    let model = config
        .embedding_model
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(match config.mode {
            ModelConnectionMode::Local => DEFAULT_LOCAL_EMBEDDING_MODEL,
            ModelConnectionMode::Api => HASH_EMBEDDING_MODEL,
        });
    if model == HASH_EMBEDDING_MODEL {
        return Ok(RuntimeEmbedder::Hash(HashEmbedder::default()));
    }
    let endpoint = config
        .embedding_endpoint
        .as_deref()
        .unwrap_or(&config.endpoint);
    let runtime = config
        .embedding_runtime_preset
        .clone()
        .unwrap_or_else(|| config.runtime_preset.clone());
    match runtime {
        ModelRuntimePreset::Ollama => Ok(RuntimeEmbedder::Ollama(OllamaEmbedder::new(
            endpoint, model,
        )?)),
        ModelRuntimePreset::Mlx
        | ModelRuntimePreset::LlamaCpp
        | ModelRuntimePreset::CustomOpenAi => Ok(RuntimeEmbedder::OpenAi(
            OpenAiCompatibleEmbedder::new(endpoint, model)?,
        )),
    }
}

fn load_ready_memory(store_root: &Path) -> anyhow::Result<PersistedMemory> {
    let store = FileMemoryStore::new(store_root);
    let memory = store.load()?;
    if memory.documents.is_empty() {
        return Err(anyhow!("store {} is empty", store.root().display()));
    }
    Ok(memory)
}

fn summarize(memory: &PersistedMemory) -> MemorySummary {
    MemorySummary {
        documents: memory.documents.len(),
        chunks: memory.chunks.len(),
        regions: memory.regions.len(),
        links: memory.links.len(),
        map_bytes: memory
            .memory_map
            .as_ref()
            .map(|map| map.serialized.len())
            .unwrap_or(0),
    }
}

fn excerpt_window(text: &str, start: usize, end: usize, pad: usize) -> String {
    let chars = text.chars().collect::<Vec<_>>();
    let left = start.saturating_sub(pad);
    let right = (end + pad).min(chars.len());
    chars[left..right].iter().collect()
}

fn truncate(value: &str, max: usize) -> String {
    value.chars().take(max).collect()
}

fn chunk_label(text: &str, max: usize) -> String {
    let cleaned = text
        .split_whitespace()
        .filter(|part| !part.trim().is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    let label = cleaned
        .trim_matches(|ch: char| !ch.is_alphanumeric())
        .to_string();
    truncate(
        if label.is_empty() {
            text.trim()
        } else {
            &label
        },
        max,
    )
}

fn display_regions(memory: &PersistedMemory) -> Vec<Region> {
    memory
        .regions
        .iter()
        .map(|region| {
            let terms = meaningful_terms_for_chunks(
                memory
                    .chunks
                    .iter()
                    .filter(|chunk| chunk.region_id == region.id),
                4,
            );
            if terms.is_empty() {
                return region.clone();
            }
            let mut display = region.clone();
            display.label = format_region_label(&terms);
            display.summary = format!("Vector region around {}", terms.join(", "));
            display
                .filters
                .insert("topic".into(), display.label.clone());
            display
        })
        .collect()
}

fn node_key(node: &NodeRef) -> String {
    match node {
        NodeRef::Document(id) => format!("document:{id}"),
        NodeRef::Chunk(id) => format!("chunk:{id}"),
        NodeRef::Region(id) => format!("region:{id}"),
    }
}

fn vector_positions(chunks: &[Chunk]) -> HashMap<String, GraphPosition> {
    if chunks.is_empty() {
        return HashMap::new();
    }
    let basis = projection_basis(chunks);
    let mut raw = chunks
        .iter()
        .map(|chunk| {
            let x = dot(&chunk.embedding, &basis[0]);
            let y = dot(&chunk.embedding, &basis[1]);
            let z = dot(&chunk.embedding, &basis[2]);
            (chunk.id.clone(), GraphPosition { x, y, z })
        })
        .collect::<Vec<_>>();
    normalize_positions(&mut raw);
    raw.into_iter().collect()
}

fn projection_basis(chunks: &[Chunk]) -> [Vec<f32>; 3] {
    let dimensions = chunks
        .iter()
        .map(|chunk| chunk.embedding.len())
        .find(|length| *length > 0)
        .unwrap_or(1);
    [
        random_unit_vector(dimensions, 0x9E37_79B9_7F4A_7C15),
        random_unit_vector(dimensions, 0xC2B2_AE3D_27D4_EB4F),
        random_unit_vector(dimensions, 0x1656_67B1_9E37_79F9),
    ]
}

fn random_unit_vector(dimensions: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut vector = Vec::with_capacity(dimensions);
    for _ in 0..dimensions {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let value = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        let unit = (value as f64 / u64::MAX as f64) as f32;
        vector.push(unit * 2.0 - 1.0);
    }
    normalize_vector(&mut vector);
    vector
}

fn normalize_vector(vector: &mut [f32]) {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in vector {
            *value /= norm;
        }
    }
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn normalize_positions(points: &mut [(String, GraphPosition)]) {
    if points.is_empty() {
        return;
    }
    let count = points.len() as f32;
    let center = points
        .iter()
        .fold(GraphPosition::default(), |mut acc, (_, point)| {
            acc.x += point.x;
            acc.y += point.y;
            acc.z += point.z;
            acc
        });
    let center = GraphPosition {
        x: center.x / count,
        y: center.y / count,
        z: center.z / count,
    };
    let mut max_radius = 0.001_f32;
    for (_, point) in points.iter_mut() {
        point.x -= center.x;
        point.y -= center.y;
        point.z -= center.z;
        max_radius =
            max_radius.max((point.x * point.x + point.y * point.y + point.z * point.z).sqrt());
    }
    let scale = VISUAL_SCALE / max_radius;
    for (_, point) in points {
        point.x *= scale;
        point.y *= scale;
        point.z *= scale;
    }
}

fn centroid_position(points: &[GraphPosition]) -> Option<GraphPosition> {
    if points.is_empty() {
        return None;
    }
    let count = points.len() as f32;
    let sum = points
        .iter()
        .fold(GraphPosition::default(), |mut acc, point| {
            acc.x += point.x;
            acc.y += point.y;
            acc.z += point.z;
            acc
        });
    Some(GraphPosition {
        x: sum.x / count,
        y: sum.y / count,
        z: sum.z / count,
    })
}

fn add_visual_similarity_edges(chunks: &[Chunk], edges: &mut Vec<GraphEdge>) {
    for chunk in chunks {
        let mut neighbors = chunks
            .iter()
            .filter(|other| other.id != chunk.id)
            .map(|other| {
                (
                    other.id.clone(),
                    cosine_similarity(&chunk.embedding, &other.embedding),
                )
            })
            .filter(|(_, score)| *score > 0.35)
            .collect::<Vec<_>>();
        neighbors.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.0.cmp(&right.0))
        });
        for (neighbor_id, score) in neighbors.into_iter().take(MAX_VISUAL_CHUNK_NEIGHBORS) {
            let (left, right) = if chunk.id < neighbor_id {
                (chunk.id.as_str(), neighbor_id.as_str())
            } else {
                (neighbor_id.as_str(), chunk.id.as_str())
            };
            edges.push(GraphEdge {
                id: format!("visual-sim:{left}:{right}"),
                source: format!("chunk:{}", chunk.id),
                target: format!("chunk:{neighbor_id}"),
                label: "semantic proximity".into(),
                weight: score,
            });
        }
    }
    let mut seen = HashSet::new();
    edges.retain(|edge| seen.insert(edge.id.clone()));
}

fn current_excerpt(memory: &PersistedMemory, node: Option<&NodeRef>) -> Option<String> {
    match node? {
        NodeRef::Document(document_id) => memory
            .documents
            .iter()
            .find(|document| &document.id == document_id)
            .map(|document| truncate(&document.text, 240)),
        NodeRef::Chunk(chunk_id) => memory
            .chunks
            .iter()
            .find(|chunk| &chunk.id == chunk_id)
            .map(|chunk| truncate(&chunk.text, 240)),
        NodeRef::Region(region_id) => memory
            .regions
            .iter()
            .find(|region| &region.id == region_id)
            .map(|region| region.summary.clone()),
    }
}

fn append_chat_message(
    store: &FileMemoryStore,
    session: &mut ChatSession,
    role: ChatRole,
    content: String,
) -> anyhow::Result<ChatMessage> {
    if content.trim().is_empty() {
        anyhow::bail!("chat message is empty");
    }
    let created_at = now_millis();
    let id = unique_id("message");
    let anchor = chat_source_anchor(&session.id, &id, &content, created_at);
    let message = ChatMessage {
        id: id.clone(),
        session_id: session.id.clone(),
        role: role.clone(),
        content: content.clone(),
        created_at,
        token_estimate: estimate_tokens(&content),
        source_anchor: Some(anchor.clone()),
    };
    store.insert_chat_message(&message)?;

    session.updated_at = created_at;
    session.hotness = 1.0;
    store.update_chat_session_timestamp(&session.id, session.updated_at, session.hotness)?;
    store.insert_audit_event(&AuditEvent {
        id: unique_id("audit"),
        session_id: Some(session.id.clone()),
        event_type: "chat_message.write".into(),
        target_id: id,
        actor: match role {
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
            ChatRole::System => "system",
            ChatRole::Tool => "tool",
        }
        .into(),
        payload_json: serde_json::to_string(&message)?,
        created_at,
    })?;
    Ok(message)
}

fn refresh_chat_session_document(
    store_root: &Path,
    store: &FileMemoryStore,
    session_id: &str,
    config: &ModelConfig,
) -> anyhow::Result<()> {
    sync_chat_documents_with_config(store_root, store, config, Some(session_id))
}

fn sync_chat_documents(store_root: &Path) -> anyhow::Result<()> {
    let store = FileMemoryStore::new(store_root);
    let config = load_model_config(store_root)?;
    sync_chat_documents_with_config(store_root, &store, &config, None)
}

fn sync_chat_documents_with_config(
    store_root: &Path,
    store: &FileMemoryStore,
    config: &ModelConfig,
    session_filter: Option<&str>,
) -> anyhow::Result<()> {
    let sessions = store
        .list_chat_sessions()?
        .into_iter()
        .filter(|session| session_filter.map(|id| id == session.id).unwrap_or(true))
        .collect::<Vec<_>>();
    if sessions.is_empty() {
        return Ok(());
    }
    let mut chat_documents = Vec::new();
    for session in &sessions {
        let messages = store.list_chat_messages(&session.id)?;
        if !messages.is_empty() {
            chat_documents.push(chat_session_document(session, &messages));
        }
    }
    if chat_documents.is_empty() {
        return Ok(());
    }
    let existing = store.load()?;
    let chat_document_ids = chat_documents
        .iter()
        .map(|document| document.id.clone())
        .collect::<HashSet<_>>();
    let needs_rebuild = chat_documents.iter().any(|document| {
        existing
            .documents
            .iter()
            .find(|candidate| candidate.id == document.id)
            .map(|candidate| {
                candidate.text != document.text || candidate.content_hash != document.content_hash
            })
            .unwrap_or(true)
    });
    let has_legacy_transcript_chunks =
        store.list_transcript_chunks(session_filter)?.is_empty() == false;
    if !needs_rebuild {
        if has_legacy_transcript_chunks {
            store.delete_transcript_chunks(session_filter)?;
        }
        return Ok(());
    }
    let mut documents = existing
        .documents
        .into_iter()
        .filter(|existing| !chat_document_ids.contains(&existing.id))
        .collect::<Vec<_>>();
    documents.extend(chat_documents);
    documents.sort_by(|left, right| left.id.cmp(&right.id));
    let (memory, _) = rebuild_from_documents(store_root, documents, &existing.chunks, config)?;
    store.save(&memory)?;
    if has_legacy_transcript_chunks {
        store.delete_transcript_chunks(session_filter)?;
    }
    Ok(())
}

fn sync_web_findings(store_root: &Path) -> anyhow::Result<()> {
    let store = FileMemoryStore::new(store_root);
    let config = load_model_config(store_root)?;
    sync_web_findings_with_config(store_root, &store, &config)
}

fn sync_web_findings_with_config(
    store_root: &Path,
    store: &FileMemoryStore,
    config: &ModelConfig,
) -> anyhow::Result<()> {
    let findings = store.list_web_findings(None)?;
    if findings.is_empty() {
        return Ok(());
    }
    let mut seen_web_document_ids = HashSet::new();
    let web_documents = findings
        .iter()
        .filter_map(|finding| {
            let id = web_finding_document_id(finding);
            seen_web_document_ids
                .insert(id)
                .then(|| web_finding_document(finding))
        })
        .collect::<Vec<_>>();
    let existing = store.load()?;
    let web_document_ids = web_documents
        .iter()
        .map(|document| document.id.clone())
        .collect::<HashSet<_>>();
    let needs_rebuild = web_documents.iter().any(|document| {
        existing
            .documents
            .iter()
            .find(|candidate| candidate.id == document.id)
            .map(|candidate| {
                candidate.text != document.text || candidate.content_hash != document.content_hash
            })
            .unwrap_or(true)
    });
    if !needs_rebuild {
        return Ok(());
    }
    let mut documents = existing
        .documents
        .into_iter()
        .filter(|existing| !web_document_ids.contains(&existing.id))
        .collect::<Vec<_>>();
    documents.extend(web_documents);
    documents.sort_by(|left, right| left.id.cmp(&right.id));
    let (memory, _) = rebuild_from_documents(store_root, documents, &existing.chunks, config)?;
    store.save(&memory)?;
    Ok(())
}

fn sync_derived_memories_with_config(
    store_root: &Path,
    store: &FileMemoryStore,
    config: &ModelConfig,
) -> anyhow::Result<()> {
    let derived = store.list_derived_memories(None)?;
    if derived.is_empty() {
        return Ok(());
    }
    let derived_documents = derived
        .iter()
        .map(derived_memory_document)
        .collect::<Vec<_>>();
    let existing = store.load()?;
    let derived_document_ids = derived_documents
        .iter()
        .map(|document| document.id.clone())
        .collect::<HashSet<_>>();
    let needs_rebuild = derived_documents.iter().any(|document| {
        existing
            .documents
            .iter()
            .find(|candidate| candidate.id == document.id)
            .map(|candidate| {
                candidate.text != document.text || candidate.content_hash != document.content_hash
            })
            .unwrap_or(true)
    });
    if !needs_rebuild {
        return Ok(());
    }
    let mut documents = existing
        .documents
        .into_iter()
        .filter(|existing| !derived_document_ids.contains(&existing.id))
        .collect::<Vec<_>>();
    documents.extend(derived_documents);
    documents.sort_by(|left, right| left.id.cmp(&right.id));
    let (memory, _) = rebuild_from_documents(store_root, documents, &existing.chunks, config)?;
    store.save(&memory)?;
    Ok(())
}

fn sync_brain_artifacts_with_config(
    store_root: &Path,
    store: &FileMemoryStore,
    config: &ModelConfig,
) -> anyhow::Result<()> {
    let artifacts = store.list_brain_artifacts(None)?;
    if artifacts.is_empty() {
        return Ok(());
    }
    let artifact_documents = artifacts
        .iter()
        .map(brain_artifact_document)
        .collect::<Vec<_>>();
    let existing = store.load()?;
    let artifact_document_ids = artifact_documents
        .iter()
        .map(|document| document.id.clone())
        .collect::<HashSet<_>>();
    let needs_rebuild = artifact_documents.iter().any(|document| {
        existing
            .documents
            .iter()
            .find(|candidate| candidate.id == document.id)
            .map(|candidate| {
                candidate.text != document.text || candidate.content_hash != document.content_hash
            })
            .unwrap_or(true)
    });
    if !needs_rebuild {
        return Ok(());
    }
    let mut documents = existing
        .documents
        .into_iter()
        .filter(|existing| !artifact_document_ids.contains(&existing.id))
        .collect::<Vec<_>>();
    documents.extend(artifact_documents);
    documents.sort_by(|left, right| left.id.cmp(&right.id));
    let (memory, _) = rebuild_from_documents(store_root, documents, &existing.chunks, config)?;
    store.save(&memory)?;
    Ok(())
}

fn chat_session_document(session: &ChatSession, messages: &[ChatMessage]) -> Document {
    let id = chat_document_id(&session.id);
    let text = chat_transcript_text(session, messages);
    let content_hash = hash_text(&text);
    let path = format!("imprint://chat/{}", session.id);
    let mut metadata = BTreeMap::new();
    metadata.insert("path".into(), path.clone());
    metadata.insert("source".into(), "chat".into());
    metadata.insert("source_type".into(), "chat".into());
    metadata.insert("chat_session_id".into(), session.id.clone());
    metadata.insert("content_hash".into(), content_hash.clone());
    metadata.insert("parser_version".into(), "1".into());
    let anchor = SourceAnchor {
        id: format!("{id}:document"),
        document_id: id.clone(),
        chunk_id: None,
        path,
        content_hash: content_hash.clone(),
        start: 0,
        end: text.chars().count(),
        page: None,
        section: Some("Chat transcript".into()),
        parser_version: 1,
    };
    Document {
        id,
        title: session.title.clone(),
        text,
        metadata,
        source_anchor: Some(anchor),
        content_hash: Some(content_hash),
        parser_version: Some(1),
    }
}

fn chat_document_id(session_id: &str) -> String {
    format!("chat:{session_id}")
}

fn web_finding_document(finding: &WebFinding) -> Document {
    let id = web_finding_document_id(finding);
    let text = format!(
        "Web finding: {}\nURL: {}\nQuery: {}\nRetrieved at: {}\nConfidence: {:.2}\n\n{}",
        finding.title,
        finding.url,
        finding.query,
        finding.retrieved_at,
        finding.confidence,
        finding.summary
    );
    let content_hash = hash_text(&text);
    let mut metadata = BTreeMap::new();
    metadata.insert("path".into(), finding.url.clone());
    metadata.insert("source".into(), "web".into());
    metadata.insert("source_type".into(), "web_finding".into());
    metadata.insert("web_finding_id".into(), finding.id.clone());
    metadata.insert(
        "session_id".into(),
        finding.session_id.clone().unwrap_or_default(),
    );
    metadata.insert("query".into(), finding.query.clone());
    metadata.insert("url".into(), finding.url.clone());
    metadata.insert("retrieved_at".into(), finding.retrieved_at.to_string());
    metadata.insert("confidence".into(), format!("{:.2}", finding.confidence));
    metadata.insert("content_hash".into(), content_hash.clone());
    metadata.insert("parser_version".into(), PARSER_VERSION.to_string());
    let anchor = SourceAnchor {
        id: format!("{id}:document"),
        document_id: id.clone(),
        chunk_id: None,
        path: finding.url.clone(),
        content_hash: content_hash.clone(),
        start: 0,
        end: text.chars().count(),
        page: None,
        section: Some("Web finding".into()),
        parser_version: PARSER_VERSION,
    };
    Document {
        id,
        title: finding.title.clone(),
        text,
        metadata,
        source_anchor: Some(anchor),
        content_hash: Some(content_hash),
        parser_version: Some(PARSER_VERSION),
    }
}

fn web_finding_document_id(finding: &WebFinding) -> String {
    format!("web:{}", hash_text(&finding.url))
}

fn derived_memory_document(memory: &DerivedMemory) -> Document {
    let id = derived_memory_document_id(memory);
    let text = format!(
        "Derived memory: {:?}\nActor: {}\nConfidence: {:.2}\n\n{}",
        memory.kind, memory.actor, memory.confidence, memory.text
    );
    let content_hash = hash_text(&text);
    let path = format!("imprint://derived/{}", memory.id);
    let mut metadata = BTreeMap::new();
    metadata.insert("path".into(), path.clone());
    metadata.insert("source".into(), "derived".into());
    metadata.insert("source_type".into(), "derived_memory".into());
    metadata.insert("derived_memory_id".into(), memory.id.clone());
    metadata.insert("actor".into(), memory.actor.clone());
    metadata.insert("confidence".into(), format!("{:.2}", memory.confidence));
    metadata.insert("content_hash".into(), content_hash.clone());
    metadata.insert("parser_version".into(), PARSER_VERSION.to_string());
    let anchor = SourceAnchor {
        id: format!("{id}:document"),
        document_id: id.clone(),
        chunk_id: None,
        path,
        content_hash: content_hash.clone(),
        start: 0,
        end: text.chars().count(),
        page: None,
        section: Some("Derived memory".into()),
        parser_version: PARSER_VERSION,
    };
    Document {
        id,
        title: format!("Derived memory {}", memory.id),
        text,
        metadata,
        source_anchor: Some(anchor),
        content_hash: Some(content_hash),
        parser_version: Some(PARSER_VERSION),
    }
}

fn derived_memory_document_id(memory: &DerivedMemory) -> String {
    format!("derived:{}", hash_text(&memory.id))
}

fn brain_artifact_derived_memory(artifact: &BrainArtifact) -> DerivedMemory {
    DerivedMemory {
        id: artifact.id.clone(),
        session_id: None,
        kind: DerivedMemoryKind::Summary,
        text: artifact.body.clone(),
        source_message_ids: Vec::new(),
        actor: "memory-compiler".into(),
        confidence: artifact.confidence as f32 / 100.0,
        created_at: artifact.created_at,
        provenance: artifact.provenance.clone(),
    }
}

fn brain_artifact_document(artifact: &BrainArtifact) -> Document {
    let id = brain_artifact_document_id(artifact);
    let text = format!(
        "Brain artifact: {}\nKind: {:?}\nConfidence: {}\n\n{}",
        artifact.title, artifact.kind, artifact.confidence, artifact.body
    );
    let content_hash = hash_text(&text);
    let path = format!("imprint://brain-artifact/{}", artifact.id);
    let mut metadata = BTreeMap::new();
    metadata.insert("path".into(), path.clone());
    metadata.insert("source".into(), "brain".into());
    metadata.insert("source_type".into(), "brain_artifact".into());
    metadata.insert("brain_artifact_id".into(), artifact.id.clone());
    metadata.insert("brain_artifact_kind".into(), format!("{:?}", artifact.kind));
    metadata.insert("content_hash".into(), content_hash.clone());
    metadata.insert("parser_version".into(), PARSER_VERSION.to_string());
    let anchor = SourceAnchor {
        id: format!("{id}:document"),
        document_id: id.clone(),
        chunk_id: None,
        path,
        content_hash: content_hash.clone(),
        start: 0,
        end: text.chars().count(),
        page: None,
        section: Some("Brain artifact".into()),
        parser_version: PARSER_VERSION,
    };
    Document {
        id,
        title: artifact.title.clone(),
        text,
        metadata,
        source_anchor: Some(anchor),
        content_hash: Some(content_hash),
        parser_version: Some(PARSER_VERSION),
    }
}

fn brain_artifact_document_id(artifact: &BrainArtifact) -> String {
    format!("brain-artifact:{}", hash_text(&artifact.id))
}

fn chat_transcript_text(session: &ChatSession, messages: &[ChatMessage]) -> String {
    let mut lines = vec![format!("Chat: {}", session.title)];
    for message in messages {
        lines.push(String::new());
        lines.push(format!(
            "{} {}: {}",
            chat_role_label(&message.role),
            message.id,
            message.content
        ));
    }
    lines.join("\n")
}

fn chat_role_label(role: &ChatRole) -> &'static str {
    match role {
        ChatRole::System => "System",
        ChatRole::User => "User",
        ChatRole::Assistant => "Assistant",
        ChatRole::Tool => "Tool",
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct WebSearchResult {
    title: String,
    url: String,
    snippet: String,
    body: String,
}

trait WebSearcher {
    fn search(&self, query: &str, max_results: usize) -> anyhow::Result<Vec<WebSearchResult>>;
}

#[derive(Debug, Clone, Copy)]
struct LiveWebSearcher;

impl WebSearcher for LiveWebSearcher {
    #[cfg(test)]
    fn search(&self, _query: &str, _max_results: usize) -> anyhow::Result<Vec<WebSearchResult>> {
        Ok(Vec::new())
    }

    #[cfg(not(test))]
    fn search(&self, query: &str, max_results: usize) -> anyhow::Result<Vec<WebSearchResult>> {
        live_web_search(query, max_results)
    }
}

fn build_chat_context_trace(
    store_root: &Path,
    store: &FileMemoryStore,
    embedder: &RuntimeEmbedder,
    config: &ModelConfig,
    session_id: &str,
    user_message: &ChatMessage,
) -> anyhow::Result<ChatContextTrace> {
    build_chat_context_trace_with_searcher(
        store_root,
        store,
        embedder,
        config,
        session_id,
        user_message,
        &LiveWebSearcher,
    )
}

fn build_chat_context_trace_with_searcher<S: WebSearcher>(
    store_root: &Path,
    store: &FileMemoryStore,
    embedder: &RuntimeEmbedder,
    config: &ModelConfig,
    session_id: &str,
    user_message: &ChatMessage,
    web_searcher: &S,
) -> anyhow::Result<ChatContextTrace> {
    let query_embedding = embedder.embed(&user_message.content)?;
    let memory = FileMemoryStore::new(store_root).load().ok();
    let attention_marks = store.list_attention_marks(None).unwrap_or_default();
    let now = now_millis();
    let memory_accesses = store
        .list_memory_accesses(None, Some(now.saturating_sub(MEMORY_ACCESS_HORIZON_MILLIS)))
        .unwrap_or_default();
    let hot_snippets = memory
        .as_ref()
        .map(|memory| {
            memory
                .chunks
                .iter()
                .filter(|chunk| {
                    chunk.metadata.get("source_type").map(String::as_str) == Some("chat")
                })
                .filter(|chunk| !chunk.text.contains(&user_message.id))
                .map(|chunk| {
                    let similarity = cosine_similarity(&query_embedding, &chunk.embedding);
                    ChatContextSnippet {
                        id: chunk.id.clone(),
                        source_kind: "chunk".into(),
                        source_id: chunk.id.clone(),
                        excerpt: truncate(&chunk.text, 360),
                        score: similarity + 0.35,
                        hotness: 1.0,
                        source_anchor: chunk.source_anchor.clone(),
                    }
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let mut snippets = hot_snippets.clone();
    let mut searched_web = false;
    let mut tool_trace = vec!["planner: read compact map and active session hot memory".into()];
    let mandatory_hits = execute_memory_search(
        embedder,
        memory.as_ref(),
        &attention_marks,
        &memory_accesses,
        &user_message.content,
        6,
        5,
        "retrieval_floor",
        0.75,
    );
    publish_agent_presence_from_snippets(
        store_root,
        "chat",
        1,
        DEFAULT_PLANNER_STEPS + 2,
        "Agent is searching memory",
        &mandatory_hits,
    );
    tool_trace.push(format!(
        "retrieval floor: memory_search query={:?} hits={}",
        user_message.content,
        mandatory_hits.len()
    ));
    snippets.extend(mandatory_hits);
    let actions = match plan_memory_actions(config, user_message, memory.as_ref(), &hot_snippets) {
        Ok(actions) => actions,
        Err(error) => {
            tool_trace.push(format!("planner fallback: {error}"));
            default_planner_actions(&user_message.content)
        }
    };

    let mut executed_actions = Vec::new();
    for (index, action) in actions.into_iter().take(DEFAULT_PLANNER_STEPS).enumerate() {
        let step = index + 1;
        executed_actions.push(action.tool.clone());
        match action.tool.as_str() {
            "memory_search" => {
                let query = action.query.unwrap_or_else(|| user_message.content.clone());
                let max_chunks = action.max_chunks.unwrap_or(4).clamp(1, 8);
                let max_regions = action.max_regions.unwrap_or(5).clamp(1, 8);
                tool_trace.push(format!("planner step {step}: memory_search query={query:?} max_regions={max_regions} max_chunks={max_chunks}"));
                let hits = execute_memory_search(
                    embedder,
                    memory.as_ref(),
                    &attention_marks,
                    &memory_accesses,
                    &query,
                    max_chunks,
                    max_regions,
                    "planner_search",
                    0.25,
                );
                publish_agent_presence_from_snippets(
                    store_root,
                    "chat",
                    step + 1,
                    DEFAULT_PLANNER_STEPS + 2,
                    "Agent is searching memory",
                    &hits,
                );
                snippets.extend(hits);
            }
            "memory_open" => {
                let target = resolve_action_node(&action, &snippets);
                tool_trace.push(format!(
                    "planner step {step}: memory_open target={}",
                    target
                        .as_ref()
                        .map(node_ref_label)
                        .unwrap_or_else(|| "none".into())
                ));
                if let Some(node) = target {
                    publish_agent_presence_for_node(
                        store_root,
                        "chat",
                        step + 1,
                        DEFAULT_PLANNER_STEPS + 2,
                        "Agent is opening memory",
                        &node,
                    );
                    if let Ok(opened) = surf_open(store_root, &node) {
                        snippets.extend(snippets_from_opened(&opened, "planner_open", 0.82, 0.30));
                    }
                }
            }
            "memory_neighbors" => {
                let target = resolve_action_node(&action, &snippets);
                let max_results = action.max_results.unwrap_or(5).clamp(1, 8);
                tool_trace.push(format!(
                    "planner step {step}: memory_neighbors target={} max_results={max_results}",
                    target
                        .as_ref()
                        .map(node_ref_label)
                        .unwrap_or_else(|| "none".into())
                ));
                if let Some(node) = target {
                    publish_agent_presence_for_node(
                        store_root,
                        "chat",
                        step + 1,
                        DEFAULT_PLANNER_STEPS + 2,
                        "Agent is reading neighbors",
                        &node,
                    );
                    if let Ok(neighbors) = surf_neighbors(store_root, &node, max_results) {
                        snippets.extend(neighbors.into_iter().map(|neighbor| {
                            snippet_from_neighbor(neighbor, "planner_neighbor", 0.22)
                        }));
                    }
                }
            }
            "memory_expand" => {
                let target = action
                    .chunk_id
                    .or(action.target_id)
                    .or_else(|| best_chunk_id(&snippets));
                let mode = parse_expand_mode(action.mode.as_deref()).unwrap_or(ExpandMode::Window);
                let window = action.window.unwrap_or(520).clamp(80, 2400);
                tool_trace.push(format!(
                    "planner step {step}: memory_expand target={} mode={mode:?} window={window}",
                    target.as_deref().unwrap_or("none")
                ));
                if let Some(chunk_id) = target {
                    publish_agent_presence_for_node(
                        store_root,
                        "chat",
                        step + 1,
                        DEFAULT_PLANNER_STEPS + 2,
                        "Agent is expanding source context",
                        &NodeRef::Chunk(strip_node_prefix(&chunk_id)),
                    );
                    if let Ok(expanded) =
                        surf_expand(store_root, &strip_node_prefix(&chunk_id), mode, window)
                    {
                        snippets.push(snippet_from_expansion(
                            expanded,
                            "planner_expand",
                            0.95,
                            0.35,
                        ));
                    }
                }
            }
            "memory_jump_to_anchor" => {
                let anchor_id = action.anchor_id.or_else(|| best_anchor_id(&snippets));
                let window = action.window.unwrap_or(520).clamp(80, 2400);
                tool_trace.push(format!(
                    "planner step {step}: memory_jump_to_anchor anchor={} window={window}",
                    anchor_id.as_deref().unwrap_or("none")
                ));
                if let Some(anchor_id) = anchor_id {
                    if let Ok(expanded) = surf_jump_to_anchor(store_root, &anchor_id, window) {
                        snippets.push(snippet_from_expansion(expanded, "planner_jump", 0.90, 0.30));
                    }
                }
            }
            "mark_attention" => {
                tool_trace.push(format!(
                    "planner step {step}: mark_attention action={} reason={}",
                    action.action.as_deref().unwrap_or("promote"),
                    action
                        .reason
                        .as_deref()
                        .unwrap_or("planner-selected context")
                ));
                mark_context_attention(
                    store_root,
                    &snippets,
                    action
                        .reason
                        .as_deref()
                        .unwrap_or("planner-selected context"),
                );
            }
            "web_search" => {
                let query = action.query.unwrap_or_else(|| user_message.content.clone());
                let max_results = action
                    .max_results
                    .unwrap_or(WEB_SEARCH_MAX_RESULTS)
                    .clamp(1, 8);
                tool_trace.push(format!(
                    "planner step {step}: web_search query={query:?} max_results={max_results}"
                ));
                match search_web_into_memory(
                    store_root,
                    store,
                    embedder,
                    config,
                    session_id,
                    &query,
                    max_results,
                    web_searcher,
                    &mut tool_trace,
                ) {
                    Ok(web_snippets) => {
                        searched_web = true;
                        snippets.extend(web_snippets);
                    }
                    Err(error) => {
                        tool_trace.push(format!("planner step {step}: web_search skipped: {error}"))
                    }
                }
            }
            other => {
                tool_trace.push(format!(
                    "planner step {step}: ignored unsupported tool {other}"
                ));
            }
        }
    }
    let cortex_trace = if config.cortex_enabled {
        Some(build_cortex_trace(
            config,
            &mut tool_trace,
            &executed_actions,
            &snippets,
        ))
    } else {
        None
    };

    if !searched_web && should_search_web(&user_message.content, &snippets) {
        match search_web_into_memory(
            store_root,
            store,
            embedder,
            config,
            session_id,
            &user_message.content,
            WEB_SEARCH_MAX_RESULTS,
            web_searcher,
            &mut tool_trace,
        ) {
            Ok(web_snippets) => {
                snippets.extend(web_snippets);
            }
            Err(error) => tool_trace.push(format!("web_search skipped: {error}")),
        }
    }

    dedupe_snippets(&mut snippets);
    snippets.sort_by(|a, b| {
        snippet_rank(b)
            .partial_cmp(&snippet_rank(a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    snippets.truncate(MAX_RESPONSE_CONTEXT_SNIPPETS);
    if snippets.is_empty() {
        snippets.push(ChatContextSnippet {
            id: format!("hot-session:{session_id}"),
            source_kind: "session".into(),
            source_id: session_id.into(),
            excerpt: "No prior memory matched this turn yet; the active chat session is hot."
                .into(),
            score: 0.0,
            hotness: 1.0,
            source_anchor: None,
        });
    }
    publish_agent_presence_from_snippets(
        store_root,
        "chat",
        DEFAULT_PLANNER_STEPS + 2,
        DEFAULT_PLANNER_STEPS + 2,
        "Agent is drafting from selected memory",
        &snippets,
    );

    let created_at = now_millis();
    Ok(ChatContextTrace {
        id: unique_id("trace"),
        session_id: session_id.into(),
        user_message_id: user_message.id.clone(),
        tool_trace: {
            tool_trace.push(format!(
                "response_context: selected {} bounded snippet(s)",
                snippets.len()
            ));
            tool_trace
        },
        snippets,
        cortex_trace,
        created_at,
    })
}

fn complete_chat_response(
    store_root: &Path,
    config: &ModelConfig,
    user_message: &str,
    trace: &ChatContextTrace,
) -> anyhow::Result<String> {
    if let Some(answer) = structured_table_answer(store_root, user_message, trace) {
        return Ok(answer);
    }
    let model = config
        .response_model
        .as_deref()
        .or(config.chat_model.as_deref())
        .unwrap_or(DEFAULT_RESPONSE_MODEL);
    if model == HASH_EMBEDDING_MODEL {
        let first = trace
            .snippets
            .first()
            .map(|snippet| snippet.excerpt.as_str())
            .unwrap_or("No context available.");
        return Ok(format!(
            "Saved to memory. Retrieved {} hot/context snippet(s) for this turn. Most relevant: {}",
            trace.snippets.len(),
            first
        ));
    }

    let endpoint = openai_base_url(&config.endpoint);
    let context = trace
        .snippets
        .iter()
        .enumerate()
        .map(|(index, snippet)| {
            format!(
                "[{}] kind={} id={} hotness={:.2}\n{}",
                index + 1,
                snippet.source_kind,
                snippet.source_id,
                snippet.hotness,
                snippet.excerpt
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    let request = OpenAiChatRequest {
        model,
        stream: false,
        temperature: 0.2,
        messages: vec![
            OpenAiChatMessage {
                role: "system",
                content: "You are imprint's memory agent. You receive source-grounded context from the local DB and from web findings that the tool planner retrieves with `web_search(query,max_results)`. Answer from the supplied snippets, use web-backed snippets when present, and cite snippet numbers when relevant.",
            },
            OpenAiChatMessage {
                role: "system",
                content: &context,
            },
            OpenAiChatMessage {
                role: "user",
                content: user_message,
            },
        ],
    };
    let response = Client::builder()
        .timeout(std::time::Duration::from_secs(90))
        .build()?
        .post(format!("{endpoint}/chat/completions"))
        .bearer_auth("not-needed")
        .json(&request)
        .send()
        .with_context(|| format!("calling local response model {model}"))?
        .error_for_status()
        .with_context(|| format!("local response model {model} returned an error"))?
        .json::<OpenAiChatResponse>()
        .context("decoding local chat response")?;
    response
        .choices
        .into_iter()
        .next()
        .map(|choice| choice.message.content)
        .filter(|content| !content.trim().is_empty())
        .context("local response model returned no content")
}

fn build_cortex_trace(
    config: &ModelConfig,
    tool_trace: &mut Vec<String>,
    executed_actions: &[String],
    snippets: &[ChatContextSnippet],
) -> CortexTrace {
    let round_count = config.cortex_rounds.clamp(1, 4);
    let sufficient = snippets
        .iter()
        .any(|snippet| snippet.source_anchor.is_some())
        || !snippets.is_empty();
    let gap = if sufficient {
        "anchored evidence selected".to_string()
    } else {
        "no anchored evidence found".to_string()
    };
    let note = if sufficient {
        "Recursive cortex found enough source-grounded context for the response model.".to_string()
    } else {
        "Recursive cortex could not find source-grounded context; answer should be cautious."
            .to_string()
    };
    let actions = if executed_actions.is_empty() {
        vec!["memory_search".to_string()]
    } else {
        executed_actions.to_vec()
    };
    let rounds = (1..=round_count)
        .map(|round| {
            let snippets_before = if round == 1 {
                0
            } else {
                snippets.len().saturating_sub(round_count - round + 1)
            };
            let snippets_after = snippets.len();
            tool_trace.push(format!(
                "cortex round {round}: actions={} critique_sufficient={} gap={}",
                actions.join(","),
                sufficient,
                gap
            ));
            CortexRoundTrace {
                round,
                actions: actions.clone(),
                snippets_before,
                snippets_after,
                critique: CortexCritique {
                    sufficient,
                    gap: gap.clone(),
                    note: note.clone(),
                },
            }
        })
        .collect::<Vec<_>>();
    CortexTrace {
        enabled: true,
        rounds,
        final_note: note,
    }
}

fn plan_memory_actions(
    config: &ModelConfig,
    user_message: &ChatMessage,
    memory: Option<&PersistedMemory>,
    hot_snippets: &[ChatContextSnippet],
) -> anyhow::Result<Vec<PlannerAction>> {
    let model = config
        .planner_model
        .as_deref()
        .unwrap_or(DEFAULT_PLANNER_MODEL);
    if model == HASH_EMBEDDING_MODEL {
        return Ok(default_planner_actions(&user_message.content));
    }
    let map = memory
        .and_then(|memory| memory.memory_map.as_ref())
        .map(|map| map.serialized.as_str())
        .unwrap_or("Memory map unavailable.");
    let hot = hot_snippets
        .iter()
        .take(6)
        .map(|snippet| {
            format!(
                "- {} {} hot={:.2}: {}",
                snippet.source_kind,
                snippet.source_id,
                snippet.hotness,
                truncate(&snippet.excerpt, 180)
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let prompt = format!(
        "You are imprint's tool planner. Read the compact map and hot conversation memory, then choose up to 6 small navigation actions before the response model answers. You do have web search access. Use `web_search(query,max_results)` if the answer is not in the local DB or if the user asks for current/live web information. Available tools:\n- memory_search(query,max_chunks)\n- web_search(query,max_results)\n- memory_open(node_kind,node_id) for kind chunk/document/region\n- memory_neighbors(node_kind,node_id,max_results)\n- memory_expand(chunk_id,mode,window) where mode is window/page/section/document\n- memory_jump_to_anchor(anchor_id,window)\n- mark_attention(action,reason)\nUse memory_search to check the local DB first. Use web_search to pull external results into memory, where they will be embedded and searched before the response model answers. Open/neighbor calls inspect linked memory, expand/jump reads original source context, and mark_attention records useful or weak context. Return strict JSON only: {{\"actions\":[{{\"tool\":\"memory_search\",\"query\":\"...\",\"max_chunks\":4}},{{\"tool\":\"web_search\",\"query\":\"...\",\"max_results\":4}},{{\"tool\":\"memory_open\"}},{{\"tool\":\"memory_neighbors\",\"max_results\":5}},{{\"tool\":\"memory_expand\",\"mode\":\"window\",\"window\":520}},{{\"tool\":\"mark_attention\",\"action\":\"promote\",\"reason\":\"...\"}}]}}.\n\nMAP:\n{map}\n\nHOT MEMORY:\n{hot}\n\nUSER QUERY:\n{}",
        user_message.content
    );
    let endpoint = openai_base_url(
        config
            .planner_endpoint
            .as_deref()
            .unwrap_or(&config.endpoint),
    );
    let request = OpenAiChatRequest {
        model,
        stream: false,
        temperature: 0.0,
        messages: vec![
            OpenAiChatMessage {
                role: "system",
                content: "Return strict JSON only. Do not answer the user.",
            },
            OpenAiChatMessage {
                role: "user",
                content: &prompt,
            },
        ],
    };
    let content = Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?
        .post(format!("{endpoint}/chat/completions"))
        .bearer_auth("not-needed")
        .json(&request)
        .send()
        .with_context(|| format!("calling local planner model {model}"))?
        .error_for_status()
        .with_context(|| format!("local planner model {model} returned an error"))?
        .json::<OpenAiChatResponse>()
        .context("decoding local planner response")?
        .choices
        .into_iter()
        .next()
        .map(|choice| choice.message.content)
        .context("local planner model returned no content")?;
    let json = extract_json_object(&content).context("planner returned no JSON object")?;
    let plan: PlannerPlan = serde_json::from_str(json).context("decoding planner JSON")?;
    if plan.actions.is_empty() {
        Ok(default_planner_actions(&user_message.content))
    } else {
        Ok(plan.actions)
    }
}

fn execute_memory_search(
    embedder: &RuntimeEmbedder,
    memory: Option<&PersistedMemory>,
    attention_marks: &[AttentionMark],
    memory_accesses: &[MemoryAccess],
    query: &str,
    max_chunks: usize,
    max_regions: usize,
    source_kind: &str,
    hotness: f32,
) -> Vec<ChatContextSnippet> {
    let Some(memory) = memory else {
        return Vec::new();
    };
    if memory.memory_map.is_none() || memory.chunks.is_empty() {
        return Vec::new();
    }
    let ann = RegionIndexer.rebuild(&memory.chunks, &memory.regions);
    MemoryQueryEngine
        .execute_with_signals(
            embedder,
            memory,
            &ann,
            QueryRequest {
                text: query.into(),
                filters: BTreeMap::new(),
                max_regions: max_regions.max(1),
                max_chunks: max_chunks.max(1),
            },
            attention_marks,
            memory_accesses,
            now_millis(),
        )
        .map(|result| {
            result
                .hits
                .into_iter()
                .map(|hit| ChatContextSnippet {
                    id: format!("{source_kind}:{}", hit.hit_id),
                    source_kind: "chunk".into(),
                    source_id: hit.chunk_id,
                    excerpt: hit.excerpt,
                    score: hit.score,
                    hotness,
                    source_anchor: hit.source_anchor,
                })
                .collect()
        })
        .unwrap_or_default()
}

fn should_search_web(query: &str, snippets: &[ChatContextSnippet]) -> bool {
    if user_requested_web_search(query) {
        return true;
    }
    let strongest_local_score = snippets
        .iter()
        .filter(|snippet| {
            snippet
                .source_anchor
                .as_ref()
                .is_some_and(|anchor| !anchor.path.starts_with("imprint://chat/"))
        })
        .map(|snippet| snippet.score)
        .fold(0.0, f32::max);
    strongest_local_score < WEB_SEARCH_MIN_LOCAL_SCORE
}

fn user_requested_web_search(query: &str) -> bool {
    let lowered = query.to_ascii_lowercase();
    lowered.contains("search the web")
        || lowered.contains("web search")
        || lowered.contains("look it up")
        || lowered.contains("google it")
        || lowered.contains("browse")
        || lowered.contains("internet")
}

fn search_web_into_memory<S: WebSearcher>(
    store_root: &Path,
    store: &FileMemoryStore,
    embedder: &RuntimeEmbedder,
    config: &ModelConfig,
    session_id: &str,
    query: &str,
    max_results: usize,
    web_searcher: &S,
    tool_trace: &mut Vec<String>,
) -> anyhow::Result<Vec<ChatContextSnippet>> {
    publish_agent_presence_from_snippets(
        store_root,
        "chat",
        DEFAULT_PLANNER_STEPS + 1,
        DEFAULT_PLANNER_STEPS + 3,
        "Agent is searching the web",
        &[],
    );
    let max_results = max_results.clamp(1, 8);
    let results = web_searcher.search(query, max_results)?;
    let downloaded_pages = results
        .iter()
        .filter(|result| !result.body.trim().is_empty())
        .count();
    tool_trace.push(format!(
        "web_search query={query:?} results={} pages_downloaded={downloaded_pages}",
        results.len()
    ));
    if results.is_empty() {
        return Ok(Vec::new());
    }

    let mut findings = Vec::new();
    for result in results.into_iter().take(max_results) {
        let finding = web_search_result_finding(session_id, query, result);
        store.insert_web_finding(&finding)?;
        audit_write(
            store,
            finding.session_id.clone(),
            "web_finding.write",
            &finding.id,
            &finding.actor,
            &finding,
        )?;
        findings.push(finding);
    }
    if let Err(error) = sync_web_findings_with_config(store_root, store, config) {
        let snippets = snippets_from_web_findings(&findings, "web_finding_fallback");
        tool_trace.push(format!(
            "web_research fallback: embedding web findings failed; using {} fetched source snippet(s): {error}",
            snippets.len()
        ));
        return Ok(snippets);
    }
    let refreshed = store.load()?;
    let attention_marks = store.list_attention_marks(None).unwrap_or_default();
    let now = now_millis();
    let memory_accesses = store
        .list_memory_accesses(None, Some(now.saturating_sub(MEMORY_ACCESS_HORIZON_MILLIS)))
        .unwrap_or_default();
    let mut snippets = execute_memory_search(
        embedder,
        Some(&refreshed),
        &attention_marks,
        &memory_accesses,
        query,
        6,
        6,
        "web_research",
        0.90,
    );
    tool_trace.push(format!(
        "web_research: embedded web findings and re-ran memory_search hits={}",
        snippets.len()
    ));
    if snippets.is_empty() {
        snippets = snippets_from_web_findings(&findings, "web_finding");
        tool_trace.push(format!(
            "web_research fallback: memory_search returned no web hits; using {} fetched source snippet(s)",
            snippets.len()
        ));
    }
    publish_agent_presence_from_snippets(
        store_root,
        "chat",
        DEFAULT_PLANNER_STEPS + 2,
        DEFAULT_PLANNER_STEPS + 3,
        "Agent is reading web findings from memory",
        &snippets,
    );
    Ok(snippets)
}

fn snippets_from_web_findings(findings: &[WebFinding], prefix: &str) -> Vec<ChatContextSnippet> {
    findings
        .iter()
        .map(|finding| {
            let document = web_finding_document(finding);
            ChatContextSnippet {
                id: format!("{prefix}:{}", finding.id),
                source_kind: "web_finding".into(),
                source_id: finding.id.clone(),
                excerpt: truncate(&document.text, MAX_CONTEXT_SNIPPET_CHARS),
                score: finding.confidence,
                hotness: 0.9,
                source_anchor: document.source_anchor,
            }
        })
        .collect()
}

fn web_search_result_finding(session_id: &str, query: &str, result: WebSearchResult) -> WebFinding {
    let created_at = now_millis();
    let summary = truncate(
        &format!(
            "Search result snippet:\n{}\n\nFetched page text:\n{}",
            result.snippet.trim(),
            result.body.trim()
        ),
        MAX_WEB_SUMMARY_CHARS,
    );
    WebFinding {
        id: unique_id("web"),
        session_id: Some(session_id.into()),
        query: query.into(),
        url: result.url.clone(),
        title: if result.title.trim().is_empty() {
            result.url.clone()
        } else {
            result.title
        },
        summary,
        retrieved_at: now_secs(),
        confidence: 0.72,
        actor: "web-search-agent".into(),
        created_at,
        provenance: ProvenanceRecord {
            actor: "web-search-agent".into(),
            reason: "Automatic web search fallback after weak local memory retrieval.".into(),
            created_at,
            source_refs: vec![result.url],
        },
    }
}

fn default_planner_actions(query: &str) -> Vec<PlannerAction> {
    vec![
        PlannerAction {
            query: Some(query.into()),
            max_chunks: Some(4),
            reason: Some("search vector memory from map route".into()),
            ..PlannerAction::tool("memory_search")
        },
        PlannerAction {
            reason: Some("open the most relevant source node to inspect its local context".into()),
            ..PlannerAction::tool("memory_open")
        },
        PlannerAction {
            max_results: Some(5),
            reason: Some(
                "inspect linked chunks, same-document moves, region, and semantic neighbors".into(),
            ),
            ..PlannerAction::tool("memory_neighbors")
        },
        PlannerAction {
            mode: Some("window".into()),
            window: Some(520),
            reason: Some("dig to original source window if a source hit exists".into()),
            ..PlannerAction::tool("memory_expand")
        },
        PlannerAction {
            window: Some(520),
            reason: Some("verify provenance by jumping through the best source anchor".into()),
            ..PlannerAction::tool("memory_jump_to_anchor")
        },
        PlannerAction {
            action: Some("promote".into()),
            reason: Some("promote useful context and decay weak context".into()),
            ..PlannerAction::tool("mark_attention")
        },
    ]
}

fn snippets_from_opened(
    opened: &SurfOpenResult,
    prefix: &str,
    score: f32,
    hotness: f32,
) -> Vec<ChatContextSnippet> {
    let mut snippets = vec![ChatContextSnippet {
        id: format!("{prefix}:{}", node_ref_label(&opened.node)),
        source_kind: snippet_kind_for_node(&opened.node).into(),
        source_id: node_ref_id(&opened.node),
        excerpt: truncate(&opened.excerpt, MAX_CONTEXT_SNIPPET_CHARS),
        score,
        hotness,
        source_anchor: opened.source_anchor.clone(),
    }];
    snippets.extend(opened.passages.iter().take(3).map(|passage| {
        snippet_from_passage(
            passage,
            prefix,
            score * passage.score.max(0.35),
            hotness * 0.85,
        )
    }));
    snippets
}

fn snippet_from_passage(
    passage: &surf::SurfPassage,
    prefix: &str,
    score: f32,
    hotness: f32,
) -> ChatContextSnippet {
    ChatContextSnippet {
        id: format!("{prefix}:passage:{}", node_ref_label(&passage.node)),
        source_kind: snippet_kind_for_node(&passage.node).into(),
        source_id: node_ref_id(&passage.node),
        excerpt: truncate(&passage.excerpt, MAX_CONTEXT_SNIPPET_CHARS),
        score,
        hotness,
        source_anchor: passage.source_anchor.clone(),
    }
}

fn snippet_from_neighbor(neighbor: SurfNeighbor, prefix: &str, hotness: f32) -> ChatContextSnippet {
    ChatContextSnippet {
        id: format!("{prefix}:{}", node_ref_label(&neighbor.node)),
        source_kind: snippet_kind_for_node(&neighbor.node).into(),
        source_id: node_ref_id(&neighbor.node),
        excerpt: truncate(&neighbor.excerpt, MAX_CONTEXT_SNIPPET_CHARS),
        score: neighbor.score,
        hotness,
        source_anchor: neighbor.source_anchor,
    }
}

fn snippet_from_expansion(
    expanded: SurfExpansion,
    prefix: &str,
    score: f32,
    hotness: f32,
) -> ChatContextSnippet {
    let expanded_chunk_id = expanded.chunk_id.clone();
    ChatContextSnippet {
        id: format!("{prefix}:{expanded_chunk_id}:{:?}", expanded.mode),
        source_kind: "source_window".into(),
        source_id: expanded_chunk_id,
        excerpt: truncate(&expanded.excerpt, MAX_CONTEXT_SNIPPET_CHARS),
        score,
        hotness,
        source_anchor: expanded.source_anchor,
    }
}

fn publish_agent_presence_from_snippets(
    store_root: &Path,
    phase: &str,
    completed: usize,
    total: usize,
    message: &str,
    snippets: &[ChatContextSnippet],
) {
    let mut seen = HashSet::new();
    let active_node_ids = snippets
        .iter()
        .filter_map(snippet_node_label)
        .filter(|id| seen.insert(id.clone()))
        .take(5)
        .collect::<Vec<_>>();
    let label = snippets
        .iter()
        .find(|snippet| snippet_node_label(snippet).is_some())
        .map(|snippet| truncate(&snippet.excerpt, 52));
    let _ = write_progress_with_presence(
        store_root,
        phase,
        completed,
        total,
        message,
        &active_node_ids,
        label.as_deref(),
    );
}

fn publish_agent_presence_for_node(
    store_root: &Path,
    phase: &str,
    completed: usize,
    total: usize,
    message: &str,
    node: &NodeRef,
) {
    let active_node_ids = vec![node_ref_label(node)];
    let _ = write_progress_with_presence(
        store_root,
        phase,
        completed,
        total,
        message,
        &active_node_ids,
        Some(&node_ref_label(node)),
    );
}

fn snippet_node_label(snippet: &ChatContextSnippet) -> Option<String> {
    match snippet.source_kind.as_str() {
        "chunk" | "source_window" => Some(format!("chunk:{}", snippet.source_id)),
        "document" => Some(format!("document:{}", snippet.source_id)),
        "region" => Some(format!("region:{}", snippet.source_id)),
        _ => None,
    }
}

fn resolve_action_node(action: &PlannerAction, snippets: &[ChatContextSnippet]) -> Option<NodeRef> {
    let id = action
        .node_id
        .as_deref()
        .or(action.target_id.as_deref())
        .or(action.chunk_id.as_deref());
    if let Some(id) = id {
        if let Some(node) = parse_node_ref(id, action.node_kind.as_deref()) {
            return Some(node);
        }
    }
    best_node_ref(snippets)
}

fn parse_node_ref(raw: &str, kind: Option<&str>) -> Option<NodeRef> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    if let Some(id) = raw.strip_prefix("chunk:") {
        return Some(NodeRef::Chunk(id.into()));
    }
    if let Some(id) = raw.strip_prefix("document:") {
        return Some(NodeRef::Document(id.into()));
    }
    if let Some(id) = raw.strip_prefix("region:") {
        return Some(NodeRef::Region(id.into()));
    }
    match kind.unwrap_or("chunk").to_ascii_lowercase().as_str() {
        "chunk" | "source_window" => Some(NodeRef::Chunk(raw.into())),
        "document" | "doc" => Some(NodeRef::Document(raw.into())),
        "region" => Some(NodeRef::Region(raw.into())),
        _ => None,
    }
}

fn best_node_ref(snippets: &[ChatContextSnippet]) -> Option<NodeRef> {
    snippets
        .iter()
        .filter_map(|snippet| {
            let node = match snippet.source_kind.as_str() {
                "chunk" | "source_window" => Some(NodeRef::Chunk(snippet.source_id.clone())),
                "document" => Some(NodeRef::Document(snippet.source_id.clone())),
                "region" => Some(NodeRef::Region(snippet.source_id.clone())),
                _ => None,
            }?;
            Some((
                snippet.score + snippet.hotness + snippet_rank(snippet),
                node,
            ))
        })
        .max_by(|left, right| {
            left.0
                .partial_cmp(&right.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(_, node)| node)
}

fn node_ref_label(node: &NodeRef) -> String {
    match node {
        NodeRef::Document(id) => format!("document:{id}"),
        NodeRef::Chunk(id) => format!("chunk:{id}"),
        NodeRef::Region(id) => format!("region:{id}"),
    }
}

fn node_ref_id(node: &NodeRef) -> String {
    match node {
        NodeRef::Document(id) | NodeRef::Chunk(id) | NodeRef::Region(id) => id.clone(),
    }
}

fn snippet_kind_for_node(node: &NodeRef) -> &'static str {
    match node {
        NodeRef::Document(_) => "document",
        NodeRef::Chunk(_) => "chunk",
        NodeRef::Region(_) => "region",
    }
}

fn strip_node_prefix(value: &str) -> String {
    value
        .strip_prefix("chunk:")
        .or_else(|| value.strip_prefix("document:"))
        .or_else(|| value.strip_prefix("region:"))
        .unwrap_or(value)
        .into()
}

fn parse_expand_mode(raw: Option<&str>) -> Option<ExpandMode> {
    match raw.unwrap_or("window").to_ascii_lowercase().as_str() {
        "window" => Some(ExpandMode::Window),
        "page" => Some(ExpandMode::Page),
        "section" => Some(ExpandMode::Section),
        "document" | "doc" | "full_document" => Some(ExpandMode::Document),
        _ => None,
    }
}

fn best_anchor_id(snippets: &[ChatContextSnippet]) -> Option<String> {
    snippets
        .iter()
        .filter_map(|snippet| {
            snippet.source_anchor.as_ref().map(|anchor| {
                (
                    snippet.score + snippet.hotness + snippet_rank(snippet),
                    anchor.id.clone(),
                )
            })
        })
        .max_by(|left, right| {
            left.0
                .partial_cmp(&right.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(_, anchor_id)| anchor_id)
}

fn best_chunk_id(snippets: &[ChatContextSnippet]) -> Option<String> {
    snippets
        .iter()
        .filter(|snippet| snippet.source_kind == "chunk" || snippet.source_kind == "source_window")
        .max_by(|left, right| {
            (left.score + left.hotness)
                .partial_cmp(&(right.score + right.hotness))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|snippet| snippet.source_id.clone())
}

fn mark_context_attention(store_root: &Path, snippets: &[ChatContextSnippet], reason: &str) {
    let mut ranked = snippets.to_vec();
    ranked.sort_by(|left, right| {
        (right.score + right.hotness)
            .partial_cmp(&(left.score + left.hotness))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for snippet in ranked.iter().take(3) {
        let _ = apply_attention_mark(
            store_root,
            AttentionMarkWrite {
                target_id: snippet.source_id.clone(),
                target_kind: attention_target_for_snippet(snippet),
                action: AttentionAction::Promote,
                reason: reason.into(),
                actor: "planner".into(),
            },
        );
    }
    for snippet in ranked.iter().rev().take(2) {
        let _ = apply_attention_mark(
            store_root,
            AttentionMarkWrite {
                target_id: snippet.source_id.clone(),
                target_kind: attention_target_for_snippet(snippet),
                action: AttentionAction::Decay,
                reason: "Planner judged this context weak for the current query.".into(),
                actor: "planner".into(),
            },
        );
    }
}

fn attention_target_for_snippet(snippet: &ChatContextSnippet) -> AttentionTargetKind {
    match snippet.source_kind.as_str() {
        "transcript" => AttentionTargetKind::ChatMessage,
        "chunk" | "source_window" => AttentionTargetKind::Chunk,
        _ => AttentionTargetKind::DerivedMemory,
    }
}

fn dedupe_snippets(snippets: &mut Vec<ChatContextSnippet>) {
    let mut seen = HashSet::new();
    snippets.retain(|snippet| {
        seen.insert((
            snippet.source_kind.clone(),
            snippet.source_id.clone(),
            snippet.excerpt.clone(),
        ))
    });
}

fn snippet_rank(snippet: &ChatContextSnippet) -> f32 {
    let source_weight = match snippet.source_kind.as_str() {
        "source_window" => 2.0,
        "chunk" => 1.5,
        "transcript" => 0.0,
        _ => 0.4,
    };
    snippet.score + snippet.hotness * 0.4 + source_weight
}

fn structured_table_answer(
    store_root: &Path,
    user_message: &str,
    trace: &ChatContextTrace,
) -> Option<String> {
    if !looks_like_table_listing_request(user_message) {
        return None;
    }
    let memory = FileMemoryStore::new(store_root).load().ok()?;
    let query_tokens = crate::index::tokenize(user_message);
    let mut candidate_ids = trace
        .snippets
        .iter()
        .filter_map(|snippet| {
            snippet
                .source_anchor
                .as_ref()
                .map(|anchor| anchor.document_id.clone())
        })
        .collect::<Vec<_>>();
    candidate_ids.sort();
    candidate_ids.dedup();

    let mut candidates = memory
        .documents
        .iter()
        .filter(|document| candidate_ids.iter().any(|id| id == &document.id))
        .filter_map(|document| {
            let table = parse_markdown_like_table(&document.text)?;
            let score = structured_document_score(document, &table, &query_tokens);
            (score > 0).then_some((score, document, table))
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        right
            .0
            .cmp(&left.0)
            .then_with(|| left.1.id.cmp(&right.1.id))
    });
    let (_, document, table) = candidates.into_iter().next()?;
    Some(format_table_answer(document, &table))
}

fn looks_like_table_listing_request(message: &str) -> bool {
    let tokens = crate::index::tokenize(message);
    let has_list_intent = tokens
        .iter()
        .any(|token| matches!(token.as_str(), "list" | "show" | "display" | "all" | "who"));
    let has_table_subject = tokens.iter().any(|token| {
        matches!(
            token.as_str(),
            "employee"
                | "employees"
                | "people"
                | "contacts"
                | "users"
                | "customers"
                | "accounts"
                | "rows"
                | "records"
        )
    });
    has_list_intent && has_table_subject
}

struct ParsedTable {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
}

fn parse_markdown_like_table(text: &str) -> Option<ParsedTable> {
    let mut lines = text.lines().filter(|line| !line.trim().is_empty());
    let header_line = lines.find(|line| line.matches(',').count() >= 2)?;
    let headers = parse_csv_record(header_line);
    if headers.len() < 2 {
        return None;
    }

    let mut rows = Vec::new();
    let mut seen = HashSet::new();
    for line in lines {
        if line.matches(',').count() < headers.len().saturating_sub(1).min(2) {
            continue;
        }
        let mut row = parse_csv_record(line);
        if row.len() < headers.len() {
            continue;
        }
        row.truncate(headers.len());
        if seen.insert(row.join("\u{1f}")) {
            rows.push(row);
        }
    }
    (!rows.is_empty()).then_some(ParsedTable { headers, rows })
}

fn parse_csv_record(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut chars = line.chars().peekable();
    let mut in_quotes = false;
    while let Some(ch) = chars.next() {
        match ch {
            '"' if in_quotes && chars.peek() == Some(&'"') => {
                current.push('"');
                let _ = chars.next();
            }
            '"' => in_quotes = !in_quotes,
            ',' if !in_quotes => {
                fields.push(current.trim().to_string());
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    fields.push(current.trim().to_string());
    fields
}

fn structured_document_score(
    document: &Document,
    table: &ParsedTable,
    query_tokens: &[String],
) -> usize {
    let title = document.title.to_ascii_lowercase();
    let headers = table.headers.join(" ").to_ascii_lowercase();
    query_tokens
        .iter()
        .filter(|token| {
            title.contains(token.as_str())
                || headers.contains(token.as_str())
                || table
                    .rows
                    .iter()
                    .take(8)
                    .any(|row| row.join(" ").to_ascii_lowercase().contains(token.as_str()))
        })
        .count()
}

fn format_table_answer(document: &Document, table: &ParsedTable) -> String {
    let path = document
        .metadata
        .get("path")
        .map(String::as_str)
        .unwrap_or(document.id.as_str());
    let first_name_index = header_index(&table.headers, &["first name", "firstname", "first"]);
    let last_name_index = header_index(&table.headers, &["last name", "lastname", "last"]);
    let email_index = header_index(&table.headers, &["email", "e-mail"]);
    let department_index = header_index(&table.headers, &["department", "dept"]);
    let position_index = header_index(&table.headers, &["position", "title", "role"]);

    let mut out = format!(
        "Found {} row(s) in `{}`.\n\nSource: `{}`\n\n",
        table.rows.len(),
        document.title,
        path
    );
    out.push_str("| Name | Email | Department | Position |\n");
    out.push_str("| --- | --- | --- | --- |\n");
    for row in table.rows.iter().take(200) {
        let name = match (first_name_index, last_name_index) {
            (Some(first), Some(last)) => format!("{} {}", field(row, first), field(row, last))
                .trim()
                .to_string(),
            _ => field(row, 0).to_string(),
        };
        let email = email_index.map(|index| field(row, index)).unwrap_or("");
        let department = department_index
            .map(|index| field(row, index))
            .unwrap_or("");
        let position = position_index.map(|index| field(row, index)).unwrap_or("");
        out.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            escape_markdown_table_cell(&name),
            escape_markdown_table_cell(email),
            escape_markdown_table_cell(department),
            escape_markdown_table_cell(position)
        ));
    }
    if table.rows.len() > 200 {
        out.push_str(&format!(
            "\nShowing the first 200 of {} rows.\n",
            table.rows.len()
        ));
    }
    out
}

fn header_index(headers: &[String], names: &[&str]) -> Option<usize> {
    headers.iter().position(|header| {
        let normalized = header.to_ascii_lowercase().replace('_', " ");
        names
            .iter()
            .any(|name| normalized == *name || normalized.contains(*name))
    })
}

fn field(row: &[String], index: usize) -> &str {
    row.get(index).map(String::as_str).unwrap_or("")
}

fn escape_markdown_table_cell(value: &str) -> String {
    value.replace('|', "\\|").replace('\n', " ")
}

fn extract_json_object(content: &str) -> Option<&str> {
    let start = content.find('{')?;
    let end = content.rfind('}')?;
    if end <= start {
        return None;
    }
    Some(&content[start..=end])
}

#[derive(Debug, Clone, Deserialize)]
struct PlannerPlan {
    actions: Vec<PlannerAction>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct PlannerAction {
    tool: String,
    #[serde(default)]
    query: Option<String>,
    #[serde(default)]
    max_regions: Option<usize>,
    #[serde(default)]
    max_chunks: Option<usize>,
    #[serde(default)]
    target_id: Option<String>,
    #[serde(default)]
    node_id: Option<String>,
    #[serde(default)]
    node_kind: Option<String>,
    #[serde(default)]
    chunk_id: Option<String>,
    #[serde(default)]
    anchor_id: Option<String>,
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    window: Option<usize>,
    #[serde(default)]
    max_results: Option<usize>,
    #[serde(default)]
    action: Option<String>,
    #[serde(default)]
    reason: Option<String>,
}

impl PlannerAction {
    fn tool(name: &str) -> Self {
        Self {
            tool: name.into(),
            ..Default::default()
        }
    }
}

fn extract_turn_memories(
    store_root: &Path,
    session_id: &str,
    user_message: &ChatMessage,
    assistant_message: &ChatMessage,
) -> anyhow::Result<Vec<DerivedMemory>> {
    let mut memories = Vec::new();
    let summary_text = format!(
        "User: {} Assistant: {}",
        truncate(&user_message.content, 240),
        truncate(&assistant_message.content, 240)
    );
    memories.push(write_derived_memory(
        store_root,
        DerivedMemoryWrite {
            session_id: Some(session_id.into()),
            kind: DerivedMemoryKind::Summary,
            text: summary_text,
            source_message_ids: vec![user_message.id.clone(), assistant_message.id.clone()],
            actor: "assistant".into(),
            confidence: 0.72,
        },
    )?);

    for sentence in user_message.content.split(['.', '\n']).map(str::trim) {
        let lower = sentence.to_ascii_lowercase();
        let kind = if lower.contains("decided") || lower.contains("decision") {
            Some(DerivedMemoryKind::Decision)
        } else if lower.contains("todo") || lower.contains("task") || lower.contains("need to") {
            Some(DerivedMemoryKind::Task)
        } else if lower.contains("remember") || lower.contains("fact") {
            Some(DerivedMemoryKind::Fact)
        } else {
            None
        };
        if let Some(kind) = kind {
            memories.push(write_derived_memory(
                store_root,
                DerivedMemoryWrite {
                    session_id: Some(session_id.into()),
                    kind,
                    text: sentence.into(),
                    source_message_ids: vec![user_message.id.clone()],
                    actor: "assistant".into(),
                    confidence: 0.66,
                },
            )?);
        }
    }
    Ok(memories)
}

fn audit_write<T: Serialize>(
    store: &FileMemoryStore,
    session_id: Option<String>,
    event_type: &str,
    target_id: &str,
    actor: &str,
    payload: &T,
) -> anyhow::Result<()> {
    store.insert_audit_event(&AuditEvent {
        id: unique_id("audit"),
        session_id,
        event_type: event_type.into(),
        target_id: target_id.into(),
        actor: actor.into(),
        payload_json: serde_json::to_string(payload)?,
        created_at: now_millis(),
    })
}

fn record_query_hit_accesses(store: &FileMemoryStore, result: &QueryResult) {
    for hit in result.hits.iter().take(3) {
        record_memory_access(
            store,
            AttentionTargetKind::Chunk,
            hit.chunk_id.clone(),
            MemoryAccessKind::QueryHit,
            "Returned as a top source recall hit.",
            "memory-runtime",
        );
    }
    for region_id in result.routed.region_ids.iter().take(3) {
        record_memory_access(
            store,
            AttentionTargetKind::Region,
            region_id.clone(),
            MemoryAccessKind::QueryHit,
            "Selected as a query route candidate.",
            "memory-runtime",
        );
    }
}

fn record_memory_access(
    store: &FileMemoryStore,
    target_kind: AttentionTargetKind,
    target_id: String,
    access_kind: MemoryAccessKind,
    reason: &str,
    actor: &str,
) {
    let access = MemoryAccess {
        id: unique_id("access"),
        target_id,
        target_kind,
        access_kind,
        reason: reason.into(),
        actor: actor.into(),
        accessed_at: now_millis(),
    };
    let _ = store.insert_memory_access(&access);
}

fn access_target_from_node_ref(node: &NodeRef) -> (AttentionTargetKind, String) {
    match node {
        NodeRef::Document(id) => (AttentionTargetKind::Document, id.clone()),
        NodeRef::Chunk(id) => (AttentionTargetKind::Chunk, id.clone()),
        NodeRef::Region(id) => (AttentionTargetKind::Region, id.clone()),
    }
}

fn attention_target_session_id(
    store: &FileMemoryStore,
    target_kind: &AttentionTargetKind,
    target_id: &str,
) -> anyhow::Result<Option<String>> {
    match target_kind {
        AttentionTargetKind::ChatSession => Ok(Some(target_id.into())),
        AttentionTargetKind::ChatMessage => {
            Ok(store.list_chat_sessions()?.into_iter().find_map(|session| {
                store
                    .list_chat_messages(&session.id)
                    .ok()?
                    .into_iter()
                    .find(|message| message.id == target_id)
                    .map(|message| message.session_id)
            }))
        }
        AttentionTargetKind::TranscriptChunk => Ok(store
            .list_transcript_chunks(None)?
            .into_iter()
            .find(|chunk| chunk.id == target_id || chunk.message_id == target_id)
            .map(|chunk| chunk.session_id)),
        AttentionTargetKind::DerivedMemory => Ok(store
            .list_derived_memories(None)?
            .into_iter()
            .find(|memory| memory.id == target_id)
            .and_then(|memory| memory.session_id)),
        AttentionTargetKind::WebFinding
        | AttentionTargetKind::Document
        | AttentionTargetKind::Chunk
        | AttentionTargetKind::Region
        | AttentionTargetKind::Link => Ok(None),
    }
}

fn chat_source_anchor(
    session_id: &str,
    message_id: &str,
    content: &str,
    created_at: u64,
) -> SourceAnchor {
    SourceAnchor {
        id: format!("anchor:{message_id}"),
        document_id: chat_document_id(session_id),
        chunk_id: None,
        path: format!("imprint://chat/{session_id}/{message_id}"),
        content_hash: hash_text(content),
        start: 0,
        end: content.chars().count(),
        page: None,
        section: Some("Chat transcript".into()),
        parser_version: created_at as u32,
    }
}

fn estimate_tokens(text: &str) -> usize {
    (text.split_whitespace().count() as f32 * 1.35).ceil() as usize
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn unique_id(prefix: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{prefix}:{nanos}")
}

fn hash_text(text: &str) -> String {
    let mut hash = 14695981039346656037u64;
    for byte in text.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    format!("{hash:016x}")
}

fn openai_base_url(endpoint: &str) -> String {
    let trimmed = endpoint.trim_end_matches('/');
    if trimmed.ends_with("/v1") {
        trimmed.into()
    } else {
        format!("{trimmed}/v1")
    }
}

#[cfg(not(test))]
fn live_web_search(query: &str, max_results: usize) -> anyhow::Result<Vec<WebSearchResult>> {
    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(12))
        .build()?;
    let html = client
        .get(format!(
            "https://duckduckgo.com/html/?q={}",
            url_encode(query)
        ))
        .header("user-agent", "imprint-memory-agent/0.1")
        .send()
        .context("calling DuckDuckGo web search")?
        .error_for_status()
        .context("DuckDuckGo web search returned an error")?
        .text()
        .context("reading DuckDuckGo web search response")?;
    let mut results = parse_duckduckgo_results(&html, max_results);
    if results.is_empty() {
        let bing_html = client
            .get(format!(
                "https://www.bing.com/search?q={}",
                url_encode(query)
            ))
            .header("user-agent", "Mozilla/5.0 imprint-memory-agent/0.1")
            .send()
            .context("calling Bing web search fallback")?
            .error_for_status()
            .context("Bing web search fallback returned an error")?
            .text()
            .context("reading Bing web search fallback response")?;
        results = parse_bing_results(&bing_html, max_results);
    }
    for result in &mut results {
        if result.body.trim().is_empty() {
            result.body = fetch_web_page_text(&client, &result.url)
                .unwrap_or_else(|_| result.snippet.clone());
        }
    }
    Ok(results)
}

#[cfg(not(test))]
fn parse_duckduckgo_results(html: &str, max_results: usize) -> Vec<WebSearchResult> {
    let mut results = Vec::new();
    for part in html.split("result__a").skip(1) {
        let Some(href) = extract_attr(part, "href").and_then(clean_search_url) else {
            continue;
        };
        if results
            .iter()
            .any(|result: &WebSearchResult| result.url == href)
        {
            continue;
        }
        let title = part
            .find('>')
            .and_then(|start| {
                part[start + 1..]
                    .find("</a>")
                    .map(|end| &part[start + 1..start + 1 + end])
            })
            .map(strip_html_tags)
            .map(|value| html_decode(&value))
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| href.clone());
        let snippet = part
            .split("result__snippet")
            .nth(1)
            .and_then(|snippet_part| {
                snippet_part.find('>').and_then(|start| {
                    snippet_part[start + 1..]
                        .find("</")
                        .map(|end| &snippet_part[start + 1..start + 1 + end])
                })
            })
            .map(strip_html_tags)
            .map(|value| html_decode(&value))
            .unwrap_or_default();
        results.push(WebSearchResult {
            title: collapse_whitespace(&title),
            url: href,
            snippet: collapse_whitespace(&snippet),
            body: String::new(),
        });
        if results.len() >= max_results {
            break;
        }
    }
    results
}

#[cfg(not(test))]
fn parse_bing_results(html: &str, max_results: usize) -> Vec<WebSearchResult> {
    let mut results = Vec::new();
    for part in html.split("<li class=\"b_algo").skip(1) {
        let Some(anchor_start) = part.find("<a ") else {
            continue;
        };
        let anchor = &part[anchor_start..];
        let Some(url) = extract_attr(anchor, "href")
            .and_then(clean_search_url)
            .filter(|url| url.starts_with("http://") || url.starts_with("https://"))
        else {
            continue;
        };
        if results
            .iter()
            .any(|result: &WebSearchResult| result.url == url)
        {
            continue;
        }
        let title = anchor
            .find('>')
            .and_then(|start| {
                anchor[start + 1..]
                    .find("</a>")
                    .map(|end| &anchor[start + 1..start + 1 + end])
            })
            .map(strip_html_tags)
            .map(|value| html_decode(&value))
            .map(|value| collapse_whitespace(&value))
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| url.clone());
        let snippet = part
            .find("<p")
            .and_then(|start| part[start..].find('>').map(|end| start + end + 1))
            .and_then(|start| {
                part[start..]
                    .find("</p>")
                    .map(|end| &part[start..start + end])
            })
            .map(strip_html_tags)
            .map(|value| html_decode(&value))
            .map(|value| collapse_whitespace(&value))
            .unwrap_or_default();
        results.push(WebSearchResult {
            title,
            url,
            snippet,
            body: String::new(),
        });
        if results.len() >= max_results {
            break;
        }
    }
    results
}

#[cfg(not(test))]
fn fetch_web_page_text(client: &Client, url: &str) -> anyhow::Result<String> {
    let raw = client
        .get(url)
        .header("user-agent", "imprint-memory-agent/0.1")
        .send()
        .with_context(|| format!("fetching web result {url}"))?
        .error_for_status()
        .with_context(|| format!("web result {url} returned an error"))?
        .text()
        .with_context(|| format!("reading web result {url}"))?;
    let without_scripts = remove_html_blocks(&raw, "script");
    let without_styles = remove_html_blocks(&without_scripts, "style");
    let text = html_decode(&strip_html_tags(&without_styles));
    Ok(truncate(&collapse_whitespace(&text), MAX_WEB_BODY_CHARS))
}

#[cfg(not(test))]
fn extract_attr(input: &str, attr: &str) -> Option<String> {
    let needle = format!("{attr}=\"");
    let start = input.find(&needle)? + needle.len();
    let end = input[start..].find('"')?;
    Some(html_decode(&input[start..start + end]))
}

#[cfg(not(test))]
fn clean_search_url(raw: String) -> Option<String> {
    let raw = raw.trim();
    let raw = raw
        .strip_prefix("//")
        .map(|value| format!("https://{value}"))
        .unwrap_or_else(|| raw.to_string());
    if raw.contains("bing.com/ck/a") {
        if let Ok(url) = reqwest::Url::parse(&raw) {
            if let Some((_, target)) = url.query_pairs().find(|(key, _)| key == "u") {
                return decode_bing_target(&target);
            }
        }
    }
    if raw.contains("duckduckgo.com/l/") {
        if let Ok(url) = reqwest::Url::parse(&raw) {
            if let Some((_, target)) = url.query_pairs().find(|(key, _)| key == "uddg") {
                return Some(target.into_owned());
            }
        }
    }
    (raw.starts_with("http://") || raw.starts_with("https://")).then_some(raw)
}

#[cfg(not(test))]
fn decode_bing_target(raw: &str) -> Option<String> {
    use base64::Engine;
    let encoded = raw.strip_prefix("a1").unwrap_or(raw);
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(encoded)
        .ok()?;
    let decoded = String::from_utf8(bytes).ok()?;
    (decoded.starts_with("http://") || decoded.starts_with("https://")).then_some(decoded)
}

#[cfg(not(test))]
fn strip_html_tags(input: &str) -> String {
    let mut output = String::new();
    let mut in_tag = false;
    for ch in input.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => {
                in_tag = false;
                output.push(' ');
            }
            _ if !in_tag => output.push(ch),
            _ => {}
        }
    }
    output
}

#[cfg(not(test))]
fn remove_html_blocks(input: &str, tag: &str) -> String {
    let mut remaining = input;
    let mut output = String::new();
    let open = format!("<{tag}");
    let close = format!("</{tag}>");
    while let Some(start) = remaining.to_ascii_lowercase().find(&open) {
        output.push_str(&remaining[..start]);
        let after_open = &remaining[start..];
        if let Some(end) = after_open.to_ascii_lowercase().find(&close) {
            remaining = &after_open[end + close.len()..];
        } else {
            return output;
        }
    }
    output.push_str(remaining);
    output
}

#[cfg(not(test))]
fn html_decode(input: &str) -> String {
    input
        .replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&nbsp;", " ")
}

#[cfg(not(test))]
fn collapse_whitespace(input: &str) -> String {
    input.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(not(test))]
fn url_encode(input: &str) -> String {
    let mut output = String::new();
    for byte in input.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                output.push(byte as char)
            }
            b' ' => output.push('+'),
            _ => output.push_str(&format!("%{byte:02X}")),
        }
    }
    output
}

#[derive(Serialize)]
struct OpenAiChatRequest<'a> {
    model: &'a str,
    messages: Vec<OpenAiChatMessage<'a>>,
    stream: bool,
    temperature: f32,
}

#[derive(Serialize)]
struct OpenAiChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct OpenAiChatResponse {
    choices: Vec<OpenAiChatChoice>,
}

#[derive(Deserialize)]
struct OpenAiChatChoice {
    message: OpenAiChatResponseMessage,
}

#[derive(Deserialize)]
struct OpenAiChatResponseMessage {
    content: String,
}

fn model_config_path(store_root: &Path) -> PathBuf {
    store_root.join("model_config.json")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_store_root(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("ai-memory-{name}"));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("create temp root");
        save_model_config(
            &root,
            &ModelConfig {
                mode: ModelConnectionMode::Local,
                endpoint: DEFAULT_LOCAL_ENDPOINT.into(),
                api_key_name: None,
                chat_model: None,
                planner_model: Some(HASH_EMBEDDING_MODEL.into()),
                response_model: Some(HASH_EMBEDDING_MODEL.into()),
                planner_endpoint: Some(DEFAULT_LOCAL_ENDPOINT.into()),
                runtime_preset: ModelRuntimePreset::CustomOpenAi,
                cortex_enabled: true,
                cortex_rounds: 3,
                critic_model: Some(HASH_EMBEDDING_MODEL.into()),
                critic_endpoint: Some(DEFAULT_LOCAL_ENDPOINT.into()),
                compiler_model: Some(HASH_EMBEDDING_MODEL.into()),
                embedding_model: Some(HASH_EMBEDDING_MODEL.into()),
                embedding_endpoint: Some(DEFAULT_LOCAL_ENDPOINT.into()),
                embedding_runtime_preset: Some(ModelRuntimePreset::Ollama),
                health: None,
            },
        )
        .expect("save hash config");
        root
    }

    #[test]
    fn progress_payload_can_publish_live_agent_node_presence() {
        let root = temp_store_root("agent-presence-progress");

        write_progress_with_presence(
            &root,
            "chat",
            2,
            6,
            "Agent is reading Plantar fascia",
            &["chunk:plantar-fascia".into()],
            Some("Plantar fascia"),
        )
        .expect("write progress");

        let raw = fs::read_to_string(progress_path(&root)).expect("read progress");
        let progress: OperationProgress = serde_json::from_str(&raw).expect("decode progress");
        assert_eq!(progress.active_node_ids, vec!["chunk:plantar-fascia"]);
        assert_eq!(
            progress.active_node_label.as_deref(),
            Some("Plantar fascia")
        );
    }

    #[test]
    fn visualization_snapshot_contains_nodes_and_edges() {
        let root = temp_store_root("snapshot");
        let input = root.join("docs");
        fs::create_dir_all(&input).expect("docs dir");
        fs::write(
            input.join("foot.txt"),
            "The foot contains the heel, arch, toes, talus, calcaneus, and plantar fascia.",
        )
        .expect("write input");
        ingest_paths(&root, &[input]).expect("ingest");
        let snapshot = get_visualization_snapshot(&root).expect("snapshot");
        assert!(!snapshot.nodes.is_empty());
        assert!(!snapshot.edges.is_empty());
    }

    #[test]
    fn chat_history_is_indexed_as_normal_document_memory_without_imported_documents() {
        let root = temp_store_root("chat-snapshot");
        let session = create_chat_session(&root, Some("Graph chat".into())).expect("create chat");
        send_chat_turn(
            &root,
            ChatTurnRequest {
                session_id: session.id.clone(),
                message: "Graph-visible chat history should appear as embedded memory.".into(),
            },
        )
        .expect("send chat turn");

        let snapshot = get_visualization_snapshot(&root).expect("snapshot");
        let document_id = chat_document_id(&session.id);
        assert!(snapshot.nodes.iter().any(|node| {
            node.id == format!("document:{document_id}")
                && node.kind == GraphNodeKind::Document
                && node.label == "Graph chat"
        }));
        assert!(snapshot.nodes.iter().any(|node| {
            node.id.starts_with(&format!("chunk:{document_id}:chunk:"))
                && node.kind == GraphNodeKind::Chunk
        }));

        let result = run_query(
            &root,
            QueryRequest {
                text: "Graph-visible chat history".into(),
                filters: Default::default(),
                max_regions: 3,
                max_chunks: 5,
            },
        )
        .expect("query");
        assert!(result.hits.iter().any(|hit| hit.document_id == document_id));
    }

    #[test]
    fn visualization_snapshot_syncs_existing_chat_messages_into_normal_document_memory() {
        let root = temp_store_root("chat-snapshot-existing");
        let store = FileMemoryStore::new(&root);
        let session =
            create_chat_session(&root, Some("Existing chat".into())).expect("create chat");
        let created_at = now_millis();
        let message = ChatMessage {
            id: "message:existing".into(),
            session_id: session.id.clone(),
            role: ChatRole::User,
            content: "Existing chat history was embedded before the unified corpus path.".into(),
            created_at,
            token_estimate: 12,
            source_anchor: Some(chat_source_anchor(
                &session.id,
                "message:existing",
                "Existing chat history was embedded before the unified corpus path.",
                created_at,
            )),
        };
        store
            .insert_chat_message(&message)
            .expect("insert old message");
        store
            .insert_transcript_chunk(&TranscriptChunk {
                id: "transcript:legacy".into(),
                session_id: session.id.clone(),
                message_id: message.id.clone(),
                ordinal: 0,
                text: message.content.clone(),
                embedding: vec![1.0, 0.0, 0.0],
                embedding_provider: "legacy".into(),
                embedding_model: "legacy".into(),
                embedding_endpoint: "legacy".into(),
                source_anchor: message.source_anchor.clone().expect("anchor"),
                attention_state: AttentionState::Hot,
                hotness: 1.0,
                created_at,
            })
            .expect("insert legacy transcript chunk");

        let snapshot = get_visualization_snapshot(&root).expect("snapshot");
        let document_id = chat_document_id(&session.id);
        assert!(snapshot
            .nodes
            .iter()
            .any(|node| node.id == format!("document:{document_id}")));
        assert!(run_query(
            &root,
            QueryRequest {
                text: "unified corpus path".into(),
                filters: Default::default(),
                max_regions: 3,
                max_chunks: 5,
            },
        )
        .expect("query")
        .hits
        .iter()
        .any(|hit| hit.document_id == document_id));
        assert!(store
            .list_transcript_chunks(Some(session.id.as_str()))
            .expect("legacy transcript chunks")
            .is_empty());
    }

    #[test]
    fn ingest_paths_appends_and_replaces_by_document_id() {
        let root = temp_store_root("append");
        let first = root.join("first");
        let second = root.join("second");
        fs::create_dir_all(&first).expect("first dir");
        fs::create_dir_all(&second).expect("second dir");
        let first_doc = first.join("alpha.txt");
        let second_doc = second.join("beta.txt");
        fs::write(&first_doc, "alpha foot ankle heel").expect("write alpha");
        fs::write(&second_doc, "beta cortex neuron memory").expect("write beta");

        let first_result =
            ingest_paths(&root, std::slice::from_ref(&first_doc)).expect("first ingest");
        assert_eq!(first_result.imported_count, 1);
        assert_eq!(first_result.summary.documents, 1);

        let second_result =
            ingest_paths(&root, std::slice::from_ref(&second_doc)).expect("second ingest");
        assert_eq!(second_result.imported_count, 1);
        assert_eq!(second_result.replaced_count, 0);
        assert_eq!(second_result.summary.documents, 2);

        fs::write(&first_doc, "alpha updated plantar fascia").expect("update alpha");
        let replace_result =
            ingest_paths(&root, std::slice::from_ref(&first_doc)).expect("replace ingest");
        assert_eq!(replace_result.imported_count, 0);
        assert_eq!(replace_result.replaced_count, 1);
        assert_eq!(replace_result.summary.documents, 2);
    }

    #[test]
    fn pdf_imports_text_and_reports_invalid_pdf_skips() {
        let root = temp_store_root("pdf");
        let pdf_path = root.join("memory.pdf");
        let invalid_path = root.join("broken.pdf");
        fs::write(&pdf_path, simple_pdf("Hello PDF memory alpha")).expect("write pdf");
        fs::write(&invalid_path, b"%PDF broken").expect("write invalid pdf");

        let result = ingest_paths(&root, &[pdf_path.clone(), invalid_path]).expect("pdf ingest");
        assert_eq!(result.imported_count, 1);
        assert_eq!(result.skipped_count, 1);
        let query = run_query(
            &root,
            QueryRequest {
                text: "alpha".into(),
                filters: Default::default(),
                max_regions: 3,
                max_chunks: 3,
            },
        )
        .expect("query pdf");
        assert!(query
            .hits
            .iter()
            .any(|hit| hit.excerpt.contains("Hello PDF memory alpha")));
    }

    #[test]
    fn query_results_are_deduplicated_by_chunk() {
        let root = temp_store_root("dedupe");
        let input = root.join("single.txt");
        fs::write(&input, "alpha alpha alpha memory").expect("write input");
        ingest_paths(&root, std::slice::from_ref(&input)).expect("ingest");
        let query = run_query(
            &root,
            QueryRequest {
                text: "alpha".into(),
                filters: Default::default(),
                max_regions: 3,
                max_chunks: 5,
            },
        )
        .expect("query");
        let unique = query
            .hits
            .iter()
            .map(|hit| hit.chunk_id.clone())
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(query.hits.len(), unique.len());
    }

    #[test]
    fn unchanged_rebuild_reuses_cached_embeddings() {
        let root = temp_store_root("reuse");
        let input = root.join("doc.txt");
        fs::write(&input, "alpha memory foot ankle heel ".repeat(120)).expect("write input");

        let first = ingest_paths(&root, std::slice::from_ref(&input)).expect("first ingest");
        assert!(first.embedded_count > 0);
        assert_eq!(first.reused_embedding_count, 0);

        let second = rebuild_memory(&root).expect("rebuild");
        assert_eq!(second.embedded_count, 0);
        assert_eq!(second.reused_embedding_count, first.summary.chunks);
    }

    #[test]
    fn importing_new_document_embeds_only_new_chunks() {
        let root = temp_store_root("new-doc");
        let first = root.join("first.txt");
        let second = root.join("second.txt");
        fs::write(&first, "alpha memory foot ankle heel ".repeat(80)).expect("write first");
        let first_result = ingest_paths(&root, std::slice::from_ref(&first)).expect("first ingest");

        fs::write(&second, "beta cortex neuron synapse ".repeat(40)).expect("write second");
        let second_result =
            ingest_paths(&root, std::slice::from_ref(&second)).expect("second ingest");
        assert_eq!(
            second_result.reused_embedding_count,
            first_result.summary.chunks
        );
        assert!(second_result.embedded_count > 0);
        assert_eq!(
            second_result.reused_embedding_count + second_result.embedded_count,
            second_result.summary.chunks
        );
    }

    #[test]
    fn changing_document_reembeds_only_changed_chunks() {
        let root = temp_store_root("changed-doc");
        let first = root.join("first.txt");
        let second = root.join("second.txt");
        fs::write(&first, "alpha memory foot ankle heel ".repeat(80)).expect("write first");
        fs::write(&second, "beta cortex neuron synapse ".repeat(80)).expect("write second");
        let initial =
            ingest_paths(&root, &[first.clone(), second.clone()]).expect("initial ingest");

        fs::write(&second, "gamma cortex neuron retrieval ".repeat(80)).expect("change second");
        let changed = ingest_paths(&root, std::slice::from_ref(&second)).expect("changed ingest");
        assert!(changed.reused_embedding_count > 0);
        assert!(changed.embedded_count > 0);
        assert_eq!(changed.summary.documents, initial.summary.documents);
        assert_eq!(
            changed.reused_embedding_count + changed.embedded_count,
            changed.summary.chunks
        );
    }

    #[test]
    fn old_chunks_without_embedding_metadata_are_reembedded() {
        let root = temp_store_root("old-chunks");
        let input = root.join("doc.txt");
        fs::write(&input, "alpha memory foot ankle heel ".repeat(60)).expect("write input");
        ingest_paths(&root, std::slice::from_ref(&input)).expect("ingest");

        let store = FileMemoryStore::new(&root);
        let mut memory = store.load().expect("load memory");
        for chunk in &mut memory.chunks {
            chunk.embedding_text_hash = None;
            chunk.embedding_provider = None;
            chunk.embedding_model = None;
            chunk.embedding_endpoint = None;
            chunk.embedding_dimension = None;
            chunk.chunking_version = None;
        }
        store.save(&memory).expect("save old-style memory");

        let rebuilt = rebuild_memory(&root).expect("rebuild");
        assert_eq!(rebuilt.reused_embedding_count, 0);
        assert_eq!(rebuilt.embedded_count, rebuilt.summary.chunks);
    }

    #[test]
    fn model_config_round_trip_does_not_require_api_key() {
        let root = temp_store_root("config");
        let config = ModelConfig {
            mode: ModelConnectionMode::Api,
            endpoint: "https://example.com".into(),
            api_key_name: Some("memory-app-api-key".into()),
            chat_model: Some("gpt-test".into()),
            planner_model: Some("planner-test".into()),
            response_model: Some("response-test".into()),
            planner_endpoint: Some("http://planner.example.com/v1".into()),
            runtime_preset: ModelRuntimePreset::CustomOpenAi,
            cortex_enabled: true,
            cortex_rounds: 2,
            critic_model: Some("critic-test".into()),
            critic_endpoint: Some("http://critic.example.com/v1".into()),
            compiler_model: Some("compiler-test".into()),
            embedding_model: Some("embed-test".into()),
            embedding_endpoint: Some("http://embedding.example.com".into()),
            embedding_runtime_preset: Some(ModelRuntimePreset::Ollama),
            health: None,
        };
        save_model_config(&root, &config).expect("save config");
        let loaded = load_model_config(&root).expect("load config");
        assert_eq!(loaded.api_key_name.as_deref(), Some("memory-app-api-key"));
        assert_eq!(loaded.planner_model.as_deref(), Some("planner-test"));
        assert_eq!(loaded.response_model.as_deref(), Some("response-test"));
        assert_eq!(
            loaded.planner_endpoint.as_deref(),
            Some("http://planner.example.com/v1")
        );
        assert_eq!(loaded.runtime_preset, ModelRuntimePreset::CustomOpenAi);
        assert!(loaded.cortex_enabled);
        assert_eq!(loaded.cortex_rounds, 2);
        assert_eq!(loaded.critic_model.as_deref(), Some("critic-test"));
        assert_eq!(
            loaded.critic_endpoint.as_deref(),
            Some("http://critic.example.com/v1")
        );
        assert_eq!(loaded.compiler_model.as_deref(), Some("compiler-test"));
        assert_eq!(
            loaded.embedding_endpoint.as_deref(),
            Some("http://embedding.example.com")
        );
        assert_eq!(
            loaded.embedding_runtime_preset,
            Some(ModelRuntimePreset::Ollama)
        );
    }

    #[test]
    fn chat_context_trace_records_recursive_cortex_rounds() {
        let root = temp_store_root("recursive-cortex");
        let input = root.join("source.txt");
        fs::write(
            &input,
            "Recursive memory cortex should route through compact maps, inspect source anchors, critique evidence, and preserve citations.",
        )
        .expect("write source");
        ingest_paths(&root, std::slice::from_ref(&input)).expect("ingest");

        let mut config = load_model_config(&root).expect("config");
        config.planner_model = Some(HASH_EMBEDDING_MODEL.into());
        config.response_model = Some(HASH_EMBEDDING_MODEL.into());
        config.cortex_enabled = true;
        config.cortex_rounds = 3;
        save_model_config(&root, &config).expect("save config");

        let session = create_chat_session(&root, Some("Cortex".into())).expect("create chat");
        let result = send_chat_turn(
            &root,
            ChatTurnRequest {
                session_id: session.id.clone(),
                message: "How should recursive memory cortex preserve citations?".into(),
            },
        )
        .expect("send chat turn");

        let cortex = result
            .context_trace
            .cortex_trace
            .as_ref()
            .expect("cortex trace");
        assert_eq!(cortex.rounds.len(), 3);
        assert!(cortex.rounds.iter().all(|round| !round.actions.is_empty()));
        assert!(cortex.rounds.iter().any(|round| round.critique.sufficient));
        assert!(result
            .context_trace
            .tool_trace
            .iter()
            .any(|line| line.contains("cortex round 1")));

        let traces = list_chat_context_traces(&root, &session.id).expect("traces");
        assert!(traces
            .first()
            .and_then(|trace| trace.cortex_trace.as_ref())
            .is_some());
    }

    #[test]
    fn memory_compile_writes_idempotent_searchable_brain_artifacts() {
        let root = temp_store_root("brain-compile");
        let input = root.join("source.txt");
        fs::write(
            &input,
            "Tiny local models become useful when memory is compiled into compact maps, entity hints, and source-grounded routing instincts.",
        )
        .expect("write source");
        ingest_paths(&root, std::slice::from_ref(&input)).expect("ingest");

        let first = compile_memory_brain(&root).expect("compile");
        let second = compile_memory_brain(&root).expect("compile again");
        assert!(first.artifacts_written > 0);
        assert_eq!(first.artifact_ids, second.artifact_ids);
        assert_eq!(first.training_records_written, first.artifacts_written);
        assert!(first
            .export_files
            .iter()
            .any(|file| file.ends_with("route_region.train.jsonl")));
        assert!(first
            .export_files
            .iter()
            .any(|file| file.ends_with("route_region.eval.jsonl")));
        let prepared_adapter = first.adapter_state.as_ref().expect("adapter state");
        assert_eq!(prepared_adapter.freshness, "fresh");
        assert_eq!(prepared_adapter.status, "prepared");
        assert_eq!(
            prepared_adapter.base_model.as_deref(),
            Some(HASH_EMBEDDING_MODEL)
        );
        assert!(prepared_adapter
            .manifest_path
            .as_deref()
            .is_some_and(|path| path.contains("/adapters/prepared-")));
        assert_eq!(
            prepared_adapter.train_records,
            Some(first.artifacts_written * 4)
        );
        assert!(std::fs::read_to_string(&first.training_records_path)
            .expect("training records")
            .contains("\"task\":\"memory_routing\""));
        let route_train = first
            .export_files
            .iter()
            .find(|file| file.ends_with("route_region.train.jsonl"))
            .expect("route train file");
        let route_eval = first
            .export_files
            .iter()
            .find(|file| file.ends_with("route_region.eval.jsonl"))
            .expect("route eval file");
        let train_raw = std::fs::read_to_string(route_train).expect("read train");
        let eval_raw = std::fs::read_to_string(route_eval).expect("read eval");
        assert!(train_raw.contains("\"schema_version\":1"));
        assert!(train_raw.contains("\"task\":\"route_region\""));
        assert!(train_raw.contains("\"artifact_ids\""));
        assert_ne!(train_raw, eval_raw);
        let adapter_dir = root.join("adapters").join("check");
        fs::create_dir_all(&adapter_dir).expect("adapter dir");
        fs::write(
            adapter_dir.join("adapter_manifest.json"),
            serde_json::json!({
                "status": "prepared",
                "base_model": "tiny-memory-model",
                "adapter_path": adapter_dir.display().to_string(),
                "dataset_hash": "prepared-hash",
                "source_dataset_hash": prepared_adapter.current_source_dataset_hash,
                "train_records": 1,
                "valid_records": 1,
                "test_records": 1,
                "iters": 25
            })
            .to_string(),
        )
        .expect("adapter manifest");
        let with_adapter = compile_memory_brain(&root).expect("compile with adapter");
        let adapter_state = with_adapter.adapter_state.as_ref().expect("adapter state");
        assert_eq!(adapter_state.freshness, "fresh");
        assert_eq!(adapter_state.status, "prepared");
        assert_eq!(
            adapter_state.base_model.as_deref(),
            Some("tiny-memory-model")
        );
        let persisted_snapshot =
            load_cortex_adapter_snapshot(&root).expect("load persisted adapter state");
        let persisted_adapter = persisted_snapshot
            .adapter_state
            .as_ref()
            .expect("persisted adapter state");
        assert_eq!(persisted_adapter, adapter_state);

        let memories = list_derived_memories(&root, None).expect("derived memories");
        assert!(memories.iter().any(|memory| {
            memory.id.starts_with("brain-region:")
                && memory.provenance.reason.contains("Compiled")
                && memory.text.contains("brain artifact")
        }));

        let memory = FileMemoryStore::new(&root).load().expect("load memory");
        assert!(memory.documents.iter().any(|document| {
            document.metadata.get("source_type").map(String::as_str) == Some("derived_memory")
                && document.text.contains("brain artifact")
        }));
        let artifacts = FileMemoryStore::new(&root)
            .list_brain_artifacts(None)
            .expect("brain artifacts");
        assert!(artifacts
            .iter()
            .any(|artifact| artifact.kind == BrainArtifactKind::RegionCard));
        assert!(artifacts
            .iter()
            .any(|artifact| artifact.kind == BrainArtifactKind::RoutingRule));
        assert!(artifacts
            .iter()
            .all(|artifact| !artifact.source_refs.is_empty()));
        assert!(run_query(
            &root,
            QueryRequest {
                text: "routing instincts compact maps".into(),
                filters: BTreeMap::new(),
                max_regions: 4,
                max_chunks: 6,
            },
        )
        .expect("query")
        .hits
        .iter()
        .any(|hit| hit
            .source_anchor
            .as_ref()
            .is_some_and(|anchor| anchor.path.starts_with("imprint://derived/"))));
    }

    #[test]
    fn chat_turns_are_persisted_indexed_hot_and_survive_rebuilds() {
        let root = temp_store_root("chat-memory");
        let session = create_chat_session(&root, Some("Milestone A".into())).expect("create chat");

        let turn = send_chat_turn(
            &root,
            ChatTurnRequest {
                session_id: session.id.clone(),
                message: "Remember that Milestone A keeps chat history hot for retrieval.".into(),
            },
        )
        .expect("send chat turn");

        assert_eq!(turn.session.id, session.id);
        assert_eq!(turn.messages.len(), 2);
        assert!(turn
            .context_trace
            .snippets
            .iter()
            .any(|snippet| snippet.hotness > 0.0));

        let messages = list_chat_messages(&root, &session.id).expect("messages");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, ChatRole::User);
        assert_eq!(messages[1].role, ChatRole::Assistant);

        let memory = FileMemoryStore::new(&root).load().expect("load memory");
        let chat_doc_id = chat_document_id(&session.id);
        assert!(memory.documents.iter().any(|document| {
            document.id == chat_doc_id
                && document.metadata.get("source_type").map(String::as_str) == Some("chat")
                && document.text.contains("Milestone A keeps chat history hot")
        }));
        assert!(memory.chunks.iter().any(|chunk| {
            chunk.document_id == chat_doc_id
                && chunk.metadata.get("source_type").map(String::as_str) == Some("chat")
                && chunk
                    .source_anchor
                    .as_ref()
                    .is_some_and(|anchor| anchor.path.starts_with("imprint://chat/"))
        }));

        let marks = list_attention_marks(&root, None).expect("attention marks");
        assert!(marks
            .iter()
            .any(|mark| mark.action == AttentionAction::Active && mark.target_id == session.id));

        let input = root.join("source.txt");
        fs::write(&input, "alpha memory foot ankle heel ".repeat(60)).expect("write input");
        ingest_paths(&root, std::slice::from_ref(&input)).expect("ingest");
        rebuild_memory(&root).expect("rebuild");

        let after_rebuild = list_chat_messages(&root, &session.id).expect("messages after rebuild");
        assert_eq!(after_rebuild.len(), 2);
    }

    #[test]
    fn chat_context_trace_uses_agentic_planner_steps_and_marks_attention() {
        let root = temp_store_root("planner-loop");
        let input = root.join("source.txt");
        fs::write(
            &input,
            "The source of truth says retrieval should read the map, search vector memory, expand original source context, and mark useful memories.",
        )
        .expect("write source");
        ingest_paths(&root, std::slice::from_ref(&input)).expect("ingest");

        let session = create_chat_session(&root, Some("Planner".into())).expect("create chat");
        let result = send_chat_turn(
            &root,
            ChatTurnRequest {
                session_id: session.id.clone(),
                message: "How should retrieval use the map and source context?".into(),
            },
        )
        .expect("send chat turn");

        assert!(result
            .context_trace
            .tool_trace
            .iter()
            .any(|line| line.contains("planner step 1")));
        assert!(result
            .context_trace
            .tool_trace
            .iter()
            .any(|line| line.contains("memory_search")));
        assert!(result
            .context_trace
            .tool_trace
            .iter()
            .any(|line| line.contains("memory_open")));
        assert!(result
            .context_trace
            .tool_trace
            .iter()
            .any(|line| line.contains("memory_neighbors")));
        assert!(result
            .context_trace
            .tool_trace
            .iter()
            .any(|line| line.contains("memory_expand")));
        assert!(result.context_trace.snippets.len() <= MAX_RESPONSE_CONTEXT_SNIPPETS);
        assert!(result
            .context_trace
            .snippets
            .iter()
            .any(|snippet| snippet.source_anchor.is_some()));
        assert!(result
            .context_trace
            .snippets
            .iter()
            .any(|snippet| snippet.source_kind == "source_window"));

        let marks = list_attention_marks(&root, None).expect("attention marks");
        assert!(marks
            .iter()
            .any(|mark| mark.action == AttentionAction::Promote));
        assert!(marks
            .iter()
            .any(|mark| mark.action == AttentionAction::Decay));
    }

    #[test]
    fn chat_lists_structured_csv_rows_without_hallucinating_records() {
        let root = temp_store_root("chat-csv-list");
        let input = root.join("employees-current-any-date.csv");
        fs::write(
            &input,
            "First Name,Last Name,Xage Email,Department,Position\n\
Benjamin,Paul,benjamin@xage.com,Development,Sr. Software Engineer\n\
Amit,Pawar,amit@xage.com,Sales,VP Consulting and Services\n\
Kevin,Tang,ktang@xage.com,Development,Senior Software Engineer\n",
        )
        .expect("write csv");
        ingest_paths(&root, std::slice::from_ref(&input)).expect("ingest");

        let session = create_chat_session(&root, Some("CSV".into())).expect("create chat");
        let result = send_chat_turn(
            &root,
            ChatTurnRequest {
                session_id: session.id.clone(),
                message: "please list all of the xage employees for me".into(),
            },
        )
        .expect("send chat turn");
        let answer = &result.messages[1].content;
        assert!(answer.contains("Benjamin Paul"));
        assert!(answer.contains("Amit Pawar"));
        assert!(answer.contains("Kevin Tang"));
        assert!(answer.contains("benjamin@xage.com"));
        assert!(!answer.contains("Michael Graham"));
        assert!(!answer.contains("@xage.gov"));
    }

    #[test]
    fn derived_memories_and_web_findings_are_audited_writebacks() {
        let root = temp_store_root("writebacks");
        let session = create_chat_session(&root, Some("Writebacks".into())).expect("create chat");

        let summary = write_derived_memory(
            &root,
            DerivedMemoryWrite {
                session_id: Some(session.id.clone()),
                kind: DerivedMemoryKind::Summary,
                text: "The user decided chat transcripts should become searchable hot memory."
                    .into(),
                source_message_ids: Vec::new(),
                actor: "assistant".into(),
                confidence: 0.82,
            },
        )
        .expect("write summary");
        assert_eq!(summary.kind, DerivedMemoryKind::Summary);

        let finding = write_web_finding(
            &root,
            WebFindingWrite {
                session_id: Some(session.id.clone()),
                query: "LFM2.5 MLX OpenAI compatible".into(),
                url: "https://docs.liquid.ai/lfm/inference/mlx".into(),
                title: "Liquid MLX docs".into(),
                summary: "MLX can serve LFM models through an OpenAI-compatible API.".into(),
                retrieved_at: 1_777_311_476,
                confidence: 0.9,
                actor: "assistant".into(),
            },
        )
        .expect("write web finding");
        assert_eq!(finding.url, "https://docs.liquid.ai/lfm/inference/mlx");

        let mark = apply_attention_mark(
            &root,
            AttentionMarkWrite {
                target_id: summary.id.clone(),
                target_kind: AttentionTargetKind::DerivedMemory,
                action: AttentionAction::Promote,
                reason: "Useful default for future routing.".into(),
                actor: "assistant".into(),
            },
        )
        .expect("attention mark");
        assert_eq!(mark.action, AttentionAction::Promote);

        let audit = list_audit_events(&root, Some(session.id)).expect("audit");
        assert!(audit
            .iter()
            .any(|event| event.event_type == "derived_memory.write"));
        assert!(audit
            .iter()
            .any(|event| event.event_type == "web_finding.write"));
        assert!(audit
            .iter()
            .any(|event| event.event_type == "attention.mark"));
    }

    #[test]
    fn web_findings_are_indexed_as_source_anchored_memory() {
        let root = temp_store_root("web-finding-index");
        let session = create_chat_session(&root, Some("Web".into())).expect("create chat");

        let finding = write_web_finding(
            &root,
            WebFindingWrite {
                session_id: Some(session.id),
                query: "current memory filesystem research".into(),
                url: "https://example.com/current-memory-filesystem".into(),
                title: "Current memory filesystem research".into(),
                summary: "Fresh web evidence says semantic filesystems should preserve URL provenance and retrieval dates.".into(),
                retrieved_at: 1_777_311_476,
                confidence: 0.86,
                actor: "assistant".into(),
            },
        )
        .expect("write web finding");
        write_web_finding(
            &root,
            WebFindingWrite {
                session_id: None,
                query: "duplicate current memory filesystem research".into(),
                url: "https://example.com/current-memory-filesystem".into(),
                title: "Updated current memory filesystem research".into(),
                summary: "Updated fresh web evidence says semantic filesystems still preserve URL provenance and retrieval dates.".into(),
                retrieved_at: 1_777_311_500,
                confidence: 0.88,
                actor: "assistant".into(),
            },
        )
        .expect("rewrite same URL without duplicate document failure");

        let query = run_query(
            &root,
            QueryRequest {
                text: "semantic filesystems URL provenance retrieval dates".into(),
                filters: Default::default(),
                max_regions: 5,
                max_chunks: 5,
            },
        )
        .expect("query web memory");

        let hit = query
            .hits
            .iter()
            .find(|hit| hit.document_id == web_finding_document_id(&finding))
            .expect("web finding hit");
        let anchor = hit.source_anchor.as_ref().expect("web source anchor");
        assert_eq!(anchor.path, "https://example.com/current-memory-filesystem");
    }

    #[derive(Debug)]
    struct FixtureWebSearcher {
        results: Vec<WebSearchResult>,
    }

    impl WebSearcher for FixtureWebSearcher {
        fn search(
            &self,
            _query: &str,
            _max_results: usize,
        ) -> anyhow::Result<Vec<WebSearchResult>> {
            Ok(self.results.clone())
        }
    }

    #[test]
    fn chat_context_falls_back_to_web_search_and_researches_embedded_results_when_memory_is_weak() {
        let root = temp_store_root("chat-web-fallback");
        let store = FileMemoryStore::new(&root);
        let mut session =
            create_chat_session(&root, Some("Web fallback".into())).expect("create chat");
        let config = load_model_config(&root).expect("config");
        let embedder = embedder_for_config(&config).expect("embedder");
        let user_message = append_chat_message(
            &store,
            &mut session,
            ChatRole::User,
            "What is the fresh remote answer about semantic filesystem provenance?".into(),
        )
        .expect("append user");
        refresh_chat_session_document(&root, &store, &session.id, &config)
            .expect("refresh chat document");

        let trace = build_chat_context_trace_with_searcher(
            &root,
            &store,
            &embedder,
            &config,
            &session.id,
            &user_message,
            &FixtureWebSearcher {
                results: vec![WebSearchResult {
                    title: "Fresh remote answer".into(),
                    url: "https://example.com/fresh-remote-answer".into(),
                    snippet: "Fresh remote answer for semantic filesystem provenance.".into(),
                    body: "Fresh remote answer says agent web search should be written into memory, embedded, searched again, and cited from URL anchors.".into(),
                }],
            },
        )
        .expect("trace");

        assert!(trace
            .tool_trace
            .iter()
            .any(|line| line.contains("web_search")));
        assert!(trace
            .tool_trace
            .iter()
            .any(|line| line.contains("web_research")));
        assert!(trace.snippets.iter().any(|snippet| {
            snippet
                .source_anchor
                .as_ref()
                .is_some_and(|anchor| anchor.path == "https://example.com/fresh-remote-answer")
        }));
    }

    #[test]
    fn weak_local_document_hits_do_not_suppress_web_search() {
        let anchor = SourceAnchor {
            id: "anchor:weak-local".into(),
            document_id: "doc:employees".into(),
            chunk_id: Some("chunk:employees".into()),
            path: "/tmp/xage-employees.csv".into(),
            content_hash: "hash".into(),
            start: 0,
            end: 80,
            page: None,
            section: None,
            parser_version: 1,
        };
        let snippets = vec![ChatContextSnippet {
            id: "weak-local".into(),
            source_kind: "chunk".into(),
            source_id: "chunk:employees".into(),
            excerpt: "An unrelated local employee list matched only the word employees.".into(),
            score: 0.12,
            hotness: 0.25,
            source_anchor: Some(anchor),
        }];

        assert!(should_search_web(
            "how many employees does google have?",
            &snippets
        ));
    }

    #[test]
    fn explicit_web_search_requests_bypass_local_confidence_gate() {
        let anchor = SourceAnchor {
            id: "anchor:strong-local".into(),
            document_id: "doc:cached".into(),
            chunk_id: Some("chunk:cached".into()),
            path: "/tmp/cached.md".into(),
            content_hash: "hash".into(),
            start: 0,
            end: 80,
            page: None,
            section: None,
            parser_version: 1,
        };
        let snippets = vec![ChatContextSnippet {
            id: "strong-local".into(),
            source_kind: "chunk".into(),
            source_id: "chunk:cached".into(),
            excerpt: "A strong local cached result should not block an explicit web request."
                .into(),
            score: 0.98,
            hotness: 0.25,
            source_anchor: Some(anchor),
        }];

        assert!(should_search_web("please search the web", &snippets));
    }

    #[test]
    fn web_search_returns_source_snippets_when_embedding_refresh_fails() {
        let root = temp_store_root("web-search-embedding-fallback");
        let store = FileMemoryStore::new(&root);
        let session =
            create_chat_session(&root, Some("Embedding fallback".into())).expect("session");
        let good_config = load_model_config(&root).expect("good config");
        let embedder = embedder_for_config(&good_config).expect("hash embedder");
        let mut bad_embedding_config = good_config;
        bad_embedding_config.embedding_model = Some("missing-embedding-model".into());
        bad_embedding_config.embedding_endpoint = Some("http://127.0.0.1:9".into());
        bad_embedding_config.embedding_runtime_preset = Some(ModelRuntimePreset::CustomOpenAi);
        let mut tool_trace = Vec::new();

        let snippets = search_web_into_memory(
            &root,
            &store,
            &embedder,
            &bad_embedding_config,
            &session.id,
            "Google employee count",
            WEB_SEARCH_MAX_RESULTS,
            &FixtureWebSearcher {
                results: vec![WebSearchResult {
                    title: "Alphabet annual report workforce".into(),
                    url: "https://example.com/alphabet-workforce".into(),
                    snippet: "Alphabet reported its employee count in its annual report.".into(),
                    body: "Alphabet reported 182,502 employees as of December 31, 2023.".into(),
                }],
            },
            &mut tool_trace,
        )
        .expect("web snippets despite embedding failure");

        assert!(tool_trace
            .iter()
            .any(|line| line.contains("web_research fallback")));
        assert!(snippets.iter().any(|snippet| {
            snippet.excerpt.contains("182,502")
                && snippet
                    .source_anchor
                    .as_ref()
                    .is_some_and(|anchor| anchor.path == "https://example.com/alphabet-workforce")
        }));
    }

    #[test]
    fn summary_still_loads_existing_documents_when_web_finding_sync_fails() {
        let root = temp_store_root("summary-web-sync-failure");
        let input = root.join("local.txt");
        fs::write(
            &input,
            "local documents should remain visible when web sync fails",
        )
        .expect("write local doc");
        ingest_paths(&root, std::slice::from_ref(&input)).expect("ingest local");
        let store = FileMemoryStore::new(&root);
        let created_at = now_millis();
        store
            .insert_web_finding(&WebFinding {
                id: "web:pending".into(),
                session_id: None,
                query: "pending web finding".into(),
                url: "https://example.com/pending".into(),
                title: "Pending web finding".into(),
                summary: "A pending web finding should not blank the library if embedding fails."
                    .into(),
                retrieved_at: now_secs(),
                confidence: 0.7,
                actor: "test".into(),
                created_at,
                provenance: ProvenanceRecord {
                    actor: "test".into(),
                    reason: "regression test".into(),
                    created_at,
                    source_refs: vec!["https://example.com/pending".into()],
                },
            })
            .expect("insert web finding");
        let mut bad_config = load_model_config(&root).expect("config");
        bad_config.embedding_model = Some("missing-embedding-model".into());
        bad_config.embedding_endpoint = Some("http://127.0.0.1:9".into());
        bad_config.embedding_runtime_preset = Some(ModelRuntimePreset::CustomOpenAi);
        save_model_config(&root, &bad_config).expect("save bad config");

        let summary = get_memory_summary(&root).expect("summary should degrade gracefully");

        assert!(summary.documents >= 1);
        assert!(summary.chunks >= 1);
    }

    #[test]
    fn sqlite_store_round_trip_preserves_source_anchors() {
        let root = temp_store_root("sqlite-anchors");
        let input = root.join("anchored.md");
        fs::write(&input, "# Intro\nalpha memory foot ankle heel").expect("write input");

        ingest_paths(&root, std::slice::from_ref(&input)).expect("ingest");
        assert!(root.join("memory.sqlite").exists());

        let store = FileMemoryStore::new(&root);
        let memory = store.load().expect("load sqlite memory");
        let chunk = memory.chunks.first().expect("chunk");
        let anchor = chunk.source_anchor.as_ref().expect("source anchor");
        assert_eq!(anchor.path, input.display().to_string());
        assert_eq!(anchor.page, Some(1));
        assert_eq!(anchor.section.as_deref(), Some("Intro"));
        assert_eq!(
            chunk.metadata.get("section").map(String::as_str),
            Some("Intro")
        );
    }

    #[test]
    fn search_results_return_valid_anchors_and_can_expand_section() {
        let root = temp_store_root("surf-section");
        let input = root.join("notes.md");
        fs::write(
            &input,
            "# Alpha\nalpha memory foot ankle heel\n\n# Beta\nbeta cortex neuron synapse",
        )
        .expect("write input");
        ingest_paths(&root, std::slice::from_ref(&input)).expect("ingest");

        let query = run_query(
            &root,
            QueryRequest {
                text: "alpha ankle".into(),
                filters: Default::default(),
                max_regions: 3,
                max_chunks: 3,
            },
        )
        .expect("query");
        let hit = query.hits.first().expect("hit");
        assert!(hit.source_anchor.is_some());

        let expansion =
            surf_expand(&root, &hit.chunk_id, ExpandMode::Section, 120).expect("expand");
        assert!(expansion.excerpt.contains("Alpha"));
        assert_eq!(
            expansion
                .source_anchor
                .as_ref()
                .and_then(|anchor| anchor.section.as_deref()),
            Some("Alpha")
        );
    }

    #[test]
    fn surf_neighbors_exposes_document_and_semantic_moves() {
        let root = temp_store_root("surf-neighbors");
        let input = root.join("long.txt");
        fs::write(&input, "alpha memory foot ankle heel ".repeat(120)).expect("write input");
        ingest_paths(&root, std::slice::from_ref(&input)).expect("ingest");

        let store = FileMemoryStore::new(&root);
        let memory = store.load().expect("load memory");
        let chunk = memory.chunks.first().expect("chunk");
        let neighbors = surf::neighbors(&memory, &NodeRef::Chunk(chunk.id.clone()), 8);
        assert!(neighbors
            .iter()
            .any(|neighbor| matches!(neighbor.node, NodeRef::Document(_))));
        assert!(neighbors.iter().any(|neighbor| neighbor.link_type
            == Some(LinkType::SameDocument)
            || neighbor.link_type == Some(LinkType::SemanticNeighbor)));
    }

    #[test]
    fn surf_open_chunk_includes_exact_selected_chunk_passage() {
        let root = temp_store_root("surf-chunk-passage");
        let input = root.join("long.txt");
        fs::write(&input, "alpha memory foot ankle heel ".repeat(80)).expect("write input");
        ingest_paths(&root, std::slice::from_ref(&input)).expect("ingest");

        let store = FileMemoryStore::new(&root);
        let memory = store.load().expect("load memory");
        let chunk = memory.chunks.first().expect("chunk");
        let opened = surf_open(&root, &NodeRef::Chunk(chunk.id.clone())).expect("open chunk");
        let passage = opened.passages.first().expect("passage");
        assert_eq!(passage.role, crate::surf::SurfPassageRole::SelectedChunk);
        assert_eq!(passage.excerpt, chunk.text);
        assert_eq!(passage.start, chunk.start);
        assert_eq!(passage.end, chunk.end);
    }

    #[test]
    fn surf_open_region_includes_representative_chunk_passages() {
        let root = temp_store_root("surf-region-passages");
        let input = root.join("docs.txt");
        fs::write(
            &input,
            format!(
                "{}{}",
                "identity privilege access policy ".repeat(80),
                "neuron cortex memory retrieval ".repeat(80)
            ),
        )
        .expect("write input");
        ingest_paths(&root, std::slice::from_ref(&input)).expect("ingest");

        let store = FileMemoryStore::new(&root);
        let memory = store.load().expect("load memory");
        let region = memory.regions.first().expect("region");
        let opened = surf_open(&root, &NodeRef::Region(region.id.clone())).expect("open region");
        assert!(!opened.passages.is_empty());
        assert!(opened
            .passages
            .iter()
            .all(|passage| passage.role == crate::surf::SurfPassageRole::RepresentativeChunk));
        assert!(opened
            .passages
            .windows(2)
            .all(|window| window[0].score >= window[1].score));
    }

    #[test]
    fn surf_open_document_includes_first_document_chunks() {
        let root = temp_store_root("surf-document-passages");
        let input = root.join("book.txt");
        fs::write(&input, "alpha memory foot ankle heel ".repeat(240)).expect("write input");
        ingest_paths(&root, std::slice::from_ref(&input)).expect("ingest");

        let store = FileMemoryStore::new(&root);
        let memory = store.load().expect("load memory");
        let document = memory.documents.first().expect("document");
        let opened =
            surf_open(&root, &NodeRef::Document(document.id.clone())).expect("open document");
        let expected = memory
            .chunks
            .iter()
            .filter(|chunk| chunk.document_id == document.id)
            .take(5)
            .map(|chunk| NodeRef::Chunk(chunk.id.clone()))
            .collect::<Vec<_>>();
        let actual = opened
            .passages
            .iter()
            .map(|passage| passage.node.clone())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        assert!(opened
            .passages
            .iter()
            .all(|passage| passage.role == crate::surf::SurfPassageRole::DocumentChunk));
    }

    #[test]
    fn visualization_region_labels_skip_common_glue_words() {
        let root = temp_store_root("region-labels");
        let input = root.join("labels.txt");
        fs::write(
            &input,
            "the and for our access policy identity privilege security ".repeat(120),
        )
        .expect("write input");
        ingest_paths(&root, std::slice::from_ref(&input)).expect("ingest");

        let snapshot = get_visualization_snapshot(&root).expect("snapshot");
        let labels = snapshot
            .nodes
            .iter()
            .filter(|node| node.kind == GraphNodeKind::Region)
            .map(|node| node.label.as_str())
            .collect::<Vec<_>>();
        assert!(!labels.is_empty());
        assert!(labels
            .iter()
            .all(|label| *label != "and" && *label != "the"));
    }

    fn simple_pdf(text: &str) -> Vec<u8> {
        let escaped = text
            .replace('\\', "\\\\")
            .replace('(', "\\(")
            .replace(')', "\\)");
        let stream = format!("BT /F1 18 Tf 72 120 Td ({escaped}) Tj ET\n");
        let objects = vec![
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>".to_vec(),
            format!("<< /Length {} >>\nstream\n{stream}endstream", stream.len()).into_bytes(),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
        ];
        let mut out = b"%PDF-1.4\n".to_vec();
        let mut offsets = Vec::new();
        for (index, object) in objects.iter().enumerate() {
            offsets.push(out.len());
            out.extend_from_slice(format!("{} 0 obj\n", index + 1).as_bytes());
            out.extend_from_slice(object);
            out.extend_from_slice(b"\nendobj\n");
        }
        let xref = out.len();
        out.extend_from_slice(
            format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).as_bytes(),
        );
        for offset in offsets {
            out.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
        }
        out.extend_from_slice(
            format!(
                "trailer\n<< /Root 1 0 R /Size {} >>\nstartxref\n{xref}\n%%EOF\n",
                objects.len() + 1
            )
            .as_bytes(),
        );
        out
    }
}
