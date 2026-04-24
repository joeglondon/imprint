use crate::extract::{Extractor, MemoryExtractor};
use crate::graph::GraphBuilder;
use crate::index::{cosine_similarity, Embedder, HashEmbedder, Indexer, OllamaEmbedder, RegionIndexer, RuntimeEmbedder};
use crate::ingest::{format_region_label, meaningful_terms_for_chunks, EmbeddingReuseStats, ImportSkip, Ingester};
use crate::map::MapBuilder;
use crate::navigation::{MemoryNavigator, Navigator};
use crate::query::MemoryQueryEngine;
use crate::store::{FileMemoryStore, MemoryStore};
use crate::surf::{self, ExpandMode, SurfAction, SurfExpansion, SurfNeighbor, SurfOpenResult, SurfStepResult};
use crate::types::*;
use anyhow::{anyhow, Context};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_WINDOW: usize = 90;
const DEFAULT_LOCAL_ENDPOINT: &str = "http://localhost:11434";
const DEFAULT_LOCAL_EMBEDDING_MODEL: &str = "embeddinggemma:300m";
const HASH_EMBEDDING_MODEL: &str = "hash";
const VISUAL_SCALE: f32 = 18.0;
const MAX_VISUAL_CHUNK_NEIGHBORS: usize = 2;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelConnectionMode {
    Local,
    Api,
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
    pub embedding_model: Option<String>,
    pub health: Option<ModelHealth>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelConnectionTestRequest {
    pub mode: ModelConnectionMode,
    pub endpoint: String,
    pub api_key: Option<String>,
    pub chat_model: Option<String>,
    pub embedding_model: Option<String>,
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
}

pub fn ingest_paths(store_root: &Path, paths: &[PathBuf]) -> anyhow::Result<ImportResult> {
    write_progress(store_root, "extract", 0, 1, "Reading selected files")?;
    let store = FileMemoryStore::new(store_root);
    let existing = store.load()?;
    let config = load_model_config(store_root)?;
    let ingester = Ingester::new(embedder_for_config(&config)?);
    let batch = ingester.extract_documents(paths)?;
    let existing_ids = existing
        .documents
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

    let mut documents = existing
        .documents
        .into_iter()
        .filter(|document| !incoming_ids.contains(&document.id))
        .collect::<Vec<_>>();
    documents.extend(batch.documents);
    let (memory, stats) = rebuild_from_documents(store_root, documents, &existing.chunks, &config)?;
    write_progress(store_root, "save", 99, 100, "Saving memory store")?;
    store.save(&memory)?;
    write_progress(store_root, "complete", 100, 100, "Memory updated")?;
    Ok(ImportResult {
        summary: summarize(&memory),
        imported_count: imported_paths.len(),
        replaced_count: replaced_paths.len(),
        skipped_count: batch.skipped_paths.len(),
        embedded_count: stats.embedded_count,
        reused_embedding_count: stats.reused_embedding_count,
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
    let docs = memory.documents;
    let reusable_chunks = memory.chunks;
    let config = load_model_config(store_root)?;
    let (rebuilt, stats) = rebuild_from_documents(store_root, docs, &reusable_chunks, &config)?;
    write_progress(store_root, "save", 99, 100, "Saving rebuilt memory")?;
    store.save(&rebuilt)?;
    write_progress(store_root, "complete", 100, 100, "Memory rebuilt")?;
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
    })
}

pub fn get_memory_summary(store_root: &Path) -> anyhow::Result<MemorySummary> {
    let store = FileMemoryStore::new(store_root);
    let memory = store.load()?;
    Ok(summarize(&memory))
}

pub fn get_visualization_snapshot(store_root: &Path) -> anyhow::Result<VisualizationSnapshot> {
    let memory = load_ready_memory(store_root)?;
    memory.memory_map.as_ref().context("memory map missing")?;
    let display_regions = display_regions(&memory);
    let display_region_by_id = display_regions
        .iter()
        .map(|region| (region.id.clone(), region.clone()))
        .collect::<HashMap<_, _>>();
    let map = MapBuilder::default().build(&display_regions);
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let chunk_positions = vector_positions(&memory.chunks);
    let document_chunk_counts = memory.chunks.iter().fold(HashMap::<String, usize>::new(), |mut counts, chunk| {
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
            detail: format!("{} chunks · {}", region.chunk_ids.len(), display_region.summary),
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
            document_positions.insert(document.id.clone(), GraphPosition { y: position.y + 2.2, ..position });
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
                document_chunk_counts.get(&document.id).copied().unwrap_or(0),
                truncate(&document.text, 80)
            ),
            score: 0.9,
            position,
            region_id,
        });
    }

    add_visual_similarity_edges(&memory.chunks, &mut edges);
    for link in memory
        .links
        .iter()
        .filter(|link| matches!(link.link_type, LinkType::SemanticNeighbor | LinkType::EntityOverlap | LinkType::CitationReference))
    {
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
    let memory = load_ready_memory(store_root)?;
    let config = load_model_config(store_root)?;
    let embedder = embedder_for_config(&config)?;
    let ann = RegionIndexer.rebuild(&memory.chunks, &memory.regions);
    MemoryQueryEngine.execute(&embedder, &memory, &ann, request)
}

pub fn grep_region(store_root: &Path, region_id: &str, needle: &str) -> anyhow::Result<Vec<ExtractHit>> {
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
    Ok(DocumentExcerpt {
        document_id: document.id.clone(),
        title: document.title.clone(),
        chunk_id: chunk.id.clone(),
        excerpt: excerpt_window(&document.text, chunk.start, chunk.end, 140),
        start: chunk.start,
        end: chunk.end,
        source_anchor: chunk.source_anchor.clone(),
    })
}

pub fn grep_document(store_root: &Path, document_id: &str, needle: &str) -> anyhow::Result<Vec<ExtractHit>> {
    let memory = load_ready_memory(store_root)?;
    Ok(MemoryExtractor.grep_document(
        &memory,
        document_id,
        needle,
        DEFAULT_WINDOW,
    ))
}

pub fn semantic_document_search(
    store_root: &Path,
    document_id: &str,
    query: &str,
) -> anyhow::Result<Vec<ExtractHit>> {
    let memory = load_ready_memory(store_root)?;
    let config = load_model_config(store_root)?;
    let embedder = embedder_for_config(&config)?;
    MemoryExtractor.semantic_document(
        &embedder,
        &memory,
        document_id,
        query,
        DEFAULT_WINDOW,
    )
}

pub fn surf_open(store_root: &Path, node: &NodeRef) -> anyhow::Result<SurfOpenResult> {
    let memory = load_ready_memory(store_root)?;
    surf::open(&memory, node).context("node not found")
}

pub fn surf_neighbors(store_root: &Path, node: &NodeRef, max_results: usize) -> anyhow::Result<Vec<SurfNeighbor>> {
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
    surf::expand_chunk(&memory, chunk_id, mode, window).context("chunk not found")
}

pub fn surf_jump_to_anchor(
    store_root: &Path,
    anchor_id: &str,
    window: usize,
) -> anyhow::Result<SurfExpansion> {
    let memory = load_ready_memory(store_root)?;
    surf::jump_to_anchor(&memory, anchor_id, window).context("anchor not found")
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

pub fn backtrack_navigation(store_root: &Path, session: SessionState) -> anyhow::Result<NavigationResult> {
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
            endpoint: DEFAULT_LOCAL_ENDPOINT.into(),
            api_key_name: None,
            chat_model: None,
            embedding_model: Some(DEFAULT_LOCAL_EMBEDDING_MODEL.into()),
            health: Some(ModelHealth {
                status: "disconnected".into(),
                message: "Using local Ollama embeddings".into(),
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
    Ok(config)
}

pub fn test_model_connection(request: ModelConnectionTestRequest) -> anyhow::Result<ModelHealth> {
    let client = Client::builder().timeout(std::time::Duration::from_secs(8)).build()?;
    let endpoint = request.endpoint.trim_end_matches('/');
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
            let embedder = OllamaEmbedder::new(endpoint, model)?;
            embedder.embed("AI memory embedding health check")?;
            return Ok(ModelHealth {
                status: "connected".into(),
                message: format!("Ollama embedding model {model} responded"),
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
    let (mut memory, stats) = ingester.ingest_documents_reusing_embeddings(documents, reusable_chunks, |completed, total| {
        let _ = write_progress(
            store_root,
            "embed",
            completed,
            total.max(1),
            &format!("Embedding/reusing chunks {completed}/{}", total.max(1)),
        );
    })?;
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
    std::fs::create_dir_all(store_root)?;
    let total = total.max(1);
    let percent = ((completed as f32 / total as f32) * 100.0).clamp(0.0, 100.0);
    let payload = OperationProgress {
        phase: phase.into(),
        completed,
        total,
        percent,
        message: message.into(),
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
    match config.mode {
        ModelConnectionMode::Local if model != HASH_EMBEDDING_MODEL => {
            Ok(RuntimeEmbedder::Ollama(OllamaEmbedder::new(&config.endpoint, model)?))
        }
        _ => Ok(RuntimeEmbedder::Hash(HashEmbedder::default())),
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
    truncate(if label.is_empty() { text.trim() } else { &label }, max)
}

fn display_regions(memory: &PersistedMemory) -> Vec<Region> {
    memory
        .regions
        .iter()
        .map(|region| {
            let terms = meaningful_terms_for_chunks(
                memory.chunks.iter().filter(|chunk| chunk.region_id == region.id),
                4,
            );
            if terms.is_empty() {
                return region.clone();
            }
            let mut display = region.clone();
            display.label = format_region_label(&terms);
            display.summary = format!("Vector region around {}", terms.join(", "));
            display.filters.insert("topic".into(), display.label.clone());
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
    left.iter().zip(right).map(|(left, right)| left * right).sum()
}

fn normalize_positions(points: &mut [(String, GraphPosition)]) {
    if points.is_empty() {
        return;
    }
    let count = points.len() as f32;
    let center = points.iter().fold(GraphPosition::default(), |mut acc, (_, point)| {
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
        max_radius = max_radius.max((point.x * point.x + point.y * point.y + point.z * point.z).sqrt());
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
    let sum = points.iter().fold(GraphPosition::default(), |mut acc, point| {
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
            .map(|other| (other.id.clone(), cosine_similarity(&chunk.embedding, &other.embedding)))
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
                embedding_model: Some(HASH_EMBEDDING_MODEL.into()),
                health: None,
            },
        )
        .expect("save hash config");
        root
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

        let first_result = ingest_paths(&root, std::slice::from_ref(&first_doc)).expect("first ingest");
        assert_eq!(first_result.imported_count, 1);
        assert_eq!(first_result.summary.documents, 1);

        let second_result = ingest_paths(&root, std::slice::from_ref(&second_doc)).expect("second ingest");
        assert_eq!(second_result.imported_count, 1);
        assert_eq!(second_result.replaced_count, 0);
        assert_eq!(second_result.summary.documents, 2);

        fs::write(&first_doc, "alpha updated plantar fascia").expect("update alpha");
        let replace_result = ingest_paths(&root, std::slice::from_ref(&first_doc)).expect("replace ingest");
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
        assert!(query.hits.iter().any(|hit| hit.excerpt.contains("Hello PDF memory alpha")));
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
        let second_result = ingest_paths(&root, std::slice::from_ref(&second)).expect("second ingest");
        assert_eq!(second_result.reused_embedding_count, first_result.summary.chunks);
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
        let initial = ingest_paths(&root, &[first.clone(), second.clone()]).expect("initial ingest");

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
            embedding_model: Some("embed-test".into()),
            health: None,
        };
        save_model_config(&root, &config).expect("save config");
        let loaded = load_model_config(&root).expect("load config");
        assert_eq!(loaded.api_key_name.as_deref(), Some("memory-app-api-key"));
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
        assert_eq!(chunk.metadata.get("section").map(String::as_str), Some("Intro"));
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

        let expansion = surf_expand(&root, &hit.chunk_id, ExpandMode::Section, 120).expect("expand");
        assert!(expansion.excerpt.contains("Alpha"));
        assert_eq!(expansion.source_anchor.as_ref().and_then(|anchor| anchor.section.as_deref()), Some("Alpha"));
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
        assert!(neighbors.iter().any(|neighbor| matches!(neighbor.node, NodeRef::Document(_))));
        assert!(neighbors.iter().any(|neighbor| neighbor.link_type == Some(LinkType::SameDocument) || neighbor.link_type == Some(LinkType::SemanticNeighbor)));
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
        assert!(opened.passages.iter().all(|passage| passage.role == crate::surf::SurfPassageRole::RepresentativeChunk));
        assert!(opened.passages.windows(2).all(|window| window[0].score >= window[1].score));
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
        let opened = surf_open(&root, &NodeRef::Document(document.id.clone())).expect("open document");
        let expected = memory
            .chunks
            .iter()
            .filter(|chunk| chunk.document_id == document.id)
            .take(5)
            .map(|chunk| NodeRef::Chunk(chunk.id.clone()))
            .collect::<Vec<_>>();
        let actual = opened.passages.iter().map(|passage| passage.node.clone()).collect::<Vec<_>>();
        assert_eq!(actual, expected);
        assert!(opened.passages.iter().all(|passage| passage.role == crate::surf::SurfPassageRole::DocumentChunk));
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
        assert!(labels.iter().all(|label| *label != "and" && *label != "the"));
    }

    fn simple_pdf(text: &str) -> Vec<u8> {
        let escaped = text.replace('\\', "\\\\").replace('(', "\\(").replace(')', "\\)");
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
        out.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).as_bytes());
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
