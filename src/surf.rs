use crate::index::cosine_similarity;
use crate::navigation::{MemoryNavigator, Navigator};
use crate::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExpandMode {
    Window,
    Page,
    Section,
    Document,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SurfOpenResult {
    pub node: NodeRef,
    pub label: String,
    pub excerpt: String,
    #[serde(default)]
    pub source_anchor: Option<SourceAnchor>,
    #[serde(default)]
    pub passages: Vec<SurfPassage>,
    pub links: Vec<Link>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SurfPassageRole {
    SelectedChunk,
    RepresentativeChunk,
    DocumentChunk,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SurfPassage {
    pub node: NodeRef,
    pub label: String,
    pub excerpt: String,
    pub start: usize,
    pub end: usize,
    pub score: f32,
    #[serde(default)]
    pub source_anchor: Option<SourceAnchor>,
    pub role: SurfPassageRole,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SurfNeighbor {
    pub node: NodeRef,
    pub score: f32,
    pub label: String,
    pub excerpt: String,
    #[serde(default)]
    pub source_anchor: Option<SourceAnchor>,
    #[serde(default)]
    pub link_type: Option<LinkType>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SurfExpansion {
    pub chunk_id: ChunkId,
    pub mode: ExpandMode,
    pub excerpt: String,
    pub start: usize,
    pub end: usize,
    #[serde(default)]
    pub source_anchor: Option<SourceAnchor>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SurfAction {
    Open(NodeRef),
    FollowLink(String),
    Backtrack,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SurfStepResult {
    pub session: SessionState,
    #[serde(default)]
    pub current: Option<SurfOpenResult>,
}

pub fn open(memory: &PersistedMemory, node: &NodeRef) -> Option<SurfOpenResult> {
    let links = MemoryNavigator.list_links(memory, node);
    match node {
        NodeRef::Document(id) => {
            let document = memory
                .documents
                .iter()
                .find(|document| &document.id == id)?;
            let passages = document_passages(memory, id);
            Some(SurfOpenResult {
                node: node.clone(),
                label: document.title.clone(),
                excerpt: excerpt_window(
                    &document.text,
                    0,
                    document.text.chars().count().min(520),
                    0,
                ),
                source_anchor: document.source_anchor.clone(),
                passages,
                links,
            })
        }
        NodeRef::Chunk(id) => {
            let chunk = memory.chunks.iter().find(|chunk| &chunk.id == id)?;
            let excerpt = memory
                .documents
                .iter()
                .find(|document| document.id == chunk.document_id)
                .map(|document| excerpt_window(&document.text, chunk.start, chunk.end, 220))
                .unwrap_or_else(|| chunk.text.clone());
            Some(SurfOpenResult {
                node: node.clone(),
                label: chunk
                    .metadata
                    .get("document_title")
                    .cloned()
                    .unwrap_or_else(|| chunk.id.clone()),
                excerpt,
                source_anchor: chunk.source_anchor.clone(),
                passages: vec![chunk_passage(chunk, 1.0, SurfPassageRole::SelectedChunk)],
                links,
            })
        }
        NodeRef::Region(id) => {
            let region = memory.regions.iter().find(|region| &region.id == id)?;
            let passages = representative_region_passages(memory, region, 8);
            Some(SurfOpenResult {
                node: node.clone(),
                label: region.label.clone(),
                excerpt: region.summary.clone(),
                source_anchor: None,
                passages,
                links,
            })
        }
    }
}

pub fn neighbors(
    memory: &PersistedMemory,
    node: &NodeRef,
    max_results: usize,
) -> Vec<SurfNeighbor> {
    let mut out = Vec::new();
    let mut seen = HashSet::<NodeRef>::new();
    for link in MemoryNavigator.list_links(memory, node) {
        let target = if &link.source == node {
            link.target.clone()
        } else {
            link.source.clone()
        };
        if seen.insert(target.clone()) {
            if let Some(opened) = open(memory, &target) {
                out.push(SurfNeighbor {
                    node: target,
                    score: link.score,
                    label: opened.label,
                    excerpt: opened.excerpt,
                    source_anchor: opened.source_anchor,
                    link_type: Some(link.link_type),
                });
            }
        }
    }
    if let NodeRef::Chunk(chunk_id) = node {
        if let Some(chunk) = memory.chunks.iter().find(|chunk| &chunk.id == chunk_id) {
            let mut semantic = memory
                .chunks
                .iter()
                .filter(|candidate| candidate.id != chunk.id)
                .map(|candidate| {
                    (
                        candidate,
                        cosine_similarity(&chunk.embedding, &candidate.embedding),
                    )
                })
                .collect::<Vec<_>>();
            semantic.sort_by(|left, right| {
                right
                    .1
                    .partial_cmp(&left.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| left.0.id.cmp(&right.0.id))
            });
            for (candidate, score) in semantic.into_iter().take(max_results.max(1)) {
                let node_ref = NodeRef::Chunk(candidate.id.clone());
                if seen.insert(node_ref.clone()) {
                    out.push(SurfNeighbor {
                        node: node_ref,
                        score,
                        label: candidate
                            .metadata
                            .get("document_title")
                            .cloned()
                            .unwrap_or_else(|| candidate.id.clone()),
                        excerpt: candidate.text.clone(),
                        source_anchor: candidate.source_anchor.clone(),
                        link_type: Some(LinkType::SemanticNeighbor),
                    });
                }
            }
        }
    }
    out.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.label.cmp(&right.label))
    });
    out.truncate(max_results.max(1));
    out
}

fn document_passages(memory: &PersistedMemory, document_id: &str) -> Vec<SurfPassage> {
    let mut chunks = memory
        .chunks
        .iter()
        .filter(|chunk| chunk.document_id == document_id)
        .collect::<Vec<_>>();
    chunks.sort_by(|left, right| {
        left.ordinal
            .cmp(&right.ordinal)
            .then_with(|| left.id.cmp(&right.id))
    });
    chunks
        .into_iter()
        .take(5)
        .map(|chunk| chunk_passage(chunk, 1.0, SurfPassageRole::DocumentChunk))
        .collect()
}

fn representative_region_passages(
    memory: &PersistedMemory,
    region: &Region,
    max_results: usize,
) -> Vec<SurfPassage> {
    let mut chunks = memory
        .chunks
        .iter()
        .filter(|chunk| chunk.region_id == region.id)
        .map(|chunk| (chunk, cosine_similarity(&region.centroid, &chunk.embedding)))
        .collect::<Vec<_>>();
    chunks.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.id.cmp(&right.0.id))
    });
    chunks
        .into_iter()
        .take(max_results.max(1))
        .map(|(chunk, score)| chunk_passage(chunk, score, SurfPassageRole::RepresentativeChunk))
        .collect()
}

fn chunk_passage(chunk: &Chunk, score: f32, role: SurfPassageRole) -> SurfPassage {
    SurfPassage {
        node: NodeRef::Chunk(chunk.id.clone()),
        label: chunk
            .metadata
            .get("document_title")
            .cloned()
            .unwrap_or_else(|| chunk.document_id.clone()),
        excerpt: chunk.text.clone(),
        start: chunk.start,
        end: chunk.end,
        score,
        source_anchor: chunk.source_anchor.clone(),
        role,
    }
}

pub fn expand_chunk(
    memory: &PersistedMemory,
    chunk_id: &str,
    mode: ExpandMode,
    window: usize,
) -> Option<SurfExpansion> {
    let chunk = memory.chunks.iter().find(|chunk| chunk.id == chunk_id)?;
    let document = memory
        .documents
        .iter()
        .find(|document| document.id == chunk.document_id)?;
    let (start, end) = match mode {
        ExpandMode::Window => (
            chunk.start.saturating_sub(window),
            (chunk.end + window).min(document.text.chars().count()),
        ),
        ExpandMode::Page => {
            span_for_metadata(memory, chunk, "page").unwrap_or((chunk.start, chunk.end))
        }
        ExpandMode::Section => {
            span_for_metadata(memory, chunk, "section").unwrap_or((chunk.start, chunk.end))
        }
        ExpandMode::Document => (0, document.text.chars().count()),
    };
    Some(SurfExpansion {
        chunk_id: chunk.id.clone(),
        mode,
        excerpt: excerpt_window(&document.text, start, end, 0),
        start,
        end,
        source_anchor: chunk.source_anchor.as_ref().map(|anchor| SourceAnchor {
            id: format!("{}:expanded", anchor.id),
            document_id: anchor.document_id.clone(),
            chunk_id: anchor.chunk_id.clone(),
            path: anchor.path.clone(),
            content_hash: anchor.content_hash.clone(),
            start,
            end,
            page: anchor.page,
            section: anchor.section.clone(),
            parser_version: anchor.parser_version,
        }),
    })
}

pub fn jump_to_anchor(
    memory: &PersistedMemory,
    anchor_id: &str,
    window: usize,
) -> Option<SurfExpansion> {
    if let Some(chunk) = memory.chunks.iter().find(|chunk| {
        chunk
            .source_anchor
            .as_ref()
            .map(|anchor| anchor.id.as_str())
            == Some(anchor_id)
    }) {
        return expand_chunk(memory, &chunk.id, ExpandMode::Window, window);
    }
    let document = memory.documents.iter().find(|document| {
        document
            .source_anchor
            .as_ref()
            .map(|anchor| anchor.id.as_str())
            == Some(anchor_id)
    })?;
    let first_chunk = memory
        .chunks
        .iter()
        .find(|chunk| chunk.document_id == document.id)?;
    expand_chunk(memory, &first_chunk.id, ExpandMode::Document, window)
}

pub fn session_step(
    memory: &PersistedMemory,
    mut session: SessionState,
    action: SurfAction,
) -> Option<SurfStepResult> {
    match action {
        SurfAction::Open(node) => MemoryNavigator.open(&mut session, node),
        SurfAction::FollowLink(link_id) => {
            MemoryNavigator.step(memory, &mut session, &link_id)?;
        }
        SurfAction::Backtrack => {
            MemoryNavigator.backtrack(&mut session)?;
        }
    }
    let current = session.current.as_ref().and_then(|node| open(memory, node));
    Some(SurfStepResult { session, current })
}

fn span_for_metadata(memory: &PersistedMemory, chunk: &Chunk, key: &str) -> Option<(usize, usize)> {
    let value = chunk.metadata.get(key)?;
    let mut starts = Vec::new();
    let mut ends = Vec::new();
    for candidate in memory
        .chunks
        .iter()
        .filter(|candidate| candidate.document_id == chunk.document_id)
        .filter(|candidate| candidate.metadata.get(key) == Some(value))
    {
        starts.push(candidate.start);
        ends.push(candidate.end);
    }
    Some((*starts.iter().min()?, *ends.iter().max()?))
}

fn excerpt_window(text: &str, start: usize, end: usize, pad: usize) -> String {
    let chars = text.chars().collect::<Vec<_>>();
    let left = start.saturating_sub(pad);
    let right = (end + pad).min(chars.len());
    chars[left..right].iter().collect()
}
