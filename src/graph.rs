use crate::index::{cosine_similarity, tokenize};
use crate::types::*;
use std::collections::{BTreeMap, HashMap, HashSet};

const DOC_SEMANTIC_THRESHOLD: f32 = 0.58;
const MAX_DOC_SEMANTIC_LINKS_PER_DOCUMENT: usize = 3;
const CHUNK_SEMANTIC_THRESHOLD: f32 = 0.72;
const MAX_CHUNK_SEMANTIC_LINKS_PER_CHUNK: usize = 3;

#[derive(Debug, Clone, Default)]
pub struct GraphBuilder;

impl GraphBuilder {
    pub fn build(&self, memory: &mut PersistedMemory) {
        let chunk_by_id = memory
            .chunks
            .iter()
            .map(|chunk| (chunk.id.clone(), chunk.clone()))
            .collect::<HashMap<_, _>>();
        let doc_by_id = memory
            .documents
            .iter()
            .map(|document| (document.id.clone(), document.clone()))
            .collect::<HashMap<_, _>>();
        let mut links = Vec::new();
        let mut region_neighbors = HashMap::<String, HashSet<String>>::new();
        for region in &memory.regions {
            for chunk_id in &region.chunk_ids {
                links.push(Link {
                    id: format!("link:{}:{}", region.id, chunk_id),
                    source: NodeRef::Region(region.id.clone()),
                    target: NodeRef::Chunk(chunk_id.clone()),
                    link_type: LinkType::RegionMembership,
                    score: 1.0,
                    label: format!("member of {}", region.label),
                });
            }
        }
        for document in &memory.documents {
            let doc_chunks = memory
                .chunks
                .iter()
                .filter(|chunk| chunk.document_id == document.id)
                .collect::<Vec<_>>();
            for window in doc_chunks.windows(2) {
                let left = window[0];
                let right = window[1];
                links.push(Link {
                    id: format!("link:{}:{}", left.id, right.id),
                    source: NodeRef::Chunk(left.id.clone()),
                    target: NodeRef::Chunk(right.id.clone()),
                    link_type: LinkType::SameDocument,
                    score: 0.9,
                    label: "same document".into(),
                });
            }
            for cited in parse_citations(&document.text) {
                if doc_by_id.contains_key(&cited) {
                    links.push(Link {
                        id: format!("citation:{}:{}", document.id, cited),
                        source: NodeRef::Document(document.id.clone()),
                        target: NodeRef::Document(cited.clone()),
                        link_type: LinkType::CitationReference,
                        score: 0.8,
                        label: "citation".into(),
                    });
                }
            }
            for linked in parse_markdown_links(&document.text) {
                if let Some(target) = resolve_document_reference(&linked, &doc_by_id) {
                    links.push(Link {
                        id: format!("explicit-link:{}:{}", document.id, target),
                        source: NodeRef::Document(document.id.clone()),
                        target: NodeRef::Document(target),
                        link_type: LinkType::CitationReference,
                        score: 0.82,
                        label: "explicit reference".into(),
                    });
                }
            }
        }
        for chunk in &memory.chunks {
            let mut candidates = memory
                .chunks
                .iter()
                .filter(|other| other.id != chunk.id)
                .map(|other| (other.id.clone(), cosine_similarity(&chunk.embedding, &other.embedding)))
                .filter(|(_, score)| *score >= CHUNK_SEMANTIC_THRESHOLD)
                .collect::<Vec<_>>();
            candidates.sort_by(|left, right| {
                right
                    .1
                    .partial_cmp(&left.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| left.0.cmp(&right.0))
            });
            for (other_id, score) in candidates.into_iter().take(MAX_CHUNK_SEMANTIC_LINKS_PER_CHUNK) {
                let (left, right) = if chunk.id < other_id {
                    (chunk.id.clone(), other_id)
                } else {
                    (other_id, chunk.id.clone())
                };
                links.push(Link {
                    id: format!("chunk-semantic:{left}:{right}"),
                    source: NodeRef::Chunk(left),
                    target: NodeRef::Chunk(right),
                    link_type: LinkType::SemanticNeighbor,
                    score,
                    label: "semantic neighbor".into(),
                });
            }
        }
        let doc_vectors = document_centroids(memory);
        let doc_regions = document_regions(memory);
        for document in &memory.documents {
            let Some(left_vector) = doc_vectors.get(&document.id) else {
                continue;
            };
            let mut candidates = Vec::new();
            for other in &memory.documents {
                if other.id == document.id {
                    continue;
                }
                let Some(right_vector) = doc_vectors.get(&other.id) else {
                    continue;
                };
                let semantic_score = cosine_similarity(left_vector, right_vector);
                if semantic_score >= DOC_SEMANTIC_THRESHOLD {
                    candidates.push((other.id.clone(), LinkType::SemanticNeighbor, semantic_score, "semantic neighbor".to_string()));
                    continue;
                }
                if document_entity_overlap(document, other) {
                    candidates.push((other.id.clone(), LinkType::EntityOverlap, 0.45, "entity overlap".to_string()));
                }
            }
            candidates.sort_by(|left, right| {
                right
                    .2
                    .partial_cmp(&left.2)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| left.0.cmp(&right.0))
            });
            for (other_id, link_type, score, label) in candidates
                .into_iter()
                .take(MAX_DOC_SEMANTIC_LINKS_PER_DOCUMENT)
            {
                links.push(Link {
                    id: format!("doc-link:{}:{}", document.id, other_id),
                    source: NodeRef::Document(document.id.clone()),
                    target: NodeRef::Document(other_id.clone()),
                    link_type,
                    score,
                    label,
                });
                if let (Some(left_region), Some(right_region)) =
                    (doc_regions.get(&document.id), doc_regions.get(&other_id))
                {
                    if left_region != right_region {
                        region_neighbors
                            .entry(left_region.clone())
                            .or_default()
                            .insert(right_region.clone());
                        region_neighbors
                            .entry(right_region.clone())
                            .or_default()
                            .insert(left_region.clone());
                    }
                }
            }
        }
        let mut deduped = BTreeMap::new();
        for link in links {
            deduped.entry(link.id.clone()).or_insert(link);
        }
        memory.links = deduped.into_values().collect();

        for link in &memory.links {
            if let (NodeRef::Chunk(left), NodeRef::Chunk(right)) = (&link.source, &link.target) {
                let Some(left_chunk) = chunk_by_id.get(left) else {
                    continue;
                };
                let Some(right_chunk) = chunk_by_id.get(right) else {
                    continue;
                };
                if left_chunk.region_id != right_chunk.region_id {
                    region_neighbors
                        .entry(left_chunk.region_id.clone())
                        .or_default()
                        .insert(right_chunk.region_id.clone());
                    region_neighbors
                        .entry(right_chunk.region_id.clone())
                        .or_default()
                        .insert(left_chunk.region_id.clone());
                }
            }
        }
        for region in &mut memory.regions {
            let mut neighbors = region.neighbors.clone();
            if let Some(extra) = region_neighbors.get(&region.id) {
                neighbors.extend(extra.iter().cloned());
                neighbors.sort();
                neighbors.dedup();
            }
            region.neighbors = neighbors;
        }
    }
}

fn parse_citations(text: &str) -> Vec<String> {
    text.split_whitespace()
        .filter_map(|token| token.strip_prefix("cite:"))
        .map(|value| value.trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '-').to_string())
        .collect()
}

fn parse_markdown_links(text: &str) -> Vec<String> {
    let mut refs = Vec::new();
    let bytes = text.as_bytes();
    let mut index = 0usize;
    while index + 3 < bytes.len() {
        if bytes[index] == b'[' && bytes.get(index + 1) == Some(&b'[') {
            if let Some(end) = text[index + 2..].find("]]") {
                refs.push(text[index + 2..index + 2 + end].trim().to_string());
                index += end + 4;
                continue;
            }
        }
        if bytes[index] == b']' && bytes.get(index + 1) == Some(&b'(') {
            if let Some(end) = text[index + 2..].find(')') {
                refs.push(text[index + 2..index + 2 + end].trim().to_string());
                index += end + 3;
                continue;
            }
        }
        index += 1;
    }
    refs
}

fn resolve_document_reference(raw: &str, documents: &HashMap<String, Document>) -> Option<String> {
    let normalized = normalize_ref(raw);
    documents
        .iter()
        .find(|(id, document)| {
            normalize_ref(id) == normalized
                || normalize_ref(&document.title) == normalized
                || document
                    .metadata
                    .get("path")
                    .map(|path| normalize_ref(path).contains(&normalized))
                    .unwrap_or(false)
        })
        .map(|(id, _)| id.clone())
}

fn normalize_ref(raw: &str) -> String {
    raw.trim()
        .trim_matches(|ch: char| ch == '/' || ch == '"' || ch == '\'')
        .trim_end_matches(".md")
        .trim_end_matches(".markdown")
        .to_lowercase()
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || *ch == '-')
        .collect()
}

fn document_centroids(memory: &PersistedMemory) -> HashMap<String, Vec<f32>> {
    let mut sums = HashMap::<String, Vec<f32>>::new();
    let mut counts = HashMap::<String, f32>::new();
    for chunk in &memory.chunks {
        let vector = sums
            .entry(chunk.document_id.clone())
            .or_insert_with(|| vec![0.0; chunk.embedding.len()]);
        for (slot, value) in vector.iter_mut().zip(&chunk.embedding) {
            *slot += *value;
        }
        *counts.entry(chunk.document_id.clone()).or_default() += 1.0;
    }
    for (document_id, vector) in &mut sums {
        let count = counts.get(document_id).copied().unwrap_or(1.0);
        for slot in vector {
            *slot /= count;
        }
    }
    sums
}

fn document_regions(memory: &PersistedMemory) -> HashMap<String, String> {
    let mut regions = HashMap::new();
    for chunk in &memory.chunks {
        regions
            .entry(chunk.document_id.clone())
            .or_insert_with(|| chunk.region_id.clone());
    }
    regions
}

fn document_entity_overlap(left: &Document, right: &Document) -> bool {
    let left_tokens = tokenize(&left.text)
        .into_iter()
        .filter(|token| token.len() > 6)
        .collect::<HashSet<_>>();
    let right_tokens = tokenize(&right.text)
        .into_iter()
        .filter(|token| token.len() > 6)
        .collect::<HashSet<_>>();
    left_tokens.intersection(&right_tokens).next().is_some()
}
