use crate::index::{cosine_similarity, tokenize};
use crate::types::*;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

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
            let mut doc_chunks = doc_chunks;
            doc_chunks.sort_by(|left, right| {
                left.ordinal
                    .cmp(&right.ordinal)
                    .then_with(|| left.start.cmp(&right.start))
                    .then_with(|| left.id.cmp(&right.id))
            });
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
                .map(|other| {
                    (
                        other.id.clone(),
                        cosine_similarity(&chunk.embedding, &other.embedding),
                    )
                })
                .filter(|(_, score)| *score >= CHUNK_SEMANTIC_THRESHOLD)
                .collect::<Vec<_>>();
            candidates.sort_by(|left, right| {
                right
                    .1
                    .partial_cmp(&left.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| left.0.cmp(&right.0))
            });
            for (other_id, score) in candidates
                .into_iter()
                .take(MAX_CHUNK_SEMANTIC_LINKS_PER_CHUNK)
            {
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
                    label: semantic_neighbor_label(score),
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
                    candidates.push((
                        other.id.clone(),
                        LinkType::SemanticNeighbor,
                        semantic_score,
                        "semantic neighbor".to_string(),
                    ));
                    continue;
                }
                let shared_entities = shared_document_entities(document, other);
                if !shared_entities.is_empty() {
                    let score = entity_overlap_score(shared_entities.len());
                    candidates.push((
                        other.id.clone(),
                        LinkType::EntityOverlap,
                        score,
                        format!(
                            "entity overlap: {}",
                            shared_entities
                                .iter()
                                .take(3)
                                .cloned()
                                .collect::<Vec<_>>()
                                .join(", ")
                        ),
                    ));
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
    let mut refs = BTreeSet::new();
    for token in text.split_whitespace() {
        if let Some(value) = token.strip_prefix("cite:") {
            add_citation_ref(value, &mut refs);
        }
        if let Some(value) = token.strip_prefix('@') {
            add_citation_ref(value, &mut refs);
        }
    }

    for value in parse_wrapped_refs(text, "cite(", ")") {
        add_citation_ref(&value, &mut refs);
    }
    for value in parse_bracketed_citation_refs(text) {
        add_citation_ref(&value, &mut refs);
    }

    refs.into_iter().collect()
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

fn parse_wrapped_refs(text: &str, open: &str, close: &str) -> Vec<String> {
    let mut refs = Vec::new();
    let mut cursor = 0usize;
    while let Some(start) = text[cursor..].find(open) {
        let value_start = cursor + start + open.len();
        let Some(end) = text[value_start..].find(close) else {
            break;
        };
        refs.push(text[value_start..value_start + end].trim().to_string());
        cursor = value_start + end + close.len();
    }
    refs
}

fn parse_bracketed_citation_refs(text: &str) -> Vec<String> {
    let mut refs = Vec::new();
    let mut cursor = 0usize;
    while let Some(start) = text[cursor..].find("[@") {
        let value_start = cursor + start + 2;
        let Some(end) = text[value_start..].find(']') else {
            break;
        };
        let group = &text[value_start..value_start + end];
        for part in group.split([';', ',']) {
            let trimmed = part.trim().trim_start_matches('@');
            if !trimmed.is_empty() {
                refs.push(trimmed.to_string());
            }
        }
        cursor = value_start + end + 1;
    }
    refs
}

fn add_citation_ref(raw: &str, refs: &mut BTreeSet<String>) {
    let value = raw
        .trim()
        .trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '-' && ch != '_' && ch != '/')
        .trim_end_matches(".md")
        .trim_end_matches(".markdown");
    if !value.is_empty() {
        refs.insert(value.to_string());
    }
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

fn semantic_neighbor_label(score: f32) -> String {
    let confidence = if score >= 0.88 {
        "high"
    } else if score >= 0.78 {
        "medium"
    } else {
        "low"
    };
    format!("semantic neighbor ({confidence} confidence)")
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

fn shared_document_entities(left: &Document, right: &Document) -> Vec<String> {
    let left_entities = document_entities(left);
    let right_entities = document_entities(right);
    left_entities
        .intersection(&right_entities)
        .take(8)
        .cloned()
        .collect()
}

fn document_entities(document: &Document) -> BTreeSet<String> {
    let mut entities = BTreeSet::new();
    for key in ["entities", "entity", "people", "person", "tags", "keywords"] {
        if let Some(raw) = document.metadata.get(key) {
            for part in raw.split([',', ';', '|']) {
                add_entity(part, &mut entities);
            }
        }
    }
    for token in tokenize(&document.title) {
        add_normalized_entity(&token, &mut entities);
    }
    for token in document
        .text
        .split(|ch: char| !ch.is_alphanumeric() && ch != '-' && ch != '_')
    {
        if looks_like_named_entity(token) {
            add_entity(token, &mut entities);
        }
    }
    entities
}

fn add_entity(raw: &str, entities: &mut BTreeSet<String>) {
    let normalized = normalize_entity(raw);
    add_normalized_entity(&normalized, entities);
}

fn add_normalized_entity(normalized: &str, entities: &mut BTreeSet<String>) {
    if normalized.len() >= 4 && !ENTITY_STOPWORDS.contains(&normalized) {
        entities.insert(normalized.to_string());
    }
}

fn normalize_entity(raw: &str) -> String {
    raw.trim()
        .trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '-' && ch != '_')
        .to_lowercase()
}

fn looks_like_named_entity(token: &str) -> bool {
    let trimmed = token.trim();
    if trimmed.len() < 4 || ENTITY_STOPWORDS.contains(&trimmed.to_lowercase().as_str()) {
        return false;
    }
    let mut chars = trimmed.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first.is_ascii_uppercase() && chars.any(|ch| ch.is_ascii_lowercase()))
        || (trimmed.len() >= 2 && trimmed.chars().all(|ch| ch.is_ascii_uppercase()))
}

fn entity_overlap_score(shared_count: usize) -> f32 {
    (0.48 + (shared_count.min(5) as f32 * 0.04)).min(0.68)
}

const ENTITY_STOPWORDS: &[&str] = &[
    "about", "after", "also", "before", "between", "document", "from", "into", "memory", "source",
    "that", "their", "there", "these", "this", "with",
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn document(id: &str, title: &str, text: &str) -> Document {
        Document {
            id: id.into(),
            title: title.into(),
            text: text.into(),
            metadata: BTreeMap::new(),
            source_anchor: None,
            content_hash: None,
            parser_version: None,
        }
    }

    fn chunk(id: &str, document_id: &str, ordinal: usize, start: usize) -> Chunk {
        Chunk {
            id: id.into(),
            document_id: document_id.into(),
            region_id: "region-a".into(),
            ordinal,
            start,
            end: start + 10,
            text: format!("chunk {id}"),
            metadata: BTreeMap::new(),
            source_anchor: None,
            embedding: vec![1.0, 0.0, 0.0],
            embedding_text_hash: None,
            embedding_provider: None,
            embedding_model: None,
            embedding_endpoint: None,
            embedding_dimension: Some(3),
            chunking_version: None,
        }
    }

    fn region() -> Region {
        Region {
            id: "region-a".into(),
            label: "Region A".into(),
            summary: "Test region".into(),
            filters: BTreeMap::new(),
            chunk_ids: Vec::new(),
            centroid: vec![1.0, 0.0, 0.0],
            neighbors: Vec::new(),
        }
    }

    #[test]
    fn same_document_links_follow_chunk_ordinal_order() {
        let mut memory = PersistedMemory {
            documents: vec![document("doc-a", "Doc A", "body")],
            chunks: vec![
                chunk("chunk-2", "doc-a", 2, 20),
                chunk("chunk-0", "doc-a", 0, 0),
                chunk("chunk-1", "doc-a", 1, 10),
            ],
            regions: vec![region()],
            links: Vec::new(),
            memory_map: None,
        };

        GraphBuilder.build(&mut memory);

        let same_doc_links = memory
            .links
            .iter()
            .filter(|link| link.link_type == LinkType::SameDocument)
            .map(|link| (&link.source, &link.target))
            .collect::<Vec<_>>();
        assert!(same_doc_links.contains(&(
            &NodeRef::Chunk("chunk-0".into()),
            &NodeRef::Chunk("chunk-1".into())
        )));
        assert!(same_doc_links.contains(&(
            &NodeRef::Chunk("chunk-1".into()),
            &NodeRef::Chunk("chunk-2".into())
        )));
    }

    #[test]
    fn citation_parser_handles_common_inline_forms() {
        let refs = parse_citations(
            "See cite:alpha-doc, cite(beta_doc) [@gamma-doc; @delta/doc] and @epsilon.",
        );

        assert_eq!(
            refs,
            vec![
                "alpha-doc".to_string(),
                "beta_doc".to_string(),
                "delta/doc".to_string(),
                "epsilon".to_string(),
                "gamma-doc".to_string(),
            ]
        );
    }

    #[test]
    fn entity_overlap_requires_named_or_metadata_entities() {
        let mut left = document(
            "left",
            "Imprint Cortex",
            "RecursiveMAS routes source recall through CortexIndex.",
        );
        left.metadata
            .insert("entities".into(), "RecursiveMAS, CortexIndex".into());
        let right = document(
            "right",
            "Research Notes",
            "The RecursiveMAS critic reviews CortexIndex route sketches.",
        );

        let shared = shared_document_entities(&left, &right);

        assert!(shared.contains(&"recursivemas".to_string()));
        assert!(shared.contains(&"cortexindex".to_string()));
        assert!(entity_overlap_score(shared.len()) > 0.48);
    }

    #[test]
    fn semantic_neighbor_label_exposes_confidence_band() {
        assert_eq!(
            semantic_neighbor_label(0.9),
            "semantic neighbor (high confidence)"
        );
        assert_eq!(
            semantic_neighbor_label(0.8),
            "semantic neighbor (medium confidence)"
        );
        assert_eq!(
            semantic_neighbor_label(0.72),
            "semantic neighbor (low confidence)"
        );
    }
}
