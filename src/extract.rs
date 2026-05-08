use crate::index::{cosine_similarity, tokenize, Embedder};
use crate::types::*;
use std::collections::{BTreeMap, HashMap};

pub trait Extractor {
    fn grep_document(
        &self,
        memory: &PersistedMemory,
        document_id: &str,
        needle: &str,
        window: usize,
    ) -> Vec<ExtractHit>;
    fn grep_region(
        &self,
        memory: &PersistedMemory,
        region_id: &str,
        needle: &str,
        window: usize,
    ) -> Vec<ExtractHit>;
    fn semantic_document<E: Embedder>(
        &self,
        embedder: &E,
        memory: &PersistedMemory,
        document_id: &str,
        query: &str,
        window: usize,
    ) -> anyhow::Result<Vec<ExtractHit>>;
    fn excerpt_for_hit(
        &self,
        memory: &PersistedMemory,
        hit_id: &str,
        window: usize,
    ) -> Option<String>;
}

#[derive(Debug, Clone, Default)]
pub struct MemoryExtractor;

impl Extractor for MemoryExtractor {
    fn grep_document(
        &self,
        memory: &PersistedMemory,
        document_id: &str,
        needle: &str,
        window: usize,
    ) -> Vec<ExtractHit> {
        let Some(document) = memory.documents.iter().find(|doc| doc.id == document_id) else {
            return Vec::new();
        };
        grep_text(&document.text, needle, window)
            .into_iter()
            .enumerate()
            .map(|(index, (start, end, excerpt))| ExtractHit {
                hit_id: format!("doc-hit:{document_id}|{needle}|{index}"),
                node: NodeRef::Document(document_id.to_string()),
                score: 1.0,
                excerpt,
                start,
                end,
                metadata: document.metadata.clone(),
                source_anchor: document.source_anchor.as_ref().map(|anchor| SourceAnchor {
                    id: format!("{}:hit:{index}", anchor.id),
                    document_id: anchor.document_id.clone(),
                    chunk_id: None,
                    source_artifact_id: anchor.source_artifact_id.clone(),
                    path: anchor.path.clone(),
                    content_hash: anchor.content_hash.clone(),
                    start,
                    end,
                    byte_start: Some(byte_offset_for_char(&document.text, start)),
                    byte_end: Some(byte_offset_for_char(&document.text, end)),
                    char_start: Some(start),
                    char_end: Some(end),
                    page: anchor.page,
                    section: anchor.section.clone(),
                    section_hierarchy: anchor.section_hierarchy.clone(),
                    paragraph_index: anchor.paragraph_index,
                    parser_version: anchor.parser_version,
                }),
            })
            .collect()
    }

    fn grep_region(
        &self,
        memory: &PersistedMemory,
        region_id: &str,
        needle: &str,
        window: usize,
    ) -> Vec<ExtractHit> {
        let chunks = memory
            .chunks
            .iter()
            .filter(|chunk| chunk.region_id == region_id)
            .collect::<Vec<_>>();
        let mut hits = Vec::new();
        for chunk in chunks {
            let found = grep_text(&chunk.text, needle, window);
            for (index, (start, end, excerpt)) in found.into_iter().enumerate() {
                hits.push(ExtractHit {
                    hit_id: format!("region-hit:{region_id}|{}|{needle}|{index}", chunk.id),
                    node: NodeRef::Chunk(chunk.id.clone()),
                    score: 1.0,
                    excerpt,
                    start: chunk.start + start,
                    end: chunk.start + end,
                    metadata: chunk.metadata.clone(),
                    source_anchor: chunk.source_anchor.clone(),
                });
            }
        }
        hits
    }

    fn semantic_document<E: Embedder>(
        &self,
        embedder: &E,
        memory: &PersistedMemory,
        document_id: &str,
        query: &str,
        window: usize,
    ) -> anyhow::Result<Vec<ExtractHit>> {
        let chunks = memory
            .chunks
            .iter()
            .filter(|chunk| chunk.document_id == document_id)
            .collect::<Vec<_>>();
        let query_embedding = embedder.embed(query)?;
        let mut hits = chunks
            .into_iter()
            .enumerate()
            .map(|(index, chunk)| {
                let score = cosine_similarity(&query_embedding, &chunk.embedding);
                ExtractHit {
                    hit_id: format!("semantic-hit:{document_id}|{}|{index}", chunk.id),
                    node: NodeRef::Chunk(chunk.id.clone()),
                    score,
                    excerpt: excerpt(chunk.text.as_str(), 0, chunk.text.len(), window),
                    start: chunk.start,
                    end: chunk.end,
                    metadata: chunk.metadata.clone(),
                    source_anchor: chunk.source_anchor.clone(),
                }
            })
            .collect::<Vec<_>>();
        hits.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(5);
        Ok(hits)
    }

    fn excerpt_for_hit(
        &self,
        memory: &PersistedMemory,
        hit_id: &str,
        window: usize,
    ) -> Option<String> {
        if let Some(stripped) = hit_id.strip_prefix("doc-hit:") {
            let (document_id, remainder) = stripped.split_once('|')?;
            let (needle, index) = remainder.rsplit_once('|')?;
            let index = index.parse::<usize>().ok()?;
            let document = memory
                .documents
                .iter()
                .find(|document| document.id == document_id)?;
            let hits = grep_text(&document.text, needle, window);
            return hits.get(index).map(|(_, _, excerpt)| excerpt.clone());
        }
        let chunk_lookup = memory
            .chunks
            .iter()
            .map(|chunk| (chunk.id.clone(), chunk))
            .collect::<HashMap<_, _>>();
        if let Some(stripped) = hit_id.strip_prefix("region-hit:") {
            let (_, remainder) = stripped.split_once('|')?;
            let (chunk_id, _) = remainder.rsplit_once('|')?;
            let (chunk_id, _) = chunk_id.rsplit_once('|').unwrap_or((chunk_id, ""));
            if let Some(chunk) = chunk_lookup.get(chunk_id) {
                return Some(excerpt(chunk.text.as_str(), 0, chunk.text.len(), window));
            }
        }
        if let Some(stripped) = hit_id.strip_prefix("semantic-hit:") {
            let (_, remainder) = stripped.split_once('|')?;
            let (chunk_id, _) = remainder.rsplit_once('|')?;
            if let Some(chunk) = chunk_lookup.get(chunk_id) {
                return Some(excerpt(chunk.text.as_str(), 0, chunk.text.len(), window));
            }
        }
        None
    }
}

fn grep_text(text: &str, needle: &str, window: usize) -> Vec<(usize, usize, String)> {
    if needle.is_empty() {
        return vec![(0, text.len(), excerpt(text, 0, text.len(), window))];
    }
    let lower = text.to_lowercase();
    let needle_lower = needle.to_lowercase();
    let mut start_at = 0usize;
    let mut hits = Vec::new();
    while let Some(found) = lower[start_at..].find(&needle_lower) {
        let start = start_at + found;
        let end = start + needle_lower.len();
        hits.push((start, end, excerpt(text, start, end, window)));
        start_at = end;
    }
    hits
}

fn excerpt(text: &str, start: usize, end: usize, window: usize) -> String {
    let chars = text.chars().collect::<Vec<_>>();
    let left = start.saturating_sub(window);
    let right = (end + window).min(chars.len());
    chars[left..right].iter().collect()
}

fn byte_offset_for_char(text: &str, offset: usize) -> usize {
    if offset == 0 {
        return 0;
    }
    text.char_indices()
        .nth(offset)
        .map(|(byte, _)| byte)
        .unwrap_or(text.len())
}

#[allow(dead_code)]
fn _metadata_filter(
    metadata: &BTreeMap<String, String>,
    filters: &BTreeMap<String, String>,
) -> bool {
    filters.iter().all(|(key, value)| {
        metadata
            .get(key)
            .map(|candidate| candidate == value)
            .unwrap_or(false)
    })
}

#[allow(dead_code)]
fn _token_overlap(query: &str, text: &str) -> usize {
    let query_tokens = tokenize(query);
    let text_tokens = tokenize(text);
    query_tokens
        .iter()
        .filter(|token| text_tokens.contains(token))
        .count()
}
