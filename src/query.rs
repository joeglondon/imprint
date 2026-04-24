use crate::index::{cosine_similarity, tokenize, Embedder, RegionAnnIndex};
use crate::types::*;
use std::collections::{BTreeMap, HashMap, HashSet};

pub trait Router {
    fn route(&self, memory_map: &MemoryMap, request: &QueryRequest) -> RoutedQuery;
}

#[derive(Debug, Clone, Default)]
pub struct MemoryQueryEngine;

impl Router for MemoryQueryEngine {
    fn route(&self, memory_map: &MemoryMap, request: &QueryRequest) -> RoutedQuery {
        let tokens = tokenize(&request.text);
        let mut scored = memory_map
            .entries
            .iter()
            .map(|entry| {
                let mut score = 0usize;
                for token in &tokens {
                    if entry.label.contains(token) || entry.summary.contains(token) {
                        score += 2;
                    }
                    if entry.filters.values().any(|value| value.contains(token)) {
                        score += 1;
                    }
                }
                (entry.region_id.clone(), entry.label.clone(), score)
            })
            .collect::<Vec<_>>();
        scored.sort_by(|left, right| right.2.cmp(&left.2).then_with(|| left.0.cmp(&right.0)));
        let chosen = scored
            .into_iter()
            .take(request.max_regions.max(1))
            .map(|(id, label, score)| (id, format!("{label} matched score {score}")))
            .collect::<Vec<_>>();
        let region_ids = chosen.iter().map(|(id, _)| id.clone()).collect::<Vec<_>>();
        let rationale = chosen
            .iter()
            .map(|(_, why)| why.clone())
            .collect::<Vec<_>>()
            .join("; ");
        RoutedQuery {
            query: request.text.clone(),
            region_ids,
            filters: request.filters.clone(),
            rationale,
        }
    }
}

impl MemoryQueryEngine {
    pub fn execute<E: Embedder>(
        &self,
        embedder: &E,
        memory: &PersistedMemory,
        ann_index: &RegionAnnIndex,
        request: QueryRequest,
    ) -> anyhow::Result<QueryResult> {
        let routed = self.route(memory.memory_map.as_ref().expect("memory map missing"), &request);
        let query_embedding = embedder.embed(&request.text)?;
        let chunks_by_id = memory
            .chunks
            .iter()
            .map(|chunk| (chunk.id.clone(), chunk))
            .collect::<HashMap<_, _>>();
        let documents_by_id = memory
            .documents
            .iter()
            .map(|document| (document.id.clone(), document))
            .collect::<HashMap<_, _>>();

        let mut hits = Vec::new();
        for region_id in &routed.region_ids {
            let candidates = ann_index.search_region(region_id, &query_embedding, request.max_chunks.max(1));
            for (chunk_id, ann_score) in candidates {
                let Some(chunk) = chunks_by_id.get(&chunk_id) else {
                    continue;
                };
                if !matches_filters(&request.filters, &chunk.metadata) {
                    continue;
                }
                let Some(document) = documents_by_id.get(&chunk.document_id) else {
                    continue;
                };
                let lexical = lexical_overlap_score(&request.text, &chunk.text);
                let score = ann_score * 0.7 + lexical * 0.3;
                let excerpt = excerpt_window(&document.text, chunk.start, chunk.end, 60);
                hits.push(ChunkHit {
                    hit_id: format!("hit:{chunk_id}"),
                    chunk_id: chunk.id.clone(),
                    document_id: chunk.document_id.clone(),
                    region_id: chunk.region_id.clone(),
                    score,
                    excerpt,
                    start: chunk.start,
                    end: chunk.end,
                    source_anchor: chunk.source_anchor.clone(),
                });
            }
        }
        hits.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut seen_chunks = HashSet::new();
        hits.retain(|hit| seen_chunks.insert(hit.chunk_id.clone()));
        hits.truncate(request.max_chunks.max(1));
        Ok(QueryResult { routed, hits })
    }

    pub fn trace(&self, memory_map: &MemoryMap, query: &str, max_regions: usize) -> QueryTrace {
        let routed = self.route(
            memory_map,
            &QueryRequest {
                text: query.into(),
                filters: BTreeMap::new(),
                max_regions,
                max_chunks: 5,
            },
        );
        QueryTrace {
            query: query.into(),
            chosen_regions: routed.region_ids,
            reasons: vec![routed.rationale],
        }
    }
}

fn matches_filters(filters: &BTreeMap<String, String>, metadata: &BTreeMap<String, String>) -> bool {
    filters
        .iter()
        .all(|(key, value)| metadata.get(key).map(|candidate| candidate == value).unwrap_or(false))
}

fn lexical_overlap_score(left: &str, right: &str) -> f32 {
    let left = tokenize(left);
    let right = tokenize(right);
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let overlap = left.iter().filter(|token| right.contains(token)).count() as f32;
    overlap / left.len().max(1) as f32
}

fn excerpt_window(text: &str, start: usize, end: usize, pad: usize) -> String {
    let chars = text.chars().collect::<Vec<_>>();
    let left = start.saturating_sub(pad);
    let right = (end + pad).min(chars.len());
    chars[left..right].iter().collect()
}

#[allow(dead_code)]
fn _region_score(query_embedding: &[f32], region: &Region) -> f32 {
    cosine_similarity(query_embedding, &region.centroid)
}
