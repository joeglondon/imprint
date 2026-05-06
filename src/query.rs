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
                let label = entry.label.to_lowercase();
                let summary = entry.summary.to_lowercase();
                let filter_values = entry
                    .filters
                    .values()
                    .map(|value| value.to_lowercase())
                    .collect::<Vec<_>>();
                let mut score = 0.0f32;
                let mut matched_terms = Vec::new();
                for token in &tokens {
                    let mut matched = false;
                    if label.contains(token) || summary.contains(token) {
                        score += 2.0;
                        matched = true;
                    }
                    if filter_values.iter().any(|value| value.contains(token)) {
                        score += 1.0;
                        matched = true;
                    }
                    if matched {
                        matched_terms.push(token.clone());
                    }
                }
                RouteCandidate {
                    region_id: entry.region_id.clone(),
                    label: entry.label.clone(),
                    score,
                    matched_terms,
                    reason: format!("{} matched route score {score:.1}", entry.label),
                }
            })
            .collect::<Vec<_>>();
        scored.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.region_id.cmp(&right.region_id))
        });
        let candidates = scored
            .into_iter()
            .take(request.max_regions.max(1))
            .collect::<Vec<_>>();
        let region_ids = candidates
            .iter()
            .map(|candidate| candidate.region_id.clone())
            .collect::<Vec<_>>();
        let rationale = candidates
            .iter()
            .map(|candidate| candidate.reason.clone())
            .collect::<Vec<_>>()
            .join("; ");
        let next_steps = route_next_steps(&candidates);
        RoutedQuery {
            query: request.text.clone(),
            region_ids,
            filters: request.filters.clone(),
            rationale,
            route_plan: RoutePlan {
                candidates,
                next_steps,
            },
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
        let routed = self.route(
            memory.memory_map.as_ref().expect("memory map missing"),
            &request,
        );
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
            let candidates =
                ann_index.search_region(region_id, &query_embedding, request.max_chunks.max(1));
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
                let lexical = lexical_overlap_score(
                    &request.text,
                    &format!(
                        "{} {} {}",
                        document.title,
                        chunk
                            .metadata
                            .get("document_title")
                            .map(String::as_str)
                            .unwrap_or_default(),
                        chunk.text
                    ),
                );
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

            for chunk in memory
                .chunks
                .iter()
                .filter(|chunk| &chunk.region_id == region_id)
            {
                if !matches_filters(&request.filters, &chunk.metadata) {
                    continue;
                }
                let Some(document) = documents_by_id.get(&chunk.document_id) else {
                    continue;
                };
                let lexical = lexical_overlap_score(
                    &request.text,
                    &format!(
                        "{} {} {}",
                        document.title,
                        chunk
                            .metadata
                            .get("document_title")
                            .map(String::as_str)
                            .unwrap_or_default(),
                        chunk.text
                    ),
                );
                if lexical <= 0.0 {
                    continue;
                }
                let excerpt = excerpt_window(&document.text, chunk.start, chunk.end, 60);
                hits.push(ChunkHit {
                    hit_id: format!("hit:{chunk_id}", chunk_id = chunk.id),
                    chunk_id: chunk.id.clone(),
                    document_id: chunk.document_id.clone(),
                    region_id: chunk.region_id.clone(),
                    score: lexical.max(0.05),
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

fn matches_filters(
    filters: &BTreeMap<String, String>,
    metadata: &BTreeMap<String, String>,
) -> bool {
    filters.iter().all(|(key, value)| {
        metadata
            .get(key)
            .map(|candidate| candidate == value)
            .unwrap_or(false)
    })
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

fn route_next_steps(candidates: &[RouteCandidate]) -> Vec<String> {
    if candidates.is_empty() {
        return vec!["memory_search with a broader query before opening source context".into()];
    }
    let region_list = candidates
        .iter()
        .map(|candidate| candidate.region_id.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    vec![
        format!("memory_search within candidate regions: {region_list}"),
        "memory_open the strongest hit, then memory_neighbors for surfable links".into(),
        "memory_expand or memory_jump_to_anchor before citing source truth".into(),
    ]
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

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(region_id: &str, label: &str, summary: &str) -> MapEntry {
        MapEntry {
            region_id: region_id.into(),
            label: label.into(),
            summary: summary.into(),
            filters: BTreeMap::new(),
        }
    }

    #[test]
    fn route_returns_structured_candidates_for_agent_planning() {
        let memory_map = MemoryMap {
            budget_bytes: 2048,
            serialized: String::new(),
            entries: vec![
                entry(
                    "source-truth",
                    "Source Truth",
                    "Anchored citations and original documents",
                ),
                entry(
                    "agent-routing",
                    "Agent Routing",
                    "Tool contracts for memory navigation",
                ),
            ],
        };
        let request = QueryRequest {
            text: "How should an agent navigate source citations?".into(),
            filters: BTreeMap::new(),
            max_regions: 2,
            max_chunks: 5,
        };

        let routed = MemoryQueryEngine.route(&memory_map, &request);

        assert_eq!(routed.route_plan.candidates.len(), 2);
        assert_eq!(routed.route_plan.candidates[0].region_id, "source-truth");
        assert!(routed.route_plan.candidates[0].score > 0.0);
        assert!(routed.route_plan.candidates[0]
            .matched_terms
            .contains(&"source".into()));
        assert!(routed
            .route_plan
            .next_steps
            .iter()
            .any(|step| step.contains("memory_search")));
    }
}
