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
        self.route_with_region_scores(memory_map, request, &HashMap::new(), &HashMap::new())
    }
}

impl MemoryQueryEngine {
    fn route_with_cortex_index_scores(
        &self,
        cortex_index: &CortexIndex,
        fallback_memory_map: Option<&MemoryMap>,
        request: &QueryRequest,
        semantic_scores: &HashMap<RegionId, f32>,
        attention_scores: &HashMap<RegionId, f32>,
    ) -> RoutedQuery {
        let tokens = tokenize(&request.text);
        let mut scored = cortex_index
            .regions
            .iter()
            .map(|region| {
                let label = region.label.to_lowercase();
                let summary = region.summary.to_lowercase();
                let route_examples = region
                    .route_examples
                    .iter()
                    .map(|example| example.to_lowercase())
                    .collect::<Vec<_>>();
                let source_refs = region
                    .source_refs
                    .iter()
                    .map(|source| source.to_lowercase())
                    .collect::<Vec<_>>();
                let artifact_ids = region
                    .artifact_ids
                    .iter()
                    .map(|artifact| artifact.to_lowercase())
                    .collect::<Vec<_>>();
                let mut score = 0.0f32;
                let mut matched_terms = Vec::new();
                for token in &tokens {
                    let mut matched = false;
                    if label.contains(token) || summary.contains(token) {
                        score += 2.0;
                        matched = true;
                    }
                    if route_examples.iter().any(|example| example.contains(token)) {
                        score += 1.25;
                        matched = true;
                    }
                    if source_refs.iter().any(|source| source.contains(token))
                        || artifact_ids.iter().any(|artifact| artifact.contains(token))
                    {
                        score += 0.5;
                        matched = true;
                    }
                    if matched {
                        matched_terms.push(token.clone());
                    }
                }
                let semantic = semantic_scores
                    .get(&region.region_id)
                    .copied()
                    .unwrap_or_default()
                    .max(0.0);
                let attention = attention_scores
                    .get(&region.region_id)
                    .copied()
                    .unwrap_or_default();
                score += semantic * 3.0;
                score += attention;
                let reason = format!(
                    "{} matched cortex-index route score {score:.2} (semantic {semantic:.2}, attention {attention:.2}, examples {}, sources {})",
                    region.label,
                    region.route_examples.len(),
                    region.source_refs.len()
                );
                RouteCandidate {
                    region_id: region.region_id.clone(),
                    label: region.label.clone(),
                    score,
                    matched_terms,
                    reason,
                }
            })
            .collect::<Vec<_>>();
        let cortex_region_ids = cortex_index
            .regions
            .iter()
            .map(|region| region.region_id.clone())
            .collect::<HashSet<_>>();
        if let Some(memory_map) = fallback_memory_map {
            let mut fallback_request = request.clone();
            fallback_request.max_regions = memory_map.entries.len().max(request.max_regions);
            scored.extend(
                self.route_with_region_scores(
                    memory_map,
                    &fallback_request,
                    semantic_scores,
                    attention_scores,
                )
                .route_plan
                .candidates
                .into_iter()
                .filter(|candidate| !cortex_region_ids.contains(&candidate.region_id))
                .map(|mut candidate| {
                    candidate.reason = format!("legacy fallback: {}", candidate.reason);
                    candidate
                }),
            );
        }
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

    fn route_with_region_scores(
        &self,
        memory_map: &MemoryMap,
        request: &QueryRequest,
        semantic_scores: &HashMap<RegionId, f32>,
        attention_scores: &HashMap<RegionId, f32>,
    ) -> RoutedQuery {
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
                let semantic = semantic_scores
                    .get(&entry.region_id)
                    .copied()
                    .unwrap_or_default()
                    .max(0.0);
                let attention = attention_scores
                    .get(&entry.region_id)
                    .copied()
                    .unwrap_or_default();
                score += semantic * 3.0;
                score += attention;
                let reason = if semantic > 0.0 || attention != 0.0 {
                    format!(
                        "{} matched route score {score:.2} (semantic {semantic:.2}, attention {attention:.2})",
                        entry.label
                    )
                } else {
                    format!("{} matched route score {score:.1}", entry.label)
                };
                RouteCandidate {
                    region_id: entry.region_id.clone(),
                    label: entry.label.clone(),
                    score,
                    matched_terms,
                    reason,
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

    pub fn execute<E: Embedder>(
        &self,
        embedder: &E,
        memory: &PersistedMemory,
        ann_index: &RegionAnnIndex,
        request: QueryRequest,
    ) -> anyhow::Result<QueryResult> {
        self.execute_with_attention(embedder, memory, ann_index, request, &[])
    }

    pub fn execute_with_attention<E: Embedder>(
        &self,
        embedder: &E,
        memory: &PersistedMemory,
        ann_index: &RegionAnnIndex,
        request: QueryRequest,
        attention_marks: &[AttentionMark],
    ) -> anyhow::Result<QueryResult> {
        self.execute_with_signals(
            embedder,
            memory,
            ann_index,
            request,
            attention_marks,
            &[],
            0,
            None,
        )
    }

    pub fn execute_with_signals<E: Embedder>(
        &self,
        embedder: &E,
        memory: &PersistedMemory,
        ann_index: &RegionAnnIndex,
        request: QueryRequest,
        attention_marks: &[AttentionMark],
        memory_accesses: &[MemoryAccess],
        now: u64,
        cortex_index: Option<&CortexIndex>,
    ) -> anyhow::Result<QueryResult> {
        let query_embedding = embedder.embed(&request.text)?;
        let mut attention_scores = attention_scores(attention_marks);
        merge_access_scores(&mut attention_scores, memory_accesses, now);
        let region_attention_scores = attention_scores
            .iter()
            .filter_map(|(key, score)| {
                key.strip_prefix("region:")
                    .map(|region_id| (region_id.to_string(), *score))
            })
            .collect::<HashMap<_, _>>();
        let semantic_scores = memory
            .regions
            .iter()
            .map(|region| {
                (
                    region.id.clone(),
                    cosine_similarity(&query_embedding, &region.centroid),
                )
            })
            .collect::<HashMap<_, _>>();
        let routed = if let Some(cortex_index) = cortex_index {
            self.route_with_cortex_index_scores(
                cortex_index,
                memory.memory_map.as_ref(),
                &request,
                &semantic_scores,
                &region_attention_scores,
            )
        } else {
            self.route_with_region_scores(
                memory.memory_map.as_ref().expect("memory map missing"),
                &request,
                &semantic_scores,
                &region_attention_scores,
            )
        };
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
                let score = apply_recall_score(
                    ann_score * 0.7 + lexical * 0.3,
                    chunk,
                    document,
                    &attention_scores,
                    now,
                );
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
                    score: apply_recall_score(
                        lexical.max(0.05),
                        chunk,
                        document,
                        &attention_scores,
                        now,
                    ),
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

fn apply_recall_score(
    base_score: f32,
    chunk: &Chunk,
    document: &Document,
    attention_scores: &HashMap<String, f32>,
    now: u64,
) -> f32 {
    let boost = attention_scores
        .get(&attention_key(&AttentionTargetKind::Chunk, &chunk.id))
        .copied()
        .unwrap_or_default()
        + attention_scores
            .get(&attention_key(
                &AttentionTargetKind::Document,
                &chunk.document_id,
            ))
            .copied()
            .unwrap_or_default()
        + attention_scores
            .get(&attention_key(
                &AttentionTargetKind::Region,
                &chunk.region_id,
            ))
            .copied()
            .unwrap_or_default();
    let source_signal = source_recall_signal(document, chunk, now);
    (base_score + boost.clamp(-0.45, 0.55) + source_signal).max(0.001)
}

fn source_recall_signal(document: &Document, chunk: &Chunk, now: u64) -> f32 {
    let mut signal = match metadata_value(document, chunk, "source_type").as_deref() {
        Some("web_finding") => -0.01,
        Some("derived_memory") | Some("brain_artifact") => -0.04,
        Some("chat") => 0.01,
        _ => 0.04,
    };

    if let Some(trust) = parse_metadata_f32(document, chunk, "source_trust") {
        signal += (trust.clamp(0.0, 1.0) - 0.5) * 0.18;
    }

    if let Some(confidence) = parse_metadata_f32(document, chunk, "confidence") {
        signal += (confidence.clamp(0.0, 1.0) - 0.5) * 0.12;
    }

    if let Some(retrieved_at) = parse_metadata_u64(document, chunk, "retrieved_at") {
        signal += freshness_signal(retrieved_at, now);
    }

    if let Some(expires_at) = parse_metadata_u64(document, chunk, "freshness_expires_at") {
        let expires_at = normalize_epoch_millis(expires_at);
        if now > 0 && now > expires_at {
            signal -= 0.12;
        }
    }

    signal.clamp(-0.18, 0.18)
}

fn freshness_signal(retrieved_at: u64, now: u64) -> f32 {
    if now == 0 {
        return 0.0;
    }
    let retrieved_at = normalize_epoch_millis(retrieved_at);
    let age = now.saturating_sub(retrieved_at);
    let day = 24 * 60 * 60 * 1000;
    if age <= 7 * day {
        0.08
    } else if age <= 30 * day {
        0.02
    } else if age <= 180 * day {
        -0.03
    } else {
        -0.08
    }
}

fn normalize_epoch_millis(value: u64) -> u64 {
    if value < 10_000_000_000 {
        value * 1000
    } else {
        value
    }
}

fn parse_metadata_f32(document: &Document, chunk: &Chunk, key: &str) -> Option<f32> {
    metadata_value(document, chunk, key).and_then(|value| value.parse::<f32>().ok())
}

fn parse_metadata_u64(document: &Document, chunk: &Chunk, key: &str) -> Option<u64> {
    metadata_value(document, chunk, key).and_then(|value| value.parse::<u64>().ok())
}

fn metadata_value(document: &Document, chunk: &Chunk, key: &str) -> Option<String> {
    chunk
        .metadata
        .get(key)
        .or_else(|| document.metadata.get(key))
        .cloned()
}

fn attention_scores(marks: &[AttentionMark]) -> HashMap<String, f32> {
    let mut scores = HashMap::new();
    for mark in marks.iter().filter(|mark| mark.reverted_at.is_none()) {
        let delta = match mark.action {
            AttentionAction::Pin => 0.35,
            AttentionAction::Promote => 0.22,
            AttentionAction::Active => 0.14,
            AttentionAction::Decay => -0.14,
            AttentionAction::Suppress => -0.40,
        };
        *scores
            .entry(attention_key(&mark.target_kind, &mark.target_id))
            .or_insert(0.0) += delta;
    }
    scores
}

fn merge_access_scores(scores: &mut HashMap<String, f32>, accesses: &[MemoryAccess], now: u64) {
    for access in accesses {
        let mut score = match access.access_kind {
            MemoryAccessKind::QueryHit => 0.03,
            MemoryAccessKind::Open => 0.08,
            MemoryAccessKind::Expand | MemoryAccessKind::JumpToAnchor => 0.12,
            MemoryAccessKind::Cite => 0.18,
        };
        if now > 0 {
            let age = now.saturating_sub(access.accessed_at);
            let week = 7 * 24 * 60 * 60 * 1000;
            if age >= week {
                score *= 0.25;
            } else {
                let freshness = 1.0 - (age as f32 / week as f32);
                score *= 0.45 + freshness * 0.55;
            }
        }
        scores
            .entry(attention_key(&access.target_kind, &access.target_id))
            .and_modify(|current| *current = (*current + score).clamp(-0.6, 0.6))
            .or_insert(score);
    }
}

fn attention_key(kind: &AttentionTargetKind, target_id: &str) -> String {
    let kind = match kind {
        AttentionTargetKind::ChatSession => "chat_session",
        AttentionTargetKind::ChatMessage => "chat_message",
        AttentionTargetKind::TranscriptChunk => "transcript_chunk",
        AttentionTargetKind::DerivedMemory => "derived_memory",
        AttentionTargetKind::WebFinding => "web_finding",
        AttentionTargetKind::Document => "document",
        AttentionTargetKind::Chunk => "chunk",
        AttentionTargetKind::Region => "region",
        AttentionTargetKind::Link => "link",
    };
    format!("{kind}:{target_id}")
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
    use crate::index::Indexer;

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

    #[test]
    fn execute_routes_by_region_centroid_when_map_words_do_not_match() {
        let embedder = crate::index::HashEmbedder::default();
        let target_text = "Plantar fascia rehabilitation uses calf stretching and arch loading.";
        let distractor_text = "Soup recipes use onion stock and gentle simmering.";
        let target_embedding = embedder.embed(target_text).expect("target embedding");
        let distractor_embedding = embedder
            .embed(distractor_text)
            .expect("distractor embedding");
        let memory = PersistedMemory {
            documents: vec![
                Document {
                    id: "doc-target".into(),
                    title: "Foot note".into(),
                    text: target_text.into(),
                    metadata: BTreeMap::new(),
                    source_anchor: None,
                    content_hash: None,
                    parser_version: None,
                },
                Document {
                    id: "doc-distractor".into(),
                    title: "Kitchen note".into(),
                    text: distractor_text.into(),
                    metadata: BTreeMap::new(),
                    source_anchor: None,
                    content_hash: None,
                    parser_version: None,
                },
            ],
            chunks: vec![
                Chunk {
                    id: "chunk-target".into(),
                    document_id: "doc-target".into(),
                    region_id: "region-hidden".into(),
                    ordinal: 0,
                    start: 0,
                    end: target_text.chars().count(),
                    text: target_text.into(),
                    metadata: BTreeMap::new(),
                    source_anchor: None,
                    embedding: target_embedding.clone(),
                    embedding_text_hash: None,
                    embedding_provider: None,
                    embedding_model: None,
                    embedding_endpoint: None,
                    embedding_dimension: None,
                    chunking_version: None,
                },
                Chunk {
                    id: "chunk-distractor".into(),
                    document_id: "doc-distractor".into(),
                    region_id: "region-distractor".into(),
                    ordinal: 0,
                    start: 0,
                    end: distractor_text.chars().count(),
                    text: distractor_text.into(),
                    metadata: BTreeMap::new(),
                    source_anchor: None,
                    embedding: distractor_embedding.clone(),
                    embedding_text_hash: None,
                    embedding_provider: None,
                    embedding_model: None,
                    embedding_endpoint: None,
                    embedding_dimension: None,
                    chunking_version: None,
                },
            ],
            regions: vec![
                Region {
                    id: "region-hidden".into(),
                    label: "Archive A".into(),
                    summary: "Opaque bucket".into(),
                    filters: BTreeMap::new(),
                    chunk_ids: vec!["chunk-target".into()],
                    centroid: target_embedding,
                    neighbors: Vec::new(),
                },
                Region {
                    id: "region-distractor".into(),
                    label: "Archive B".into(),
                    summary: "Opaque bucket".into(),
                    filters: BTreeMap::new(),
                    chunk_ids: vec!["chunk-distractor".into()],
                    centroid: distractor_embedding,
                    neighbors: Vec::new(),
                },
            ],
            links: Vec::new(),
            memory_map: Some(MemoryMap {
                budget_bytes: 2048,
                serialized: String::new(),
                entries: vec![
                    entry("region-distractor", "Archive B", "Opaque bucket"),
                    entry("region-hidden", "Archive A", "Opaque bucket"),
                ],
            }),
        };
        let ann = crate::index::RegionIndexer.rebuild(&memory.chunks, &memory.regions);

        let result = MemoryQueryEngine
            .execute(
                &embedder,
                &memory,
                &ann,
                QueryRequest {
                    text: "plantar fascia".into(),
                    filters: BTreeMap::new(),
                    max_regions: 1,
                    max_chunks: 3,
                },
            )
            .expect("query");

        assert_eq!(result.routed.region_ids, vec!["region-hidden"]);
        assert!(result.routed.rationale.contains("semantic"));
        assert_eq!(
            result.hits.first().map(|hit| hit.chunk_id.as_str()),
            Some("chunk-target")
        );
    }

    #[test]
    fn route_can_use_cortex_index_sketches_instead_of_legacy_map_entries() {
        let memory_map = MemoryMap {
            budget_bytes: 2048,
            serialized: String::new(),
            entries: vec![
                entry(
                    "legacy-distractor",
                    "Plantar Fascia",
                    "Legacy projection points at the old bucket",
                ),
                entry("cortex-target", "Archive", "Opaque summary"),
            ],
        };
        let cortex_index = CortexIndex {
            id: "cortex-index:current".into(),
            schema_version: 1,
            corpus_hash: "hash".into(),
            created_at: 1,
            compiler: "test".into(),
            source_refs: vec!["imprint://chunk/target".into()],
            artifact_ids: vec!["artifact:target".into()],
            regions: vec![
                CortexRegionSketch {
                    region_id: "cortex-target".into(),
                    label: "Archive".into(),
                    summary: "Opaque summary".into(),
                    source_refs: vec!["imprint://chunk/target".into()],
                    artifact_ids: vec!["artifact:target".into()],
                    route_examples: vec!["plantar fascia rehabilitation source".into()],
                },
                CortexRegionSketch {
                    region_id: "legacy-distractor".into(),
                    label: "Old bucket".into(),
                    summary: "Unrelated summary".into(),
                    source_refs: Vec::new(),
                    artifact_ids: Vec::new(),
                    route_examples: Vec::new(),
                },
            ],
            compatibility_map: memory_map.clone(),
        };
        let request = QueryRequest {
            text: "plantar fascia".into(),
            filters: BTreeMap::new(),
            max_regions: 1,
            max_chunks: 5,
        };

        let legacy = MemoryQueryEngine.route_with_region_scores(
            &memory_map,
            &request,
            &HashMap::new(),
            &HashMap::new(),
        );
        let cortex = MemoryQueryEngine.route_with_cortex_index_scores(
            &cortex_index,
            None,
            &request,
            &HashMap::new(),
            &HashMap::new(),
        );

        assert_eq!(legacy.region_ids, vec!["legacy-distractor"]);
        assert_eq!(cortex.region_ids, vec!["cortex-target"]);
        assert!(cortex.rationale.contains("cortex-index"));
    }

    #[test]
    fn execute_uses_attention_marks_to_rank_source_recall() {
        let embedder = crate::index::HashEmbedder::default();
        let text = "shared routing memory";
        let embedding = embedder.embed(text).expect("embedding");
        let document = |id: &str| Document {
            id: id.into(),
            title: id.into(),
            text: text.into(),
            metadata: BTreeMap::new(),
            source_anchor: None,
            content_hash: None,
            parser_version: None,
        };
        let chunk = |id: &str, document_id: &str, ordinal: usize| Chunk {
            id: id.into(),
            document_id: document_id.into(),
            region_id: "region".into(),
            ordinal,
            start: 0,
            end: text.chars().count(),
            text: text.into(),
            metadata: BTreeMap::new(),
            source_anchor: None,
            embedding: embedding.clone(),
            embedding_text_hash: None,
            embedding_provider: None,
            embedding_model: None,
            embedding_endpoint: None,
            embedding_dimension: None,
            chunking_version: None,
        };
        let memory = PersistedMemory {
            documents: vec![document("doc-a"), document("doc-b")],
            chunks: vec![chunk("chunk-a", "doc-a", 0), chunk("chunk-b", "doc-b", 1)],
            regions: vec![Region {
                id: "region".into(),
                label: "Region".into(),
                summary: text.into(),
                filters: BTreeMap::new(),
                chunk_ids: vec!["chunk-a".into(), "chunk-b".into()],
                centroid: embedding,
                neighbors: Vec::new(),
            }],
            links: Vec::new(),
            memory_map: Some(MemoryMap {
                budget_bytes: 2048,
                serialized: String::new(),
                entries: vec![entry("region", "Region", text)],
            }),
        };
        let ann = crate::index::RegionIndexer.rebuild(&memory.chunks, &memory.regions);
        let marks = vec![AttentionMark {
            id: "attention-1".into(),
            target_id: "chunk-b".into(),
            target_kind: AttentionTargetKind::Chunk,
            action: AttentionAction::Pin,
            reason: "User pinned this passage as the preferred routing address.".into(),
            actor: "test".into(),
            created_at: 1,
            reverted_at: None,
        }];

        let result = MemoryQueryEngine
            .execute_with_attention(
                &embedder,
                &memory,
                &ann,
                QueryRequest {
                    text: text.into(),
                    filters: BTreeMap::new(),
                    max_regions: 1,
                    max_chunks: 2,
                },
                &marks,
            )
            .expect("query");

        assert_eq!(
            result.hits.first().map(|hit| hit.chunk_id.as_str()),
            Some("chunk-b")
        );
        assert!(result.hits[0].score > result.hits[1].score);
    }

    #[test]
    fn execute_uses_recent_access_history_as_weak_attention() {
        let embedder = crate::index::HashEmbedder::default();
        let text = "shared routing memory";
        let embedding = embedder.embed(text).expect("embedding");
        let document = |id: &str| Document {
            id: id.into(),
            title: id.into(),
            text: text.into(),
            metadata: BTreeMap::new(),
            source_anchor: None,
            content_hash: None,
            parser_version: None,
        };
        let chunk = |id: &str, document_id: &str, ordinal: usize| Chunk {
            id: id.into(),
            document_id: document_id.into(),
            region_id: "region".into(),
            ordinal,
            start: 0,
            end: text.chars().count(),
            text: text.into(),
            metadata: BTreeMap::new(),
            source_anchor: None,
            embedding: embedding.clone(),
            embedding_text_hash: None,
            embedding_provider: None,
            embedding_model: None,
            embedding_endpoint: None,
            embedding_dimension: None,
            chunking_version: None,
        };
        let memory = PersistedMemory {
            documents: vec![document("doc-a"), document("doc-b")],
            chunks: vec![chunk("chunk-a", "doc-a", 0), chunk("chunk-b", "doc-b", 1)],
            regions: vec![Region {
                id: "region".into(),
                label: "Region".into(),
                summary: text.into(),
                filters: BTreeMap::new(),
                chunk_ids: vec!["chunk-a".into(), "chunk-b".into()],
                centroid: embedding,
                neighbors: Vec::new(),
            }],
            links: Vec::new(),
            memory_map: Some(MemoryMap {
                budget_bytes: 2048,
                serialized: String::new(),
                entries: vec![entry("region", "Region", text)],
            }),
        };
        let ann = crate::index::RegionIndexer.rebuild(&memory.chunks, &memory.regions);
        let accesses = vec![
            MemoryAccess {
                id: "access-1".into(),
                target_id: "chunk-b".into(),
                target_kind: AttentionTargetKind::Chunk,
                access_kind: MemoryAccessKind::Expand,
                reason: "Previously expanded as source truth.".into(),
                actor: "test".into(),
                accessed_at: 1_000,
            },
            MemoryAccess {
                id: "access-2".into(),
                target_id: "chunk-b".into(),
                target_kind: AttentionTargetKind::Chunk,
                access_kind: MemoryAccessKind::Open,
                reason: "Previously opened by the agent.".into(),
                actor: "test".into(),
                accessed_at: 1_100,
            },
        ];

        let result = MemoryQueryEngine
            .execute_with_signals(
                &embedder,
                &memory,
                &ann,
                QueryRequest {
                    text: text.into(),
                    filters: BTreeMap::new(),
                    max_regions: 1,
                    max_chunks: 2,
                },
                &[],
                &accesses,
                1_200,
                None,
            )
            .expect("query");

        assert_eq!(
            result.hits.first().map(|hit| hit.chunk_id.as_str()),
            Some("chunk-b")
        );
        assert!(result.hits[0].score > result.hits[1].score);
    }

    #[test]
    fn execute_uses_source_metadata_as_recall_signal() {
        let embedder = crate::index::HashEmbedder::default();
        let text = "same evidence about source grounded recall";
        let embedding = embedder.embed(text).expect("embedding");
        let mut stale_web_metadata = BTreeMap::new();
        stale_web_metadata.insert("source_type".into(), "web_finding".into());
        stale_web_metadata.insert("confidence".into(), "0.20".into());
        stale_web_metadata.insert("retrieved_at".into(), "1000".into());
        let mut trusted_local_metadata = BTreeMap::new();
        trusted_local_metadata.insert("source".into(), "local".into());
        trusted_local_metadata.insert("source_trust".into(), "0.90".into());
        let document = |id: &str, metadata: BTreeMap<String, String>| Document {
            id: id.into(),
            title: id.into(),
            text: text.into(),
            metadata,
            source_anchor: None,
            content_hash: None,
            parser_version: None,
        };
        let chunk = |id: &str, document_id: &str, ordinal: usize| Chunk {
            id: id.into(),
            document_id: document_id.into(),
            region_id: "region".into(),
            ordinal,
            start: 0,
            end: text.chars().count(),
            text: text.into(),
            metadata: BTreeMap::new(),
            source_anchor: None,
            embedding: embedding.clone(),
            embedding_text_hash: None,
            embedding_provider: None,
            embedding_model: None,
            embedding_endpoint: None,
            embedding_dimension: None,
            chunking_version: None,
        };
        let memory = PersistedMemory {
            documents: vec![
                document("doc-stale-web", stale_web_metadata),
                document("doc-trusted-local", trusted_local_metadata),
            ],
            chunks: vec![
                chunk("chunk-stale-web", "doc-stale-web", 0),
                chunk("chunk-trusted-local", "doc-trusted-local", 1),
            ],
            regions: vec![Region {
                id: "region".into(),
                label: "Region".into(),
                summary: text.into(),
                filters: BTreeMap::new(),
                chunk_ids: vec!["chunk-stale-web".into(), "chunk-trusted-local".into()],
                centroid: embedding,
                neighbors: Vec::new(),
            }],
            links: Vec::new(),
            memory_map: Some(MemoryMap {
                budget_bytes: 2048,
                serialized: String::new(),
                entries: vec![entry("region", "Region", text)],
            }),
        };
        let ann = crate::index::RegionIndexer.rebuild(&memory.chunks, &memory.regions);

        let result = MemoryQueryEngine
            .execute_with_signals(
                &embedder,
                &memory,
                &ann,
                QueryRequest {
                    text: text.into(),
                    filters: BTreeMap::new(),
                    max_regions: 1,
                    max_chunks: 2,
                },
                &[],
                &[],
                1_777_311_476_000,
                None,
            )
            .expect("query");

        assert_eq!(
            result.hits.first().map(|hit| hit.chunk_id.as_str()),
            Some("chunk-trusted-local")
        );
        assert!(result.hits[0].score > result.hits[1].score);
    }
}
