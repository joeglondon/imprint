use crate::types::*;
use anyhow::{Context, Result};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::Duration;

pub trait Embedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>>;
    fn identity(&self) -> EmbedderIdentity;

    fn embed_many(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|text| self.embed(text)).collect()
    }
}

#[derive(Debug, Clone)]
pub struct HashEmbedder {
    dimensions: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbedderIdentity {
    pub provider: String,
    pub model: String,
    pub endpoint: String,
}

impl HashEmbedder {
    pub fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }
}

impl Default for HashEmbedder {
    fn default() -> Self {
        Self::new(64)
    }
}

impl Embedder for HashEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut vector = vec![0.0; self.dimensions];
        for token in tokenize(text) {
            let mut hash = 14695981039346656037u64;
            for byte in token.as_bytes() {
                hash ^= *byte as u64;
                hash = hash.wrapping_mul(1099511628211);
            }
            let index = (hash as usize) % self.dimensions;
            vector[index] += 1.0;
        }
        normalize(&mut vector);
        Ok(vector)
    }

    fn identity(&self) -> EmbedderIdentity {
        EmbedderIdentity {
            provider: "hash".into(),
            model: format!("hash-{}d", self.dimensions),
            endpoint: "local".into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct OllamaEmbedder {
    endpoint: String,
    model: String,
    client: Client,
}

impl OllamaEmbedder {
    pub fn new(endpoint: impl Into<String>, model: impl Into<String>) -> Result<Self> {
        Ok(Self {
            endpoint: endpoint.into().trim_end_matches('/').to_string(),
            model: model.into(),
            client: Client::builder().timeout(Duration::from_secs(60)).build()?,
        })
    }
}

impl Embedder for OllamaEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let response = self
            .client
            .post(format!("{}/api/embed", self.endpoint))
            .json(&OllamaEmbedRequest {
                model: &self.model,
                input: text,
            })
            .send()
            .with_context(|| format!("calling Ollama embedding model {}", self.model))?
            .error_for_status()
            .with_context(|| format!("Ollama embedding model {} returned an error", self.model))?
            .json::<OllamaEmbedResponse>()
            .context("decoding Ollama embedding response")?;
        response
            .embeddings
            .into_iter()
            .next()
            .filter(|embedding| !embedding.is_empty())
            .context("Ollama returned no embedding values")
    }

    fn embed_many(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let response = self
            .client
            .post(format!("{}/api/embed", self.endpoint))
            .json(&OllamaEmbedManyRequest {
                model: &self.model,
                input: texts,
            })
            .send()
            .with_context(|| format!("calling Ollama embedding model {}", self.model))?
            .error_for_status()
            .with_context(|| format!("Ollama embedding model {} returned an error", self.model))?
            .json::<OllamaEmbedResponse>()
            .context("decoding Ollama embedding response")?;
        if response.embeddings.len() != texts.len() {
            anyhow::bail!(
                "Ollama returned {} embeddings for {} inputs",
                response.embeddings.len(),
                texts.len()
            );
        }
        if response.embeddings.iter().any(Vec::is_empty) {
            anyhow::bail!("Ollama returned empty embedding values");
        }
        Ok(response.embeddings)
    }

    fn identity(&self) -> EmbedderIdentity {
        EmbedderIdentity {
            provider: "ollama".into(),
            model: self.model.clone(),
            endpoint: self.endpoint.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct OpenAiCompatibleEmbedder {
    endpoint: String,
    model: String,
    client: Client,
}

impl OpenAiCompatibleEmbedder {
    pub fn new(endpoint: impl Into<String>, model: impl Into<String>) -> Result<Self> {
        let raw = endpoint.into();
        let trimmed = raw.trim_end_matches('/');
        let endpoint = if trimmed.ends_with("/v1") {
            trimmed.to_string()
        } else {
            format!("{trimmed}/v1")
        };
        Ok(Self {
            endpoint,
            model: model.into(),
            client: Client::builder().timeout(Duration::from_secs(60)).build()?,
        })
    }
}

impl Embedder for OpenAiCompatibleEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let response = self
            .client
            .post(format!("{}/embeddings", self.endpoint))
            .bearer_auth("not-needed")
            .json(&OpenAiEmbeddingRequest {
                model: &self.model,
                input: text,
            })
            .send()
            .with_context(|| format!("calling OpenAI-compatible embedding model {}", self.model))?
            .error_for_status()
            .with_context(|| {
                format!(
                    "OpenAI-compatible embedding model {} returned an error",
                    self.model
                )
            })?
            .json::<OpenAiEmbeddingResponse>()
            .context("decoding OpenAI-compatible embedding response")?;
        response
            .data
            .into_iter()
            .next()
            .map(|item| item.embedding)
            .filter(|embedding| !embedding.is_empty())
            .context("OpenAI-compatible endpoint returned no embedding values")
    }

    fn identity(&self) -> EmbedderIdentity {
        EmbedderIdentity {
            provider: "openai-compatible".into(),
            model: self.model.clone(),
            endpoint: self.endpoint.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum RuntimeEmbedder {
    Hash(HashEmbedder),
    Ollama(OllamaEmbedder),
    OpenAi(OpenAiCompatibleEmbedder),
}

impl Embedder for RuntimeEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        match self {
            RuntimeEmbedder::Hash(embedder) => embedder.embed(text),
            RuntimeEmbedder::Ollama(embedder) => embedder.embed(text),
            RuntimeEmbedder::OpenAi(embedder) => embedder.embed(text),
        }
    }

    fn identity(&self) -> EmbedderIdentity {
        match self {
            RuntimeEmbedder::Hash(embedder) => embedder.identity(),
            RuntimeEmbedder::Ollama(embedder) => embedder.identity(),
            RuntimeEmbedder::OpenAi(embedder) => embedder.identity(),
        }
    }
}

#[derive(Serialize)]
struct OllamaEmbedRequest<'a> {
    model: &'a str,
    input: &'a str,
}

#[derive(Serialize)]
struct OllamaEmbedManyRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(Deserialize)]
struct OllamaEmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

#[derive(Serialize)]
struct OpenAiEmbeddingRequest<'a> {
    model: &'a str,
    input: &'a str,
}

#[derive(Deserialize)]
struct OpenAiEmbeddingResponse {
    data: Vec<OpenAiEmbeddingItem>,
}

#[derive(Deserialize)]
struct OpenAiEmbeddingItem {
    embedding: Vec<f32>,
}

pub const VECTOR_INDEX_KIND: &str = "region-graph-ann";
pub const VECTOR_INDEX_VERSION: u32 = 1;

pub trait Indexer {
    fn rebuild(&self, chunks: &[Chunk], regions: &[Region]) -> RegionAnnIndex;
}

#[derive(Debug, Clone, Default)]
pub struct RegionIndexer;

impl Indexer for RegionIndexer {
    fn rebuild(&self, chunks: &[Chunk], regions: &[Region]) -> RegionAnnIndex {
        RegionAnnIndex::build(chunks, regions)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VectorIndexHealth {
    pub index_kind: String,
    pub index_version: u32,
    pub embedding_provider: Option<String>,
    pub embedding_model: Option<String>,
    pub embedding_endpoint: Option<String>,
    pub dimension: Option<usize>,
    pub corpus_hash: String,
    pub region_count: usize,
    pub chunk_count: usize,
    pub last_rebuild_at: u64,
}

impl VectorIndexHealth {
    pub fn for_memory(chunks: &[Chunk], regions: &[Region], last_rebuild_at: u64) -> Self {
        Self {
            index_kind: VECTOR_INDEX_KIND.into(),
            index_version: VECTOR_INDEX_VERSION,
            embedding_provider: unique_chunk_value(chunks, |chunk| {
                chunk.embedding_provider.as_deref()
            }),
            embedding_model: unique_chunk_value(chunks, |chunk| chunk.embedding_model.as_deref()),
            embedding_endpoint: unique_chunk_value(chunks, |chunk| {
                chunk.embedding_endpoint.as_deref()
            }),
            dimension: unique_chunk_dimension(chunks),
            corpus_hash: vector_corpus_hash(chunks, regions),
            region_count: regions.len(),
            chunk_count: chunks.len(),
            last_rebuild_at,
        }
    }

    pub fn matches_memory(&self, chunks: &[Chunk], regions: &[Region]) -> bool {
        let current = Self::for_memory(chunks, regions, self.last_rebuild_at);
        self.index_kind == current.index_kind
            && self.index_version == current.index_version
            && self.embedding_provider == current.embedding_provider
            && self.embedding_model == current.embedding_model
            && self.embedding_endpoint == current.embedding_endpoint
            && self.dimension == current.dimension
            && self.corpus_hash == current.corpus_hash
            && self.region_count == current.region_count
            && self.chunk_count == current.chunk_count
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegionAnnIndex {
    pub entries: BTreeMap<RegionId, RegionIndex>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegionIndex {
    pub entry_chunk: ChunkId,
    #[serde(default)]
    pub signature: String,
    pub adjacency: BTreeMap<ChunkId, Vec<ChunkId>>,
    pub embeddings: BTreeMap<ChunkId, Vec<f32>>,
}

impl RegionAnnIndex {
    pub fn build(chunks: &[Chunk], regions: &[Region]) -> Self {
        let mut by_region: HashMap<&str, Vec<&Chunk>> = HashMap::new();
        for chunk in chunks {
            by_region
                .entry(chunk.region_id.as_str())
                .or_default()
                .push(chunk);
        }
        let mut entries = BTreeMap::new();
        for region in regions {
            if let Some(region_chunks) = by_region.get(region.id.as_str()) {
                let entry_chunk = region_chunks
                    .iter()
                    .min_by_key(|chunk| chunk.ordinal)
                    .map(|chunk| chunk.id.clone())
                    .unwrap_or_else(|| format!("{}:entry", region.id));
                let mut embeddings = BTreeMap::new();
                let mut adjacency = BTreeMap::new();
                for chunk in region_chunks {
                    embeddings.insert(chunk.id.clone(), chunk.embedding.clone());
                    let mut neighbors = region_chunks
                        .iter()
                        .filter(|other| other.id != chunk.id)
                        .map(|other| {
                            (
                                other.id.clone(),
                                cosine_similarity(&chunk.embedding, &other.embedding),
                            )
                        })
                        .collect::<Vec<_>>();
                    neighbors.sort_by(desc_score);
                    adjacency.insert(
                        chunk.id.clone(),
                        neighbors.into_iter().take(4).map(|(id, _)| id).collect(),
                    );
                }
                entries.insert(
                    region.id.clone(),
                    RegionIndex {
                        entry_chunk,
                        signature: region_signature(region_chunks),
                        adjacency,
                        embeddings,
                    },
                );
            }
        }
        Self { entries }
    }

    pub fn build_incremental(
        previous: Option<&RegionAnnIndex>,
        chunks: &[Chunk],
        regions: &[Region],
    ) -> Self {
        let Some(previous) = previous else {
            return Self::build(chunks, regions);
        };
        let mut by_region: HashMap<&str, Vec<&Chunk>> = HashMap::new();
        for chunk in chunks {
            by_region
                .entry(chunk.region_id.as_str())
                .or_default()
                .push(chunk);
        }
        let mut entries = BTreeMap::new();
        for region in regions {
            let Some(region_chunks) = by_region.get(region.id.as_str()) else {
                continue;
            };
            let signature = region_signature(region_chunks);
            if let Some(existing) = previous.entries.get(&region.id) {
                if existing.signature == signature {
                    entries.insert(region.id.clone(), existing.clone());
                    continue;
                }
            }
            if let Some(rebuilt) = Self::build_region(region, region_chunks) {
                entries.insert(region.id.clone(), rebuilt);
            }
        }
        Self { entries }
    }

    fn build_region(region: &Region, region_chunks: &[&Chunk]) -> Option<RegionIndex> {
        if region_chunks.is_empty() {
            return None;
        }
        let entry_chunk = region_chunks
            .iter()
            .min_by_key(|chunk| chunk.ordinal)
            .map(|chunk| chunk.id.clone())
            .unwrap_or_else(|| format!("{}:entry", region.id));
        let mut embeddings = BTreeMap::new();
        let mut adjacency = BTreeMap::new();
        for chunk in region_chunks {
            embeddings.insert(chunk.id.clone(), chunk.embedding.clone());
            let mut neighbors = region_chunks
                .iter()
                .filter(|other| other.id != chunk.id)
                .map(|other| {
                    (
                        other.id.clone(),
                        cosine_similarity(&chunk.embedding, &other.embedding),
                    )
                })
                .collect::<Vec<_>>();
            neighbors.sort_by(desc_score);
            adjacency.insert(
                chunk.id.clone(),
                neighbors.into_iter().take(4).map(|(id, _)| id).collect(),
            );
        }
        Some(RegionIndex {
            entry_chunk,
            signature: region_signature(region_chunks),
            adjacency,
            embeddings,
        })
    }

    pub fn search_region(
        &self,
        region_id: &str,
        query_embedding: &[f32],
        max_results: usize,
    ) -> Vec<(ChunkId, f32)> {
        let Some(region) = self.entries.get(region_id) else {
            return Vec::new();
        };
        if region.embeddings.is_empty() {
            return Vec::new();
        }
        let mut best = region.entry_chunk.clone();
        let mut best_score = region
            .embeddings
            .get(&best)
            .map(|v| cosine_similarity(query_embedding, v))
            .unwrap_or(0.0);
        let mut improved = true;
        let mut seen = HashSet::new();
        while improved {
            improved = false;
            seen.insert(best.clone());
            if let Some(neighbors) = region.adjacency.get(&best) {
                for neighbor in neighbors {
                    if seen.contains(neighbor) {
                        continue;
                    }
                    let score = region
                        .embeddings
                        .get(neighbor)
                        .map(|v| cosine_similarity(query_embedding, v))
                        .unwrap_or(0.0);
                    if score > best_score {
                        best = neighbor.clone();
                        best_score = score;
                        improved = true;
                    }
                }
            }
        }
        let mut candidates = vec![(best.clone(), best_score)];
        let mut frontier = region.adjacency.get(&best).cloned().unwrap_or_default();
        frontier.push(best);
        frontier.sort();
        frontier.dedup();
        for candidate in frontier {
            if let Some(embedding) = region.embeddings.get(&candidate) {
                candidates.push((candidate, cosine_similarity(query_embedding, embedding)));
            }
        }
        candidates.sort_by(desc_score);
        candidates.truncate(max_results);
        candidates
    }
}

pub fn vector_corpus_hash(chunks: &[Chunk], regions: &[Region]) -> String {
    let mut parts = Vec::new();
    for region in regions {
        let mut chunk_ids = region.chunk_ids.clone();
        chunk_ids.sort();
        parts.push(format!(
            "region:{}:{}:{}",
            region.id,
            stable_hash(&region.centroid),
            chunk_ids.join(",")
        ));
    }
    for chunk in chunks {
        parts.push(format!(
            "chunk:{}:{}:{}:{}:{}:{}:{}",
            chunk.id,
            chunk.region_id,
            chunk.embedding_text_hash.as_deref().unwrap_or(""),
            chunk.embedding_provider.as_deref().unwrap_or(""),
            chunk.embedding_model.as_deref().unwrap_or(""),
            chunk.embedding_endpoint.as_deref().unwrap_or(""),
            stable_hash(&chunk.embedding)
        ));
    }
    parts.sort();
    stable_text_hash(&parts.join("\n"))
}

pub fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|ch: char| !ch.is_alphanumeric())
        .filter(|token| token.len() > 2)
        .map(ToOwned::to_owned)
        .collect()
}

pub fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    if left.is_empty() || right.is_empty() || left.len() != right.len() {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;
    for (l, r) in left.iter().zip(right) {
        dot += l * r;
        left_norm += l * l;
        right_norm += r * r;
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        return 0.0;
    }
    dot / (left_norm.sqrt() * right_norm.sqrt())
}

fn normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in vector {
            *value /= norm;
        }
    }
}

fn desc_score(left: &(String, f32), right: &(String, f32)) -> Ordering {
    right
        .1
        .partial_cmp(&left.1)
        .unwrap_or(Ordering::Equal)
        .then_with(|| left.0.cmp(&right.0))
}

fn unique_chunk_value<F>(chunks: &[Chunk], read: F) -> Option<String>
where
    F: Fn(&Chunk) -> Option<&str>,
{
    let mut value = None::<String>;
    for chunk in chunks {
        let Some(current) = read(chunk) else {
            continue;
        };
        if let Some(existing) = &value {
            if existing != current {
                return Some("mixed".into());
            }
        } else {
            value = Some(current.to_string());
        }
    }
    value
}

fn unique_chunk_dimension(chunks: &[Chunk]) -> Option<usize> {
    let mut dimension = None;
    for chunk in chunks {
        let current = chunk
            .embedding_dimension
            .unwrap_or_else(|| chunk.embedding.len());
        if let Some(existing) = dimension {
            if existing != current {
                return None;
            }
        } else {
            dimension = Some(current);
        }
    }
    dimension
}

fn region_signature(region_chunks: &[&Chunk]) -> String {
    let mut parts = region_chunks
        .iter()
        .map(|chunk| {
            format!(
                "{}:{}:{}:{}:{}",
                chunk.id,
                chunk.ordinal,
                chunk.embedding_text_hash.as_deref().unwrap_or(""),
                chunk
                    .embedding_dimension
                    .unwrap_or_else(|| chunk.embedding.len()),
                stable_hash(&chunk.embedding)
            )
        })
        .collect::<Vec<_>>();
    parts.sort();
    stable_text_hash(&parts.join("\n"))
}

fn stable_hash(values: &[f32]) -> String {
    let mut hash = 14695981039346656037u64;
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(1099511628211);
        }
    }
    format!("{hash:016x}")
}

fn stable_text_hash(text: &str) -> String {
    let mut hash = 14695981039346656037u64;
    for byte in text.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    format!("{hash:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn vector_index_health_detects_embedding_dimension_changes() {
        let chunks = vec![chunk("chunk-a", "region-a", vec![1.0, 0.0])];
        let regions = vec![region("region-a", vec!["chunk-a".into()], vec![1.0, 0.0])];
        let health = VectorIndexHealth::for_memory(&chunks, &regions, 42);
        assert!(health.matches_memory(&chunks, &regions));

        let changed = vec![chunk("chunk-a", "region-a", vec![1.0, 0.0, 0.0])];
        assert!(!health.matches_memory(&changed, &regions));
    }

    #[test]
    fn incremental_rebuild_reuses_unchanged_regions() {
        let chunks = vec![
            chunk("chunk-a", "region-a", vec![1.0, 0.0]),
            chunk("chunk-b", "region-b", vec![0.0, 1.0]),
        ];
        let regions = vec![
            region("region-a", vec!["chunk-a".into()], vec![1.0, 0.0]),
            region("region-b", vec!["chunk-b".into()], vec![0.0, 1.0]),
        ];
        let original = RegionAnnIndex::build(&chunks, &regions);
        let mut changed_chunks = chunks.clone();
        changed_chunks[1].embedding = vec![0.5, 0.5];
        changed_chunks[1].embedding_dimension = Some(2);
        let updated = RegionAnnIndex::build_incremental(Some(&original), &changed_chunks, &regions);

        assert_eq!(
            original.entries["region-a"].signature,
            updated.entries["region-a"].signature
        );
        assert_ne!(
            original.entries["region-b"].signature,
            updated.entries["region-b"].signature
        );
    }

    fn chunk(id: &str, region_id: &str, embedding: Vec<f32>) -> Chunk {
        Chunk {
            id: id.into(),
            document_id: format!("doc-{id}"),
            region_id: region_id.into(),
            ordinal: 0,
            start: 0,
            end: 4,
            text: id.into(),
            metadata: BTreeMap::new(),
            source_anchor: None,
            embedding_dimension: Some(embedding.len()),
            embedding,
            embedding_text_hash: Some(format!("hash-{id}")),
            embedding_provider: Some("hash".into()),
            embedding_model: Some("hash-2d".into()),
            embedding_endpoint: Some("local".into()),
            chunking_version: Some(1),
        }
    }

    fn region(id: &str, chunk_ids: Vec<String>, centroid: Vec<f32>) -> Region {
        Region {
            id: id.into(),
            label: id.into(),
            summary: id.into(),
            filters: BTreeMap::new(),
            chunk_ids,
            centroid,
            neighbors: Vec::new(),
        }
    }
}
