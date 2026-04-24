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
pub enum RuntimeEmbedder {
    Hash(HashEmbedder),
    Ollama(OllamaEmbedder),
}

impl Embedder for RuntimeEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        match self {
            RuntimeEmbedder::Hash(embedder) => embedder.embed(text),
            RuntimeEmbedder::Ollama(embedder) => embedder.embed(text),
        }
    }

    fn identity(&self) -> EmbedderIdentity {
        match self {
            RuntimeEmbedder::Hash(embedder) => embedder.identity(),
            RuntimeEmbedder::Ollama(embedder) => embedder.identity(),
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

#[derive(Debug, Clone)]
pub struct RegionAnnIndex {
    pub entries: BTreeMap<RegionId, RegionIndex>,
}

#[derive(Debug, Clone)]
pub struct RegionIndex {
    pub entry_chunk: ChunkId,
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
                        .map(|other| (other.id.clone(), cosine_similarity(&chunk.embedding, &other.embedding)))
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
                        adjacency,
                        embeddings,
                    },
                );
            }
        }
        Self { entries }
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
