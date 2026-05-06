use crate::map::MapBuilder;
use crate::types::*;
use std::collections::{BTreeMap, BTreeSet};

const CORTEX_INDEX_SCHEMA_VERSION: u32 = 1;

pub fn build_brain_artifacts(memory: &PersistedMemory, created_at: u64) -> Vec<BrainArtifact> {
    let mut artifacts = Vec::new();
    let regions = memory
        .regions
        .iter()
        .filter(|region| region_has_source_chunks(memory, region))
        .take(12)
        .collect::<Vec<_>>();
    if !regions.is_empty() {
        artifacts.push(library_map_artifact(&regions, created_at));
    }
    for region in regions {
        artifacts.push(region_card_artifact(memory, region, created_at));
        artifacts.push(routing_rule_artifact(region, created_at));
    }
    artifacts
}

pub fn build_cortex_index(
    memory: &PersistedMemory,
    artifacts: &[BrainArtifact],
    created_at: u64,
) -> CortexIndex {
    let source_regions = memory
        .regions
        .iter()
        .filter(|region| region_has_source_chunks(memory, region))
        .collect::<Vec<_>>();
    let compatibility_map = MapBuilder::default().build(
        &source_regions
            .iter()
            .map(|region| (*region).clone())
            .collect::<Vec<_>>(),
    );
    let mut artifact_ids = artifacts
        .iter()
        .map(|artifact| artifact.id.clone())
        .collect::<Vec<_>>();
    artifact_ids.sort();
    let artifact_ids_by_region = artifacts_by_region(artifacts);
    let regions = source_regions
        .iter()
        .map(|region| cortex_region_sketch(memory, region, &artifact_ids_by_region))
        .collect::<Vec<_>>();
    let source_refs = regions
        .iter()
        .flat_map(|region| region.source_refs.iter().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    CortexIndex {
        id: "cortex-index:current".into(),
        schema_version: CORTEX_INDEX_SCHEMA_VERSION,
        corpus_hash: cortex_corpus_hash(memory, &artifact_ids),
        created_at,
        compiler: "imprint-memory-compiler".into(),
        source_refs,
        artifact_ids,
        regions,
        compatibility_map,
    }
}

fn library_map_artifact(regions: &[&Region], created_at: u64) -> BrainArtifact {
    let source_refs = regions
        .iter()
        .flat_map(|region| region.chunk_ids.iter().take(4))
        .map(|id| format!("imprint://chunk/{id}"))
        .collect::<Vec<_>>();
    let body = format!(
        "Library cognitive map. Major regions: {}. Tiny local models should choose a region first, then search chunks and expand source anchors.",
        regions
            .iter()
            .map(|region| format!("{} ({})", region.label, region.summary))
            .collect::<Vec<_>>()
            .join("; ")
    );
    artifact(
        "brain-library:map".into(),
        BrainArtifactKind::LibraryMap,
        "Library cognitive map".into(),
        body,
        source_refs,
        created_at,
        84,
    )
}

fn region_card_artifact(
    memory: &PersistedMemory,
    region: &Region,
    created_at: u64,
) -> BrainArtifact {
    let sample_titles = region
        .chunk_ids
        .iter()
        .take(5)
        .filter_map(|chunk_id| memory.chunks.iter().find(|chunk| &chunk.id == chunk_id))
        .filter_map(|chunk| {
            memory
                .documents
                .iter()
                .find(|document| document.id == chunk.document_id)
                .map(|document| document.title.clone())
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let body = format!(
        "Compiled brain artifact for region '{}'. Routing hints: {}. Representative sources: {}. Use this as a compact map for tiny local models, then expand original source anchors before citing.",
        region.label,
        region.summary,
        if sample_titles.is_empty() { "none".into() } else { sample_titles.join("; ") }
    );
    artifact(
        format!("brain-region:{}", region.id),
        BrainArtifactKind::RegionCard,
        format!("Region card: {}", region.label),
        body,
        region
            .chunk_ids
            .iter()
            .take(12)
            .map(|id| format!("imprint://chunk/{id}"))
            .collect(),
        created_at,
        82,
    )
}

fn routing_rule_artifact(region: &Region, created_at: u64) -> BrainArtifact {
    let body = format!(
        "Routing rule for '{}': if a query matches '{}', search this region, open the best chunk, read neighbors, and expand an anchor before answering.",
        region.label, region.summary
    );
    artifact(
        format!("brain-routing:{}", region.id),
        BrainArtifactKind::RoutingRule,
        format!("Routing rule: {}", region.label),
        body,
        region
            .chunk_ids
            .iter()
            .take(12)
            .map(|id| format!("imprint://chunk/{id}"))
            .collect(),
        created_at,
        78,
    )
}

fn artifact(
    id: String,
    kind: BrainArtifactKind,
    title: String,
    body: String,
    source_refs: Vec<String>,
    created_at: u64,
    confidence: u8,
) -> BrainArtifact {
    BrainArtifact {
        id,
        kind,
        title,
        content_hash: stable_hash(&body),
        body,
        source_refs: source_refs.clone(),
        provenance: ProvenanceRecord {
            actor: "memory-compiler".into(),
            reason: "Compiled into a persistent cognitive map for tiny local models.".into(),
            created_at,
            source_refs,
        },
        confidence,
        created_at,
        updated_at: created_at,
    }
}

fn cortex_region_sketch(
    memory: &PersistedMemory,
    region: &Region,
    artifact_ids_by_region: &BTreeMap<String, Vec<String>>,
) -> CortexRegionSketch {
    let source_refs = region
        .chunk_ids
        .iter()
        .filter_map(|chunk_id| source_chunk_ref(memory, chunk_id))
        .take(12)
        .collect::<Vec<_>>();
    let route_examples = representative_terms(&region.summary)
        .into_iter()
        .take(4)
        .map(|term| format!("{term} -> search region {}", region.id))
        .collect::<Vec<_>>();
    CortexRegionSketch {
        region_id: region.id.clone(),
        label: region.label.clone(),
        summary: region.summary.clone(),
        source_refs,
        artifact_ids: artifact_ids_by_region
            .get(&region.id)
            .cloned()
            .unwrap_or_default(),
        route_examples,
    }
}

fn artifacts_by_region(artifacts: &[BrainArtifact]) -> BTreeMap<String, Vec<String>> {
    let mut by_region = BTreeMap::<String, Vec<String>>::new();
    for artifact in artifacts {
        let Some(region_id) = artifact
            .id
            .strip_prefix("brain-region:")
            .or_else(|| artifact.id.strip_prefix("brain-routing:"))
        else {
            continue;
        };
        by_region
            .entry(region_id.to_string())
            .or_default()
            .push(artifact.id.clone());
    }
    by_region
}

fn source_chunk_ref(memory: &PersistedMemory, chunk_id: &str) -> Option<String> {
    let chunk = memory.chunks.iter().find(|chunk| chunk.id == chunk_id)?;
    if matches!(
        chunk.metadata.get("source_type").map(String::as_str),
        Some("derived_memory") | Some("brain_artifact")
    ) {
        return None;
    }
    Some(format!("imprint://chunk/{chunk_id}"))
}

fn representative_terms(summary: &str) -> Vec<String> {
    let mut terms = summary
        .split(|character: char| !character.is_alphanumeric())
        .map(str::trim)
        .filter(|term| term.len() >= 4)
        .map(|term| term.to_lowercase())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    if terms.is_empty() {
        terms.push("memory".into());
    }
    terms
}

fn cortex_corpus_hash(memory: &PersistedMemory, artifact_ids: &[String]) -> String {
    let mut parts = Vec::new();
    for document in &memory.documents {
        if is_source_document(document) {
            parts.push(format!(
                "doc:{}:{}",
                document.id,
                document
                    .content_hash
                    .as_deref()
                    .unwrap_or(document.text.as_str())
            ));
        }
    }
    for chunk in &memory.chunks {
        if !matches!(
            chunk.metadata.get("source_type").map(String::as_str),
            Some("derived_memory") | Some("brain_artifact")
        ) {
            parts.push(format!(
                "chunk:{}:{}:{}",
                chunk.id,
                chunk.region_id,
                chunk.embedding_text_hash.as_deref().unwrap_or("")
            ));
        }
    }
    for artifact_id in artifact_ids {
        parts.push(format!("artifact:{artifact_id}"));
    }
    parts.sort();
    stable_hash(&parts.join("\n"))
}

fn is_source_document(document: &Document) -> bool {
    !matches!(
        document.metadata.get("source_type").map(String::as_str),
        Some("derived_memory") | Some("brain_artifact")
    )
}

fn region_has_source_chunks(memory: &PersistedMemory, region: &Region) -> bool {
    region.chunk_ids.iter().any(|chunk_id| {
        memory
            .chunks
            .iter()
            .find(|chunk| &chunk.id == chunk_id)
            .is_some_and(|chunk| {
                !matches!(
                    chunk.metadata.get("source_type").map(String::as_str),
                    Some("derived_memory") | Some("brain_artifact")
                )
            })
    })
}

fn stable_hash(text: &str) -> String {
    let mut hash = 14695981039346656037u64;
    for byte in text.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    format!("{hash:016x}")
}
