use crate::types::*;
use std::collections::BTreeSet;

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
