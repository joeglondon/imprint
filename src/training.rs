use crate::types::{BrainArtifact, BrainArtifactKind};
use anyhow::Context;
use serde_json::json;
use std::fs;
use std::path::{Path, PathBuf};

const TRAINING_SCHEMA_VERSION: u32 = 1;

pub fn write_training_exports(
    store_root: &Path,
    artifacts: &[BrainArtifact],
) -> anyhow::Result<Vec<String>> {
    let training_dir = store_root.join("training");
    fs::create_dir_all(&training_dir).with_context(|| {
        format!(
            "creating training export directory {}",
            training_dir.display()
        )
    })?;
    let tasks = [
        "route_region",
        "choose_tool",
        "critique_evidence",
        "collaboration_pattern",
    ];
    let mut files = Vec::new();
    for task in tasks {
        files.push(write_split(&training_dir, task, "train", artifacts)?);
        files.push(write_split(&training_dir, task, "eval", artifacts)?);
    }
    Ok(files
        .into_iter()
        .map(|path| path.display().to_string())
        .collect())
}

fn write_split(
    training_dir: &Path,
    task: &str,
    split: &str,
    artifacts: &[BrainArtifact],
) -> anyhow::Result<PathBuf> {
    let path = training_dir.join(format!("{task}.{split}.jsonl"));
    let mut lines = Vec::new();
    for artifact in artifacts {
        lines.push(training_record(task, split, artifact).to_string());
    }
    fs::write(&path, lines.join("\n")).with_context(|| format!("writing {}", path.display()))?;
    Ok(path)
}

fn training_record(task: &str, split: &str, artifact: &BrainArtifact) -> serde_json::Value {
    let target = match task {
        "route_region" => format!("Route to {}", artifact.title),
        "choose_tool" => "Use memory_search, then memory_expand before answering".to_string(),
        "critique_evidence" => {
            "Check source anchors and ask for another round if evidence is weak".to_string()
        }
        "collaboration_pattern" => {
            "Planner gathers, critic checks, response model answers from bounded evidence"
                .to_string()
        }
        _ => "Use source-grounded memory".to_string(),
    };
    let split_prefix = if split == "eval" {
        "Held-out evaluation: "
    } else {
        ""
    };
    json!({
        "schema_version": TRAINING_SCHEMA_VERSION,
        "task": task,
        "split": split,
        "id": format!("{task}:{split}:{}", artifact.id),
        "input": format!("{split_prefix}{}", artifact.body),
        "target": target,
        "artifact_ids": [artifact.id.clone()],
        "artifact_kind": artifact_kind_label(&artifact.kind),
        "source_refs": artifact.source_refs,
        "created_at": artifact.created_at,
    })
}

fn artifact_kind_label(kind: &BrainArtifactKind) -> &'static str {
    match kind {
        BrainArtifactKind::LibraryMap => "library_map",
        BrainArtifactKind::RegionCard => "region_card",
        BrainArtifactKind::EntityCard => "entity_card",
        BrainArtifactKind::ProjectCard => "project_card",
        BrainArtifactKind::PreferenceCard => "preference_card",
        BrainArtifactKind::RoutingRule => "routing_rule",
        BrainArtifactKind::ToolPattern => "tool_pattern",
        BrainArtifactKind::CritiquePattern => "critique_pattern",
    }
}
