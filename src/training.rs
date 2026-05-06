use crate::types::{BrainArtifact, BrainArtifactKind, CortexAdapterState};
use anyhow::Context;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

const TRAINING_SCHEMA_VERSION: u32 = 1;
const DEFAULT_ADAPTER_ITERS: usize = 100;

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

pub fn training_source_hash(training_dir: &Path) -> anyhow::Result<String> {
    let mut files = jsonl_files(training_dir)?;
    files.sort();
    let mut digest = Sha256::new();
    for path in files {
        if let Some(name) = path.file_name().and_then(|name| name.to_str()) {
            digest.update(name.as_bytes());
            digest.update(b"\0");
            for line in fs::read_to_string(&path)
                .with_context(|| format!("reading {}", path.display()))?
                .lines()
                .filter(|line| !line.trim().is_empty())
            {
                let mut value: serde_json::Value = serde_json::from_str(line)
                    .with_context(|| format!("parsing {}", path.display()))?;
                if let Some(object) = value.as_object_mut() {
                    object.remove("created_at");
                }
                digest.update(serde_json::to_string(&value)?.as_bytes());
                digest.update(b"\n");
            }
            digest.update(b"\0");
        }
    }
    Ok(format!("{:x}", digest.finalize()))
}

pub fn prepare_cortex_adapter_dataset(
    store_root: &Path,
    base_model: &str,
    source_dataset_hash: &str,
) -> anyhow::Result<Option<PathBuf>> {
    let training_dir = store_root.join("training");
    let adapter_dir = store_root.join("adapters").join(format!(
        "prepared-{}",
        &source_dataset_hash[..12.min(source_dataset_hash.len())]
    ));
    let data_dir = adapter_dir.join("mlx-data");
    fs::create_dir_all(&data_dir)
        .with_context(|| format!("creating adapter data directory {}", data_dir.display()))?;

    let train_records = mlx_records_for_split(&training_dir, "train")?;
    if train_records.is_empty() {
        return Ok(None);
    }
    let mut valid_records = mlx_records_for_split(&training_dir, "eval")?;
    if valid_records.is_empty() {
        valid_records = train_records.clone();
    }

    write_jsonl(&data_dir.join("train.jsonl"), &train_records)?;
    write_jsonl(&data_dir.join("valid.jsonl"), &valid_records)?;
    write_jsonl(&data_dir.join("test.jsonl"), &valid_records)?;
    let prepared_dataset_hash = training_source_hash(&data_dir)?;
    let manifest_path = adapter_dir.join("adapter_manifest.json");
    let manifest = json!({
        "status": "prepared",
        "base_model": base_model,
        "dataset": data_dir.display().to_string(),
        "dataset_hash": prepared_dataset_hash,
        "source_dataset": training_dir.display().to_string(),
        "source_dataset_hash": source_dataset_hash,
        "adapter_path": adapter_dir.display().to_string(),
        "iters": DEFAULT_ADAPTER_ITERS,
        "train_records": train_records.len(),
        "valid_records": valid_records.len(),
        "test_records": valid_records.len(),
    });
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).context("serializing adapter manifest")?,
    )
    .with_context(|| format!("writing {}", manifest_path.display()))?;
    Ok(Some(manifest_path))
}

pub fn read_cortex_adapter_state(
    store_root: &Path,
    current_source_dataset_hash: String,
    checked_at: u64,
) -> anyhow::Result<CortexAdapterState> {
    let Some(manifest_path) = newest_adapter_manifest(&store_root.join("adapters"))? else {
        return Ok(CortexAdapterState {
            freshness: "missing".into(),
            status: "missing".into(),
            reason: Some("No adapter_manifest.json found under store adapters directory".into()),
            base_model: None,
            adapter_path: None,
            manifest_path: None,
            source_dataset_hash: None,
            current_source_dataset_hash,
            prepared_dataset_hash: None,
            train_records: None,
            valid_records: None,
            test_records: None,
            iters: None,
            checked_at,
        });
    };

    let raw = fs::read_to_string(&manifest_path)
        .with_context(|| format!("reading {}", manifest_path.display()))?;
    let manifest: serde_json::Value = serde_json::from_str(&raw)
        .with_context(|| format!("parsing {}", manifest_path.display()))?;
    let source_dataset_hash = string_field(&manifest, "source_dataset_hash");
    let status = string_field(&manifest, "status").unwrap_or_else(|| "unknown".into());
    let freshness = match source_dataset_hash.as_deref() {
        Some(hash) if hash == current_source_dataset_hash => "fresh",
        Some(_) => "stale",
        None => "unknown",
    }
    .to_string();
    let reason = match freshness.as_str() {
        "fresh" => Some("Adapter source dataset hash matches current training exports".into()),
        "stale" => Some("Adapter source dataset hash does not match current training exports".into()),
        _ => Some("Manifest does not include source_dataset_hash; rerun train_mlx_lora.py to make freshness inspectable".into()),
    };

    Ok(CortexAdapterState {
        freshness,
        status,
        reason,
        base_model: string_field(&manifest, "base_model"),
        adapter_path: string_field(&manifest, "adapter_path"),
        manifest_path: Some(manifest_path.display().to_string()),
        source_dataset_hash,
        current_source_dataset_hash,
        prepared_dataset_hash: string_field(&manifest, "dataset_hash"),
        train_records: usize_field(&manifest, "train_records"),
        valid_records: usize_field(&manifest, "valid_records"),
        test_records: usize_field(&manifest, "test_records"),
        iters: usize_field(&manifest, "iters"),
        checked_at,
    })
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

fn mlx_records_for_split(
    training_dir: &Path,
    split: &str,
) -> anyhow::Result<Vec<serde_json::Value>> {
    let suffix = format!(".{split}.jsonl");
    let mut files = jsonl_files(training_dir)?
        .into_iter()
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with(&suffix))
        })
        .collect::<Vec<_>>();
    files.sort();

    let mut records = Vec::new();
    for path in files {
        for line in fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?
            .lines()
            .filter(|line| !line.trim().is_empty())
        {
            let record: serde_json::Value = serde_json::from_str(line)
                .with_context(|| format!("parsing {}", path.display()))?;
            records.push(json!({
                "prompt": format!(
                    "You are imprint's tiny-model memory cortex. Task: {}\n\n{}",
                    record.get("task").and_then(|value| value.as_str()).unwrap_or("memory"),
                    record.get("input").and_then(|value| value.as_str()).unwrap_or("")
                ),
                "completion": record
                    .get("target")
                    .and_then(|value| value.as_str())
                    .unwrap_or(""),
            }));
        }
    }
    Ok(records)
}

fn write_jsonl(path: &Path, records: &[serde_json::Value]) -> anyhow::Result<()> {
    let lines = records
        .iter()
        .map(serde_json::to_string)
        .collect::<Result<Vec<_>, _>>()
        .context("serializing prepared JSONL")?;
    fs::write(path, lines.join("\n")).with_context(|| format!("writing {}", path.display()))
}

fn jsonl_files(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut files = Vec::new();
    for entry in fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let path = entry?.path();
        if path.extension().and_then(|extension| extension.to_str()) == Some("jsonl") {
            files.push(path);
        }
    }
    Ok(files)
}

fn newest_adapter_manifest(adapters_dir: &Path) -> anyhow::Result<Option<PathBuf>> {
    let mut manifests = Vec::new();
    collect_adapter_manifests(adapters_dir, &mut manifests)?;
    manifests.sort_by_key(|path| {
        fs::metadata(path)
            .and_then(|metadata| metadata.modified())
            .ok()
    });
    Ok(manifests.pop())
}

fn collect_adapter_manifests(dir: &Path, manifests: &mut Vec<PathBuf>) -> anyhow::Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let path = entry?.path();
        if path.is_dir() {
            collect_adapter_manifests(&path, manifests)?;
        } else if path.file_name().and_then(|name| name.to_str()) == Some("adapter_manifest.json") {
            manifests.push(path);
        }
    }
    Ok(())
}

fn string_field(value: &serde_json::Value, key: &str) -> Option<String> {
    value.get(key)?.as_str().map(ToString::to_string)
}

fn usize_field(value: &serde_json::Value, key: &str) -> Option<usize> {
    value
        .get(key)?
        .as_u64()
        .and_then(|number| number.try_into().ok())
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
