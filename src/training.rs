use crate::store::FileMemoryStore;
use crate::types::{BrainArtifact, BrainArtifactKind, CortexAdapterJob, CortexAdapterState};
use anyhow::Context;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const TRAINING_SCHEMA_VERSION: u32 = 1;
const DEFAULT_ADAPTER_ITERS: usize = 100;
const DEFAULT_ADAPTER_TRAINING_TIMEOUT_MILLIS: u64 = 30 * 60 * 1000;

#[derive(Debug, Clone)]
pub struct CortexAdapterTrainingOptions {
    pub dry_run: bool,
    pub timeout_millis: u64,
    pub iters: usize,
    pub python: Option<String>,
    pub script_path: Option<PathBuf>,
    pub output_dir: Option<PathBuf>,
    pub log_path: Option<PathBuf>,
}

impl Default for CortexAdapterTrainingOptions {
    fn default() -> Self {
        Self {
            dry_run: false,
            timeout_millis: DEFAULT_ADAPTER_TRAINING_TIMEOUT_MILLIS,
            iters: DEFAULT_ADAPTER_ITERS,
            python: None,
            script_path: None,
            output_dir: None,
            log_path: None,
        }
    }
}

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
        "dataset_hash": prepared_dataset_hash.clone(),
        "prepared_dataset_hash": prepared_dataset_hash,
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

pub fn run_cortex_adapter_training_job(
    store_root: &Path,
    base_model: &str,
    source_dataset_hash: &str,
    options: CortexAdapterTrainingOptions,
) -> anyhow::Result<CortexAdapterJob> {
    let store = FileMemoryStore::new(store_root);
    let queued_at = now_millis();
    let hash_prefix = &source_dataset_hash[..12.min(source_dataset_hash.len())];
    let output_dir = options.output_dir.clone().unwrap_or_else(|| {
        store_root
            .join("adapters")
            .join(format!("trained-{hash_prefix}"))
    });
    let log_path = options.log_path.clone().unwrap_or_else(|| {
        store_root
            .join("adapters")
            .join(format!("train-{hash_prefix}.log"))
    });
    fs::create_dir_all(output_dir.parent().unwrap_or(store_root))
        .with_context(|| format!("creating adapter directory {}", output_dir.display()))?;
    if let Some(parent) = log_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating adapter log directory {}", parent.display()))?;
    }

    let training_dir = store_root.join("training");
    let current_state =
        read_cortex_adapter_state(store_root, source_dataset_hash.to_string(), queued_at)?;
    let script_path = options
        .script_path
        .clone()
        .unwrap_or_else(default_trainer_script);
    let python = options
        .python
        .clone()
        .or_else(|| std::env::var("PYTHON").ok())
        .unwrap_or_else(|| "python3".into());
    let mut command = vec![
        python,
        script_path.display().to_string(),
        "--model".into(),
        base_model.into(),
        "--dataset".into(),
        training_dir.display().to_string(),
        "--output".into(),
        output_dir.display().to_string(),
        "--iters".into(),
        options.iters.to_string(),
    ];
    if options.dry_run {
        command.push("--dry-run".into());
    }

    let mut payload = BTreeMap::new();
    payload.insert("dataset_dir".into(), training_dir.display().to_string());
    payload.insert("dry_run".into(), options.dry_run.to_string());
    payload.insert("timeout_millis".into(), options.timeout_millis.to_string());

    let mut job = CortexAdapterJob {
        id: format!("adapter-job:{hash_prefix}:{queued_at}"),
        status: "queued".into(),
        source_dataset_hash: source_dataset_hash.into(),
        prepared_dataset_hash: current_state.prepared_dataset_hash.clone(),
        base_model: Some(base_model.into()),
        adapter_output_path: output_dir.display().to_string(),
        manifest_path: None,
        train_records: current_state.train_records,
        valid_records: current_state.valid_records,
        test_records: current_state.test_records,
        iters: Some(options.iters),
        command,
        log_path: Some(log_path.display().to_string()),
        failure_reason: None,
        payload,
        created_at: queued_at,
        updated_at: queued_at,
        started_at: None,
        finished_at: None,
    };
    store.upsert_cortex_adapter_job(&job)?;

    let started_at = now_millis();
    job.status = "training".into();
    job.updated_at = started_at;
    job.started_at = Some(started_at);
    store.upsert_cortex_adapter_job(&job)?;

    let result = execute_training_command(&job.command, options.timeout_millis);
    let finished_at = now_millis();
    write_training_log(&log_path, &job.command, started_at, finished_at, &result)?;
    job.finished_at = Some(finished_at);
    job.updated_at = finished_at;

    match result {
        TrainingCommandResult::Completed {
            code: 0,
            stdout: _,
            stderr: _,
        } => {
            let manifest_path = output_dir.join("adapter_manifest.json");
            let manifest = read_manifest(&manifest_path)?;
            job.status = string_field(&manifest, "status").unwrap_or_else(|| "trained".into());
            job.manifest_path = Some(manifest_path.display().to_string());
            job.prepared_dataset_hash = string_field(&manifest, "dataset_hash");
            job.train_records = usize_field(&manifest, "train_records");
            job.valid_records = usize_field(&manifest, "valid_records");
            job.test_records = usize_field(&manifest, "test_records");
            job.iters = usize_field(&manifest, "iters").or(job.iters);
            let adapter_state = read_cortex_adapter_state(
                store_root,
                source_dataset_hash.to_string(),
                finished_at,
            )?;
            store.save_cortex_adapter_state(&adapter_state)?;
        }
        TrainingCommandResult::Completed { code, stderr, .. } => {
            job.status = "failed".into();
            job.failure_reason = Some(format!(
                "trainer exited with status {code}: {}",
                stderr.trim()
            ));
        }
        TrainingCommandResult::TimedOut { timeout_millis, .. } => {
            job.status = "failed".into();
            job.failure_reason = Some(format!(
                "trainer exceeded timeout of {timeout_millis}ms and was stopped"
            ));
        }
        TrainingCommandResult::SpawnFailed { error } => {
            job.status = "failed".into();
            job.failure_reason = Some(error);
        }
    }
    store.upsert_cortex_adapter_job(&job)?;
    Ok(job)
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
            data_freshness: "missing".into(),
            training_status: "missing".into(),
            activation_status: "inactive".into(),
            base_model: None,
            adapter_path: None,
            manifest_path: None,
            source_dataset_hash: None,
            current_source_dataset_hash,
            trained_source_dataset_hash: None,
            active_adapter_hash: None,
            prepared_dataset_hash: None,
            eval_score: None,
            failure_reason: None,
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
    let training_status = adapter_training_status(&status);
    let activation_status = adapter_activation_status(&status, &manifest);
    let trained_source_dataset_hash = string_field(&manifest, "trained_source_dataset_hash")
        .or_else(|| match status.as_str() {
            "trained" | "active" => source_dataset_hash.clone(),
            _ => None,
        });
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
        freshness: freshness.clone(),
        status,
        reason,
        data_freshness: freshness,
        training_status,
        activation_status,
        base_model: string_field(&manifest, "base_model"),
        adapter_path: string_field(&manifest, "adapter_path"),
        manifest_path: Some(manifest_path.display().to_string()),
        source_dataset_hash,
        current_source_dataset_hash,
        trained_source_dataset_hash,
        active_adapter_hash: string_field(&manifest, "active_adapter_hash")
            .or_else(|| string_field(&manifest, "adapter_hash")),
        prepared_dataset_hash: string_field(&manifest, "prepared_dataset_hash")
            .or_else(|| string_field(&manifest, "dataset_hash")),
        eval_score: f64_field(&manifest, "eval_score"),
        failure_reason: string_field(&manifest, "failure_reason"),
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

fn default_trainer_script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("training")
        .join("train_mlx_lora.py")
}

#[derive(Debug)]
enum TrainingCommandResult {
    Completed {
        code: i32,
        stdout: String,
        stderr: String,
    },
    TimedOut {
        timeout_millis: u64,
        stdout: String,
        stderr: String,
    },
    SpawnFailed {
        error: String,
    },
}

fn execute_training_command(command: &[String], timeout_millis: u64) -> TrainingCommandResult {
    if command.is_empty() {
        return TrainingCommandResult::SpawnFailed {
            error: "empty trainer command".into(),
        };
    }
    let mut child = match Command::new(&command[0])
        .args(&command[1..])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            return TrainingCommandResult::SpawnFailed {
                error: error.to_string(),
            };
        }
    };

    let timeout = Duration::from_millis(timeout_millis);
    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_status)) => match child.wait_with_output() {
                Ok(output) => {
                    return TrainingCommandResult::Completed {
                        code: output.status.code().unwrap_or(-1),
                        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                    };
                }
                Err(error) => {
                    return TrainingCommandResult::SpawnFailed {
                        error: error.to_string(),
                    };
                }
            },
            Ok(None) if started.elapsed() >= timeout => {
                let _ = child.kill();
                return match child.wait_with_output() {
                    Ok(output) => TrainingCommandResult::TimedOut {
                        timeout_millis,
                        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                    },
                    Err(error) => TrainingCommandResult::SpawnFailed {
                        error: error.to_string(),
                    },
                };
            }
            Ok(None) => thread::sleep(Duration::from_millis(50)),
            Err(error) => {
                return TrainingCommandResult::SpawnFailed {
                    error: error.to_string(),
                };
            }
        }
    }
}

fn write_training_log(
    path: &Path,
    command: &[String],
    started_at: u64,
    finished_at: u64,
    result: &TrainingCommandResult,
) -> anyhow::Result<()> {
    let payload = match result {
        TrainingCommandResult::Completed {
            code,
            stdout,
            stderr,
        } => json!({
            "status": "completed",
            "exit_code": code,
            "command": command,
            "started_at": started_at,
            "finished_at": finished_at,
            "stdout": stdout,
            "stderr": stderr,
        }),
        TrainingCommandResult::TimedOut {
            timeout_millis,
            stdout,
            stderr,
        } => json!({
            "status": "timed_out",
            "timeout_millis": timeout_millis,
            "command": command,
            "started_at": started_at,
            "finished_at": finished_at,
            "stdout": stdout,
            "stderr": stderr,
        }),
        TrainingCommandResult::SpawnFailed { error } => json!({
            "status": "spawn_failed",
            "command": command,
            "started_at": started_at,
            "finished_at": finished_at,
            "error": error,
        }),
    };
    fs::write(path, serde_json::to_string_pretty(&payload)?)
        .with_context(|| format!("writing trainer log {}", path.display()))
}

fn read_manifest(path: &Path) -> anyhow::Result<serde_json::Value> {
    let raw = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&raw)
        .with_context(|| format!("parsing adapter manifest {}", path.display()))
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

fn f64_field(value: &serde_json::Value, key: &str) -> Option<f64> {
    value.get(key)?.as_f64()
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or_default()
}

fn adapter_training_status(status: &str) -> String {
    match status {
        "missing" => "missing",
        "prepared" => "prepared",
        "queued" => "queued",
        "training" => "training",
        "trained" | "active" => "trained",
        "eval_failed" => "eval_failed",
        "failed" => "failed",
        _ => "unknown",
    }
    .into()
}

fn adapter_activation_status(status: &str, manifest: &serde_json::Value) -> String {
    if let Some(value) = string_field(manifest, "activation_status") {
        return value;
    }
    match status {
        "active" => "active",
        _ => "inactive",
    }
    .into()
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
