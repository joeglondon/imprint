use crate::store::FileMemoryStore;
use crate::types::{BrainArtifact, BrainArtifactKind, CortexAdapterJob, CortexAdapterState};
use anyhow::{anyhow, Context};
use serde::Serialize;
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
pub const DEFAULT_ADAPTER_ACTIVATION_MIN_SCORE: f64 = 0.8;

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

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct CortexAdapterEvalTaskScore {
    pub records: usize,
    pub passed: usize,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct CortexAdapterEvalReport {
    pub status: String,
    pub passed: bool,
    pub score: f64,
    pub minimum_score: f64,
    pub records: usize,
    pub source_ref_records: usize,
    pub task_scores: BTreeMap<String, CortexAdapterEvalTaskScore>,
    pub gates: BTreeMap<String, bool>,
    pub manifest_path: Option<String>,
    pub adapter_path: Option<String>,
    pub source_dataset_hash: String,
    pub evaluated_at: u64,
    pub failure_reason: Option<String>,
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
    let job =
        queue_cortex_adapter_training_job(store_root, base_model, source_dataset_hash, options)?;
    run_queued_cortex_adapter_training_job(store_root, &job.id)
}

pub fn queue_cortex_adapter_training_job(
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

    let job = CortexAdapterJob {
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
    Ok(job)
}

pub fn run_queued_cortex_adapter_training_job(
    store_root: &Path,
    job_id: &str,
) -> anyhow::Result<CortexAdapterJob> {
    let store = FileMemoryStore::new(store_root);
    let mut job = store
        .load_cortex_adapter_job(job_id)?
        .with_context(|| format!("cortex adapter job {job_id} not found"))?;
    if job.status == "cancelled" {
        return Ok(job);
    }
    if job.status != "queued" {
        return Err(anyhow!(
            "cortex adapter job {} is {}, not queued",
            job.id,
            job.status
        ));
    }
    let output_dir = PathBuf::from(&job.adapter_output_path);
    let log_path = job
        .log_path
        .as_deref()
        .map(PathBuf::from)
        .unwrap_or_else(|| output_dir.with_extension("log"));
    let timeout_millis = job
        .payload
        .get("timeout_millis")
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(DEFAULT_ADAPTER_TRAINING_TIMEOUT_MILLIS);

    let started_at = now_millis();
    job.status = "training".into();
    job.updated_at = started_at;
    job.started_at = Some(started_at);
    store.upsert_cortex_adapter_job(&job)?;

    let running_job_id = job.id.clone();
    let result = execute_training_command(&job.command, timeout_millis, || {
        cortex_adapter_job_is_cancelled(store_root, &running_job_id)
    });
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
                job.source_dataset_hash.clone(),
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
        TrainingCommandResult::Cancelled { .. } => {
            job.status = "cancelled".into();
            job.failure_reason = Some("trainer cancellation requested".into());
        }
        TrainingCommandResult::SpawnFailed { error } => {
            job.status = "failed".into();
            job.failure_reason = Some(error);
        }
    }
    store.upsert_cortex_adapter_job(&job)?;
    Ok(job)
}

pub fn cancel_cortex_adapter_training_job(
    store_root: &Path,
    job_id: &str,
) -> anyhow::Result<CortexAdapterJob> {
    let store = FileMemoryStore::new(store_root);
    let mut job = store
        .load_cortex_adapter_job(job_id)?
        .with_context(|| format!("cortex adapter job {job_id} not found"))?;
    match job.status.as_str() {
        "queued" | "training" => {
            let cancelled_at = now_millis();
            job.status = "cancelled".into();
            job.failure_reason = Some("cancellation requested".into());
            job.updated_at = cancelled_at;
            if job.started_at.is_none() {
                job.finished_at = Some(cancelled_at);
            }
            store.upsert_cortex_adapter_job(&job)?;
            Ok(job)
        }
        "cancelled" => Ok(job),
        status => Err(anyhow!(
            "cortex adapter job {job_id} is {status}, not cancellable"
        )),
    }
}

pub fn retry_cortex_adapter_training_job(
    store_root: &Path,
    job_id: &str,
    run_now: bool,
) -> anyhow::Result<CortexAdapterJob> {
    let store = FileMemoryStore::new(store_root);
    let job = store
        .load_cortex_adapter_job(job_id)?
        .with_context(|| format!("cortex adapter job {job_id} not found"))?;
    if !matches!(job.status.as_str(), "failed" | "cancelled" | "eval_failed") {
        return Err(anyhow!(
            "cortex adapter job {} is {}, not retryable",
            job.id,
            job.status
        ));
    }

    let retried_at = now_millis();
    let hash_prefix = &job.source_dataset_hash[..12.min(job.source_dataset_hash.len())];
    let output_dir = PathBuf::from(&job.adapter_output_path)
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| store_root.join("adapters"))
        .join(format!("retry-{hash_prefix}-{retried_at}"));
    let log_path = PathBuf::from(&job.log_path.clone().unwrap_or_else(|| {
        store_root
            .join("adapters")
            .join(format!("retry-{hash_prefix}-{retried_at}.log"))
            .display()
            .to_string()
    }))
    .parent()
    .map(Path::to_path_buf)
    .unwrap_or_else(|| store_root.join("adapters"))
    .join(format!("retry-{hash_prefix}-{retried_at}.log"));
    fs::create_dir_all(output_dir.parent().unwrap_or(store_root))
        .with_context(|| format!("creating adapter retry directory {}", output_dir.display()))?;
    if let Some(parent) = log_path.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!("creating adapter retry log directory {}", parent.display())
        })?;
    }

    let mut payload = job.payload.clone();
    payload.insert("retry_of".into(), job.id.clone());
    payload.insert("retried_at".into(), retried_at.to_string());
    let mut command = job.command.clone();
    set_command_arg(&mut command, "--output", output_dir.display().to_string());

    let retry = CortexAdapterJob {
        id: format!("adapter-job:{hash_prefix}:{retried_at}"),
        status: "queued".into(),
        source_dataset_hash: job.source_dataset_hash,
        prepared_dataset_hash: job.prepared_dataset_hash,
        base_model: job.base_model,
        adapter_output_path: output_dir.display().to_string(),
        manifest_path: None,
        train_records: job.train_records,
        valid_records: job.valid_records,
        test_records: job.test_records,
        iters: job.iters,
        command,
        log_path: Some(log_path.display().to_string()),
        failure_reason: None,
        payload,
        created_at: retried_at,
        updated_at: retried_at,
        started_at: None,
        finished_at: None,
    };
    store.upsert_cortex_adapter_job(&retry)?;
    if run_now {
        run_queued_cortex_adapter_training_job(store_root, &retry.id)
    } else {
        Ok(retry)
    }
}

pub fn evaluate_cortex_adapter(
    store_root: &Path,
    current_source_dataset_hash: &str,
    minimum_score: f64,
) -> anyhow::Result<CortexAdapterEvalReport> {
    let evaluated_at = now_millis();
    let Some(manifest_path) = newest_adapter_manifest_matching(
        &store_root.join("adapters"),
        &["trained", "active"],
        Some(current_source_dataset_hash),
    )?
    else {
        return Ok(CortexAdapterEvalReport {
            status: "missing".into(),
            passed: false,
            score: 0.0,
            minimum_score,
            records: 0,
            source_ref_records: 0,
            task_scores: BTreeMap::new(),
            gates: BTreeMap::from([
                ("trained_adapter_manifest".into(), false),
                ("overall_score".into(), false),
            ]),
            manifest_path: None,
            adapter_path: None,
            source_dataset_hash: current_source_dataset_hash.into(),
            evaluated_at,
            failure_reason: Some(
                "No trained adapter manifest matches the current source dataset hash".into(),
            ),
        });
    };

    let manifest = read_manifest(&manifest_path)?;
    let status = string_field(&manifest, "status").unwrap_or_else(|| "unknown".into());
    let adapter_path = string_field(&manifest, "adapter_path");
    let records = training_records_for_split(&store_root.join("training"), "eval")?;
    let mut task_counts: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut source_ref_records = 0;
    for record in &records {
        let task = record
            .get("task")
            .and_then(|value| value.as_str())
            .unwrap_or("unknown")
            .to_string();
        let passes = eval_record_passes_gate(record);
        let entry = task_counts.entry(task).or_insert((0, 0));
        entry.0 += 1;
        if passes {
            entry.1 += 1;
        }
        if record
            .get("source_refs")
            .and_then(|value| value.as_array())
            .is_some_and(|refs| !refs.is_empty())
        {
            source_ref_records += 1;
        }
    }

    let mut task_scores = BTreeMap::new();
    let mut passed_records = 0;
    for (task, (total, passed)) in task_counts {
        passed_records += passed;
        task_scores.insert(
            task,
            CortexAdapterEvalTaskScore {
                records: total,
                passed,
                score: ratio(passed, total),
            },
        );
    }
    let score = ratio(passed_records, records.len());
    let route_score = task_scores
        .get("route_region")
        .map(|task| task.score)
        .unwrap_or_default();
    let tool_score = task_scores
        .get("choose_tool")
        .map(|task| task.score)
        .unwrap_or_default();
    let critique_score = task_scores
        .get("critique_evidence")
        .map(|task| task.score)
        .unwrap_or_default();
    let source_refs_gate = !records.is_empty() && source_ref_records == records.len();
    let adapter_hash_gate = string_field(&manifest, "adapter_file_hash")
        .or_else(|| string_field(&manifest, "active_adapter_hash"))
        .is_some();
    let gates = BTreeMap::from([
        (
            "trained_adapter_manifest".into(),
            status == "trained" || status == "active",
        ),
        ("adapter_file_hash".into(), adapter_hash_gate),
        ("overall_score".into(), score >= minimum_score),
        ("route_region_behavior".into(), route_score >= minimum_score),
        (
            "source_expansion_behavior".into(),
            tool_score >= minimum_score,
        ),
        (
            "critique_evidence_behavior".into(),
            critique_score >= minimum_score,
        ),
        ("source_ref_boundary".into(), source_refs_gate),
    ]);
    let passed = gates.values().all(|value| *value);
    Ok(CortexAdapterEvalReport {
        status: if passed { "passed" } else { "failed" }.into(),
        passed,
        score,
        minimum_score,
        records: records.len(),
        source_ref_records,
        task_scores,
        gates,
        manifest_path: Some(manifest_path.display().to_string()),
        adapter_path,
        source_dataset_hash: current_source_dataset_hash.into(),
        evaluated_at,
        failure_reason: (!passed).then(|| {
            "Adapter did not satisfy all activation gates; leaving activation unchanged".into()
        }),
    })
}

pub fn activate_cortex_adapter(
    store_root: &Path,
    current_source_dataset_hash: &str,
    minimum_score: f64,
) -> anyhow::Result<(CortexAdapterEvalReport, CortexAdapterState)> {
    let report = evaluate_cortex_adapter(store_root, current_source_dataset_hash, minimum_score)?;
    if !report.passed {
        return Err(anyhow!(
            "{}",
            report
                .failure_reason
                .clone()
                .unwrap_or_else(|| "adapter evaluation failed".into())
        ));
    }
    let manifest_path = report
        .manifest_path
        .as_deref()
        .map(PathBuf::from)
        .context("adapter eval report did not include manifest path")?;
    let mut manifest = read_manifest(&manifest_path)?;
    let adapter_path = string_field(&manifest, "adapter_path")
        .map(PathBuf::from)
        .context("trained adapter manifest does not include adapter_path")?;
    let adapter_hash = string_field(&manifest, "adapter_file_hash")
        .or_else(|| adapter_output_hash(&adapter_path).ok().flatten())
        .context("trained adapter has no adapter_file_hash and no adapter files to hash")?;
    let activated_at = now_millis();
    if let Some(object) = manifest.as_object_mut() {
        object.insert("status".into(), json!("active"));
        object.insert("activation_status".into(), json!("active"));
        object.insert("active_adapter_hash".into(), json!(adapter_hash));
        object.insert(
            "trained_source_dataset_hash".into(),
            json!(current_source_dataset_hash),
        );
        object.insert("eval_score".into(), json!(report.score));
        object.insert("eval_records".into(), json!(report.records));
        object.insert("evaluated_at".into(), json!(report.evaluated_at));
        object.insert("activated_at".into(), json!(activated_at));
    }
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).context("serializing active adapter manifest")?,
    )
    .with_context(|| format!("writing {}", manifest_path.display()))?;
    let state =
        read_cortex_adapter_state(store_root, current_source_dataset_hash.into(), activated_at)?;
    FileMemoryStore::new(store_root).save_cortex_adapter_state(&state)?;
    Ok((report, state))
}

pub fn read_cortex_adapter_state(
    store_root: &Path,
    current_source_dataset_hash: String,
    checked_at: u64,
) -> anyhow::Result<CortexAdapterState> {
    let manifests = adapter_manifests_by_modified(&store_root.join("adapters"))?;
    let Some(newest_snapshot) = manifests.last() else {
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

    let active_snapshot = manifests.iter().rev().find(|(_, manifest)| {
        let status = string_field(manifest, "status").unwrap_or_else(|| "unknown".into());
        adapter_activation_status(&status, manifest) == "active"
    });
    let current_snapshot = manifests.iter().rev().find(|(_, manifest)| {
        string_field(manifest, "source_dataset_hash").as_deref()
            == Some(current_source_dataset_hash.as_str())
    });
    let (manifest_path, manifest) = active_snapshot.unwrap_or(newest_snapshot);
    let source_dataset_hash = string_field(manifest, "source_dataset_hash");
    let status = string_field(manifest, "status").unwrap_or_else(|| "unknown".into());
    let activation_status = adapter_activation_status(&status, manifest);
    let current_manifest = current_snapshot.map(|(_, manifest)| manifest);
    let current_status = current_manifest
        .and_then(|manifest| string_field(manifest, "status"))
        .unwrap_or_else(|| status.clone());
    let training_status = if active_snapshot.is_some()
        && current_snapshot
            .map(|(path, _)| path != manifest_path)
            .unwrap_or(false)
    {
        adapter_training_status(&current_status)
    } else {
        adapter_training_status(&status)
    };
    let trained_source_dataset_hash = string_field(manifest, "trained_source_dataset_hash")
        .or_else(|| match status.as_str() {
            "trained" | "active" => source_dataset_hash.clone(),
            _ => None,
        });
    let data_freshness = current_snapshot
        .map(|_| "fresh".to_string())
        .unwrap_or_else(|| {
            match source_dataset_hash.as_deref() {
                Some(hash) if hash == current_source_dataset_hash => "fresh",
                Some(_) => "stale",
                None => "unknown",
            }
            .to_string()
        });
    let freshness = match source_dataset_hash.as_deref() {
        Some(hash) if hash == current_source_dataset_hash => "fresh",
        Some(_) => "stale",
        None => "unknown",
    }
    .to_string();
    let reason = match (activation_status.as_str(), freshness.as_str(), data_freshness.as_str()) {
        ("active", "stale", "fresh") => Some(
            "Current adapter data is fresh, while the active adapter was trained on an older source dataset".into(),
        ),
        ("active", "stale", _) => Some(
            "Active adapter source dataset hash does not match current training exports".into(),
        ),
        (_, "fresh", _) => Some("Adapter source dataset hash matches current training exports".into()),
        (_, "stale", _) => Some("Adapter source dataset hash does not match current training exports".into()),
        _ => Some("Manifest does not include source_dataset_hash; rerun train_mlx_lora.py to make freshness inspectable".into()),
    };
    let data_manifest = current_manifest.unwrap_or(manifest);

    Ok(CortexAdapterState {
        freshness: freshness.clone(),
        status,
        reason,
        data_freshness,
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
        prepared_dataset_hash: string_field(data_manifest, "prepared_dataset_hash")
            .or_else(|| string_field(data_manifest, "dataset_hash"))
            .or_else(|| string_field(&manifest, "prepared_dataset_hash"))
            .or_else(|| string_field(&manifest, "dataset_hash")),
        eval_score: f64_field(&manifest, "eval_score"),
        failure_reason: string_field(&manifest, "failure_reason"),
        train_records: usize_field(data_manifest, "train_records")
            .or_else(|| usize_field(&manifest, "train_records")),
        valid_records: usize_field(data_manifest, "valid_records")
            .or_else(|| usize_field(&manifest, "valid_records")),
        test_records: usize_field(data_manifest, "test_records")
            .or_else(|| usize_field(&manifest, "test_records")),
        iters: usize_field(data_manifest, "iters").or_else(|| usize_field(&manifest, "iters")),
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

fn training_records_for_split(
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
            records.push(
                serde_json::from_str(line)
                    .with_context(|| format!("parsing {}", path.display()))?,
            );
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
    Cancelled {
        stdout: String,
        stderr: String,
    },
    SpawnFailed {
        error: String,
    },
}

fn execute_training_command(
    command: &[String],
    timeout_millis: u64,
    mut is_cancelled: impl FnMut() -> bool,
) -> TrainingCommandResult {
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
        if is_cancelled() {
            let _ = child.kill();
            return match child.wait_with_output() {
                Ok(output) => TrainingCommandResult::Cancelled {
                    stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                    stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                },
                Err(error) => TrainingCommandResult::SpawnFailed {
                    error: error.to_string(),
                },
            };
        }
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
        TrainingCommandResult::Cancelled { stdout, stderr } => json!({
            "status": "cancelled",
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

fn cortex_adapter_job_is_cancelled(store_root: &Path, job_id: &str) -> bool {
    FileMemoryStore::new(store_root)
        .load_cortex_adapter_job(job_id)
        .ok()
        .flatten()
        .is_some_and(|job| job.status == "cancelled")
}

fn set_command_arg(command: &mut Vec<String>, name: &str, value: String) {
    if let Some(index) = command.iter().position(|part| part == name) {
        if let Some(slot) = command.get_mut(index + 1) {
            *slot = value;
        } else {
            command.push(value);
        }
    } else {
        command.push(name.into());
        command.push(value);
    }
}

fn read_manifest(path: &Path) -> anyhow::Result<serde_json::Value> {
    let raw = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&raw)
        .with_context(|| format!("parsing adapter manifest {}", path.display()))
}

fn adapter_manifests_by_modified(
    adapters_dir: &Path,
) -> anyhow::Result<Vec<(PathBuf, serde_json::Value)>> {
    let mut paths = Vec::new();
    collect_adapter_manifests(adapters_dir, &mut paths)?;
    paths.sort_by_key(|path| {
        fs::metadata(path)
            .and_then(|metadata| metadata.modified())
            .ok()
    });
    paths
        .into_iter()
        .map(|path| {
            let manifest = read_manifest(&path)?;
            Ok((path, manifest))
        })
        .collect()
}

fn newest_adapter_manifest_matching(
    adapters_dir: &Path,
    statuses: &[&str],
    source_dataset_hash: Option<&str>,
) -> anyhow::Result<Option<PathBuf>> {
    let mut manifests = Vec::new();
    collect_adapter_manifests(adapters_dir, &mut manifests)?;
    let mut eligible = Vec::new();
    for path in manifests {
        let manifest = read_manifest(&path)?;
        let status = string_field(&manifest, "status").unwrap_or_else(|| "unknown".into());
        let source_matches = source_dataset_hash.is_none_or(|expected| {
            string_field(&manifest, "source_dataset_hash").as_deref() == Some(expected)
        });
        if statuses.contains(&status.as_str()) && source_matches {
            eligible.push(path);
        }
    }
    eligible.sort_by_key(|path| {
        fs::metadata(path)
            .and_then(|metadata| metadata.modified())
            .ok()
    });
    Ok(eligible.pop())
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

fn ratio(passed: usize, total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    ((passed as f64 / total as f64) * 10_000.0).round() / 10_000.0
}

fn eval_record_passes_gate(record: &serde_json::Value) -> bool {
    let task = record
        .get("task")
        .and_then(|value| value.as_str())
        .unwrap_or_default();
    let target = record
        .get("target")
        .and_then(|value| value.as_str())
        .unwrap_or_default()
        .to_lowercase();
    let input = record
        .get("input")
        .and_then(|value| value.as_str())
        .unwrap_or_default()
        .to_lowercase();
    match task {
        "route_region" => target.starts_with("route to ") && !input.trim().is_empty(),
        "choose_tool" => target.contains("memory_search") && target.contains("memory_expand"),
        "critique_evidence" => target.contains("source anchors") && target.contains("weak"),
        "collaboration_pattern" => {
            target.contains("planner") && target.contains("critic") && target.contains("answers")
        }
        _ => false,
    }
}

fn adapter_output_hash(output: &Path) -> anyhow::Result<Option<String>> {
    if !output.exists() {
        return Ok(None);
    }
    let mut files = Vec::new();
    collect_adapter_hash_files(output, output, &mut files)?;
    if files.is_empty() {
        return Ok(None);
    }
    files.sort();
    let mut digest = Sha256::new();
    for path in files {
        let relative = path
            .strip_prefix(output)
            .unwrap_or(&path)
            .to_string_lossy()
            .to_string();
        digest.update(relative.as_bytes());
        digest.update(b"\0");
        digest.update(fs::read(&path).with_context(|| format!("reading {}", path.display()))?);
        digest.update(b"\0");
    }
    Ok(Some(format!("{:x}", digest.finalize())))
}

fn collect_adapter_hash_files(
    root: &Path,
    dir: &Path,
    files: &mut Vec<PathBuf>,
) -> anyhow::Result<()> {
    for entry in fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let path = entry?.path();
        if path.is_dir() {
            if path.strip_prefix(root).ok().is_some_and(|relative| {
                relative
                    .components()
                    .any(|part| part.as_os_str() == "mlx-data")
            }) {
                continue;
            }
            collect_adapter_hash_files(root, &path, files)?;
        } else if path.file_name().and_then(|name| name.to_str()) != Some("adapter_manifest.json") {
            files.push(path);
        }
    }
    Ok(())
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
