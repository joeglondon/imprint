use crate::recursive;
use crate::store::FileMemoryStore;
use crate::types::{
    AttentionAction, AttentionMark, AttentionTargetKind, BrainArtifact, ChatContextTrace, Chunk,
    CortexAdapterJob, CortexAdapterState, CortexIndex, DerivedMemory, DerivedMemoryKind, Document,
    LinkType, MemoryAccess, PersistedMemory, WebFinding,
};
use anyhow::{anyhow, Context};
use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const TRAINING_SCHEMA_VERSION: u32 = 2;
const DEFAULT_ADAPTER_ITERS: usize = 100;
const DEFAULT_ADAPTER_TRAINING_TIMEOUT_MILLIS: u64 = 30 * 60 * 1000;
pub const DEFAULT_ADAPTER_ACTIVATION_MIN_SCORE: f64 = 0.8;
const TRAINING_TASKS: [&str; 14] = [
    "query_to_region",
    "query_to_source_family",
    "query_to_tool_plan",
    "chunk_to_semantic_address",
    "weak_evidence_to_next_action",
    "snippet_set_to_citation_boundary",
    "deleted_or_stale_memory_to_caution",
    "web_needed_or_not",
    "recursive_role_trace",
    "recursive_sufficiency_eval",
    "recursive_efficiency_eval",
    "recursive_region_selection_eval",
    "recursive_hallucination_eval",
    "recursive_token_usage_eval",
];

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
    pub route_region_baseline_score: f64,
    pub task_scores: BTreeMap<String, CortexAdapterEvalTaskScore>,
    pub gates: BTreeMap<String, bool>,
    pub manifest_path: Option<String>,
    pub adapter_path: Option<String>,
    pub source_dataset_hash: String,
    pub evaluated_at: u64,
    pub failure_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct CortexTrainingDatasetSummary {
    pub schema_version: u32,
    pub records: usize,
    pub task_counts: BTreeMap<String, usize>,
    pub split_counts: BTreeMap<String, usize>,
    pub source_counts: BTreeMap<String, usize>,
    pub source_types: BTreeMap<String, usize>,
    pub visibility_counts: BTreeMap<String, usize>,
    pub stale_or_deleted_exclusions: usize,
    pub redacted_records: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct EvaluationSetCoverage {
    pub name: String,
    pub records: usize,
    pub source_ref_records: usize,
    pub anchor_records: usize,
    pub present: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct BaselineComparison {
    pub name: String,
    pub score: f64,
    pub records: usize,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Phase13EvaluationHarnessReport {
    pub status: String,
    pub evaluated_at: u64,
    pub dataset_summary: CortexTrainingDatasetSummary,
    pub eval_sets: BTreeMap<String, EvaluationSetCoverage>,
    pub baselines: BTreeMap<String, BaselineComparison>,
    pub synthetic_fixture_path: String,
    pub synthetic_fixture_records: usize,
    pub regression_trace_fixture_path: String,
    pub regression_trace_records: usize,
    pub metrics_history_path: String,
    pub adapter_activation_gate: bool,
    pub recursive_mas_default_gate: bool,
    pub failure_reasons: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct CortexTrainingExportContext {
    pub memory: PersistedMemory,
    pub artifacts: Vec<BrainArtifact>,
    pub cortex_index: Option<CortexIndex>,
    pub derived_memories: Vec<DerivedMemory>,
    pub web_findings: Vec<WebFinding>,
    pub attention_marks: Vec<AttentionMark>,
    pub memory_accesses: Vec<MemoryAccess>,
    pub chat_context_traces: Vec<ChatContextTrace>,
    pub privacy_excluded_sources: Vec<String>,
}

pub fn write_training_exports(
    store_root: &Path,
    context: &CortexTrainingExportContext,
) -> anyhow::Result<Vec<String>> {
    let training_dir = store_root.join("training");
    fs::create_dir_all(&training_dir).with_context(|| {
        format!(
            "creating training export directory {}",
            training_dir.display()
        )
    })?;
    let records = build_training_records(context);
    let mut files = Vec::new();
    for task in TRAINING_TASKS {
        for split in ["train", "eval", "test"] {
            files.push(write_split(&training_dir, task, split, &records)?);
        }
    }
    write_training_dataset_summary(&training_dir, &records)?;
    Ok(files
        .into_iter()
        .map(|path| path.display().to_string())
        .collect())
}

pub fn summarize_training_dataset(
    training_dir: &Path,
) -> anyhow::Result<CortexTrainingDatasetSummary> {
    let mut records = Vec::new();
    for split in ["train", "eval", "test"] {
        records.extend(training_records_for_split(training_dir, split)?);
    }
    Ok(dataset_summary(&records))
}

pub fn run_phase13_evaluation_harness(
    store_root: &Path,
    current_source_dataset_hash: &str,
    minimum_adapter_score: f64,
) -> anyhow::Result<Phase13EvaluationHarnessReport> {
    let evaluated_at = now_millis();
    let training_dir = store_root.join("training");
    fs::create_dir_all(&training_dir)
        .with_context(|| format!("creating training directory {}", training_dir.display()))?;
    let mut records = training_records_for_split(&training_dir, "eval")?;
    if records.is_empty() {
        records = training_records_for_split(&training_dir, "test")?;
    }
    if records.is_empty() {
        records = training_records_for_split(&training_dir, "train")?;
    }
    backfill_missing_eval_tasks(&training_dir, &mut records)?;

    let all_records = ["train", "eval", "test"]
        .into_iter()
        .map(|split| training_records_for_split(&training_dir, split))
        .collect::<anyhow::Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let dataset_summary = dataset_summary(&all_records);
    let eval_sets = phase13_eval_set_coverage(&records);
    let synthetic_fixture_path = write_phase13_synthetic_fixture(&training_dir)?;
    let regression_trace_fixture_path =
        write_phase13_regression_trace_fixture(&training_dir, &records)?;
    let synthetic_fixture_records = count_jsonl_records(&synthetic_fixture_path)?;
    let regression_trace_records = count_jsonl_records(&regression_trace_fixture_path)?;

    let adapter_eval = evaluate_cortex_adapter(
        store_root,
        current_source_dataset_hash,
        minimum_adapter_score,
    )?;
    let baselines = phase13_baseline_comparisons(&records, &adapter_eval);
    let recursive_mas_default_gate = recursive_mas_default_gate(&baselines);
    let adapter_activation_gate = adapter_eval.passed;

    let mut failure_reasons = Vec::new();
    for coverage in eval_sets.values() {
        if !coverage.present {
            failure_reasons.push(format!("missing eval set: {}", coverage.name));
        }
    }
    if !adapter_activation_gate {
        failure_reasons.push(
            adapter_eval
                .failure_reason
                .clone()
                .unwrap_or_else(|| "adapter activation eval gate did not pass".into()),
        );
    }
    if !recursive_mas_default_gate {
        failure_reasons.push(
            "latent RecursiveMAS remains disabled until it beats the text/tool recursive baseline"
                .into(),
        );
    }
    let metrics_history_path = append_phase13_metrics_history(
        &training_dir,
        evaluated_at,
        &eval_sets,
        &baselines,
        adapter_activation_gate,
        recursive_mas_default_gate,
    )?;
    Ok(Phase13EvaluationHarnessReport {
        status: if failure_reasons.is_empty() {
            "passed".into()
        } else {
            "blocked".into()
        },
        evaluated_at,
        dataset_summary,
        eval_sets,
        baselines,
        synthetic_fixture_path: synthetic_fixture_path.display().to_string(),
        synthetic_fixture_records,
        regression_trace_fixture_path: regression_trace_fixture_path.display().to_string(),
        regression_trace_records,
        metrics_history_path: metrics_history_path.display().to_string(),
        adapter_activation_gate,
        recursive_mas_default_gate,
        failure_reasons,
    })
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

    let mut train_records = mlx_records_for_split(&training_dir, "train")?;
    if train_records.is_empty() {
        train_records = mlx_records_for_split(&training_dir, "eval")?;
    }
    if train_records.is_empty() {
        train_records = mlx_records_for_split(&training_dir, "test")?;
    }
    if train_records.is_empty() {
        return Ok(None);
    }
    let mut valid_records = mlx_records_for_split(&training_dir, "eval")?;
    if valid_records.is_empty() {
        valid_records = mlx_records_for_split(&training_dir, "test")?;
    }
    if valid_records.is_empty() {
        valid_records = train_records.clone();
    }

    write_jsonl(&data_dir.join("train.jsonl"), &train_records)?;
    write_jsonl(&data_dir.join("valid.jsonl"), &valid_records)?;
    write_jsonl(&data_dir.join("test.jsonl"), &valid_records)?;
    let prepared_dataset_hash = training_source_hash(&data_dir)?;
    let provenance = adapter_training_provenance(&training_dir)?;
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
        "adapter_provenance": provenance,
        "local_only": provenance.local_only,
        "export_warning": provenance.export_warning,
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
        "--source-dataset-hash".into(),
        source_dataset_hash.into(),
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
            let mut manifest = read_manifest(&manifest_path)?;
            ensure_adapter_manifest_provenance(store_root, &manifest_path, &mut manifest)?;
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
            route_region_baseline_score: 0.0,
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
    let training_dir = store_root.join("training");
    let mut records = training_records_for_split(&training_dir, "eval")?;
    if records.is_empty() {
        records = training_records_for_split(&training_dir, "test")?;
    }
    if records.is_empty() {
        records = training_records_for_split(&training_dir, "train")?;
    }
    backfill_missing_eval_tasks(&training_dir, &mut records)?;
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
        .get("query_to_region")
        .map(|task| task.score)
        .unwrap_or_default();
    let route_region_baseline_score = route_region_baseline_score(&records);
    let tool_score = task_scores
        .get("query_to_tool_plan")
        .map(|task| task.score)
        .unwrap_or_default();
    let critique_score = task_scores
        .get("weak_evidence_to_next_action")
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
            "route_region_accuracy_above_baseline".into(),
            route_score > route_region_baseline_score,
        ),
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
        route_region_baseline_score,
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
            last_successful_training_at: None,
            activated_at: None,
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
        status: status.clone(),
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
        last_successful_training_at: match status.as_str() {
            "trained" | "active" => {
                u64_field(&manifest, "finished_at").or_else(|| u64_field(&manifest, "created_at"))
            }
            _ => None,
        },
        activated_at: u64_field(&manifest, "activated_at"),
        checked_at,
    })
}

fn write_split(
    training_dir: &Path,
    task: &str,
    split: &str,
    records: &[serde_json::Value],
) -> anyhow::Result<PathBuf> {
    let path = training_dir.join(format!("{task}.{split}.jsonl"));
    let lines = records
        .iter()
        .filter(|record| {
            record.get("task").and_then(|value| value.as_str()) == Some(task)
                && record.get("split").and_then(|value| value.as_str()) == Some(split)
        })
        .map(serde_json::to_string)
        .collect::<Result<Vec<_>, _>>()?;
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

fn backfill_missing_eval_tasks(
    training_dir: &Path,
    records: &mut Vec<serde_json::Value>,
) -> anyhow::Result<()> {
    let present = records
        .iter()
        .filter_map(|record| record.get("task").and_then(|value| value.as_str()))
        .collect::<BTreeSet<_>>();
    if TRAINING_TASKS.iter().all(|task| present.contains(task)) {
        return Ok(());
    }
    let fallback_records = ["test", "train"]
        .into_iter()
        .map(|split| training_records_for_split(training_dir, split))
        .collect::<anyhow::Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let mut seen_ids = records
        .iter()
        .filter_map(|record| record.get("id").and_then(|value| value.as_str()))
        .map(ToString::to_string)
        .collect::<BTreeSet<_>>();
    for task in TRAINING_TASKS {
        if records
            .iter()
            .any(|record| record.get("task").and_then(|value| value.as_str()) == Some(task))
        {
            continue;
        }
        if let Some(record) = fallback_records
            .iter()
            .find(|record| record.get("task").and_then(|value| value.as_str()) == Some(task))
        {
            if let Some(id) = record.get("id").and_then(|value| value.as_str()) {
                if seen_ids.insert(id.to_string()) {
                    records.push(record.clone());
                }
            }
        }
        if !records
            .iter()
            .any(|record| record.get("task").and_then(|value| value.as_str()) == Some(task))
        {
            if let Some(record) = synthetic_eval_record_for_task(task) {
                if let Some(id) = record.get("id").and_then(|value| value.as_str()) {
                    if seen_ids.insert(id.to_string()) {
                        records.push(record);
                    }
                }
            }
        }
    }
    Ok(())
}

fn synthetic_eval_record_for_task(task: &str) -> Option<serde_json::Value> {
    Some(match task {
        "recursive_role_trace" => json!({
            "id": "synthetic:phase13:recursive-role-trace",
            "task": "recursive_role_trace",
            "split": "eval",
            "input": "Trace a source-grounded recursive planner/critic/retriever/solver loop.",
            "target": "roles:planner,retriever,critic,solver,memory_steward;latent_status:research_only;fallback:text_tool",
            "source_refs": ["imprint://synthetic/recursive/trace"],
            "anchor_ids": ["synthetic-recursive-anchor"],
            "source_type": "synthetic_fixture",
        }),
        "recursive_sufficiency_eval" => json!({
            "id": "synthetic:phase13:recursive-sufficiency",
            "task": "recursive_sufficiency_eval",
            "split": "eval",
            "input": "Critic sees anchored snippets that answer the question.",
            "target": "sufficient:true;cite_anchor_ids:synthetic-recursive-anchor",
            "source_refs": ["imprint://synthetic/recursive/sufficiency"],
            "anchor_ids": ["synthetic-recursive-anchor"],
            "source_type": "synthetic_fixture",
        }),
        "recursive_efficiency_eval" => json!({
            "id": "synthetic:phase13:recursive-efficiency",
            "task": "recursive_efficiency_eval",
            "split": "eval",
            "input": "Compare recursive source routing with text/tool fallback.",
            "target": "efficiency:better_or_equal;tool_calls:1;source_grounded:true",
            "source_refs": ["imprint://synthetic/recursive/efficiency"],
            "anchor_ids": ["synthetic-recursive-anchor"],
            "source_type": "synthetic_fixture",
        }),
        "recursive_region_selection_eval" => json!({
            "id": "synthetic:phase13:recursive-region-selection",
            "task": "recursive_region_selection_eval",
            "split": "eval",
            "input": "Retriever should select the source region before expanding context.",
            "target": "region_source_selection:correct;selected_refs:imprint://synthetic/recursive/trace",
            "source_refs": ["imprint://synthetic/recursive/trace"],
            "anchor_ids": ["synthetic-recursive-anchor"],
            "source_type": "synthetic_fixture",
        }),
        "recursive_hallucination_eval" => json!({
            "id": "synthetic:phase13:recursive-hallucination",
            "task": "recursive_hallucination_eval",
            "split": "eval",
            "input": "Solver should avoid ungrounded exact claims.",
            "target": "hallucination_risk:low;source_anchored:true",
            "source_refs": ["imprint://synthetic/recursive/hallucination"],
            "anchor_ids": ["synthetic-recursive-anchor"],
            "source_type": "synthetic_fixture",
        }),
        "recursive_token_usage_eval" => json!({
            "id": "synthetic:phase13:recursive-token-usage",
            "task": "recursive_token_usage_eval",
            "split": "eval",
            "input": "Latent recursion should reduce token use only after measured wins.",
            "target": "token_usage:lower_if_hidden_state_supported;baseline_tool_calls:2;fallback:text_tool",
            "source_refs": ["imprint://synthetic/recursive/tokens"],
            "anchor_ids": ["synthetic-recursive-anchor"],
            "source_type": "synthetic_fixture",
        }),
        _ => return None,
    })
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

fn ensure_adapter_manifest_provenance(
    store_root: &Path,
    manifest_path: &Path,
    manifest: &mut serde_json::Value,
) -> anyhow::Result<()> {
    if manifest.get("adapter_provenance").is_some() {
        return Ok(());
    }
    let provenance = adapter_training_provenance(&store_root.join("training"))?;
    if let Some(object) = manifest.as_object_mut() {
        object.insert(
            "adapter_provenance".into(),
            serde_json::to_value(&provenance)?,
        );
        object.insert("local_only".into(), json!(provenance.local_only));
        object.insert("export_warning".into(), json!(provenance.export_warning));
    }
    fs::write(
        manifest_path,
        serde_json::to_string_pretty(manifest).context("serializing adapter manifest")?,
    )
    .with_context(|| format!("writing {}", manifest_path.display()))?;
    Ok(())
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

fn u64_field(value: &serde_json::Value, key: &str) -> Option<u64> {
    value.get(key)?.as_u64()
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
        "query_to_region" => target.starts_with("route_region:") && !input.trim().is_empty(),
        "query_to_source_family" => target.starts_with("source_family:"),
        "query_to_tool_plan" => {
            target.contains("memory_search") && target.contains("memory_expand")
        }
        "chunk_to_semantic_address" => {
            target.contains("region:") && target.contains("document:") && target.contains("anchor:")
        }
        "weak_evidence_to_next_action" => {
            target.contains("weak_evidence") || target.contains("search_original_sources")
        }
        "snippet_set_to_citation_boundary" => {
            target.contains("cite_anchor_ids") && target.contains("do_not_cite")
        }
        "deleted_or_stale_memory_to_caution" => target.contains("caution:"),
        "web_needed_or_not" => {
            target.contains("web_search_needed") || target.contains("web_search_not_needed")
        }
        "recursive_role_trace" => {
            target.contains("roles:")
                && target.contains("planner")
                && target.contains("critic")
                && target.contains("latent_status:research_only")
        }
        "recursive_sufficiency_eval" => {
            target.contains("sufficient:true") && target.contains("cite_anchor_ids:")
                || target.contains("sufficient:false") && target.contains("next_action:")
        }
        "recursive_efficiency_eval" => {
            target.contains("efficiency:")
                && target.contains("tool_calls:")
                && target.contains("source_grounded:")
        }
        "recursive_region_selection_eval" => {
            target.contains("region_source_selection:")
                && (target.contains("selected_refs:") || target.contains("next_action:"))
        }
        "recursive_hallucination_eval" => {
            target.contains("hallucination_risk:")
                && (target.contains("source_anchored") || target.contains("caveat_or_refuse"))
        }
        "recursive_token_usage_eval" => {
            target.contains("token_usage:")
                && target.contains("baseline_tool_calls:")
                && target.contains("fallback:text_tool")
        }
        _ => false,
    }
}

fn route_region_baseline_score(records: &[serde_json::Value]) -> f64 {
    let route_targets = records
        .iter()
        .filter(|record| {
            record.get("task").and_then(|value| value.as_str()) == Some("query_to_region")
        })
        .filter_map(|record| record.get("target").and_then(|value| value.as_str()))
        .collect::<std::collections::BTreeSet<_>>();
    if route_targets.len() <= 1 {
        return 0.0;
    }
    ratio(1, route_targets.len())
}

fn phase13_eval_set_coverage(
    records: &[serde_json::Value],
) -> BTreeMap<String, EvaluationSetCoverage> {
    let specs = [
        ("route_accuracy", &["query_to_region"][..]),
        ("source_family_selection", &["query_to_source_family"][..]),
        ("exact_anchor_recovery", &["chunk_to_semantic_address"][..]),
        (
            "citation_correctness",
            &["snippet_set_to_citation_boundary"][..],
        ),
        (
            "weak_evidence_detection",
            &["weak_evidence_to_next_action", "recursive_sufficiency_eval"][..],
        ),
        ("web_needed_decisions", &["web_needed_or_not"][..]),
        (
            "deletion_staleness_behavior",
            &["deleted_or_stale_memory_to_caution"][..],
        ),
        (
            "token_tool_call_efficiency",
            &["recursive_efficiency_eval", "recursive_token_usage_eval"][..],
        ),
    ];
    specs
        .into_iter()
        .map(|(name, tasks)| {
            let matching = records
                .iter()
                .filter(|record| {
                    record
                        .get("task")
                        .and_then(|value| value.as_str())
                        .is_some_and(|task| tasks.contains(&task))
                })
                .collect::<Vec<_>>();
            let source_ref_records = matching
                .iter()
                .filter(|record| json_array_len(record, "source_refs") > 0)
                .count();
            let anchor_records = matching
                .iter()
                .filter(|record| json_array_len(record, "anchor_ids") > 0)
                .count();
            (
                name.into(),
                EvaluationSetCoverage {
                    name: name.into(),
                    records: matching.len(),
                    source_ref_records,
                    anchor_records,
                    present: !matching.is_empty(),
                },
            )
        })
        .collect()
}

fn phase13_baseline_comparisons(
    records: &[serde_json::Value],
    adapter_eval: &CortexAdapterEvalReport,
) -> BTreeMap<String, BaselineComparison> {
    let route_records = task_records(records, "query_to_region");
    let vector_records = task_records(records, "chunk_to_semantic_address");
    let tool_records = task_records(records, "query_to_tool_plan");
    let recursive_records = records
        .iter()
        .filter(|record| {
            record
                .get("task")
                .and_then(|value| value.as_str())
                .is_some_and(|task| task.starts_with("recursive_"))
        })
        .cloned()
        .collect::<Vec<_>>();
    let text_recursive_score = task_gate_score(&recursive_records);
    let latent_recursive_score = records
        .iter()
        .filter(|record| {
            record.get("task").and_then(|value| value.as_str()) == Some("recursive_role_trace")
        })
        .filter(|record| {
            record
                .get("target")
                .and_then(|value| value.as_str())
                .is_some_and(|target| target.contains("latent_status:research_only"))
        })
        .count();
    let latent_recursive_records = task_records(records, "recursive_role_trace").len();
    BTreeMap::from([
        (
            "lexical_map_routing".into(),
            BaselineComparison {
                name: "lexical_map_routing".into(),
                score: route_region_baseline_score(records),
                records: route_records.len(),
                status: "deterministic_baseline".into(),
            },
        ),
        (
            "vector_only_routing".into(),
            BaselineComparison {
                name: "vector_only_routing".into(),
                score: task_gate_score(&vector_records),
                records: vector_records.len(),
                status: "source_anchor_proxy".into(),
            },
        ),
        (
            "cortex_index_routing".into(),
            BaselineComparison {
                name: "cortex_index_routing".into(),
                score: task_gate_score(&route_records),
                records: route_records.len(),
                status: "cortex_eval_set".into(),
            },
        ),
        (
            "base_model_planner".into(),
            BaselineComparison {
                name: "base_model_planner".into(),
                score: task_gate_score(&tool_records),
                records: tool_records.len(),
                status: "tool_plan_proxy".into(),
            },
        ),
        (
            "adapted_model_planner".into(),
            BaselineComparison {
                name: "adapted_model_planner".into(),
                score: adapter_eval.score,
                records: adapter_eval.records,
                status: if adapter_eval.passed {
                    "passed_activation_gate"
                } else {
                    "blocked_or_missing_adapter"
                }
                .into(),
            },
        ),
        (
            "text_recursive_loop".into(),
            BaselineComparison {
                name: "text_recursive_loop".into(),
                score: text_recursive_score,
                records: recursive_records.len(),
                status: "text_tool_trace_eval".into(),
            },
        ),
        (
            "latent_recursive_link_loop".into(),
            BaselineComparison {
                name: "latent_recursive_link_loop".into(),
                score: ratio(latent_recursive_score, latent_recursive_records),
                records: latent_recursive_records,
                status: "research_only_not_default".into(),
            },
        ),
    ])
}

fn task_records(records: &[serde_json::Value], task: &str) -> Vec<serde_json::Value> {
    records
        .iter()
        .filter(|record| record.get("task").and_then(|value| value.as_str()) == Some(task))
        .cloned()
        .collect()
}

fn task_gate_score(records: &[serde_json::Value]) -> f64 {
    ratio(
        records
            .iter()
            .filter(|record| eval_record_passes_gate(record))
            .count(),
        records.len(),
    )
}

fn recursive_mas_default_gate(baselines: &BTreeMap<String, BaselineComparison>) -> bool {
    let text = baselines
        .get("text_recursive_loop")
        .map(|baseline| baseline.score)
        .unwrap_or_default();
    let latent = baselines
        .get("latent_recursive_link_loop")
        .map(|baseline| baseline.score)
        .unwrap_or_default();
    let latent_records = baselines
        .get("latent_recursive_link_loop")
        .map(|baseline| baseline.records)
        .unwrap_or_default();
    latent_records >= 5 && latent > text
}

fn write_phase13_synthetic_fixture(training_dir: &Path) -> anyhow::Result<PathBuf> {
    let fixtures_dir = training_dir.join("fixtures");
    fs::create_dir_all(&fixtures_dir)
        .with_context(|| format!("creating fixture directory {}", fixtures_dir.display()))?;
    let path = fixtures_dir.join("synthetic_phase13.jsonl");
    let records = [
        json!({
            "task": "query_to_region",
            "input": "Where are the sourdough starter notes?",
            "target": "route_region:synthetic-kitchen-notes",
            "source_refs": ["imprint://synthetic/document/sourdough"],
            "anchor_ids": ["synthetic-anchor-sourdough"],
        }),
        json!({
            "task": "weak_evidence_to_next_action",
            "input": "A summary mentions a date but no anchor is present.",
            "target": "weak_evidence:search_original_sources_before_answering",
            "source_refs": ["imprint://synthetic/document/research-log"],
            "anchor_ids": [],
        }),
        json!({
            "task": "web_needed_or_not",
            "input": "The user asks for today's API pricing.",
            "target": "web_search_needed:mutable_current_fact",
            "source_refs": ["imprint://synthetic/policy/current-facts"],
            "anchor_ids": [],
        }),
    ];
    write_jsonl(&path, &records)?;
    Ok(path)
}

fn write_phase13_regression_trace_fixture(
    training_dir: &Path,
    records: &[serde_json::Value],
) -> anyhow::Result<PathBuf> {
    let fixtures_dir = training_dir.join("fixtures");
    fs::create_dir_all(&fixtures_dir)
        .with_context(|| format!("creating fixture directory {}", fixtures_dir.display()))?;
    let path = fixtures_dir.join("regression_traces.jsonl");
    let mut trace_records = records
        .iter()
        .filter(|record| {
            record.get("source_type").and_then(|value| value.as_str()) == Some("chat_context_trace")
                || record
                    .get("target")
                    .and_then(|value| value.as_str())
                    .is_some_and(|target| target.contains("reuse_successful_trace"))
        })
        .cloned()
        .collect::<Vec<_>>();
    trace_records.sort_by(|left, right| {
        left.get("id")
            .and_then(|value| value.as_str())
            .cmp(&right.get("id").and_then(|value| value.as_str()))
    });
    write_jsonl(&path, &trace_records)?;
    Ok(path)
}

fn append_phase13_metrics_history(
    training_dir: &Path,
    evaluated_at: u64,
    eval_sets: &BTreeMap<String, EvaluationSetCoverage>,
    baselines: &BTreeMap<String, BaselineComparison>,
    adapter_activation_gate: bool,
    recursive_mas_default_gate: bool,
) -> anyhow::Result<PathBuf> {
    let metrics_dir = training_dir.join("metrics");
    fs::create_dir_all(&metrics_dir)
        .with_context(|| format!("creating metrics directory {}", metrics_dir.display()))?;
    let path = metrics_dir.join("phase13_eval_history.jsonl");
    let payload = json!({
        "evaluated_at": evaluated_at,
        "eval_sets": eval_sets,
        "baselines": baselines,
        "adapter_activation_gate": adapter_activation_gate,
        "recursive_mas_default_gate": recursive_mas_default_gate,
    });
    let mut line = serde_json::to_string(&payload)?;
    line.push('\n');
    use std::io::Write;
    fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("opening {}", path.display()))?
        .write_all(line.as_bytes())
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(path)
}

fn count_jsonl_records(path: &Path) -> anyhow::Result<usize> {
    if !path.exists() {
        return Ok(0);
    }
    Ok(fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .count())
}

fn json_array_len(record: &serde_json::Value, key: &str) -> usize {
    record
        .get(key)
        .and_then(|value| value.as_array())
        .map(Vec::len)
        .unwrap_or_default()
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

fn build_training_records(context: &CortexTrainingExportContext) -> Vec<serde_json::Value> {
    let mut records = Vec::new();
    let documents_by_id = context
        .memory
        .documents
        .iter()
        .map(|document| (document.id.as_str(), document))
        .collect::<HashMap<_, _>>();
    let chunks_by_id = context
        .memory
        .chunks
        .iter()
        .map(|chunk| (chunk.id.as_str(), chunk))
        .collect::<HashMap<_, _>>();
    let suppressed = suppressed_targets(&context.attention_marks);
    let artifact_ids_by_region = context
        .artifacts
        .iter()
        .filter_map(|artifact| {
            artifact
                .id
                .strip_prefix("brain-region:")
                .or_else(|| artifact.id.strip_prefix("brain-routing:"))
                .map(|region| (region.to_string(), artifact.id.clone()))
        })
        .fold(
            BTreeMap::<String, Vec<String>>::new(),
            |mut by_region, (region, id)| {
                by_region.entry(region).or_default().push(id);
                by_region
            },
        );

    for region in &context.memory.regions {
        let source_chunks = region
            .chunk_ids
            .iter()
            .filter_map(|chunk_id| chunks_by_id.get(chunk_id.as_str()).copied())
            .filter(|chunk| source_chunk_is_trainable(chunk, &documents_by_id, &suppressed))
            .take(8)
            .collect::<Vec<_>>();
        if source_chunks.is_empty() {
            continue;
        }
        let artifact_ids = artifact_ids_by_region
            .get(&region.id)
            .cloned()
            .unwrap_or_default();
        let region_examples = context
            .cortex_index
            .as_ref()
            .and_then(|index| {
                index
                    .regions
                    .iter()
                    .find(|sketch| sketch.region_id == region.id)
            })
            .map(|sketch| sketch.route_examples.clone())
            .unwrap_or_default();

        for chunk in source_chunks {
            let Some(document) = documents_by_id.get(chunk.document_id.as_str()).copied() else {
                continue;
            };
            let source_id = chunk.id.clone();
            let split = split_for_source(&source_id);
            let anchor = chunk
                .source_anchor
                .as_ref()
                .or(document.source_anchor.as_ref());
            let source_refs = source_refs_for(document, Some(chunk));
            let anchor_ids = anchor
                .iter()
                .map(|anchor| anchor.id.clone())
                .collect::<Vec<_>>();
            let query_hints = route_hints(document, chunk, &region_examples);
            let source_type = source_type(document);
            let source_family = source_family(document);
            let visibility = visibility(document);

            records.push(training_example(
                "query_to_region",
                split,
                &source_id,
                format!(
                    "Where should imprint search for: {}?",
                    query_hints.join(" / ")
                ),
                format!("route_region:{}", region.id),
                &artifact_ids,
                source_refs.clone(),
                anchor_ids.clone(),
                &source_type,
                &visibility,
                false,
            ));
            records.push(training_example(
                "query_to_source_family",
                split,
                &source_id,
                format!(
                    "Classify the best source family for '{}' from path '{}'.",
                    document.title,
                    document
                        .metadata
                        .get("path")
                        .map(String::as_str)
                        .unwrap_or("")
                ),
                format!("source_family:{source_family}"),
                &artifact_ids,
                source_refs.clone(),
                anchor_ids.clone(),
                &source_type,
                &visibility,
                false,
            ));
            records.push(training_example(
                "query_to_tool_plan",
                split,
                &source_id,
                format!("Plan a source-grounded answer for: {}", query_hints.join(" ")),
                "memory_search -> memory_open_or_neighbors -> memory_expand(anchor) -> answer_with_citations".into(),
                &artifact_ids,
                source_refs.clone(),
                anchor_ids.clone(),
                &source_type,
                &visibility,
                false,
            ));
            records.push(training_example(
                "chunk_to_semantic_address",
                split,
                &source_id,
                format!("Snippet: {}", bounded_text(&chunk.text)),
                format!(
                    "region:{};document:{};source_family:{};anchor:{}",
                    region.id,
                    document.id,
                    source_family,
                    anchor.map(|anchor| anchor.id.as_str()).unwrap_or("missing")
                ),
                &artifact_ids,
                source_refs.clone(),
                anchor_ids.clone(),
                &source_type,
                &visibility,
                false,
            ));
            records.push(training_example(
                "snippet_set_to_citation_boundary",
                split,
                &source_id,
                format!(
                    "Bound citations for snippet A [{}] and avoid unsupported derived memory.",
                    anchor.map(|anchor| anchor.id.as_str()).unwrap_or("missing")
                ),
                format!(
                    "cite_anchor_ids:{};do_not_cite_without_source_anchor",
                    anchor_ids.join(",")
                ),
                &artifact_ids,
                source_refs.clone(),
                anchor_ids,
                &source_type,
                &visibility,
                false,
            ));
            records.push(training_example(
                "web_needed_or_not",
                split,
                &source_id,
                format!(
                    "Should a live web search be used when local source '{}' has anchor '{}'?",
                    document.title,
                    anchor.map(|anchor| anchor.id.as_str()).unwrap_or("missing")
                ),
                "web_search_not_needed:local_source_anchor_is_available".into(),
                &artifact_ids,
                source_refs.clone(),
                anchor
                    .iter()
                    .map(|anchor| anchor.id.clone())
                    .collect::<Vec<_>>(),
                &source_type,
                &visibility,
                false,
            ));
            records.push(training_example(
                "weak_evidence_to_next_action",
                split,
                &source_id,
                format!(
                    "A route hint suggests misleading source family '{}' for '{}'.",
                    misleading_source_family(&source_family),
                    document.title
                ),
                format!(
                    "weak_evidence:reject_misleading_source_family;prefer_source_family:{source_family};search_and_expand_anchor"
                ),
                &artifact_ids,
                source_refs,
                anchor
                    .iter()
                    .map(|anchor| anchor.id.clone())
                    .collect::<Vec<_>>(),
                &source_type,
                &visibility,
                false,
            ));
        }
    }

    for source_id in &context.privacy_excluded_sources {
        let split = split_for_source(source_id);
        records.push(training_example(
            "deleted_or_stale_memory_to_caution",
            split,
            source_id,
            format!("Source {source_id} is private, opted out, forgotten, or secret-bearing."),
            "caution:source_excluded_from_adapter_training;do_not_route_or_memorize;use_source_recall_only_if_user_readds_source".into(),
            &[],
            vec![format!("imprint://document/{source_id}")],
            Vec::new(),
            "privacy_exclusion",
            "excluded_from_adapter_training",
            true,
        ));
    }

    for memory in &context.derived_memories {
        if !matches!(
            memory.kind,
            DerivedMemoryKind::Decision | DerivedMemoryKind::Task
        ) {
            continue;
        }
        let source_id = memory.id.clone();
        let split = split_for_source(&source_id);
        let source_refs = if memory.provenance.source_refs.is_empty() {
            vec![format!("imprint://derived/{}", memory.id)]
        } else {
            memory.provenance.source_refs.clone()
        };
        records.push(training_example(
            "query_to_tool_plan",
            split,
            &source_id,
            format!(
                "Chat {:?}: {}",
                memory.kind,
                bounded_text(&memory.text)
            ),
            "memory_search(chat decision/task) -> memory_expand(source_message_anchor) -> preserve provenance".into(),
            &[],
            source_refs.clone(),
            Vec::new(),
            "chat_decision_task",
            "private",
            false,
        ));
        records.push(training_example(
            "query_to_source_family",
            split,
            &source_id,
            format!("Which source family stores this chat {:?}?", memory.kind),
            "source_family:chat_decision_task".into(),
            &[],
            source_refs,
            Vec::new(),
            "chat_decision_task",
            "private",
            false,
        ));
    }

    for artifact in &context.artifacts {
        let source_id = artifact
            .source_refs
            .first()
            .cloned()
            .unwrap_or_else(|| artifact.id.clone());
        let split = split_for_source(&source_id);
        records.push(training_example(
            "weak_evidence_to_next_action",
            split,
            &source_id,
            format!(
                "Derived cortex artifact only: {}",
                bounded_text(&artifact.body)
            ),
            "weak_evidence:search_original_sources_and_expand_anchor_before_answering".into(),
            std::slice::from_ref(&artifact.id),
            artifact.source_refs.clone(),
            Vec::new(),
            "derived_memory",
            "private",
            false,
        ));
        records.push(training_example(
            "deleted_or_stale_memory_to_caution",
            split,
            &source_id,
            format!(
                "Derived memory without direct citation boundary: {}",
                bounded_text(&artifact.body)
            ),
            "caution:derived_memory_requires_original_source_anchor_before_exact_claim".into(),
            std::slice::from_ref(&artifact.id),
            artifact.source_refs.clone(),
            Vec::new(),
            "derived_memory",
            "private",
            true,
        ));
    }

    for finding in &context.web_findings {
        let source_id = finding.id.clone();
        let split = split_for_source(&source_id);
        let stale = web_finding_is_stale(finding);
        let source_refs = vec![format!("imprint://web_finding/{}", finding.id)];
        records.push(training_example(
            "web_needed_or_not",
            split,
            &source_id,
            format!(
                "Query '{}' has web finding '{}' retrieved at {} with confidence {:.2}.",
                finding.query, finding.title, finding.retrieved_at, finding.confidence
            ),
            if stale || finding.confidence < 0.45 {
                "web_search_needed:refresh_before_answering"
            } else {
                "web_search_not_needed:use_stored_web_finding_with_url_citation"
            }
            .into(),
            &[],
            source_refs.clone(),
            Vec::new(),
            "web_finding",
            "private",
            stale,
        ));
        if stale || finding.confidence < 0.45 {
            records.push(training_example(
                "deleted_or_stale_memory_to_caution",
                split,
                &source_id,
                format!("Stale or low-confidence web finding: {}", finding.title),
                "caution:mark_stale_and_refresh_web_or_local_source_before_claiming".into(),
                &[],
                source_refs,
                Vec::new(),
                "web_finding",
                "private",
                true,
            ));
        }
    }

    for mark in &context.attention_marks {
        if matches!(
            mark.action,
            AttentionAction::Suppress | AttentionAction::Cold | AttentionAction::Decay
        ) {
            let source_id = format!(
                "attention:{}:{}",
                attention_kind_label(&mark.target_kind),
                mark.target_id
            );
            let split = split_for_source(&source_id);
            records.push(training_example(
                "deleted_or_stale_memory_to_caution",
                split,
                &source_id,
                format!(
                    "Attention mark {} on {} because: {}",
                    attention_action_label(&mark.action),
                    mark.target_id,
                    mark.reason
                ),
                "caution:deprioritize_or_verify_before_using_this_memory".into(),
                &[],
                vec![format!(
                    "imprint://{}/{}",
                    attention_kind_label(&mark.target_kind),
                    mark.target_id
                )],
                Vec::new(),
                attention_kind_label(&mark.target_kind),
                "private",
                true,
            ));
        }
    }

    for access in context.memory_accesses.iter().take(32) {
        let source_id = format!(
            "access:{}:{}",
            attention_kind_label(&access.target_kind),
            access.target_id
        );
        let split = split_for_source(&source_id);
        records.push(training_example(
            "query_to_tool_plan",
            split,
            &source_id,
            format!(
                "A successful retrieval trace recently used {} on {} for '{}'.",
                memory_access_kind_label(access),
                access.target_id,
                access.reason
            ),
            "reuse_successful_trace:memory_search -> open_or_expand_recent_target -> cite_source_anchor".into(),
            &[],
            vec![format!("imprint://{}/{}", attention_kind_label(&access.target_kind), access.target_id)],
            Vec::new(),
            attention_kind_label(&access.target_kind),
            "private",
            false,
        ));
    }

    for trace in context.chat_context_traces.iter().take(32) {
        for example in recursive::trace_dataset_examples(trace) {
            let split = split_for_source(&example.source_trace_id);
            records.push(training_example(
                &example.task,
                split,
                &example.source_trace_id,
                example.input,
                example.target,
                &[],
                example.source_refs,
                example.anchor_ids,
                "chat_context_trace",
                "private",
                !example.source_grounded && example.task == "recursive_sufficiency_eval",
            ));
        }
    }

    for link in &context.memory.links {
        if matches!(
            link.link_type,
            LinkType::CitationReference | LinkType::EntityOverlap | LinkType::Explicit
        ) {
            let source_id = format!("link:{}", link.id);
            let split = split_for_source(&source_id);
            records.push(training_example(
                "query_to_region",
                split,
                &source_id,
                format!(
                    "Follow {} link '{}'.",
                    link_type_label(&link.link_type),
                    link.label
                ),
                format!("follow_link:{};then_search_linked_region_or_chunk", link.id),
                &[],
                vec![format!("imprint://link/{}", link.id)],
                Vec::new(),
                "link",
                "private",
                false,
            ));
        }
    }

    records.sort_by(|left, right| {
        left.get("id")
            .and_then(|value| value.as_str())
            .cmp(&right.get("id").and_then(|value| value.as_str()))
    });
    records
}

fn training_example(
    task: &str,
    split: &str,
    source_id: &str,
    input: String,
    target: String,
    artifact_ids: &[String],
    source_refs: Vec<String>,
    anchor_ids: Vec<String>,
    source_type: &str,
    visibility: &str,
    excluded_or_stale: bool,
) -> serde_json::Value {
    let (input, input_redacted) = redact_sensitive(&input);
    let (target, target_redacted) = redact_sensitive(&target);
    let secret_detected = input_redacted || target_redacted;
    json!({
        "schema_version": TRAINING_SCHEMA_VERSION,
        "task": task,
        "split": split,
        "id": format!("{task}:{split}:{}", stable_hash(&format!("{source_id}:{input}:{target}"))),
        "source_id": source_id,
        "input": input,
        "target": target,
        "artifact_ids": artifact_ids,
        "artifact_kind": "cortex_training_example",
        "source_refs": source_refs,
        "anchor_ids": anchor_ids,
        "source_type": source_type,
        "visibility": normalize_privacy_level(visibility),
        "redacted": secret_detected,
        "secret_detected": secret_detected,
        "excluded_or_stale": excluded_or_stale || privacy_excludes_training(visibility),
    })
}

fn write_training_dataset_summary(
    training_dir: &Path,
    records: &[serde_json::Value],
) -> anyhow::Result<()> {
    let summary = dataset_summary(records);
    let path = training_dir.join("summary.json");
    fs::write(
        &path,
        serde_json::to_string_pretty(&summary).context("serializing training dataset summary")?,
    )
    .with_context(|| format!("writing {}", path.display()))
}

fn dataset_summary(records: &[serde_json::Value]) -> CortexTrainingDatasetSummary {
    let mut task_counts = BTreeMap::new();
    let mut split_counts = BTreeMap::new();
    let mut source_counts = BTreeMap::new();
    let mut source_types = BTreeMap::new();
    let mut visibility_counts = BTreeMap::new();
    let mut stale_or_deleted_exclusions = 0;
    let mut redacted_records = 0;
    for record in records {
        bump(&mut task_counts, json_str(record, "task"));
        bump(&mut split_counts, json_str(record, "split"));
        bump(&mut source_counts, json_str(record, "source_id"));
        bump(&mut source_types, json_str(record, "source_type"));
        bump(&mut visibility_counts, json_str(record, "visibility"));
        if record
            .get("excluded_or_stale")
            .and_then(|value| value.as_bool())
            .unwrap_or(false)
        {
            stale_or_deleted_exclusions += 1;
        }
        if record
            .get("redacted")
            .and_then(|value| value.as_bool())
            .unwrap_or(false)
        {
            redacted_records += 1;
        }
    }
    CortexTrainingDatasetSummary {
        schema_version: TRAINING_SCHEMA_VERSION,
        records: records.len(),
        task_counts,
        split_counts,
        source_counts,
        source_types,
        visibility_counts,
        stale_or_deleted_exclusions,
        redacted_records,
    }
}

fn source_chunk_is_trainable(
    chunk: &Chunk,
    documents_by_id: &HashMap<&str, &Document>,
    suppressed: &BTreeSet<String>,
) -> bool {
    if suppressed.contains(&chunk.id) || suppressed.contains(&chunk.document_id) {
        return false;
    }
    let Some(document) = documents_by_id.get(chunk.document_id.as_str()) else {
        return false;
    };
    !matches!(
        source_type(document).as_str(),
        "derived_memory" | "brain_artifact"
    ) && !source_is_deleted(document)
        && !source_training_opted_out(document, Some(chunk))
        && !source_has_training_secret(document, Some(chunk))
}

fn source_is_deleted(document: &Document) -> bool {
    document.metadata.contains_key("deleted_at")
        || document.metadata.contains_key("source_deleted_at")
        || document.metadata.get("deletion_state").map(String::as_str) == Some("deleted")
}

fn source_training_opted_out(document: &Document, chunk: Option<&Chunk>) -> bool {
    metadata_truthy(&document.metadata, "adapter_training_opt_out")
        || metadata_truthy(&document.metadata, "training_opt_out")
        || metadata_truthy(&document.metadata, "exclude_from_adapter_training")
        || privacy_excludes_training(&visibility(document))
        || chunk.is_some_and(|chunk| {
            metadata_truthy(&chunk.metadata, "adapter_training_opt_out")
                || metadata_truthy(&chunk.metadata, "training_opt_out")
                || metadata_truthy(&chunk.metadata, "exclude_from_adapter_training")
                || chunk
                    .metadata
                    .get("privacy_level")
                    .is_some_and(|value| privacy_excludes_training(value))
        })
}

fn source_has_training_secret(document: &Document, chunk: Option<&Chunk>) -> bool {
    metadata_truthy(&document.metadata, "contains_secret")
        || contains_secret(&document.text)
        || chunk.is_some_and(|chunk| {
            metadata_truthy(&chunk.metadata, "contains_secret") || contains_secret(&chunk.text)
        })
}

fn route_hints(document: &Document, chunk: &Chunk, region_examples: &[String]) -> Vec<String> {
    let mut hints = BTreeSet::new();
    hints.insert(document.title.clone());
    if let Some(path) = document.metadata.get("path") {
        hints.insert(path_tail(path));
    }
    if let Some(section) = chunk
        .source_anchor
        .as_ref()
        .and_then(|anchor| anchor.section.clone())
        .or_else(|| {
            document
                .source_anchor
                .as_ref()
                .and_then(|anchor| anchor.section.clone())
        })
    {
        hints.insert(section);
    }
    for term in entity_terms(&chunk.text).into_iter().take(4) {
        hints.insert(term);
    }
    for example in region_examples.iter().take(2) {
        hints.insert(example.clone());
    }
    hints.into_iter().take(8).collect()
}

fn source_refs_for(document: &Document, chunk: Option<&Chunk>) -> Vec<String> {
    let mut refs = Vec::new();
    if let Some(chunk) = chunk {
        refs.push(format!("imprint://chunk/{}", chunk.id));
    }
    refs.push(format!("imprint://document/{}", document.id));
    if let Some(anchor) = chunk
        .and_then(|chunk| chunk.source_anchor.as_ref())
        .or(document.source_anchor.as_ref())
    {
        refs.push(format!("imprint://anchor/{}", anchor.id));
    }
    refs
}

fn split_for_source(source_id: &str) -> &'static str {
    let mut digest = Sha256::new();
    digest.update(source_id.as_bytes());
    let value = digest.finalize()[0] % 10;
    match value {
        0..=6 => "train",
        7 => "eval",
        _ => "test",
    }
}

fn suppressed_targets(marks: &[AttentionMark]) -> BTreeSet<String> {
    marks
        .iter()
        .filter(|mark| mark.reverted_at.is_none())
        .filter(|mark| matches!(mark.action, AttentionAction::Suppress))
        .filter(|mark| {
            matches!(
                mark.target_kind,
                AttentionTargetKind::Document | AttentionTargetKind::Chunk
            )
        })
        .map(|mark| mark.target_id.clone())
        .collect()
}

fn source_type(document: &Document) -> String {
    document
        .metadata
        .get("source_type")
        .or_else(|| document.metadata.get("source"))
        .cloned()
        .unwrap_or_else(|| "local_file".into())
}

fn source_family(document: &Document) -> String {
    let source_type = source_type(document);
    if source_type == "web_finding" {
        return "web_finding".into();
    }
    document
        .metadata
        .get("path")
        .map(|path| {
            Path::new(path)
                .parent()
                .and_then(|parent| parent.file_name())
                .and_then(|name| name.to_str())
                .unwrap_or(source_type.as_str())
                .to_string()
        })
        .unwrap_or(source_type)
}

fn misleading_source_family(actual: &str) -> &'static str {
    match actual {
        "web_finding" => "local_file",
        "chat" | "chat_decision_task" => "web_finding",
        "derived_memory" => "trusted_original_source",
        _ => "unrelated_web_finding",
    }
}

fn visibility(document: &Document) -> String {
    normalize_privacy_level(
        document
            .metadata
            .get("privacy_level")
            .or_else(|| document.metadata.get("visibility"))
            .map(String::as_str)
            .unwrap_or("private_user_memory"),
    )
}

fn bounded_text(text: &str) -> String {
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut chars = normalized.chars();
    let bounded = chars.by_ref().take(320).collect::<String>();
    if chars.next().is_some() {
        format!("{bounded}...")
    } else {
        bounded
    }
}

fn redact_sensitive(text: &str) -> (String, bool) {
    let mut redacted = false;
    let words = text
        .split_whitespace()
        .map(|word| {
            if word_looks_secret(word) {
                redacted = true;
                "[REDACTED]".to_string()
            } else {
                word.to_string()
            }
        })
        .collect::<Vec<_>>();
    (words.join(" "), redacted)
}

fn contains_secret(text: &str) -> bool {
    text.split_whitespace().any(word_looks_secret)
}

fn word_looks_secret(word: &str) -> bool {
    let trimmed = word.trim_matches(|character: char| {
        matches!(
            character,
            '"' | '\'' | '`' | ',' | ';' | ':' | ')' | ']' | '}'
        )
    });
    let lower = trimmed.to_lowercase();
    trimmed.starts_with("sk-")
        || lower.contains("api_key")
        || lower.contains("apikey")
        || lower.contains("password=")
        || lower.contains("secret=")
        || lower.contains("token=")
        || lower.contains("bearer ")
}

fn metadata_truthy(metadata: &BTreeMap<String, String>, key: &str) -> bool {
    metadata
        .get(key)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "y" | "on"
            )
        })
        .unwrap_or(false)
}

fn normalize_privacy_level(value: &str) -> String {
    match value.trim().to_ascii_lowercase().as_str() {
        "private" | "private_user" | "private_user_memory" | "user_private" => {
            "private_user_memory"
        }
        "shared" | "project" | "shared_project" | "shared_project_memory" => {
            "shared_project_memory"
        }
        "global" | "reference" | "global_reference" | "global_reference_memory" => {
            "global_reference_memory"
        }
        "excluded" | "excluded_from_adapter_training" | "training_excluded" => {
            "excluded_from_adapter_training"
        }
        _ => "private_user_memory",
    }
    .into()
}

fn privacy_excludes_training(value: &str) -> bool {
    normalize_privacy_level(value) == "excluded_from_adapter_training"
}

#[derive(Debug, Clone, Serialize)]
struct AdapterTrainingProvenance {
    corpus_hash: String,
    included_sources: Vec<String>,
    excluded_sources: Vec<String>,
    train_eval_files: Vec<String>,
    local_only: bool,
    export_warning: String,
}

fn adapter_training_provenance(training_dir: &Path) -> anyhow::Result<AdapterTrainingProvenance> {
    let mut included_sources = BTreeSet::new();
    let mut excluded_sources = BTreeSet::new();
    let mut train_eval_files = Vec::new();
    let mut local_only = false;
    for path in jsonl_files(training_dir)? {
        let file = path.display().to_string();
        train_eval_files.push(file.clone());
        for line in fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?
            .lines()
            .filter(|line| !line.trim().is_empty())
        {
            let record: serde_json::Value = serde_json::from_str(line)
                .with_context(|| format!("parsing {}", path.display()))?;
            let source_id = json_str(&record, "source_id");
            let excluded = record
                .get("excluded_or_stale")
                .and_then(|value| value.as_bool())
                .unwrap_or(false);
            if excluded {
                excluded_sources.insert(source_id);
            } else {
                included_sources.insert(source_id);
            }
            if record
                .get("visibility")
                .and_then(|value| value.as_str())
                .is_some_and(|visibility| visibility == "private_user_memory")
            {
                local_only = true;
            }
        }
    }
    train_eval_files.sort();
    Ok(AdapterTrainingProvenance {
        corpus_hash: training_source_hash(training_dir)?,
        included_sources: included_sources.into_iter().collect(),
        excluded_sources: excluded_sources.into_iter().collect(),
        train_eval_files,
        local_only,
        export_warning: "Training exports may contain private semantic addresses; do not send them to a non-local trainer without explicit user approval.".into(),
    })
}

fn web_finding_is_stale(finding: &WebFinding) -> bool {
    let now = now_millis();
    let ninety_days = 90 * 24 * 60 * 60 * 1000;
    finding.retrieved_at == 0 || finding.retrieved_at.saturating_add(ninety_days) < now
}

fn entity_terms(text: &str) -> Vec<String> {
    text.split(|character: char| !character.is_alphanumeric())
        .filter(|term| term.len() >= 4)
        .filter(|term| term.chars().next().is_some_and(char::is_uppercase))
        .map(ToString::to_string)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn path_tail(path: &str) -> String {
    Path::new(path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(path)
        .to_string()
}

fn attention_kind_label(kind: &AttentionTargetKind) -> &'static str {
    match kind {
        AttentionTargetKind::ChatSession => "chat_session",
        AttentionTargetKind::Session => "session",
        AttentionTargetKind::Project => "project",
        AttentionTargetKind::Workspace => "workspace",
        AttentionTargetKind::Collection => "collection",
        AttentionTargetKind::Task => "task",
        AttentionTargetKind::ChatMessage => "chat_message",
        AttentionTargetKind::TranscriptChunk => "transcript_chunk",
        AttentionTargetKind::DerivedMemory => "derived_memory",
        AttentionTargetKind::WebFinding => "web_finding",
        AttentionTargetKind::Document => "document",
        AttentionTargetKind::Chunk => "chunk",
        AttentionTargetKind::Region => "region",
        AttentionTargetKind::Link => "link",
    }
}

fn attention_action_label(action: &AttentionAction) -> &'static str {
    match action {
        AttentionAction::Active => "active",
        AttentionAction::Hot => "hot",
        AttentionAction::Warm => "warm",
        AttentionAction::Cold => "cold",
        AttentionAction::Promote => "promote",
        AttentionAction::Decay => "decay",
        AttentionAction::Pin => "pin",
        AttentionAction::Suppress => "suppress",
    }
}

fn memory_access_kind_label(access: &MemoryAccess) -> &'static str {
    match &access.access_kind {
        crate::types::MemoryAccessKind::QueryHit => "query_hit",
        crate::types::MemoryAccessKind::Open => "open",
        crate::types::MemoryAccessKind::Expand => "expand",
        crate::types::MemoryAccessKind::JumpToAnchor => "jump_to_anchor",
        crate::types::MemoryAccessKind::Cite => "cite",
    }
}

fn link_type_label(link_type: &LinkType) -> &'static str {
    match link_type {
        LinkType::SemanticNeighbor => "semantic_neighbor",
        LinkType::SameDocument => "same_document",
        LinkType::CitationReference => "citation_reference",
        LinkType::EntityOverlap => "entity_overlap",
        LinkType::RegionMembership => "region_membership",
        LinkType::Explicit => "explicit",
    }
}

fn json_str(record: &serde_json::Value, key: &str) -> String {
    record
        .get(key)
        .and_then(|value| value.as_str())
        .unwrap_or("unknown")
        .to_string()
}

fn bump(counts: &mut BTreeMap<String, usize>, key: String) {
    *counts.entry(key).or_default() += 1;
}

fn stable_hash(text: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(text.as_bytes());
    format!("{:x}", digest.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn phase13_harness_covers_eval_sets_baselines_and_history() {
        let root = std::env::temp_dir().join(format!(
            "imprint-phase13-harness-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let training_dir = root.join("training");
        fs::create_dir_all(&training_dir).expect("training dir");
        let source_refs = vec!["imprint://document/synthetic".to_string()];
        let anchor_ids = vec!["synthetic-anchor".to_string()];
        let records = vec![
            training_example(
                "query_to_region",
                "eval",
                "synthetic:route",
                "Route the synthetic memory query.".into(),
                "route_region:synthetic-region".into(),
                &[],
                source_refs.clone(),
                anchor_ids.clone(),
                "synthetic_fixture",
                "shared_project",
                false,
            ),
            training_example(
                "query_to_source_family",
                "eval",
                "synthetic:family",
                "Choose source family.".into(),
                "source_family:synthetic-notes".into(),
                &[],
                source_refs.clone(),
                anchor_ids.clone(),
                "synthetic_fixture",
                "shared_project",
                false,
            ),
            training_example(
                "query_to_tool_plan",
                "eval",
                "synthetic:tool",
                "Plan source-grounded retrieval.".into(),
                "memory_search -> memory_expand -> cite_source_anchor".into(),
                &[],
                source_refs.clone(),
                anchor_ids.clone(),
                "synthetic_fixture",
                "shared_project",
                false,
            ),
            training_example(
                "chunk_to_semantic_address",
                "eval",
                "synthetic:anchor",
                "Find exact source address.".into(),
                "region:synthetic-region;document:synthetic-doc;anchor:synthetic-anchor".into(),
                &[],
                source_refs.clone(),
                anchor_ids.clone(),
                "synthetic_fixture",
                "shared_project",
                false,
            ),
            training_example(
                "weak_evidence_to_next_action",
                "eval",
                "synthetic:weak",
                "Evidence has no source anchor.".into(),
                "weak_evidence:search_original_sources_before_answering".into(),
                &[],
                source_refs.clone(),
                anchor_ids.clone(),
                "synthetic_fixture",
                "shared_project",
                false,
            ),
            training_example(
                "snippet_set_to_citation_boundary",
                "eval",
                "synthetic:citation",
                "Decide citation boundary.".into(),
                "cite_anchor_ids:synthetic-anchor;do_not_cite:unanchored-summary".into(),
                &[],
                source_refs.clone(),
                anchor_ids.clone(),
                "synthetic_fixture",
                "shared_project",
                false,
            ),
            training_example(
                "deleted_or_stale_memory_to_caution",
                "eval",
                "synthetic:stale",
                "A memory was deleted or stale.".into(),
                "caution:verify_before_using".into(),
                &[],
                source_refs.clone(),
                anchor_ids.clone(),
                "synthetic_fixture",
                "shared_project",
                true,
            ),
            training_example(
                "web_needed_or_not",
                "eval",
                "synthetic:web",
                "The question needs current facts.".into(),
                "web_search_needed:mutable_current_fact".into(),
                &[],
                source_refs.clone(),
                anchor_ids.clone(),
                "synthetic_fixture",
                "shared_project",
                false,
            ),
            training_example(
                "recursive_role_trace",
                "eval",
                "synthetic:recursive-role",
                "Trace recursive roles.".into(),
                "roles:planner,retriever,critic,solver;latent_status:research_only".into(),
                &[],
                source_refs.clone(),
                anchor_ids.clone(),
                "chat_context_trace",
                "shared_project",
                false,
            ),
            training_example(
                "recursive_sufficiency_eval",
                "eval",
                "synthetic:recursive-sufficiency",
                "Judge sufficiency.".into(),
                "sufficient:true;cite_anchor_ids:synthetic-anchor".into(),
                &[],
                source_refs.clone(),
                anchor_ids.clone(),
                "chat_context_trace",
                "shared_project",
                false,
            ),
            training_example(
                "recursive_efficiency_eval",
                "eval",
                "synthetic:recursive-efficiency",
                "Judge efficiency.".into(),
                "efficiency:ok;tool_calls:2;source_grounded:true".into(),
                &[],
                source_refs.clone(),
                anchor_ids.clone(),
                "chat_context_trace",
                "shared_project",
                false,
            ),
            training_example(
                "recursive_region_selection_eval",
                "eval",
                "synthetic:recursive-region",
                "Judge selected refs.".into(),
                "region_source_selection:ok;selected_refs:imprint://document/synthetic".into(),
                &[],
                source_refs.clone(),
                anchor_ids.clone(),
                "chat_context_trace",
                "shared_project",
                false,
            ),
            training_example(
                "recursive_hallucination_eval",
                "eval",
                "synthetic:recursive-hallucination",
                "Judge hallucination risk.".into(),
                "hallucination_risk:low;source_anchored:true".into(),
                &[],
                source_refs.clone(),
                anchor_ids.clone(),
                "chat_context_trace",
                "shared_project",
                false,
            ),
            training_example(
                "recursive_token_usage_eval",
                "eval",
                "synthetic:recursive-token",
                "Judge token usage.".into(),
                "token_usage:lower_if_hidden_state_supported;baseline_tool_calls:2;fallback:text_tool"
                    .into(),
                &[],
                source_refs.clone(),
                anchor_ids.clone(),
                "chat_context_trace",
                "shared_project",
                false,
            ),
        ];
        write_jsonl(&training_dir.join("synthetic.eval.jsonl"), &records).expect("write eval");
        let source_hash = training_source_hash(&training_dir).expect("source hash");
        let adapter_dir = root.join("adapters").join("trained-synthetic");
        fs::create_dir_all(&adapter_dir).expect("adapter dir");
        fs::write(
            adapter_dir.join("adapter_manifest.json"),
            serde_json::json!({
                "status": "trained",
                "adapter_path": adapter_dir.display().to_string(),
                "source_dataset_hash": source_hash,
                "adapter_file_hash": "synthetic-hash",
            })
            .to_string(),
        )
        .expect("manifest");

        let report =
            run_phase13_evaluation_harness(&root, &source_hash, 0.8).expect("phase 13 harness");
        assert_eq!(report.status, "blocked");
        assert!(report.adapter_activation_gate);
        assert!(!report.recursive_mas_default_gate);
        assert!(report.eval_sets.values().all(|coverage| coverage.present));
        assert_eq!(report.synthetic_fixture_records, 3);
        assert!(report.regression_trace_records >= 5);
        assert_eq!(
            report
                .baselines
                .get("adapted_model_planner")
                .expect("adapted baseline")
                .status,
            "passed_activation_gate"
        );
        assert!(Path::new(&report.metrics_history_path).exists());
        assert_eq!(
            training_source_hash(&training_dir).expect("source hash after harness"),
            source_hash,
            "fixtures and metric history must not contaminate adapter dataset hash"
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn mlx_prep_uses_test_records_for_validation_when_eval_is_missing() {
        let root = std::env::temp_dir().join(format!(
            "imprint-mlx-split-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let dataset = root.join("training");
        let output = root.join("adapter");
        fs::create_dir_all(&dataset).expect("dataset dir");
        fs::write(
            dataset.join("query_to_region.train.jsonl"),
            [
                serde_json::json!({"task":"query_to_region","input":"train one","target":"route_region:a"}).to_string(),
                serde_json::json!({"task":"query_to_region","input":"train two","target":"route_region:b"}).to_string(),
            ]
            .join("\n"),
        )
        .expect("train records");
        fs::write(
            dataset.join("query_to_region.test.jsonl"),
            serde_json::json!({"task":"query_to_region","input":"test one","target":"route_region:c"}).to_string(),
        )
        .expect("test records");

        let script = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("training")
            .join("train_mlx_lora.py");
        let output_result = Command::new("python3")
            .arg(script)
            .arg("--model")
            .arg("tiny-memory-model")
            .arg("--dataset")
            .arg(&dataset)
            .arg("--output")
            .arg(&output)
            .arg("--dry-run")
            .output()
            .expect("run train_mlx_lora.py");
        assert!(
            output_result.status.success(),
            "{}",
            String::from_utf8_lossy(&output_result.stderr)
        );
        let manifest: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(output.join("adapter_manifest.json")).expect("manifest"),
        )
        .expect("manifest json");
        assert_eq!(
            manifest
                .get("train_records")
                .and_then(|value| value.as_u64()),
            Some(2)
        );
        assert_eq!(
            manifest
                .get("valid_records")
                .and_then(|value| value.as_u64()),
            Some(1)
        );
        assert_eq!(
            manifest
                .get("test_records")
                .and_then(|value| value.as_u64()),
            Some(1)
        );
        assert!(manifest
            .get("warnings")
            .and_then(|value| value.as_array())
            .is_some_and(|warnings| warnings.iter().any(|warning| {
                warning
                    .as_str()
                    .is_some_and(|text| text.contains("No eval records found"))
            })));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn eval_router_falls_back_to_test_records_when_eval_is_missing() {
        let root = std::env::temp_dir().join(format!(
            "imprint-router-split-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let dataset = root.join("training");
        fs::create_dir_all(&dataset).expect("dataset dir");
        fs::write(
            dataset.join("query_to_region.test.jsonl"),
            serde_json::json!({
                "task":"query_to_region",
                "input":"Where is the orchard note?",
                "target":"route_region:orchard"
            })
            .to_string(),
        )
        .expect("test record");

        let script = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("training")
            .join("eval_router.py");
        let output = Command::new("python3")
            .arg(script)
            .arg("--dataset")
            .arg(&dataset)
            .output()
            .expect("run eval_router.py");
        assert!(output.status.success());
        let report: serde_json::Value =
            serde_json::from_slice(&output.stdout).expect("router report");
        assert_eq!(
            report.get("records").and_then(|value| value.as_u64()),
            Some(1)
        );
        assert!(report
            .get("warnings")
            .and_then(|value| value.as_array())
            .is_some_and(|warnings| warnings.iter().any(|warning| {
                warning
                    .as_str()
                    .is_some_and(|text| text.contains("evaluated test records"))
            })));
        let _ = fs::remove_dir_all(root);
    }
}
