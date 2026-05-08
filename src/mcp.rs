use crate::store::{FileMemoryStore, MemoryStore};
use crate::surf::ExpandMode;
use crate::types::*;
use anyhow::{anyhow, Context};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

pub fn run_stdio(store_root: &Path) -> anyhow::Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    FileMemoryStore::new(store_root).load()?;
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let request = serde_json::from_str::<Value>(&line)?;
        if let Some(response) = handle_request(store_root, request) {
            writeln!(stdout, "{}", serde_json::to_string(&response)?)?;
            stdout.flush()?;
        }
    }
    Ok(())
}

fn handle_request(store_root: &Path, request: Value) -> Option<Value> {
    let id = request.get("id").cloned();
    let method = request
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let result = match method {
        "initialize" => Ok(json!({
            "protocolVersion": "2024-11-05",
            "serverInfo": { "name": "ai-memory", "version": env!("CARGO_PKG_VERSION") },
            "capabilities": { "tools": {} }
        })),
        "tools/list" => Ok(json!({ "tools": tools() })),
        "tools/call" => call_tool(
            store_root,
            request.get("params").cloned().unwrap_or(Value::Null),
        ),
        _ => Err(anyhow!("unknown MCP method {method}")),
    };
    id.map(|id| match result {
        Ok(result) => json!({ "jsonrpc": "2.0", "id": id, "result": result }),
        Err(error) => json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": { "code": -32000, "message": error.to_string() }
        }),
    })
}

fn call_tool(store_root: &Path, params: Value) -> anyhow::Result<Value> {
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .context("tool name missing")?;
    let args = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let payload = match name {
        "memory_search" => {
            let query = string_arg(&args, "query")?;
            let max_regions = usize_arg(&args, "max_regions").unwrap_or(3);
            let max_chunks = usize_arg(&args, "max_chunks").unwrap_or(8);
            serde_json::to_value(crate::app::run_query(
                store_root,
                QueryRequest {
                    text: query,
                    filters: BTreeMap::new(),
                    max_regions,
                    max_chunks,
                },
            )?)?
        }
        "memory_cortex_status" => cortex_status_payload(store_root, &args)?,
        "memory_cortex_compile" | "memory_compile" => {
            serde_json::to_value(crate::app::compile_memory_brain(store_root)?)?
        }
        "memory_cortex_train" => {
            serde_json::to_value(crate::app::train_cortex_adapter_now(store_root)?)?
        }
        "memory_cortex_route" => {
            let query = string_arg(&args, "query")?;
            let max_regions = usize_arg(&args, "max_regions").unwrap_or(3);
            let result = crate::app::run_query(
                store_root,
                QueryRequest {
                    text: query,
                    filters: BTreeMap::new(),
                    max_regions,
                    max_chunks: usize_arg(&args, "max_chunks").unwrap_or(0),
                },
            )?;
            let route_plan = result.routed.route_plan.clone();
            serde_json::to_value(json!({
                "routed": result.routed,
                "route_plan": route_plan,
            }))?
        }
        "memory_cortex_eval" => {
            let compile = crate::app::compile_memory_brain(store_root)?;
            let adapter_state = compile
                .adapter_state
                .as_ref()
                .context("compile did not return adapter state")?;
            serde_json::to_value(crate::training::evaluate_cortex_adapter(
                store_root,
                &adapter_state.current_source_dataset_hash,
                args.get("minimum_score")
                    .and_then(Value::as_f64)
                    .unwrap_or(crate::training::DEFAULT_ADAPTER_ACTIVATION_MIN_SCORE),
            )?)?
        }
        "memory_evaluation_harness" => {
            let compile = crate::app::compile_memory_brain(store_root)?;
            let adapter_state = compile
                .adapter_state
                .as_ref()
                .context("compile did not return adapter state")?;
            serde_json::to_value(crate::training::run_phase13_evaluation_harness(
                store_root,
                &adapter_state.current_source_dataset_hash,
                args.get("minimum_score")
                    .and_then(Value::as_f64)
                    .unwrap_or(crate::training::DEFAULT_ADAPTER_ACTIVATION_MIN_SCORE),
            )?)?
        }
        "memory_open" => {
            let node = node_arg(&args)?;
            serde_json::to_value(crate::app::surf_open(store_root, &node)?)?
        }
        "memory_open_source_artifact" => {
            open_source_artifact_payload(store_root, &string_arg(&args, "source_artifact_id")?)?
        }
        "memory_neighbors" => {
            let node = node_arg(&args)?;
            serde_json::to_value(crate::app::surf_neighbors(
                store_root,
                &node,
                usize_arg(&args, "max_results").unwrap_or(8),
            )?)?
        }
        "memory_links" => {
            let node = node_arg(&args)?;
            serde_json::to_value(crate::app::list_links(store_root, &node)?)?
        }
        "memory_inspect_link" => {
            let link_id = string_arg(&args, "link_id")?;
            serde_json::to_value(crate::app::inspect_link(store_root, &link_id)?)?
        }
        "memory_mark_link" => {
            let link_id = string_arg(&args, "link_id")?;
            serde_json::to_value(crate::app::mark_link_attention(
                store_root,
                &link_id,
                attention_action_arg(&args)?,
                string_arg(&args, "reason")?,
                args.get("actor")
                    .and_then(Value::as_str)
                    .unwrap_or("mcp-agent")
                    .to_string(),
            )?)?
        }
        "memory_expand" => {
            let chunk_id = string_arg(&args, "chunk_id")?;
            let mode = expand_mode_arg(&args).unwrap_or(ExpandMode::Window);
            serde_json::to_value(crate::app::surf_expand(
                store_root,
                &chunk_id,
                mode,
                usize_arg(&args, "window").unwrap_or(420),
            )?)?
        }
        "memory_jump_to_anchor" => {
            let anchor_id = string_arg(&args, "anchor_id")?;
            serde_json::to_value(crate::app::surf_jump_to_anchor(
                store_root,
                &anchor_id,
                usize_arg(&args, "window").unwrap_or(420),
            )?)?
        }
        "memory_expand_anchor" => {
            let anchor_id = string_arg(&args, "anchor_id")?;
            serde_json::to_value(crate::app::surf_jump_to_anchor(
                store_root,
                &anchor_id,
                usize_arg(&args, "window").unwrap_or(420),
            )?)?
        }
        "memory_list_provenance" => provenance_payload(
            store_root,
            &string_arg(&args, "target_id")?,
            args.get("target_kind").and_then(Value::as_str),
        )?,
        "memory_cite_hit" => {
            let actor = args
                .get("actor")
                .and_then(Value::as_str)
                .unwrap_or("mcp-agent")
                .to_string();
            serde_json::to_value(crate::app::cite_source_anchor(
                store_root,
                &string_arg(&args, "anchor_id")?,
                usize_arg(&args, "window").unwrap_or(420),
                actor,
            )?)?
        }
        "memory_save_trail" => {
            let session: SessionState =
                serde_json::from_value(args.get("session").cloned().context("session missing")?)
                    .context("decoding session")?;
            serde_json::to_value(crate::app::save_trail_from_session(
                store_root,
                &session,
                &string_arg(&args, "name")?,
            )?)?
        }
        "memory_write_summary" | "memory_save_derived_memory" => {
            let text = string_arg(&args, "text")?;
            let session_id = args
                .get("session_id")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned);
            let actor = args
                .get("actor")
                .and_then(Value::as_str)
                .unwrap_or("mcp-agent")
                .to_string();
            serde_json::to_value(crate::app::write_derived_memory(
                store_root,
                DerivedMemoryWrite {
                    session_id,
                    kind: DerivedMemoryKind::Summary,
                    text,
                    source_message_ids: Vec::new(),
                    actor,
                    confidence: args
                        .get("confidence")
                        .and_then(Value::as_f64)
                        .unwrap_or(0.75) as f32,
                },
            )?)?
        }
        "memory_write_web_finding"
        | "memory_save_web_finding"
        | "memory_capture_url"
        | "memory_save_search_result" => serde_json::to_value(crate::app::write_web_finding(
            store_root,
            WebFindingWrite {
                session_id: args
                    .get("session_id")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                query: string_arg(&args, "query")?,
                url: string_arg(&args, "url")?,
                title: args
                    .get("title")
                    .and_then(Value::as_str)
                    .unwrap_or("Web finding")
                    .to_string(),
                summary: string_arg(&args, "summary")?,
                extracted_text: args
                    .get("extracted_text")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                retrieved_at: args
                    .get("retrieved_at")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
                freshness_expires_at: args.get("freshness_expires_at").and_then(Value::as_u64),
                source_refs: args
                    .get("source_refs")
                    .and_then(Value::as_array)
                    .map(|values| {
                        values
                            .iter()
                            .filter_map(Value::as_str)
                            .map(ToOwned::to_owned)
                            .collect()
                    })
                    .unwrap_or_default(),
                source_trust: None,
                confidence: args
                    .get("confidence")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.75) as f32,
                actor: args
                    .get("actor")
                    .and_then(Value::as_str)
                    .unwrap_or("mcp-agent")
                    .to_string(),
            },
        )?)?,
        "memory_refresh_web_finding" => {
            let existing = web_finding_arg(store_root, &args)?;
            serde_json::to_value(crate::app::write_web_finding(
                store_root,
                WebFindingWrite {
                    session_id: existing.session_id,
                    query: args
                        .get("query")
                        .and_then(Value::as_str)
                        .unwrap_or(&existing.query)
                        .to_string(),
                    url: existing.url,
                    title: args
                        .get("title")
                        .and_then(Value::as_str)
                        .unwrap_or(&existing.title)
                        .to_string(),
                    summary: string_arg(&args, "summary")?,
                    extracted_text: args
                        .get("extracted_text")
                        .and_then(Value::as_str)
                        .map(ToOwned::to_owned),
                    retrieved_at: args
                        .get("retrieved_at")
                        .and_then(Value::as_u64)
                        .unwrap_or_else(now_secs),
                    freshness_expires_at: args.get("freshness_expires_at").and_then(Value::as_u64),
                    source_refs: args
                        .get("source_refs")
                        .and_then(Value::as_array)
                        .map(|values| {
                            values
                                .iter()
                                .filter_map(Value::as_str)
                                .map(ToOwned::to_owned)
                                .collect()
                        })
                        .unwrap_or_default(),
                    source_trust: None,
                    confidence: args
                        .get("confidence")
                        .and_then(Value::as_f64)
                        .unwrap_or(existing.confidence as f64)
                        as f32,
                    actor: args
                        .get("actor")
                        .and_then(Value::as_str)
                        .unwrap_or("mcp-agent")
                        .to_string(),
                },
            )?)?
        }
        "memory_set_web_freshness" => serde_json::to_value(crate::app::set_web_finding_freshness(
            store_root,
            &string_arg(&args, "web_finding_id")?,
            args.get("freshness_expires_at").and_then(Value::as_u64),
            args.get("actor")
                .and_then(Value::as_str)
                .unwrap_or("mcp-agent")
                .to_string(),
        )?)?,
        "memory_web_finding_history" => {
            serde_json::to_value(crate::app::list_web_finding_history(
                store_root,
                args.get("url")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
            )?)?
        }
        "memory_web_refresh_candidates" => {
            serde_json::to_value(crate::app::list_pinned_web_refresh_candidates(store_root)?)?
        }
        "memory_write_link" | "memory_save_agent_link" => {
            serde_json::to_value(crate::app::write_agent_link(
                store_root,
                AgentLinkWrite {
                    source_id: string_arg(&args, "source_id")?,
                    target_id: string_arg(&args, "target_id")?,
                    label: string_arg(&args, "label")?,
                    actor: args
                        .get("actor")
                        .and_then(Value::as_str)
                        .unwrap_or("mcp-agent")
                        .to_string(),
                },
            )?)?
        }
        "memory_mark_attention" => serde_json::to_value(crate::app::apply_attention_mark(
            store_root,
            AttentionMarkWrite {
                target_id: string_arg(&args, "target_id")?,
                target_kind: attention_target_kind_arg(&args)?,
                action: attention_action_arg(&args)?,
                reason: string_arg(&args, "reason")?,
                actor: args
                    .get("actor")
                    .and_then(Value::as_str)
                    .unwrap_or("mcp-agent")
                    .to_string(),
            },
        )?)?,
        "memory_revert_attention_mark" => serde_json::to_value(crate::app::revert_attention_mark(
            store_root,
            &string_arg(&args, "mark_id")?,
            args.get("actor")
                .and_then(Value::as_str)
                .unwrap_or("mcp-agent")
                .to_string(),
        )?)?,
        "memory_chat_trace" => {
            let session_id = string_arg(&args, "session_id")?;
            serde_json::to_value(crate::app::list_chat_context_traces(
                store_root,
                &session_id,
            )?)?
        }
        _ => return Err(anyhow!("unknown memory tool {name}")),
    };
    Ok(json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string_pretty(&payload)?
        }],
        "isError": false
    }))
}

fn cortex_status_payload(store_root: &Path, args: &Value) -> anyhow::Result<Value> {
    let store = FileMemoryStore::new(store_root);
    let snapshot = crate::app::load_cortex_adapter_snapshot(store_root)?;
    let cortex_index = store.load_current_cortex_index()?;
    let recent_jobs = store
        .list_cortex_adapter_jobs(None)?
        .into_iter()
        .take(usize_arg(args, "max_jobs").unwrap_or(10))
        .collect::<Vec<_>>();
    Ok(json!({
        "adapter_state": snapshot.adapter_state,
        "cortex_index": cortex_index.as_ref().map(|index| json!({
            "id": index.id,
            "schema_version": index.schema_version,
            "created_at": index.created_at,
            "corpus_hash": index.corpus_hash,
            "source_refs": index.source_refs.len(),
            "artifact_ids": index.artifact_ids.len(),
            "regions": index.regions.len(),
            "route_examples": index.regions.iter().map(|region| region.route_examples.len()).sum::<usize>(),
        })),
        "recent_jobs": recent_jobs,
    }))
}

fn open_source_artifact_payload(
    store_root: &Path,
    source_artifact_id: &str,
) -> anyhow::Result<Value> {
    let store = FileMemoryStore::new(store_root);
    let artifact = store
        .list_source_artifacts()?
        .into_iter()
        .find(|artifact| artifact.id == source_artifact_id)
        .context("source artifact not found")?;
    let memory = store.load()?;
    let documents = memory
        .documents
        .iter()
        .filter(|document| {
            document
                .metadata
                .get("source_artifact_id")
                .is_some_and(|id| id == source_artifact_id)
        })
        .map(|document| {
            json!({
                "document_id": document.id,
                "title": document.title,
                "source_anchor": document.source_anchor,
            })
        })
        .collect::<Vec<_>>();
    Ok(json!({
        "source_artifact": artifact,
        "documents": documents,
    }))
}

fn provenance_payload(
    store_root: &Path,
    target_id: &str,
    target_kind: Option<&str>,
) -> anyhow::Result<Value> {
    let store = FileMemoryStore::new(store_root);
    let memory = store.load()?;
    let artifacts = store.list_source_artifacts()?;
    let derived = store.list_derived_memories(None)?;
    let web = store.list_web_findings(None)?;
    let links = store.list_agent_links()?;

    let mut records = Vec::new();
    if target_kind.is_none() || target_kind == Some("source_artifact") {
        records.extend(artifacts.iter().filter(|item| item.id == target_id).map(|item| {
            json!({ "target_kind": "source_artifact", "target_id": item.id, "provenance": item.provenance })
        }));
    }
    if target_kind.is_none() || target_kind == Some("document") {
        records.extend(
            memory
                .documents
                .iter()
                .filter(|item| item.id == target_id)
                .map(|item| {
                    json!({
                        "target_kind": "document",
                        "target_id": item.id,
                        "source_anchor": item.source_anchor,
                        "source_artifact_id": item.metadata.get("source_artifact_id"),
                        "metadata": item.metadata,
                    })
                }),
        );
    }
    if target_kind.is_none() || target_kind == Some("chunk") {
        records.extend(
            memory
                .chunks
                .iter()
                .filter(|item| item.id == target_id)
                .map(|item| {
                    json!({
                        "target_kind": "chunk",
                        "target_id": item.id,
                        "document_id": item.document_id,
                        "source_anchor": item.source_anchor,
                        "metadata": item.metadata,
                    })
                }),
        );
    }
    if target_kind.is_none() || target_kind == Some("derived_memory") {
        records.extend(derived.iter().filter(|item| item.id == target_id).map(|item| {
            json!({ "target_kind": "derived_memory", "target_id": item.id, "provenance": item.provenance })
        }));
    }
    if target_kind.is_none() || target_kind == Some("web_finding") {
        records.extend(web.iter().filter(|item| item.id == target_id).map(|item| {
            json!({ "target_kind": "web_finding", "target_id": item.id, "provenance": item.provenance, "freshness_expires_at": item.freshness_expires_at })
        }));
    }
    if target_kind.is_none() || target_kind == Some("link") {
        records.extend(links.iter().filter(|item| item.id == target_id).map(|item| {
            json!({ "target_kind": "link", "target_id": item.id, "provenance": item.provenance })
        }));
    }
    if records.is_empty() {
        return Err(anyhow!("no provenance found for {target_id}"));
    }
    Ok(json!({ "target_id": target_id, "records": records }))
}

fn web_finding_arg(store_root: &Path, args: &Value) -> anyhow::Result<WebFinding> {
    let id = string_arg(args, "web_finding_id")?;
    FileMemoryStore::new(store_root)
        .list_web_findings(None)?
        .into_iter()
        .find(|finding| finding.id == id)
        .context("web finding not found")
}

fn tools() -> Vec<Value> {
    vec![
        tool(
            "memory_search",
            "Search local AI memory and return anchored chunks plus a typed route_plan.",
            json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "max_regions": { "type": "integer" },
                    "max_chunks": { "type": "integer" }
                },
                "required": ["query"]
            }),
        ),
        tool(
            "memory_cortex_status",
            "Return cortex index, adapter lifecycle, and recent training job status.",
            json!({
                "type": "object",
                "properties": { "max_jobs": { "type": "integer" } }
            }),
        ),
        tool(
            "memory_cortex_compile",
            "Compile the current CortexIndex and prepared adapter dataset.",
            json!({ "type": "object", "properties": {} }),
        ),
        tool(
            "memory_cortex_train",
            "Queue and start cortex adapter training for the current source dataset.",
            json!({ "type": "object", "properties": {} }),
        ),
        tool(
            "memory_cortex_route",
            "Return the cortex-guided route plan for a query without requiring answer generation.",
            json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "max_regions": { "type": "integer" },
                    "max_chunks": { "type": "integer" }
                },
                "required": ["query"]
            }),
        ),
        tool(
            "memory_cortex_eval",
            "Evaluate the current trained adapter against cortex activation gates.",
            json!({
                "type": "object",
                "properties": { "minimum_score": { "type": "number" } }
            }),
        ),
        tool(
            "memory_evaluation_harness",
            "Run the full phase 13 evaluation harness over eval sets, baselines, gates, fixtures, and metric history.",
            json!({
                "type": "object",
                "properties": { "minimum_score": { "type": "number" } }
            }),
        ),
        tool(
            "memory_open",
            "Open a document, chunk, or region node.",
            node_schema(),
        ),
        tool(
            "memory_open_source_artifact",
            "Open a source artifact provenance record and list documents imported from it.",
            json!({
                "type": "object",
                "properties": { "source_artifact_id": { "type": "string" } },
                "required": ["source_artifact_id"]
            }),
        ),
        tool(
            "memory_neighbors",
            "List semantic, document, and graph neighbors for a node.",
            json!({
                "type": "object",
                "properties": {
                    "node": node_value_schema(),
                    "max_results": { "type": "integer" }
                },
                "required": ["node"]
            }),
        ),
        tool(
            "memory_links",
            "List inspectable graph links for a document, chunk, or region node.",
            node_schema(),
        ),
        tool(
            "memory_inspect_link",
            "Explain why a graph link exists, with evidence, confidence, provenance, and attention marks.",
            json!({
                "type": "object",
                "properties": {
                    "link_id": { "type": "string" }
                },
                "required": ["link_id"]
            }),
        ),
        tool(
            "memory_mark_link",
            "Pin, promote, suppress, or otherwise mark a graph link with reversible attention.",
            json!({
                "type": "object",
                "properties": {
                    "link_id": { "type": "string" },
                    "action": { "type": "string", "enum": ["active", "hot", "warm", "cold", "promote", "decay", "pin", "suppress"] },
                    "reason": { "type": "string" },
                    "actor": { "type": "string" }
                },
                "required": ["link_id", "action", "reason"]
            }),
        ),
        tool(
            "memory_expand",
            "Expand a chunk by window, page, section, or document.",
            json!({
                "type": "object",
                "properties": {
                    "chunk_id": { "type": "string" },
                    "mode": { "type": "string", "enum": ["window", "page", "section", "document"] },
                    "window": { "type": "integer" }
                },
                "required": ["chunk_id"]
            }),
        ),
        tool(
            "memory_jump_to_anchor",
            "Jump directly to an anchored source location.",
            json!({
                "type": "object",
                "properties": {
                    "anchor_id": { "type": "string" },
                    "window": { "type": "integer" }
                },
                "required": ["anchor_id"]
            }),
        ),
        tool(
            "memory_expand_anchor",
            "Expand exact source context from an anchor id.",
            json!({
                "type": "object",
                "properties": {
                    "anchor_id": { "type": "string" },
                    "window": { "type": "integer" }
                },
                "required": ["anchor_id"]
            }),
        ),
        tool(
            "memory_list_provenance",
            "List provenance and source-anchor metadata for a memory target.",
            json!({
                "type": "object",
                "properties": {
                    "target_id": { "type": "string" },
                    "target_kind": { "type": "string", "enum": ["source_artifact", "document", "chunk", "derived_memory", "web_finding", "link"] }
                },
                "required": ["target_id"]
            }),
        ),
        tool(
            "memory_cite_hit",
            "Record a cite access and return exact source context for an anchor.",
            json!({
                "type": "object",
                "properties": {
                    "anchor_id": { "type": "string" },
                    "window": { "type": "integer" },
                    "actor": { "type": "string" }
                },
                "required": ["anchor_id"]
            }),
        ),
        tool(
            "memory_save_trail",
            "Save a reversible surf trail from a session state.",
            json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string" },
                    "session": { "type": "object" }
                },
                "required": ["name", "session"]
            }),
        ),
        tool(
            "memory_write_summary",
            "Write an agent summary back into memory with provenance.",
            json!({
                "type": "object",
                "properties": {
                    "session_id": { "type": "string" },
                    "text": { "type": "string" },
                    "actor": { "type": "string" },
                    "confidence": { "type": "number" }
                },
                "required": ["text"]
            }),
        ),
        tool(
            "memory_save_derived_memory",
            "Alias for memory_write_summary while clients migrate to explicit writeback naming.",
            json!({
                "type": "object",
                "properties": {
                    "session_id": { "type": "string" },
                    "text": { "type": "string" },
                    "actor": { "type": "string" },
                    "confidence": { "type": "number" }
                },
                "required": ["text"]
            }),
        ),
        tool(
            "memory_write_web_finding",
            "Store a supplied web finding with URL, query, retrieval date, and provenance.",
            web_finding_write_schema(),
        ),
        tool(
            "memory_save_web_finding",
            "Alias for saving supplied web findings with provenance.",
            web_finding_write_schema(),
        ),
        tool(
            "memory_capture_url",
            "Capture supplied URL evidence into memory with retrieval/freshness metadata.",
            web_finding_write_schema(),
        ),
        tool(
            "memory_save_search_result",
            "Save a search result as a source-grounded web finding.",
            web_finding_write_schema(),
        ),
        tool(
            "memory_refresh_web_finding",
            "Write a refreshed web finding version for an existing web_finding_id.",
            json!({
                "type": "object",
                "properties": {
                    "web_finding_id": { "type": "string" },
                    "query": { "type": "string" },
                    "title": { "type": "string" },
                    "summary": { "type": "string" },
                    "extracted_text": { "type": "string" },
                    "retrieved_at": { "type": "integer" },
                    "freshness_expires_at": { "type": "integer" },
                    "source_refs": {
                        "type": "array",
                        "items": { "type": "string" }
                    },
                    "confidence": { "type": "number" },
                    "actor": { "type": "string" }
                },
                "required": ["web_finding_id", "summary"]
            }),
        ),
        tool(
            "memory_set_web_freshness",
            "Set or clear a web finding freshness expiration.",
            json!({
                "type": "object",
                "properties": {
                    "web_finding_id": { "type": "string" },
                    "freshness_expires_at": { "type": "integer" },
                    "actor": { "type": "string" }
                },
                "required": ["web_finding_id"]
            }),
        ),
        tool(
            "memory_web_finding_history",
            "List content-hash revisions and simple diffs for captured web findings.",
            json!({
                "type": "object",
                "properties": {
                    "url": { "type": "string" }
                }
            }),
        ),
        tool(
            "memory_web_refresh_candidates",
            "List pinned web findings whose freshness window requires recapture.",
            json!({
                "type": "object",
                "properties": {}
            }),
        ),
        tool(
            "memory_write_link",
            "Write an inspectable agent-created link between memory targets.",
            json!({
                "type": "object",
                "properties": {
                    "source_id": { "type": "string" },
                    "target_id": { "type": "string" },
                    "label": { "type": "string" },
                    "actor": { "type": "string" }
                },
                "required": ["source_id", "target_id", "label"]
            }),
        ),
        tool(
            "memory_save_agent_link",
            "Alias for writing an inspectable agent-created memory link.",
            json!({
                "type": "object",
                "properties": {
                    "source_id": { "type": "string" },
                    "target_id": { "type": "string" },
                    "label": { "type": "string" },
                    "actor": { "type": "string" }
                },
                "required": ["source_id", "target_id", "label"]
            }),
        ),
        tool(
            "memory_mark_attention",
            "Mark memory active, hot, warm, cold, promoted, decayed, pinned, or suppressed with a reason.",
            json!({
                "type": "object",
                "properties": {
                    "target_id": { "type": "string" },
                    "target_kind": { "type": "string", "enum": ["chat_session", "session", "project", "workspace", "collection", "task", "chat_message", "transcript_chunk", "derived_memory", "web_finding", "document", "chunk", "region", "link"] },
                    "action": { "type": "string", "enum": ["active", "hot", "warm", "cold", "promote", "decay", "pin", "suppress"] },
                    "reason": { "type": "string" },
                    "actor": { "type": "string" }
                },
                "required": ["target_id", "target_kind", "action", "reason"]
            }),
        ),
        tool(
            "memory_revert_attention_mark",
            "Revert a previous attention mark by id.",
            json!({
                "type": "object",
                "properties": {
                    "mark_id": { "type": "string" },
                    "actor": { "type": "string" }
                },
                "required": ["mark_id"]
            }),
        ),
        tool(
            "memory_chat_trace",
            "List context traces showing which snippets were used for a chat session.",
            json!({
                "type": "object",
                "properties": {
                    "session_id": { "type": "string" }
                },
                "required": ["session_id"]
            }),
        ),
        tool(
            "memory_compile",
            "Compatibility alias for memory_cortex_compile.",
            json!({
                "type": "object",
                "properties": {}
            }),
        ),
    ]
}

fn tool(name: &str, description: &str, input_schema: Value) -> Value {
    json!({
        "name": name,
        "description": description,
        "inputSchema": input_schema
    })
}

fn node_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "node": node_value_schema()
        },
        "required": ["node"]
    })
}

fn node_value_schema() -> Value {
    json!({
        "oneOf": [
            { "type": "object", "properties": { "Document": { "type": "string" } }, "required": ["Document"] },
            { "type": "object", "properties": { "Chunk": { "type": "string" } }, "required": ["Chunk"] },
            { "type": "object", "properties": { "Region": { "type": "string" } }, "required": ["Region"] }
        ]
    })
}

fn web_finding_write_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "session_id": { "type": "string" },
            "query": { "type": "string" },
            "url": { "type": "string" },
            "title": { "type": "string" },
            "summary": { "type": "string" },
            "extracted_text": { "type": "string" },
            "retrieved_at": { "type": "integer" },
            "freshness_expires_at": { "type": "integer" },
            "source_refs": {
                "type": "array",
                "items": { "type": "string" }
            },
            "confidence": { "type": "number" },
            "actor": { "type": "string" }
        },
        "required": ["query", "url", "summary"]
    })
}

fn node_arg(args: &Value) -> anyhow::Result<NodeRef> {
    serde_json::from_value(args.get("node").cloned().context("node missing")?)
        .context("decoding node")
}

fn string_arg(args: &Value, key: &str) -> anyhow::Result<String> {
    args.get(key)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .with_context(|| format!("{key} missing"))
}

fn usize_arg(args: &Value, key: &str) -> Option<usize> {
    args.get(key)
        .and_then(Value::as_u64)
        .map(|value| value as usize)
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn expand_mode_arg(args: &Value) -> Option<ExpandMode> {
    match args.get("mode")?.as_str()?.to_ascii_lowercase().as_str() {
        "window" => Some(ExpandMode::Window),
        "page" => Some(ExpandMode::Page),
        "section" => Some(ExpandMode::Section),
        "document" => Some(ExpandMode::Document),
        _ => None,
    }
}

fn attention_action_arg(args: &Value) -> anyhow::Result<AttentionAction> {
    match string_arg(args, "action")?.to_ascii_lowercase().as_str() {
        "active" => Ok(AttentionAction::Active),
        "hot" => Ok(AttentionAction::Hot),
        "warm" => Ok(AttentionAction::Warm),
        "cold" => Ok(AttentionAction::Cold),
        "promote" => Ok(AttentionAction::Promote),
        "decay" => Ok(AttentionAction::Decay),
        "pin" => Ok(AttentionAction::Pin),
        "suppress" => Ok(AttentionAction::Suppress),
        value => Err(anyhow!("unknown attention action {value}")),
    }
}

fn attention_target_kind_arg(args: &Value) -> anyhow::Result<AttentionTargetKind> {
    match string_arg(args, "target_kind")?
        .to_ascii_lowercase()
        .as_str()
    {
        "chat_session" => Ok(AttentionTargetKind::ChatSession),
        "session" => Ok(AttentionTargetKind::Session),
        "project" => Ok(AttentionTargetKind::Project),
        "workspace" => Ok(AttentionTargetKind::Workspace),
        "collection" => Ok(AttentionTargetKind::Collection),
        "task" => Ok(AttentionTargetKind::Task),
        "chat_message" => Ok(AttentionTargetKind::ChatMessage),
        "transcript_chunk" => Ok(AttentionTargetKind::TranscriptChunk),
        "derived_memory" => Ok(AttentionTargetKind::DerivedMemory),
        "web_finding" => Ok(AttentionTargetKind::WebFinding),
        "document" => Ok(AttentionTargetKind::Document),
        "chunk" => Ok(AttentionTargetKind::Chunk),
        "region" => Ok(AttentionTargetKind::Region),
        "link" => Ok(AttentionTargetKind::Link),
        value => Err(anyhow!("unknown attention target kind {value}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::{save_model_config, ModelConfig, ModelConnectionMode, ModelRuntimePreset};
    use std::fs;

    fn temp_store_root(name: &str) -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!("ai-memory-mcp-{name}"));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("create temp root");
        save_model_config(
            &root,
            &ModelConfig {
                mode: ModelConnectionMode::Local,
                endpoint: "http://localhost:11434".into(),
                api_key_name: None,
                chat_model: None,
                planner_model: Some("hash".into()),
                response_model: Some("hash".into()),
                planner_endpoint: Some("http://localhost:11434".into()),
                planner_adapter_path: None,
                response_adapter_path: None,
                shared_cortex_adapter_path: None,
                active_adapter_hash: None,
                adapter_activation_policy: "automatic".into(),
                runtime_preset: ModelRuntimePreset::CustomOpenAi,
                cortex_enabled: true,
                latent_recursive_enabled: false,
                cortex_rounds: 3,
                critic_model: Some("hash".into()),
                critic_endpoint: Some("http://localhost:11434".into()),
                compiler_model: Some("hash".into()),
                embedding_model: Some("hash".into()),
                embedding_endpoint: Some("http://localhost:11434".into()),
                embedding_runtime_preset: Some(ModelRuntimePreset::Ollama),
                health: None,
            },
        )
        .expect("save config");
        root
    }

    #[test]
    fn mcp_lists_memory_write_tools() {
        let names = tools()
            .into_iter()
            .filter_map(|tool| {
                tool.get("name")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
            })
            .collect::<Vec<_>>();
        assert!(names.contains(&"memory_cortex_status".to_string()));
        assert!(names.contains(&"memory_cortex_compile".to_string()));
        assert!(names.contains(&"memory_cortex_train".to_string()));
        assert!(names.contains(&"memory_cortex_route".to_string()));
        assert!(names.contains(&"memory_cortex_eval".to_string()));
        assert!(names.contains(&"memory_open_source_artifact".to_string()));
        assert!(names.contains(&"memory_expand_anchor".to_string()));
        assert!(names.contains(&"memory_list_provenance".to_string()));
        assert!(names.contains(&"memory_cite_hit".to_string()));
        assert!(names.contains(&"memory_save_trail".to_string()));
        assert!(names.contains(&"memory_write_summary".to_string()));
        assert!(names.contains(&"memory_save_derived_memory".to_string()));
        assert!(names.contains(&"memory_write_web_finding".to_string()));
        assert!(names.contains(&"memory_save_web_finding".to_string()));
        assert!(names.contains(&"memory_capture_url".to_string()));
        assert!(names.contains(&"memory_save_search_result".to_string()));
        assert!(names.contains(&"memory_refresh_web_finding".to_string()));
        assert!(names.contains(&"memory_set_web_freshness".to_string()));
        assert!(names.contains(&"memory_web_finding_history".to_string()));
        assert!(names.contains(&"memory_web_refresh_candidates".to_string()));
        assert!(names.contains(&"memory_write_link".to_string()));
        assert!(names.contains(&"memory_save_agent_link".to_string()));
        assert!(names.contains(&"memory_links".to_string()));
        assert!(names.contains(&"memory_inspect_link".to_string()));
        assert!(names.contains(&"memory_mark_link".to_string()));
        assert!(names.contains(&"memory_mark_attention".to_string()));
        assert!(names.contains(&"memory_revert_attention_mark".to_string()));
        assert!(names.contains(&"memory_chat_trace".to_string()));
        assert!(names.contains(&"memory_compile".to_string()));
    }

    #[test]
    fn mcp_tool_schemas_are_documented_for_phase_ten_surface() {
        let docs = fs::read_to_string("docs/mcp-tools.md").expect("mcp docs");
        for name in tools()
            .into_iter()
            .filter_map(|tool| tool.get("name").and_then(Value::as_str).map(str::to_owned))
        {
            assert!(docs.contains(&name), "missing docs for {name}");
        }
    }

    #[test]
    fn mcp_write_summary_persists_audited_memory() {
        let root = temp_store_root("write-summary");
        let session = crate::app::create_chat_session(&root, Some("MCP".into())).expect("session");
        let result = call_tool(
            &root,
            json!({
                "name": "memory_write_summary",
                "arguments": {
                    "session_id": session.id,
                    "text": "Chat transcripts are indexed as hot memory.",
                    "actor": "mcp-test"
                }
            }),
        )
        .expect("call summary tool");
        let content = result["content"][0]["text"].as_str().expect("text payload");
        assert!(content.contains("Chat transcripts are indexed"));
        let memories = crate::app::list_derived_memories(&root, None).expect("derived memories");
        assert_eq!(memories.len(), 1);
        let audit = crate::app::list_audit_events(&root, None).expect("audit");
        assert!(audit
            .iter()
            .any(|event| event.event_type == "derived_memory.write"));
    }

    #[test]
    fn mcp_can_inspect_and_mark_graph_links() {
        let root = temp_store_root("inspect-link");
        let first = root.join("first.md");
        let second = root.join("second.md");
        fs::write(&first, "# First\n\nAlpha source.").expect("write first");
        fs::write(&second, "# Second\n\nBeta source.").expect("write second");
        crate::app::ingest_paths(&root, &[first, second]).expect("ingest");

        let memory = FileMemoryStore::new(&root).load().expect("memory");
        let source_id = memory.documents[0].id.clone();
        let target_id = memory.documents[1].id.clone();
        let write_result = call_tool(
            &root,
            json!({
                "name": "memory_write_link",
                "arguments": {
                    "source_id": format!("document:{source_id}"),
                    "target_id": format!("document:{target_id}"),
                    "label": "related source",
                    "actor": "mcp-test"
                }
            }),
        )
        .expect("write link");
        let link: AgentLinkMemory =
            serde_json::from_str(write_result["content"][0]["text"].as_str().expect("text"))
                .expect("decode link");

        let links_result = call_tool(
            &root,
            json!({
                "name": "memory_links",
                "arguments": {
                    "node": { "Document": source_id }
                }
            }),
        )
        .expect("list links");
        assert!(links_result["content"][0]["text"]
            .as_str()
            .expect("text")
            .contains(&link.id));

        let inspect_result = call_tool(
            &root,
            json!({
                "name": "memory_inspect_link",
                "arguments": {
                    "link_id": link.id.clone()
                }
            }),
        )
        .expect("inspect link");
        assert!(inspect_result["content"][0]["text"]
            .as_str()
            .expect("text")
            .contains("explicitly wrote"));

        let mark_result = call_tool(
            &root,
            json!({
                "name": "memory_mark_link",
                "arguments": {
                    "link_id": link.id,
                    "action": "suppress",
                    "reason": "too noisy",
                    "actor": "mcp-test"
                }
            }),
        )
        .expect("mark link");
        assert!(mark_result["content"][0]["text"]
            .as_str()
            .expect("text")
            .contains("Suppress"));
    }

    #[test]
    fn mcp_search_output_includes_route_plan() {
        let root = temp_store_root("route-plan");
        let source = root.join("memory.md");
        fs::write(
            &source,
            "# Cortex\n\nCortex routing should search, open, expand, and cite source anchors.",
        )
        .expect("write source");
        crate::app::ingest_paths(&root, &[source]).expect("ingest");

        let result = call_tool(
            &root,
            json!({
                "name": "memory_search",
                "arguments": {
                    "query": "How should cortex routing cite anchors?",
                    "max_regions": 2,
                    "max_chunks": 2
                }
            }),
        )
        .expect("search");
        let payload: Value =
            serde_json::from_str(result["content"][0]["text"].as_str().expect("text"))
                .expect("decode payload");
        assert!(payload["routed"]["route_plan"]["next_steps"]
            .as_array()
            .is_some_and(|steps| !steps.is_empty()));
    }

    #[test]
    fn mcp_source_recall_and_writeback_tools_return_compatible_json() {
        let root = temp_store_root("source-recall");
        let source = root.join("source.md");
        fs::write(
            &source,
            "# Recall\n\nExact recall should cite anchors and preserve provenance.",
        )
        .expect("write source");
        crate::app::ingest_paths(&root, &[source]).expect("ingest");
        let memory = FileMemoryStore::new(&root).load().expect("memory");
        let chunk = memory.chunks.first().expect("chunk");
        let anchor = chunk.source_anchor.as_ref().expect("anchor");
        let source_artifact_id = anchor.source_artifact_id.as_ref().expect("artifact");

        let opened = call_tool(
            &root,
            json!({
                "name": "memory_open_source_artifact",
                "arguments": { "source_artifact_id": source_artifact_id }
            }),
        )
        .expect("open artifact");
        assert!(opened["content"][0]["text"]
            .as_str()
            .expect("text")
            .contains(source_artifact_id));

        let cited = call_tool(
            &root,
            json!({
                "name": "memory_cite_hit",
                "arguments": { "anchor_id": anchor.id, "actor": "mcp-test" }
            }),
        )
        .expect("cite");
        assert!(cited["content"][0]["text"]
            .as_str()
            .expect("text")
            .contains("Exact recall"));

        let provenance = call_tool(
            &root,
            json!({
                "name": "memory_list_provenance",
                "arguments": { "target_id": chunk.id, "target_kind": "chunk" }
            }),
        )
        .expect("provenance");
        assert!(provenance["content"][0]["text"]
            .as_str()
            .expect("text")
            .contains("source_anchor"));
    }

    #[test]
    fn mcp_web_capture_and_attention_revert_are_compatible() {
        let root = temp_store_root("web-capture");
        let finding_result = call_tool(
            &root,
            json!({
                "name": "memory_capture_url",
                "arguments": {
                    "query": "semantic filesystem provenance",
                    "url": "https://example.com/imprint",
                    "summary": "Captured evidence keeps URL provenance.",
                    "retrieved_at": 1,
                    "actor": "mcp-test"
                }
            }),
        )
        .expect("capture");
        let finding: WebFinding =
            serde_json::from_str(finding_result["content"][0]["text"].as_str().expect("text"))
                .expect("decode finding");

        let refreshed = call_tool(
            &root,
            json!({
                "name": "memory_refresh_web_finding",
                "arguments": {
                    "web_finding_id": finding.id,
                    "summary": "Refreshed evidence keeps URL provenance.",
                    "retrieved_at": 2,
                    "actor": "mcp-test"
                }
            }),
        )
        .expect("refresh");
        assert!(refreshed["content"][0]["text"]
            .as_str()
            .expect("text")
            .contains("Refreshed evidence"));

        let mark_result = call_tool(
            &root,
            json!({
                "name": "memory_mark_attention",
                "arguments": {
                    "target_id": "workspace-1",
                    "target_kind": "workspace",
                    "action": "hot",
                    "reason": "active workspace",
                    "actor": "mcp-test"
                }
            }),
        )
        .expect("mark");
        let mark: AttentionMark =
            serde_json::from_str(mark_result["content"][0]["text"].as_str().expect("text"))
                .expect("decode mark");
        let reverted = call_tool(
            &root,
            json!({
                "name": "memory_revert_attention_mark",
                "arguments": { "mark_id": mark.id, "actor": "mcp-test" }
            }),
        )
        .expect("revert");
        assert!(reverted["content"][0]["text"]
            .as_str()
            .expect("text")
            .contains("reverted_at"));
    }
}
