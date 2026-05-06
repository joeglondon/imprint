use crate::store::{FileMemoryStore, MemoryStore};
use crate::surf::ExpandMode;
use crate::types::*;
use anyhow::{anyhow, Context};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::{self, BufRead, Write};
use std::path::Path;

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
        "memory_open" => {
            let node = node_arg(&args)?;
            serde_json::to_value(crate::app::surf_open(store_root, &node)?)?
        }
        "memory_neighbors" => {
            let node = node_arg(&args)?;
            serde_json::to_value(crate::app::surf_neighbors(
                store_root,
                &node,
                usize_arg(&args, "max_results").unwrap_or(8),
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
        "memory_write_summary" => {
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
        "memory_write_web_finding" => serde_json::to_value(crate::app::write_web_finding(
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
                retrieved_at: args
                    .get("retrieved_at")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
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
        "memory_write_link" => serde_json::to_value(crate::app::write_agent_link(
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
        )?)?,
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
        "memory_chat_trace" => {
            let session_id = string_arg(&args, "session_id")?;
            serde_json::to_value(crate::app::list_chat_context_traces(
                store_root,
                &session_id,
            )?)?
        }
        "memory_compile" => serde_json::to_value(crate::app::compile_memory_brain(store_root)?)?,
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

fn tools() -> Vec<Value> {
    vec![
        tool(
            "memory_search",
            "Search local AI memory and return anchored chunks.",
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
            "memory_open",
            "Open a document, chunk, or region node.",
            node_schema(),
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
            "memory_write_web_finding",
            "Store a supplied web finding with URL, query, retrieval date, and provenance.",
            json!({
                "type": "object",
                "properties": {
                    "session_id": { "type": "string" },
                    "query": { "type": "string" },
                    "url": { "type": "string" },
                    "title": { "type": "string" },
                    "summary": { "type": "string" },
                    "retrieved_at": { "type": "integer" },
                    "confidence": { "type": "number" },
                    "actor": { "type": "string" }
                },
                "required": ["query", "url", "summary"]
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
            "memory_mark_attention",
            "Mark memory active, hot, warm, cold, promoted, decayed, pinned, or suppressed with a reason.",
            json!({
                "type": "object",
                "properties": {
                    "target_id": { "type": "string" },
                    "target_kind": { "type": "string", "enum": ["chat_session", "chat_message", "transcript_chunk", "derived_memory", "web_finding", "document", "chunk", "region", "link"] },
                    "action": { "type": "string", "enum": ["active", "hot", "warm", "cold", "promote", "decay", "pin", "suppress"] },
                    "reason": { "type": "string" },
                    "actor": { "type": "string" }
                },
                "required": ["target_id", "target_kind", "action", "reason"]
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
            "Compile compact source-grounded brain artifacts for tiny local models.",
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
        assert!(names.contains(&"memory_write_summary".to_string()));
        assert!(names.contains(&"memory_write_web_finding".to_string()));
        assert!(names.contains(&"memory_write_link".to_string()));
        assert!(names.contains(&"memory_mark_attention".to_string()));
        assert!(names.contains(&"memory_chat_trace".to_string()));
        assert!(names.contains(&"memory_compile".to_string()));
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
}
