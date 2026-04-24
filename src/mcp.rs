use crate::store::{FileMemoryStore, MemoryStore};
use crate::surf::ExpandMode;
use crate::types::{NodeRef, QueryRequest};
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
    let method = request.get("method").and_then(Value::as_str).unwrap_or_default();
    let result = match method {
        "initialize" => Ok(json!({
            "protocolVersion": "2024-11-05",
            "serverInfo": { "name": "ai-memory", "version": env!("CARGO_PKG_VERSION") },
            "capabilities": { "tools": {} }
        })),
        "tools/list" => Ok(json!({ "tools": tools() })),
        "tools/call" => call_tool(store_root, request.get("params").cloned().unwrap_or(Value::Null)),
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
    let args = params.get("arguments").cloned().unwrap_or_else(|| json!({}));
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
        tool("memory_search", "Search local AI memory and return anchored chunks.", json!({
            "type": "object",
            "properties": {
                "query": { "type": "string" },
                "max_regions": { "type": "integer" },
                "max_chunks": { "type": "integer" }
            },
            "required": ["query"]
        })),
        tool("memory_open", "Open a document, chunk, or region node.", node_schema()),
        tool("memory_neighbors", "List semantic, document, and graph neighbors for a node.", json!({
            "type": "object",
            "properties": {
                "node": node_value_schema(),
                "max_results": { "type": "integer" }
            },
            "required": ["node"]
        })),
        tool("memory_expand", "Expand a chunk by window, page, section, or document.", json!({
            "type": "object",
            "properties": {
                "chunk_id": { "type": "string" },
                "mode": { "type": "string", "enum": ["window", "page", "section", "document"] },
                "window": { "type": "integer" }
            },
            "required": ["chunk_id"]
        })),
        tool("memory_jump_to_anchor", "Jump directly to an anchored source location.", json!({
            "type": "object",
            "properties": {
                "anchor_id": { "type": "string" },
                "window": { "type": "integer" }
            },
            "required": ["anchor_id"]
        })),
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
    serde_json::from_value(args.get("node").cloned().context("node missing")?).context("decoding node")
}

fn string_arg(args: &Value, key: &str) -> anyhow::Result<String> {
    args.get(key)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .with_context(|| format!("{key} missing"))
}

fn usize_arg(args: &Value, key: &str) -> Option<usize> {
    args.get(key).and_then(Value::as_u64).map(|value| value as usize)
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
