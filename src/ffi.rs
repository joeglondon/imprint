use crate::app;
use crate::surf::{ExpandMode, SurfAction};
use crate::types::{NodeRef, QueryRequest, SessionState};
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::ffi::{c_char, CStr, CString};
use std::path::PathBuf;

#[derive(Serialize)]
struct FfiResponse<T: Serialize> {
    ok: bool,
    data: Option<T>,
    error: Option<String>,
}

#[no_mangle]
pub extern "C" fn ai_memory_free_string(ptr: *mut c_char) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        drop(CString::from_raw(ptr));
    }
}

#[no_mangle]
pub extern "C" fn ai_memory_ingest_paths(
    store_path: *const c_char,
    paths_json: *const c_char,
) -> *mut c_char {
    respond(|| {
        let store = path_arg(store_path)?;
        let paths: Vec<String> = json_arg(paths_json)?;
        let as_paths = paths.into_iter().map(PathBuf::from).collect::<Vec<_>>();
        app::ingest_paths(&store, &as_paths)
    })
}

#[no_mangle]
pub extern "C" fn ai_memory_rebuild_memory(store_path: *const c_char) -> *mut c_char {
    respond(|| app::rebuild_memory(&path_arg(store_path)?))
}

#[no_mangle]
pub extern "C" fn ai_memory_get_memory_summary(store_path: *const c_char) -> *mut c_char {
    respond(|| app::get_memory_summary(&path_arg(store_path)?))
}

#[no_mangle]
pub extern "C" fn ai_memory_get_visualization_snapshot(store_path: *const c_char) -> *mut c_char {
    respond(|| app::get_visualization_snapshot(&path_arg(store_path)?))
}

#[no_mangle]
pub extern "C" fn ai_memory_run_query(
    store_path: *const c_char,
    query_json: *const c_char,
) -> *mut c_char {
    respond(|| {
        let store = path_arg(store_path)?;
        let request: QueryRequest = json_arg(query_json)?;
        app::run_query(&store, request)
    })
}

#[no_mangle]
pub extern "C" fn ai_memory_grep_region(
    store_path: *const c_char,
    region_id: *const c_char,
    needle: *const c_char,
) -> *mut c_char {
    respond(|| {
        let store = path_arg(store_path)?;
        app::grep_region(&store, &string_arg(region_id)?, &string_arg(needle)?)
    })
}

#[no_mangle]
pub extern "C" fn ai_memory_open_document_excerpt(
    store_path: *const c_char,
    chunk_id: *const c_char,
) -> *mut c_char {
    respond(|| {
        let store = path_arg(store_path)?;
        app::open_document_excerpt(&store, &string_arg(chunk_id)?)
    })
}

#[no_mangle]
pub extern "C" fn ai_memory_grep_document(
    store_path: *const c_char,
    document_id: *const c_char,
    needle: *const c_char,
) -> *mut c_char {
    respond(|| {
        let store = path_arg(store_path)?;
        app::grep_document(&store, &string_arg(document_id)?, &string_arg(needle)?)
    })
}

#[no_mangle]
pub extern "C" fn ai_memory_semantic_document_search(
    store_path: *const c_char,
    document_id: *const c_char,
    query: *const c_char,
) -> *mut c_char {
    respond(|| {
        let store = path_arg(store_path)?;
        app::semantic_document_search(&store, &string_arg(document_id)?, &string_arg(query)?)
    })
}

#[no_mangle]
pub extern "C" fn ai_memory_surf_open(
    store_path: *const c_char,
    node_json: *const c_char,
) -> *mut c_char {
    respond(|| {
        let store = path_arg(store_path)?;
        let node: NodeRef = json_arg(node_json)?;
        app::surf_open(&store, &node)
    })
}

#[no_mangle]
pub extern "C" fn ai_memory_surf_neighbors(
    store_path: *const c_char,
    node_json: *const c_char,
    max_results: usize,
) -> *mut c_char {
    respond(|| {
        let store = path_arg(store_path)?;
        let node: NodeRef = json_arg(node_json)?;
        app::surf_neighbors(&store, &node, max_results)
    })
}

#[no_mangle]
pub extern "C" fn ai_memory_surf_expand(
    store_path: *const c_char,
    chunk_id: *const c_char,
    mode_json: *const c_char,
    window: usize,
) -> *mut c_char {
    respond(|| {
        let store = path_arg(store_path)?;
        let mode: ExpandMode = json_arg(mode_json)?;
        app::surf_expand(&store, &string_arg(chunk_id)?, mode, window)
    })
}

#[no_mangle]
pub extern "C" fn ai_memory_surf_jump_to_anchor(
    store_path: *const c_char,
    anchor_id: *const c_char,
    window: usize,
) -> *mut c_char {
    respond(|| {
        let store = path_arg(store_path)?;
        app::surf_jump_to_anchor(&store, &string_arg(anchor_id)?, window)
    })
}

#[no_mangle]
pub extern "C" fn ai_memory_surf_session_step(
    store_path: *const c_char,
    session_json: *const c_char,
    action_json: *const c_char,
) -> *mut c_char {
    respond(|| {
        let store = path_arg(store_path)?;
        let session: SessionState = json_arg(session_json)?;
        let action: SurfAction = json_arg(action_json)?;
        app::surf_session_step(&store, session, action)
    })
}

#[no_mangle]
pub extern "C" fn ai_memory_list_links(
    store_path: *const c_char,
    node_json: *const c_char,
) -> *mut c_char {
    respond(|| {
        let store = path_arg(store_path)?;
        let node: NodeRef = json_arg(node_json)?;
        app::list_links(&store, &node)
    })
}

#[no_mangle]
pub extern "C" fn ai_memory_step_navigation(
    store_path: *const c_char,
    session_json: *const c_char,
    link_id: *const c_char,
) -> *mut c_char {
    respond(|| {
        let store = path_arg(store_path)?;
        let session: SessionState = json_arg(session_json)?;
        app::step_navigation(&store, session, &string_arg(link_id)?)
    })
}

#[no_mangle]
pub extern "C" fn ai_memory_backtrack_navigation(
    store_path: *const c_char,
    session_json: *const c_char,
) -> *mut c_char {
    respond(|| {
        let store = path_arg(store_path)?;
        let session: SessionState = json_arg(session_json)?;
        app::backtrack_navigation(&store, session)
    })
}

#[no_mangle]
pub extern "C" fn ai_memory_save_model_config(
    store_path: *const c_char,
    config_json: *const c_char,
) -> *mut c_char {
    respond(|| {
        let store = path_arg(store_path)?;
        let config: app::ModelConfig = json_arg(config_json)?;
        app::save_model_config(&store, &config)
    })
}

#[no_mangle]
pub extern "C" fn ai_memory_load_model_config(store_path: *const c_char) -> *mut c_char {
    respond(|| app::load_model_config(&path_arg(store_path)?))
}

#[no_mangle]
pub extern "C" fn ai_memory_test_model_connection(config_json: *const c_char) -> *mut c_char {
    respond(|| {
        let request: app::ModelConnectionTestRequest = json_arg(config_json)?;
        app::test_model_connection(request)
    })
}

#[no_mangle]
pub extern "C" fn ai_memory_create_chat_session(
    store_path: *const c_char,
    title: *const c_char,
) -> *mut c_char {
    respond(|| {
        let store = path_arg(store_path)?;
        let title = string_arg(title)?;
        app::create_chat_session(
            &store,
            if title.trim().is_empty() {
                None
            } else {
                Some(title)
            },
        )
    })
}

#[no_mangle]
pub extern "C" fn ai_memory_list_chat_sessions(store_path: *const c_char) -> *mut c_char {
    respond(|| app::list_chat_sessions(&path_arg(store_path)?))
}

#[no_mangle]
pub extern "C" fn ai_memory_list_chat_messages(
    store_path: *const c_char,
    session_id: *const c_char,
) -> *mut c_char {
    respond(|| {
        let store = path_arg(store_path)?;
        app::list_chat_messages(&store, &string_arg(session_id)?)
    })
}

#[no_mangle]
pub extern "C" fn ai_memory_send_chat_turn(
    store_path: *const c_char,
    request_json: *const c_char,
) -> *mut c_char {
    respond(|| {
        let store = path_arg(store_path)?;
        let request: crate::types::ChatTurnRequest = json_arg(request_json)?;
        app::send_chat_turn(&store, request)
    })
}

#[no_mangle]
pub extern "C" fn ai_memory_list_chat_context_traces(
    store_path: *const c_char,
    session_id: *const c_char,
) -> *mut c_char {
    respond(|| {
        let store = path_arg(store_path)?;
        app::list_chat_context_traces(&store, &string_arg(session_id)?)
    })
}

#[no_mangle]
pub extern "C" fn ai_memory_list_derived_memories(
    store_path: *const c_char,
    session_id: *const c_char,
) -> *mut c_char {
    respond(|| {
        let store = path_arg(store_path)?;
        let session = string_arg(session_id)?;
        app::list_derived_memories(
            &store,
            if session.trim().is_empty() {
                None
            } else {
                Some(session)
            },
        )
    })
}

#[no_mangle]
pub extern "C" fn ai_memory_write_derived_memory(
    store_path: *const c_char,
    write_json: *const c_char,
) -> *mut c_char {
    respond(|| {
        let store = path_arg(store_path)?;
        let write: crate::types::DerivedMemoryWrite = json_arg(write_json)?;
        app::write_derived_memory(&store, write)
    })
}

#[no_mangle]
pub extern "C" fn ai_memory_write_web_finding(
    store_path: *const c_char,
    write_json: *const c_char,
) -> *mut c_char {
    respond(|| {
        let store = path_arg(store_path)?;
        let write: crate::types::WebFindingWrite = json_arg(write_json)?;
        app::write_web_finding(&store, write)
    })
}

#[no_mangle]
pub extern "C" fn ai_memory_write_agent_link(
    store_path: *const c_char,
    write_json: *const c_char,
) -> *mut c_char {
    respond(|| {
        let store = path_arg(store_path)?;
        let write: crate::types::AgentLinkWrite = json_arg(write_json)?;
        app::write_agent_link(&store, write)
    })
}

#[no_mangle]
pub extern "C" fn ai_memory_apply_attention_mark(
    store_path: *const c_char,
    write_json: *const c_char,
) -> *mut c_char {
    respond(|| {
        let store = path_arg(store_path)?;
        let write: crate::types::AttentionMarkWrite = json_arg(write_json)?;
        app::apply_attention_mark(&store, write)
    })
}

#[no_mangle]
pub extern "C" fn ai_memory_list_attention_marks(
    store_path: *const c_char,
    target_id: *const c_char,
) -> *mut c_char {
    respond(|| {
        let store = path_arg(store_path)?;
        let target = string_arg(target_id)?;
        let target = if target.is_empty() {
            None
        } else {
            Some(target)
        };
        app::list_attention_marks(&store, target)
    })
}

#[no_mangle]
pub extern "C" fn ai_memory_revert_attention_mark(
    store_path: *const c_char,
    mark_id: *const c_char,
    actor: *const c_char,
) -> *mut c_char {
    respond(|| {
        let store = path_arg(store_path)?;
        app::revert_attention_mark(&store, &string_arg(mark_id)?, string_arg(actor)?)
    })
}

#[no_mangle]
pub extern "C" fn ai_memory_compile_memory_brain(store_path: *const c_char) -> *mut c_char {
    respond(|| app::compile_memory_brain(&path_arg(store_path)?))
}

#[no_mangle]
pub extern "C" fn ai_memory_load_cortex_adapter_snapshot(store_path: *const c_char) -> *mut c_char {
    respond(|| app::load_cortex_adapter_snapshot(&path_arg(store_path)?))
}

#[no_mangle]
pub extern "C" fn ai_memory_retry_cortex_adapter_job(
    store_path: *const c_char,
    job_id: *const c_char,
) -> *mut c_char {
    respond(|| {
        crate::training::retry_cortex_adapter_training_job(
            &path_arg(store_path)?,
            &string_arg(job_id)?,
            false,
        )
    })
}

#[no_mangle]
pub extern "C" fn ai_memory_train_cortex_adapter_now(store_path: *const c_char) -> *mut c_char {
    respond(|| app::train_cortex_adapter_now(&path_arg(store_path)?))
}

#[no_mangle]
pub extern "C" fn ai_memory_activate_last_trained_cortex_adapter(
    store_path: *const c_char,
) -> *mut c_char {
    respond(|| app::activate_last_trained_cortex_adapter(&path_arg(store_path)?))
}

#[no_mangle]
pub extern "C" fn ai_memory_disable_cortex_adapter(store_path: *const c_char) -> *mut c_char {
    respond(|| app::disable_cortex_adapter(&path_arg(store_path)?))
}

#[no_mangle]
pub extern "C" fn ai_memory_probe_cortex_adapter_route(
    store_path: *const c_char,
    query: *const c_char,
) -> *mut c_char {
    respond(|| app::probe_cortex_adapter_route(&path_arg(store_path)?, &string_arg(query)?))
}

fn respond<T: Serialize>(f: impl FnOnce() -> anyhow::Result<T>) -> *mut c_char {
    let payload: FfiResponse<Value> = match f() {
        Ok(data) => FfiResponse {
            ok: true,
            data: Some(serde_json::to_value(data).unwrap_or(Value::Null)),
            error: None,
        },
        Err(error) => FfiResponse::<Value> {
            ok: false,
            data: None,
            error: Some(error.to_string()),
        },
    };
    CString::new(
        serde_json::to_string(&payload)
            .unwrap_or_else(|_| "{\"ok\":false,\"error\":\"serialization failure\"}".into()),
    )
    .unwrap()
    .into_raw()
}

fn string_arg(ptr: *const c_char) -> anyhow::Result<String> {
    if ptr.is_null() {
        anyhow::bail!("null pointer");
    }
    let c_str = unsafe { CStr::from_ptr(ptr) };
    Ok(c_str.to_string_lossy().into_owned())
}

fn path_arg(ptr: *const c_char) -> anyhow::Result<PathBuf> {
    Ok(PathBuf::from(string_arg(ptr)?))
}

fn json_arg<T: serde::de::DeserializeOwned>(ptr: *const c_char) -> anyhow::Result<T> {
    let raw = string_arg(ptr)?;
    Ok(serde_json::from_str(&raw)?)
}

#[allow(dead_code)]
fn _empty_filters() -> BTreeMap<String, String> {
    BTreeMap::new()
}
