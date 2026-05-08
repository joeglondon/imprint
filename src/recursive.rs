use crate::types::{ChatContextSnippet, ChatContextTrace};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const LATENT_RECURSION_DEFAULT_ENABLED: bool = false;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum RecursiveRole {
    Planner,
    Retriever,
    Critic,
    Solver,
    MemorySteward,
}

impl RecursiveRole {
    pub fn as_str(self) -> &'static str {
        match self {
            RecursiveRole::Planner => "planner",
            RecursiveRole::Retriever => "retriever",
            RecursiveRole::Critic => "critic",
            RecursiveRole::Solver => "solver",
            RecursiveRole::MemorySteward => "memory_steward",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TextToolRoleContract {
    pub role: RecursiveRole,
    pub input_contract: String,
    pub output_contract: String,
    pub allowed_tools: Vec<String>,
    pub stop_condition: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LatentRoleContract {
    pub role: RecursiveRole,
    pub hidden_state_input: String,
    pub hidden_state_output: String,
    pub bridge_module: String,
    pub frozen_weights: String,
    pub observable_projection: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RecursiveRoleEvent {
    pub role: RecursiveRole,
    pub step: usize,
    pub action: String,
    pub source_refs: Vec<String>,
    pub sufficient: Option<bool>,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RecursiveTraceDatasetExample {
    pub task: String,
    pub source_trace_id: String,
    pub source_session_id: String,
    pub source_message_id: String,
    pub input: String,
    pub target: String,
    pub source_refs: Vec<String>,
    pub anchor_ids: Vec<String>,
    pub roles: Vec<RecursiveRole>,
    pub tool_calls: usize,
    pub source_grounded: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HiddenStateAccessFinding {
    pub topic: String,
    pub finding: String,
    pub implementation_note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MlxHiddenStateAccessStudy {
    pub target_model_family: String,
    pub runtime: String,
    pub status: String,
    pub findings: Vec<HiddenStateAccessFinding>,
    pub prototype_plan: Vec<String>,
    pub production_blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RecursiveLinkConfig {
    pub hidden_width: usize,
    pub bridge_rank: usize,
    pub inner_steps: usize,
    pub outer_steps: usize,
    pub roles: Vec<RecursiveRole>,
    pub frozen_base_model: bool,
    pub trainable_component: String,
    pub enabled_by_default: bool,
    pub fallback: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RecursiveLinkBridge {
    pub role: RecursiveRole,
    pub module_name: String,
    pub input_width: usize,
    pub output_width: usize,
    pub rank: usize,
    pub trainable_parameters: usize,
    pub frozen_base_model: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RecursiveLinkModule {
    pub config: RecursiveLinkConfig,
    pub bridges: Vec<RecursiveLinkBridge>,
    pub inner_transfer: String,
    pub outer_transfer: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RecursiveLatentStep {
    pub role: RecursiveRole,
    pub outer_step: usize,
    pub inner_step: usize,
    pub input_state_ref: String,
    pub output_state_ref: String,
    pub observable_reason_code: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RecursiveLatentTrace {
    pub status: String,
    pub enabled_by_default: bool,
    pub fallback_contract: String,
    pub steps: Vec<RecursiveLatentStep>,
    pub source_refs_selected: Vec<String>,
    pub hidden_states_redacted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RecursiveMasEvalReport {
    pub status: String,
    pub text_tool_baseline_tool_calls: usize,
    pub latent_estimated_tool_calls: usize,
    pub text_tool_baseline_tokens: usize,
    pub latent_estimated_tokens: usize,
    pub better_region_source_selection: bool,
    pub lower_hallucination_rate: bool,
    pub better_weak_evidence_refusal: bool,
    pub lower_token_usage: bool,
    pub same_answer_quality: bool,
    pub may_enable_latent_default: bool,
    pub reason: String,
}

pub fn text_tool_role_contracts() -> Vec<TextToolRoleContract> {
    vec![
        TextToolRoleContract {
            role: RecursiveRole::Planner,
            input_contract: "user goal, cortex route hints, hot memory, and tool budget".into(),
            output_contract: "small ordered source-recall plan with stop reason".into(),
            allowed_tools: vec![
                "memory_search".into(),
                "web_search".into(),
                "mark_attention".into(),
            ],
            stop_condition: "plan has enough retrieval actions or needs critic feedback".into(),
        },
        TextToolRoleContract {
            role: RecursiveRole::Retriever,
            input_contract: "planner action plus current route/source address".into(),
            output_contract: "bounded snippets, source refs, anchors, and access trace".into(),
            allowed_tools: vec![
                "memory_search".into(),
                "memory_open".into(),
                "memory_neighbors".into(),
                "memory_expand".into(),
                "memory_jump_to_anchor".into(),
            ],
            stop_condition: "snippets contain likely source anchors or retrieval is exhausted"
                .into(),
        },
        TextToolRoleContract {
            role: RecursiveRole::Critic,
            input_contract: "question, snippets, anchors, and retrieval trace".into(),
            output_contract: "sufficiency decision, evidence gap, and next action".into(),
            allowed_tools: Vec::new(),
            stop_condition: "evidence is sufficient or gap requires another planner/retriever pass"
                .into(),
        },
        TextToolRoleContract {
            role: RecursiveRole::Solver,
            input_contract: "question plus critic-approved source snippets".into(),
            output_contract: "answer bounded to cited source context or cautious refusal".into(),
            allowed_tools: Vec::new(),
            stop_condition: "answer is citation-safe or explicitly caveated".into(),
        },
        TextToolRoleContract {
            role: RecursiveRole::MemorySteward,
            input_contract: "useful trace outcomes, attention marks, and provenance".into(),
            output_contract: "reversible hotness/importance/writeback recommendation".into(),
            allowed_tools: vec!["mark_attention".into()],
            stop_condition: "attention/writeback decision has actor and reason".into(),
        },
    ]
}

pub fn lfm25_mlx_hidden_state_study() -> MlxHiddenStateAccessStudy {
    MlxHiddenStateAccessStudy {
        target_model_family: "LiquidAI/LFM2.5 MLX checkpoints".into(),
        runtime: "mlx-lm local Python runtime".into(),
        status: "research_only_api_shape_identified".into(),
        findings: vec![
            HiddenStateAccessFinding {
                topic: "adapter entrypoint".into(),
                finding: "The current trainer invokes `python -m mlx_lm lora`, which is a CLI path for LoRA training and does not expose per-token hidden states to Rust.".into(),
                implementation_note: "Keep training/export unchanged; hidden-state recursion needs a separate Python research runner that imports MLX model modules directly.".into(),
            },
            HiddenStateAccessFinding {
                topic: "hidden-state access".into(),
                finding: "A latent prototype should call the loaded MLX model forward path directly and request or intercept layer outputs, instead of using OpenAI-compatible chat completions.".into(),
                implementation_note: "The Rust side should treat hidden states as opaque handles/hashes and only persist observable role events, source refs, and reason codes.".into(),
            },
            HiddenStateAccessFinding {
                topic: "LoRA compatibility".into(),
                finding: "The cortex adapter can remain a normal MLX-LM LoRA artifact; RecursiveLink bridges are a separate small trainable module around frozen base/model adapter activations.".into(),
                implementation_note: "Do not merge RecursiveLink bridge weights into the personal cortex adapter until eval proves the latent loop beats text/tool recursion.".into(),
            },
            HiddenStateAccessFinding {
                topic: "production boundary".into(),
                finding: "OpenAI-compatible, Ollama, and llama.cpp endpoints cannot be assumed to return hidden states.".into(),
                implementation_note: "Runtime selection must fall back to text/tool recursion unless the MLX research runner explicitly reports hidden-state support.".into(),
            },
        ],
        prototype_plan: vec![
            "Load the MLX checkpoint and active cortex LoRA in Python.".into(),
            "Run role prompts through the frozen model and capture chosen layer hidden states as opaque tensors.".into(),
            "Apply a small role-specific RecursiveLink bridge for inner role refinement.".into(),
            "Transfer the projected hidden state to the next role bridge for outer planner/retriever/critic/solver flow.".into(),
            "Return only observable trace projections to Rust: role sequence, source refs, sufficiency, reason codes, and caution state.".into(),
        ],
        production_blockers: vec![
            "Need a checked-in Python research runner with MLX hidden-state extraction tests.".into(),
            "Need measured eval wins on region/source selection, hallucination, weak-evidence refusal, and token/tool efficiency.".into(),
            "Need a no-hidden-state fallback for all non-MLX runtimes.".into(),
        ],
    }
}

pub fn latent_role_contracts() -> Vec<LatentRoleContract> {
    text_tool_role_contracts()
        .into_iter()
        .map(|contract| LatentRoleContract {
            role: contract.role,
            hidden_state_input: format!(
                "{} hidden state plus compact source-address sketch",
                contract.role.as_str()
            ),
            hidden_state_output: format!(
                "{} hidden state delta plus observable reason-code projection",
                contract.role.as_str()
            ),
            bridge_module: format!("recursive_link_{}_bridge", contract.role.as_str()),
            frozen_weights: "base model remains frozen; only bridge/adapters are trainable".into(),
            observable_projection: vec![
                "role".into(),
                "sufficiency_outcome".into(),
                "selected_source_refs".into(),
                "reason_codes".into(),
                "confidence_or_caution".into(),
            ],
        })
        .collect()
}

pub fn default_recursive_link_config(hidden_width: usize) -> RecursiveLinkConfig {
    RecursiveLinkConfig {
        hidden_width,
        bridge_rank: 8,
        inner_steps: 2,
        outer_steps: 4,
        roles: vec![
            RecursiveRole::Planner,
            RecursiveRole::Retriever,
            RecursiveRole::Critic,
            RecursiveRole::Solver,
            RecursiveRole::MemorySteward,
        ],
        frozen_base_model: true,
        trainable_component: "role-local low-rank RecursiveLink bridges only".into(),
        enabled_by_default: LATENT_RECURSION_DEFAULT_ENABLED,
        fallback: "text_tool_recursion".into(),
    }
}

pub fn minimal_recursive_link_module(config: RecursiveLinkConfig) -> RecursiveLinkModule {
    let bridges = config
        .roles
        .iter()
        .map(|role| {
            let module_name = format!("recursive_link_{}_bridge", role.as_str());
            RecursiveLinkBridge {
                role: *role,
                module_name,
                input_width: config.hidden_width,
                output_width: config.hidden_width,
                rank: config.bridge_rank,
                trainable_parameters: config.hidden_width * config.bridge_rank * 2,
                frozen_base_model: config.frozen_base_model,
            }
        })
        .collect();
    RecursiveLinkModule {
        config,
        bridges,
        inner_transfer: "h_role_next = h_role + bridge_role(norm(h_role), source_address_sketch)"
            .into(),
        outer_transfer:
            "h_next_role = projection_role_to_role(h_role_final, observable_reason_projection)"
                .into(),
    }
}

pub fn latent_trace_prototype(
    trace: &ChatContextTrace,
    module: &RecursiveLinkModule,
) -> RecursiveLatentTrace {
    let source_refs = source_refs_from_snippets(&trace.snippets);
    let mut steps = Vec::new();
    let roles = module
        .config
        .roles
        .iter()
        .copied()
        .take(module.config.outer_steps)
        .collect::<Vec<_>>();
    for (outer_index, role) in roles.iter().enumerate() {
        for inner_step in 1..=module.config.inner_steps {
            steps.push(RecursiveLatentStep {
                role: *role,
                outer_step: outer_index + 1,
                inner_step,
                input_state_ref: format!(
                    "redacted://trace/{}/role/{}/outer/{}/inner/{}/input",
                    trace.id,
                    role.as_str(),
                    outer_index + 1,
                    inner_step
                ),
                output_state_ref: format!(
                    "redacted://trace/{}/role/{}/outer/{}/inner/{}/output",
                    trace.id,
                    role.as_str(),
                    outer_index + 1,
                    inner_step
                ),
                observable_reason_code: reason_code_for_role(
                    *role,
                    source_grounded(&trace.snippets),
                ),
            });
        }
    }
    RecursiveLatentTrace {
        status: "research_only".into(),
        enabled_by_default: module.config.enabled_by_default,
        fallback_contract: module.config.fallback.clone(),
        steps,
        source_refs_selected: source_refs,
        hidden_states_redacted: true,
    }
}

pub fn evaluate_recursive_mas_trace(trace: &ChatContextTrace) -> RecursiveMasEvalReport {
    let text_tool_calls = trace
        .tool_trace
        .iter()
        .filter(|line| line.contains("memory_") || line.contains("web_search"))
        .count();
    let grounded = source_grounded(&trace.snippets);
    let anchored_refs = source_refs_from_snippets(&trace.snippets)
        .into_iter()
        .filter(|source| source.starts_with("imprint://anchor/"))
        .count();
    let baseline_tokens = trace
        .tool_trace
        .iter()
        .map(|line| line.split_whitespace().count())
        .sum::<usize>()
        + trace
            .snippets
            .iter()
            .map(|snippet| snippet.excerpt.split_whitespace().count())
            .sum::<usize>();
    let latent_tokens = baseline_tokens
        .saturating_sub(trace.tool_trace.len().saturating_mul(6))
        .max(trace.snippets.len().saturating_mul(8));
    let latent_tool_calls = text_tool_calls.saturating_sub(if grounded { 1 } else { 0 });
    let better_region_source_selection = grounded && anchored_refs >= 1;
    let lower_hallucination_rate = grounded;
    let better_weak_evidence_refusal = grounded
        || trace
            .cortex_trace
            .as_ref()
            .is_some_and(|cortex| cortex.rounds.iter().any(|round| !round.critique.sufficient));
    let lower_token_usage = latent_tokens < baseline_tokens;
    let same_answer_quality = grounded && !trace.snippets.is_empty();
    let may_enable_latent_default = false;
    RecursiveMasEvalReport {
        status: "research_only".into(),
        text_tool_baseline_tool_calls: text_tool_calls,
        latent_estimated_tool_calls: latent_tool_calls,
        text_tool_baseline_tokens: baseline_tokens,
        latent_estimated_tokens: latent_tokens,
        better_region_source_selection,
        lower_hallucination_rate,
        better_weak_evidence_refusal,
        lower_token_usage,
        same_answer_quality,
        may_enable_latent_default,
        reason: if same_answer_quality
            && better_region_source_selection
            && lower_hallucination_rate
            && better_weak_evidence_refusal
            && lower_token_usage
        {
            "Prototype metrics are inspectable, but default enablement remains blocked until measured model eval beats the text/tool baseline.".into()
        } else {
            "Prototype metrics do not yet beat all text/tool baseline gates.".into()
        },
    }
}

pub fn role_events_from_trace(trace: &ChatContextTrace) -> Vec<RecursiveRoleEvent> {
    let mut events = Vec::new();
    let mut step = 1;
    events.push(RecursiveRoleEvent {
        role: RecursiveRole::Planner,
        step,
        action: "read_cortex_and_choose_source_recall_plan".into(),
        source_refs: Vec::new(),
        sufficient: None,
        reason: trace
            .tool_trace
            .first()
            .cloned()
            .unwrap_or_else(|| "planner trace unavailable".into()),
    });
    step += 1;

    for line in &trace.tool_trace {
        let role = role_for_tool_trace_line(line);
        if role == RecursiveRole::Planner {
            continue;
        }
        events.push(RecursiveRoleEvent {
            role,
            step,
            action: compact_action(line),
            source_refs: source_refs_from_snippets(&trace.snippets),
            sufficient: None,
            reason: line.clone(),
        });
        step += 1;
    }

    if let Some(cortex_trace) = &trace.cortex_trace {
        for round in &cortex_trace.rounds {
            events.push(RecursiveRoleEvent {
                role: RecursiveRole::Critic,
                step,
                action: format!("critique_round_{}", round.round),
                source_refs: source_refs_from_snippets(&trace.snippets),
                sufficient: Some(round.critique.sufficient),
                reason: round.critique.gap.clone(),
            });
            step += 1;
        }
    } else {
        events.push(RecursiveRoleEvent {
            role: RecursiveRole::Critic,
            step,
            action: "critique_source_sufficiency".into(),
            source_refs: source_refs_from_snippets(&trace.snippets),
            sufficient: Some(source_grounded(&trace.snippets)),
            reason: if source_grounded(&trace.snippets) {
                "source anchors available".into()
            } else {
                "no source anchors available".into()
            },
        });
        step += 1;
    }

    events.push(RecursiveRoleEvent {
        role: RecursiveRole::Solver,
        step,
        action: "answer_from_critic_approved_context".into(),
        source_refs: source_refs_from_snippets(&trace.snippets),
        sufficient: Some(source_grounded(&trace.snippets)),
        reason: "final response must cite snippets or caveat weak evidence".into(),
    });
    step += 1;
    events.push(RecursiveRoleEvent {
        role: RecursiveRole::MemorySteward,
        step,
        action: "record_attention_and_reusable_trace_outcome".into(),
        source_refs: source_refs_from_snippets(&trace.snippets),
        sufficient: None,
        reason: "trace can become training signal, not source truth".into(),
    });
    events
}

pub fn trace_dataset_examples(trace: &ChatContextTrace) -> Vec<RecursiveTraceDatasetExample> {
    let events = role_events_from_trace(trace);
    let roles = events
        .iter()
        .map(|event| event.role)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let mut source_refs = source_refs_from_snippets(&trace.snippets);
    if source_refs.is_empty() {
        source_refs.push(format!("imprint://chat_context_trace/{}", trace.id));
    }
    let anchor_ids = anchor_ids_from_snippets(&trace.snippets);
    let grounded = !anchor_ids.is_empty();
    let tool_calls = trace
        .tool_trace
        .iter()
        .filter(|line| line.contains("memory_") || line.contains("web_search"))
        .count();
    let role_sequence = roles
        .iter()
        .map(|role| role.as_str())
        .collect::<Vec<_>>()
        .join(" -> ");

    vec![
        RecursiveTraceDatasetExample {
            task: "recursive_role_trace".into(),
            source_trace_id: trace.id.clone(),
            source_session_id: trace.session_id.clone(),
            source_message_id: trace.user_message_id.clone(),
            input: format!(
                "Trace {} used {} tool step(s) and {} snippet(s). Infer the role sequence.",
                trace.id,
                tool_calls,
                trace.snippets.len()
            ),
            target: format!("roles:{role_sequence};fallback:text_tool;latent_status:research_only"),
            source_refs: source_refs.clone(),
            anchor_ids: anchor_ids.clone(),
            roles: roles.clone(),
            tool_calls,
            source_grounded: grounded,
        },
        RecursiveTraceDatasetExample {
            task: "recursive_sufficiency_eval".into(),
            source_trace_id: trace.id.clone(),
            source_session_id: trace.session_id.clone(),
            source_message_id: trace.user_message_id.clone(),
            input: format!(
                "Critique whether trace {} has enough source-grounded evidence before solving.",
                trace.id
            ),
            target: if grounded {
                format!("sufficient:true;cite_anchor_ids:{}", anchor_ids.join(","))
            } else {
                "sufficient:false;next_action:retrieve_expand_or_caveat".into()
            },
            source_refs: source_refs.clone(),
            anchor_ids: anchor_ids.clone(),
            roles: roles.clone(),
            tool_calls,
            source_grounded: grounded,
        },
        RecursiveTraceDatasetExample {
            task: "recursive_efficiency_eval".into(),
            source_trace_id: trace.id.clone(),
            source_session_id: trace.session_id.clone(),
            source_message_id: trace.user_message_id.clone(),
            input: format!(
                "Evaluate whether trace {} should reduce tool calls without losing source grounding.",
                trace.id
            ),
            target: if grounded {
                format!("efficiency:keep_or_reduce;tool_calls:{tool_calls};source_grounded:true")
            } else {
                format!("efficiency:do_not_reduce;tool_calls:{tool_calls};source_grounded:false;need_more_evidence")
            },
            source_refs: source_refs.clone(),
            anchor_ids: anchor_ids.clone(),
            roles: roles.clone(),
            tool_calls,
            source_grounded: grounded,
        },
        RecursiveTraceDatasetExample {
            task: "recursive_region_selection_eval".into(),
            source_trace_id: trace.id.clone(),
            source_session_id: trace.session_id.clone(),
            source_message_id: trace.user_message_id.clone(),
            input: format!(
                "Evaluate whether trace {} selected a useful source region or source family before solving.",
                trace.id
            ),
            target: if grounded {
                format!("region_source_selection:better_or_equal;selected_refs:{}", source_refs.join(","))
            } else {
                "region_source_selection:weak;next_action:route_search_expand".into()
            },
            source_refs: source_refs.clone(),
            anchor_ids: anchor_ids.clone(),
            roles: roles.clone(),
            tool_calls,
            source_grounded: grounded,
        },
        RecursiveTraceDatasetExample {
            task: "recursive_hallucination_eval".into(),
            source_trace_id: trace.id.clone(),
            source_session_id: trace.session_id.clone(),
            source_message_id: trace.user_message_id.clone(),
            input: format!(
                "Evaluate whether trace {} lowers hallucination risk by grounding answer boundaries.",
                trace.id
            ),
            target: if grounded {
                "hallucination_risk:lower;answer_boundary:source_anchored".into()
            } else {
                "hallucination_risk:high;answer_boundary:caveat_or_refuse".into()
            },
            source_refs: source_refs.clone(),
            anchor_ids: anchor_ids.clone(),
            roles: roles.clone(),
            tool_calls,
            source_grounded: grounded,
        },
        RecursiveTraceDatasetExample {
            task: "recursive_token_usage_eval".into(),
            source_trace_id: trace.id.clone(),
            source_session_id: trace.session_id.clone(),
            source_message_id: trace.user_message_id.clone(),
            input: format!(
                "Estimate whether latent recursion for trace {} can lower token usage versus text/tool recursion.",
                trace.id
            ),
            target: format!(
                "token_usage:lower_if_hidden_state_supported;baseline_tool_calls:{tool_calls};fallback:text_tool"
            ),
            source_refs,
            anchor_ids,
            roles,
            tool_calls,
            source_grounded: grounded,
        },
    ]
}

fn role_for_tool_trace_line(line: &str) -> RecursiveRole {
    if line.contains("mark_attention") {
        RecursiveRole::MemorySteward
    } else if line.contains("response_context") {
        RecursiveRole::Solver
    } else if line.contains("cortex round") || line.contains("critique") {
        RecursiveRole::Critic
    } else if line.contains("memory_") || line.contains("web_search") || line.contains("retrieval")
    {
        RecursiveRole::Retriever
    } else {
        RecursiveRole::Planner
    }
}

fn compact_action(line: &str) -> String {
    line.split(':')
        .nth(1)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(line)
        .split_whitespace()
        .next()
        .unwrap_or("trace_step")
        .to_string()
}

fn source_refs_from_snippets(snippets: &[ChatContextSnippet]) -> Vec<String> {
    snippets
        .iter()
        .map(|snippet| match snippet.source_anchor.as_ref() {
            Some(anchor) => format!("imprint://anchor/{}", anchor.id),
            None => format!("imprint://{}/{}", snippet.source_kind, snippet.source_id),
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn anchor_ids_from_snippets(snippets: &[ChatContextSnippet]) -> Vec<String> {
    snippets
        .iter()
        .filter_map(|snippet| {
            snippet
                .source_anchor
                .as_ref()
                .map(|anchor| anchor.id.clone())
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn source_grounded(snippets: &[ChatContextSnippet]) -> bool {
    snippets
        .iter()
        .any(|snippet| snippet.source_anchor.is_some())
}

fn reason_code_for_role(role: RecursiveRole, grounded: bool) -> String {
    match (role, grounded) {
        (RecursiveRole::Planner, _) => "route_from_cortex_sketch".into(),
        (RecursiveRole::Retriever, true) => "select_anchored_source_refs".into(),
        (RecursiveRole::Retriever, false) => "retrieve_more_or_expand".into(),
        (RecursiveRole::Critic, true) => "evidence_sufficient".into(),
        (RecursiveRole::Critic, false) => "weak_evidence_caution".into(),
        (RecursiveRole::Solver, true) => "solve_with_citations".into(),
        (RecursiveRole::Solver, false) => "caveat_or_refuse".into(),
        (RecursiveRole::MemorySteward, _) => "record_reversible_attention".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CortexCritique, CortexRoundTrace, CortexTrace, SourceAnchor};

    #[test]
    fn trace_examples_preserve_roles_and_anchor_boundary() {
        let trace = ChatContextTrace {
            id: "trace-1".into(),
            session_id: "session-1".into(),
            user_message_id: "msg-1".into(),
            snippets: vec![ChatContextSnippet {
                id: "chunk-1".into(),
                source_kind: "chunk".into(),
                source_id: "chunk-1".into(),
                excerpt: "grounded evidence".into(),
                score: 0.9,
                hotness: 0.4,
                source_anchor: Some(SourceAnchor {
                    id: "anchor-1".into(),
                    document_id: "doc-1".into(),
                    chunk_id: Some("chunk-1".into()),
                    source_artifact_id: None,
                    path: "/tmp/source.md".into(),
                    content_hash: "hash".into(),
                    start: 0,
                    end: 12,
                    byte_start: Some(0),
                    byte_end: Some(12),
                    char_start: Some(0),
                    char_end: Some(12),
                    page: None,
                    rendered_page: None,
                    pdf_selection: None,
                    email_location: None,
                    section: Some("Evidence".into()),
                    section_hierarchy: vec!["Evidence".into()],
                    paragraph_index: Some(1),
                    parser_version: 1,
                }),
            }],
            tool_trace: vec![
                "planner: read current cortex index".into(),
                "planner step 1: memory_search query=\"evidence\"".into(),
                "planner step 2: memory_expand target=chunk-1".into(),
                "response_context: selected 1 bounded snippet(s)".into(),
            ],
            cortex_trace: Some(CortexTrace {
                enabled: true,
                rounds: vec![CortexRoundTrace {
                    round: 1,
                    actions: vec!["memory_search".into(), "memory_expand".into()],
                    snippets_before: 0,
                    snippets_after: 1,
                    critique: CortexCritique {
                        sufficient: true,
                        gap: "anchored evidence selected".into(),
                        note: "ok".into(),
                    },
                }],
                final_note: "ok".into(),
            }),
            created_at: 1,
        };

        let events = role_events_from_trace(&trace);
        assert!(events
            .iter()
            .any(|event| event.role == RecursiveRole::Planner));
        assert!(events
            .iter()
            .any(|event| event.role == RecursiveRole::Retriever));
        assert!(events
            .iter()
            .any(|event| event.role == RecursiveRole::Critic));
        assert!(events
            .iter()
            .any(|event| event.role == RecursiveRole::Solver));

        let examples = trace_dataset_examples(&trace);
        assert_eq!(examples.len(), 6);
        assert!(examples
            .iter()
            .any(|example| example.task == "recursive_sufficiency_eval"
                && example.target.contains("cite_anchor_ids:anchor-1")));
        assert!(examples
            .iter()
            .any(|example| example.task == "recursive_region_selection_eval"));
        assert!(examples
            .iter()
            .any(|example| example.task == "recursive_hallucination_eval"));
        assert!(examples
            .iter()
            .any(|example| example.task == "recursive_token_usage_eval"));
        assert!(examples.iter().all(|example| example.source_grounded));
    }

    #[test]
    fn recursive_link_research_module_keeps_base_frozen_and_default_off() {
        let study = lfm25_mlx_hidden_state_study();
        assert_eq!(study.status, "research_only_api_shape_identified");
        assert!(study
            .findings
            .iter()
            .any(|finding| finding.topic == "hidden-state access"));

        let config = default_recursive_link_config(512);
        assert!(config.frozen_base_model);
        assert!(!config.enabled_by_default);
        assert_eq!(config.fallback, "text_tool_recursion");

        let module = minimal_recursive_link_module(config);
        assert_eq!(module.bridges.len(), 5);
        assert!(module
            .bridges
            .iter()
            .all(|bridge| bridge.frozen_base_model && bridge.trainable_parameters == 8192));
    }

    #[test]
    fn latent_trace_redacts_hidden_state_and_blocks_default_enablement() {
        let trace = ChatContextTrace {
            id: "trace-2".into(),
            session_id: "session-1".into(),
            user_message_id: "msg-2".into(),
            snippets: vec![ChatContextSnippet {
                id: "chunk-1".into(),
                source_kind: "chunk".into(),
                source_id: "chunk-1".into(),
                excerpt: "grounded evidence for a concise answer".into(),
                score: 0.9,
                hotness: 0.4,
                source_anchor: Some(SourceAnchor {
                    id: "anchor-2".into(),
                    document_id: "doc-1".into(),
                    chunk_id: Some("chunk-1".into()),
                    source_artifact_id: None,
                    path: "/tmp/source.md".into(),
                    content_hash: "hash".into(),
                    start: 0,
                    end: 12,
                    byte_start: Some(0),
                    byte_end: Some(12),
                    char_start: Some(0),
                    char_end: Some(12),
                    page: None,
                    rendered_page: None,
                    pdf_selection: None,
                    email_location: None,
                    section: Some("Evidence".into()),
                    section_hierarchy: vec!["Evidence".into()],
                    paragraph_index: Some(1),
                    parser_version: 1,
                }),
            }],
            tool_trace: vec![
                "planner step 1: memory_search query=\"evidence\"".into(),
                "planner step 2: memory_expand target=chunk-1".into(),
            ],
            cortex_trace: None,
            created_at: 1,
        };
        let module = minimal_recursive_link_module(default_recursive_link_config(256));
        let latent = latent_trace_prototype(&trace, &module);
        assert_eq!(latent.status, "research_only");
        assert!(!latent.enabled_by_default);
        assert!(latent.hidden_states_redacted);
        assert!(latent
            .steps
            .iter()
            .all(|step| step.input_state_ref.starts_with("redacted://")));

        let report = evaluate_recursive_mas_trace(&trace);
        assert!(report.better_region_source_selection);
        assert!(report.lower_hallucination_rate);
        assert!(report.lower_token_usage);
        assert!(!report.may_enable_latent_default);
    }
}
