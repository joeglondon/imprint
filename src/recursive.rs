use crate::types::{ChatContextSnippet, ChatContextTrace};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

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
                    path: "/tmp/source.md".into(),
                    content_hash: "hash".into(),
                    start: 0,
                    end: 12,
                    page: None,
                    section: Some("Evidence".into()),
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
        assert_eq!(examples.len(), 3);
        assert!(examples
            .iter()
            .any(|example| example.task == "recursive_sufficiency_eval"
                && example.target.contains("cite_anchor_ids:anchor-1")));
        assert!(examples.iter().all(|example| example.source_grounded));
    }
}
