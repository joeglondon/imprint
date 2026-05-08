# imprint MCP Tool Schemas

Phase 10 exposes imprint as an agent-native memory filesystem. All tools return MCP text content whose `text` field is pretty-printed JSON.

## Cortex

- `memory_cortex_status({ max_jobs? })`: adapter state, current CortexIndex summary, recent jobs.
- `memory_cortex_compile({})`: compile CortexIndex and prepared adapter data.
- `memory_cortex_train({})`: queue/start adapter training for the current source dataset.
- `memory_cortex_route({ query, max_regions?, max_chunks? })`: route plan with ranked candidates and next source-recall steps.
- `memory_cortex_eval({ minimum_score? })`: evaluate the trained adapter against activation gates.
- `memory_evaluation_harness({ minimum_score? })`: run the phase 13 eval harness across eval-set coverage, baseline comparisons, adapter/RecursiveMAS gates, synthetic fixtures, regression traces, and metric history.
- `memory_compile({})`: compatibility alias for `memory_cortex_compile`.

## Search And Source Recall

- `memory_search({ query, max_regions?, max_chunks? })`: anchored chunk hits and typed `routed.route_plan`.
- `memory_open({ node })`: open `{ "Document": id }`, `{ "Chunk": id }`, or `{ "Region": id }`.
- `memory_open_source_artifact({ source_artifact_id })`: source artifact provenance plus imported documents.
- `memory_neighbors({ node, max_results? })`: surfable graph/document/semantic neighbors.
- `memory_links({ node })`, `memory_inspect_link({ link_id })`, `memory_mark_link({ link_id, action, reason, actor? })`: inspect and tune graph links.
- `memory_expand({ chunk_id, mode?, window? })`: expand chunk context by `window`, `page`, `section`, or `document`.
- `memory_jump_to_anchor({ anchor_id, window? })` and `memory_expand_anchor({ anchor_id, window? })`: expand exact source context.
- `memory_list_provenance({ target_id, target_kind? })`: provenance/source-anchor metadata for source artifacts, documents, chunks, derived memories, web findings, and links.
- `memory_cite_hit({ anchor_id, window?, actor? })`: records a cite access and returns exact source context.
- `memory_save_trail({ name, session })`: persist a reversible surf trail from a session state.
- `memory_chat_trace({ session_id })`: list chat context traces, snippets, tool steps, and cortex trace metadata.

## Writeback

- `memory_write_summary({ text, session_id?, confidence?, actor? })`
- `memory_save_derived_memory({ text, session_id?, confidence?, actor? })`
- `memory_write_link({ source_id, target_id, label, actor? })`
- `memory_save_agent_link({ source_id, target_id, label, actor? })`
- `memory_mark_attention({ target_id, target_kind, action, reason, actor? })`
- `memory_revert_attention_mark({ mark_id, actor? })`

`target_kind` values: `chat_session`, `session`, `project`, `workspace`, `collection`, `task`, `chat_message`, `transcript_chunk`, `derived_memory`, `web_finding`, `document`, `chunk`, `region`, `link`.

`action` values: `active`, `hot`, `warm`, `cold`, `promote`, `decay`, `pin`, `suppress`.

## Web Capture

Web tools are writeback endpoints for read-only web research performed by an agent or browser client.

- `memory_write_web_finding({ query, url, summary, extracted_text?, source_refs?, title?, retrieved_at?, freshness_expires_at?, confidence?, actor?, session_id? })`
- `memory_save_web_finding(...)`
- `memory_capture_url(...)`
- `memory_save_search_result(...)`
- `memory_refresh_web_finding({ web_finding_id, summary, extracted_text?, source_refs?, query?, title?, retrieved_at?, freshness_expires_at?, confidence?, actor? })`
- `memory_set_web_freshness({ web_finding_id, freshness_expires_at?, actor? })`
- `memory_web_finding_history({ url? })`
- `memory_web_refresh_candidates({})`
