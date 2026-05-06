# imprint TODO

This is the remaining work to reach the full imprint vision: a personal memory filesystem where the model has useful latent familiarity with the user's world, while exact claims remain grounded in source recall.

Last repo-grounded status check: 2026-05-06.

## North Star

imprint should feel less like "chat plus database search" and more like a cognitive system with a library attached:

- The model carries a fuzzy, useful semantic sketch of the user's memory inside a private cortex adapter.
- The database remains the source of truth for exact chunks, citations, paths, provenance, and deletion/correction.
- The agent can reason recursively, critique weak evidence, retrieve surgically, and cite source anchors.
- The human can browse, trust, organize, and recover original sources without thinking like a retrieval engine.

V1 product contract: semantic addressability, not photographic memory. The model should know what exists and where to look. Source recall still proves exact answers.

## Current Baseline

Already implemented or partially implemented:

- [x] Versioned `CortexIndex` Rust/Swift schema.
- [x] SQLite persistence for current cortex index.
- [x] Compatibility `MemoryMap` projection inside `CortexIndex`.
- [x] Cortex-aware runtime query and chat memory routing.
- [x] CLI trace path uses cortex-aware route signals.
- [x] `CortexAdapterState` persisted and exposed through FFI/Swift.
- [x] Import/rebuild triggers cortex compile and adapter dataset preparation.
- [x] MLX-compatible prepared dataset manifest with source dataset hash.
- [x] Adapter state exposes separate data freshness, training status, and activation status.
- [x] Attention marks, access history, source metadata, and active chat sessions influence recall ranking.
- [x] Rust tests pass as of 2026-05-06.
- [x] macOS build passes as of 2026-05-06.

Important gaps:

- [ ] Actual MLX LoRA training is not automatic.
- [ ] Trained adapters are not loaded into planner/response model selection.
- [ ] Adapter freshness can mean "prepared data matches," not "trained and active."
- [ ] True latent RecursiveMAS/RecursiveLink is not implemented.
- [ ] CLI/MCP/UI still expose legacy map language in places.
- [ ] Human library, managed source artifacts, exact deep links, and import queue are still early.

## Phase 1: Make Cortex Adapter Training Real

Goal: cross from "better cortex-aware retrieval" into "some semantic memory is inside model weights."

- [x] Add a durable adapter job table or reuse `jobs` with typed payloads for cortex training.
- [x] Define adapter lifecycle states:
  - [x] `missing`
  - [x] `prepared`
  - [x] `queued`
  - [x] `training`
  - [x] `trained`
  - [x] `eval_failed`
  - [x] `active`
  - [x] `stale`
  - [x] `failed`
- [x] Split adapter state into explicit fields:
  - [x] `data_freshness`
  - [x] `training_status`
  - [x] `activation_status`
  - [x] `current_source_dataset_hash`
  - [x] `trained_source_dataset_hash`
  - [x] `active_adapter_hash`
  - [x] `base_model`
  - [x] `adapter_path`
  - [x] `eval_score`
  - [x] `failure_reason`
- [x] Wire a Rust-side trainer runner that invokes `training/train_mlx_lora.py` or `mlx_lm` directly.
- [x] Run training asynchronously after import/rebuild so source recall stays usable.
- [x] Add cancel/retry behavior for adapter jobs.
- [x] Add timeout, log capture, and failed-state persistence for training runs.
- [x] Keep the previously active adapter when a new training job fails.
- [x] Record trained adapter manifests with:
  - [x] base model
  - [x] source dataset hash
  - [x] prepared dataset hash
  - [x] adapter file hash
  - [x] record counts
  - [x] training iters
  - [x] command/version metadata
  - [x] created/finished timestamps
- [x] Add a small adapter evaluation suite before activation.
- [ ] Define minimum activation gates:
  - [ ] route-region accuracy above baseline
  - [x] source-expansion behavior present
  - [x] critique-evidence behavior present
  - [x] no regression on "answer only from source" tests
- [x] Mark adapter `active` only after eval passes.
- [ ] Surface adapter lifecycle in Swift:
  - [ ] queued/training progress
  - [ ] fresh/stale/trained/active distinction
  - [ ] last successful training time
  - [ ] failure reason and retry action
- [ ] Add CLI commands:
  - [x] `cortex status`
  - [x] `cortex compile`
  - [x] `cortex train`
  - [x] `cortex eval`
  - [x] `cortex activate`
- [x] Keep old `compile`/`map` commands as compatibility aliases.

## Phase 2: Load The Personal Adapter Into The Model Path

Goal: planner/response behavior should actually use the trained personal cortex adapter.

- [ ] Extend model configuration with adapter selection:
  - [ ] planner adapter path
  - [ ] response adapter path
  - [ ] shared cortex adapter path
  - [ ] active adapter hash
  - [ ] activation policy
- [ ] Decide how LFM2.5 is served with LoRA in the local runtime:
  - [ ] MLX-LM adapter loading path
  - [ ] fallback OpenAI-compatible endpoint behavior
  - [ ] Ollama/llama.cpp compatibility story
- [ ] Add health check that confirms the active adapter is loadable.
- [ ] Add a route-only probe that tests whether the adapted model names the expected source family before retrieval.
- [ ] Make planner prompt/context shorter when a fresh adapter is active.
- [ ] Keep full source-grounded retrieval regardless of adapter freshness.
- [ ] Add model selection rules:
  - [ ] use active adapter when fresh
  - [ ] warn when stale
  - [ ] fall back when missing/failed
  - [ ] never block exact source recall on adapter state
- [ ] Add regression tests with a fake local model endpoint that verifies adapter path selection.
- [ ] Add UI controls for:
  - [ ] train now
  - [ ] activate last trained adapter
  - [ ] disable adapter
  - [ ] compare base vs adapted routing

## Phase 3: Improve Cortex Training Data

Goal: teach semantic addressability and source discipline, not accidental memorization of long private text.

- [ ] Expand training tasks beyond current artifact-body examples:
  - [ ] `query_to_region`
  - [ ] `query_to_source_family`
  - [ ] `query_to_tool_plan`
  - [ ] `chunk_to_semantic_address`
  - [ ] `weak_evidence_to_next_action`
  - [ ] `snippet_set_to_citation_boundary`
  - [ ] `deleted_or_stale_memory_to_caution`
  - [ ] `web_needed_or_not`
- [ ] Generate route examples from:
  - [ ] document titles
  - [ ] headings/sections
  - [ ] source paths
  - [ ] entity overlaps
  - [ ] citation links
  - [ ] chat decisions/tasks
  - [ ] web finding queries
  - [ ] successful retrieval traces
- [ ] Add negative examples:
  - [ ] misleading source family
  - [ ] weak evidence
  - [ ] stale web finding
  - [ ] derived memory without source anchor
  - [ ] deleted/suppressed source
- [ ] Keep long exact source text out of default adapter training.
- [ ] Add explicit source refs and anchor IDs to every generated example when available.
- [ ] Add deterministic train/eval/test split by source ID.
- [ ] Prevent leakage between train/eval splits.
- [ ] Version the training schema and provide migrations.
- [ ] Add dataset inspection UI or CLI summary:
  - [ ] task counts
  - [ ] source counts
  - [ ] private vs shared memory counts
  - [ ] source types
  - [ ] stale/deleted exclusions
- [ ] Add export redaction policies for secrets and sensitive docs.

## Phase 4: True Latent RecursiveMAS

Goal: move from text/tool-trace recursion to latent hidden-state recursion inspired by RecursiveMAS.

- [ ] Create an isolated research module before wiring into production chat.
- [ ] Define roles:
  - [ ] planner
  - [ ] retriever
  - [ ] critic
  - [ ] solver
  - [ ] memory steward
- [ ] Define role interfaces at two levels:
  - [ ] text/tool fallback contract
  - [ ] latent hidden-state contract
- [ ] Study local LFM2.5/MLX internals needed to access hidden states.
- [ ] Implement a minimal RecursiveLink-style module:
  - [ ] inner recursion within one role
  - [ ] outer transfer between roles
  - [ ] frozen base model weights
  - [ ] small trainable bridge modules
- [ ] Build training data for latent recursion from existing traces:
  - [ ] planner actions
  - [ ] retrieved snippets
  - [ ] critic sufficiency
  - [ ] final source-grounded answer boundary
- [ ] Add eval tasks:
  - [ ] fewer tool calls for same answer quality
  - [ ] better region/source selection
  - [ ] lower hallucination rate
  - [ ] better weak-evidence refusal
  - [ ] lower token usage vs text recursion
- [ ] Keep text-mediated recursion as fallback.
- [ ] Add trace observability without leaking hidden states:
  - [ ] role sequence
  - [ ] sufficiency outcome
  - [ ] source refs selected
  - [ ] reason codes
  - [ ] confidence/caution state
- [ ] Do not ship latent recursion as default until eval beats the text/tool baseline.

## Phase 5: Source Recall And Provenance

Goal: make exact recall trustworthy, deep-linkable, and reversible.

- [ ] Make source artifacts first-class:
  - [ ] original file identity
  - [ ] managed copy or in-place reference policy
  - [ ] file hash
  - [ ] parser version
  - [ ] import timestamp
  - [ ] user-visible provenance
- [ ] Decide storage modes:
  - [ ] reference in place
  - [ ] copy into managed local store
  - [ ] both, with reconciliation
- [ ] Add source artifact migration for existing documents.
- [ ] Preserve stable IDs across moves/renames when content hash matches.
- [ ] Add durable file opening:
  - [ ] text offset
  - [ ] Markdown heading
  - [ ] PDF page
  - [ ] PDF text selection or bounding box when possible
  - [ ] email/thread location
  - [ ] browser URL/archive location
- [ ] Improve `SourceAnchor` precision:
  - [ ] rendered page metadata
  - [ ] section hierarchy
  - [ ] paragraph index
  - [ ] byte offsets plus character offsets
  - [ ] source artifact ID
- [ ] Add provenance inspector UI for every hit/chunk/document.
- [ ] Add source trust policy:
  - [ ] local source
  - [ ] user-authored note
  - [ ] imported document
  - [ ] web finding
  - [ ] generated summary
  - [ ] compiler artifact
- [ ] Make derived artifacts visibly derived, never source truth.
- [ ] Add deletion/correction semantics:
  - [ ] deleted source removed from search
  - [ ] deleted source excluded from future adapter training
  - [ ] stale adapters marked stale when deleted content was included
  - [ ] re-train/revoke workflow for personal adapters

## Phase 6: Vector Index And Graph Quality

Goal: scale beyond in-memory approximate region search while making memory surfable.

- [ ] Add a real ANN index:
  - [ ] HNSW or equivalent local ANN
  - [ ] incremental update support
  - [ ] rebuild path
  - [ ] persistence
  - [ ] compatibility with embedding dimension changes
- [ ] Add index health metadata:
  - [ ] embedding model
  - [ ] dimension
  - [ ] corpus hash
  - [ ] index version
  - [ ] last rebuild time
- [ ] Improve region derivation:
  - [ ] stable clustering
  - [ ] better labels
  - [ ] hierarchical regions
  - [ ] source-type-aware regions
  - [ ] active project/session overlays
- [ ] Improve graph links:
  - [ ] citation/reference parser quality
  - [ ] entity extraction
  - [ ] same-source ordering
  - [ ] semantic neighbor confidence
  - [ ] explicit user/agent links
  - [ ] link provenance and reversibility
- [ ] Add graph inspection tools:
  - [ ] why linked
  - [ ] source evidence
  - [ ] confidence
  - [ ] hide/suppress link
  - [ ] pin/promote link

## Phase 7: Attention, Hotness, And Ranking

Goal: make attention inspectable, reversible, and useful instead of hidden scoring magic.

- [ ] Add user-facing attention controls:
  - [ ] pin
  - [ ] promote
  - [ ] suppress
  - [ ] mark hot/warm/cold
  - [ ] explain why ranked
- [ ] Add attention inspector:
  - [ ] who set mark
  - [ ] why
  - [ ] when
  - [ ] source target
  - [ ] revert action
- [ ] Add active context models:
  - [ ] active chat
  - [ ] active project
  - [ ] active workspace/folder
  - [ ] active collection
  - [ ] current task/session
- [ ] Add source freshness policy:
  - [ ] retrieved_at
  - [ ] freshness_expires_at
  - [ ] stale warning
  - [ ] refresh needed
  - [ ] web recapture workflow
- [ ] Add learned ranking experiments after deterministic baseline is stable.
- [ ] Evaluate ranking with held-out retrieval traces and user feedback.

## Phase 8: Import, Watchers, And Library Management

Goal: make the library feel like a real local memory filesystem.

- [ ] Add import queue:
  - [ ] pending/running/succeeded/failed/cancelled states
  - [ ] resumability
  - [ ] per-file progress
  - [ ] batch progress
  - [ ] retry failed imports
- [ ] Add file watcher support.
- [ ] Add update detection:
  - [ ] changed file
  - [ ] moved file
  - [ ] deleted file
  - [ ] duplicate file
  - [ ] replaced file
- [ ] Add dedupe UX:
  - [ ] same hash
  - [ ] same path
  - [ ] similar title/content
  - [ ] choose keep/merge/replace
- [ ] Add delete workflows:
  - [ ] delete document
  - [ ] delete source artifact
  - [ ] delete derived artifacts
  - [ ] mark adapter stale when needed
- [ ] Add collections and saved views.
- [ ] Add source type filters.
- [ ] Add saved trails from surf sessions.
- [ ] Add durable "open original" actions.
- [ ] Add backup/export/import for the whole local library.

## Phase 9: Human UI As Memory Instrument

Goal: make the macOS app a calm, dense, inspectable library, not just a graph demo.

- [ ] Rename remaining Map UI concepts to Cortex or Source Recall where appropriate.
- [ ] Keep `Map` only where it literally means spatial visualization.
- [ ] Add document library view:
  - [ ] list/table
  - [ ] source type
  - [ ] date imported
  - [ ] trust/freshness
  - [ ] collections
  - [ ] open source
- [ ] Add cortex view:
  - [ ] adapter status
  - [ ] corpus hash
  - [ ] training/eval state
  - [ ] region/source-family sketches
  - [ ] route examples
  - [ ] stale reasons
- [ ] Add source recall inspector:
  - [ ] selected chunk
  - [ ] document context
  - [ ] anchors
  - [ ] graph links
  - [ ] provenance
  - [ ] ranking reasons
- [ ] Add surf session UI:
  - [ ] history
  - [ ] backtrack
  - [ ] neighborhood
  - [ ] expand source
  - [ ] cite/save trail
- [ ] Add import queue UI.
- [ ] Add settings for local-first storage, model endpoints, adapter policy, and privacy.

## Phase 10: MCP And Agent API

Goal: expose imprint as an agent-native memory filesystem.

- [ ] Add first-class cortex MCP tools:
  - [ ] `memory_cortex_status`
  - [ ] `memory_cortex_compile`
  - [ ] `memory_cortex_train`
  - [ ] `memory_cortex_route`
  - [ ] `memory_cortex_eval`
- [ ] Keep `memory_compile` as an alias until clients migrate.
- [ ] Add route-plan output to MCP search tools.
- [ ] Add explicit source-recall tools:
  - [ ] open source artifact
  - [ ] expand exact anchor
  - [ ] list provenance
  - [ ] cite hit
  - [ ] save trail
- [ ] Add writeback tools:
  - [ ] save derived memory
  - [ ] save web finding
  - [ ] save agent link
  - [ ] mark attention
  - [ ] revert attention mark
- [ ] Add web capture tools:
  - [ ] capture URL
  - [ ] save search result
  - [ ] refresh stale web finding
  - [ ] set freshness/expiration
- [ ] Document schemas for all MCP tools.
- [ ] Add compatibility tests for MCP JSON outputs.

## Phase 11: Web Findings And Browser Capture

Goal: make web search a memory expansion path, not disposable prompt context.

- [ ] Add browser capture with:
  - [ ] URL
  - [ ] title
  - [ ] retrieved_at
  - [ ] content hash
  - [ ] extracted text
  - [ ] summary
  - [ ] citations/source refs
  - [ ] freshness expiry
  - [ ] source trust
- [ ] Add refresh policy:
  - [ ] manual refresh
  - [ ] stale warning
  - [ ] scheduled refresh for pinned web findings
- [ ] Add web finding diff/history.
- [ ] Add source trust defaults by domain/source type.
- [ ] Add capture-to-cortex behavior:
  - [ ] index immediately
  - [ ] mark hot
  - [ ] include in next adapter training
  - [ ] preserve exact source provenance

## Phase 12: Privacy, Forgetting, And Safety

Goal: make personal latent memory controllable and reversible enough to trust.

- [ ] Add privacy levels:
  - [ ] private user memory
  - [ ] shared project memory
  - [ ] global/reference memory
  - [ ] excluded from adapter training
- [ ] Add per-source training opt-out.
- [ ] Add secret detection before training export.
- [ ] Add "forget this" workflow:
  - [ ] remove from search
  - [ ] remove from cortex index
  - [ ] exclude from future datasets
  - [ ] mark existing adapter stale/contaminated
  - [ ] retrain replacement adapter
- [ ] Add adapter provenance:
  - [ ] which corpus hash
  - [ ] which sources included
  - [ ] which sources excluded
  - [ ] train/eval files
- [ ] Add local-only guarantees for private adapters.
- [ ] Add export controls and warnings when training data leaves local machine.

## Phase 13: Evaluation Harness

Goal: measure whether cortex, adapters, and recursion actually improve outcomes.

- [ ] Build eval sets for:
  - [ ] route accuracy
  - [ ] source-family selection
  - [ ] exact anchor recovery
  - [ ] citation correctness
  - [ ] weak evidence detection
  - [ ] web-needed decisions
  - [ ] deletion/staleness behavior
  - [ ] token/tool-call efficiency
- [ ] Compare baselines:
  - [ ] lexical map routing
  - [ ] vector-only routing
  - [ ] cortex index routing
  - [ ] base model planner
  - [ ] adapted model planner
  - [ ] text recursive loop
  - [ ] latent RecursiveLink loop
- [ ] Add regression datasets from real successful traces.
- [ ] Add synthetic fixtures that do not leak private data.
- [ ] Track metrics over time.
- [ ] Gate adapter activation on eval.
- [ ] Gate RecursiveMAS default enablement on eval.

## Phase 14: Cloud, Sync, And Multi-Device

Goal: preserve local-first behavior while preparing for a cloud-backed memory library.

- [ ] Define local managed storage layout.
- [ ] Define cloud sync object model:
  - [ ] source artifacts
  - [ ] extracted text
  - [ ] chunks
  - [ ] embeddings
  - [ ] cortex indexes
  - [ ] derived artifacts
  - [ ] attention marks
  - [ ] adapter manifests
- [ ] Decide whether adapters sync or remain per-device.
- [ ] Add encryption/key management plan.
- [ ] Add conflict resolution:
  - [ ] edited notes
  - [ ] attention marks
  - [ ] derived memories
  - [ ] source moves/deletes
- [ ] Add web app/API architecture.
- [ ] Add shared project memory and permissions.
- [ ] Add global/reference memory boundary.

## Phase 15: Migration And Compatibility

Goal: evolve schemas without breaking existing memory stores.

- [ ] Add explicit database schema version.
- [ ] Add migrations for:
  - [ ] source artifacts
  - [ ] cortex index versions
  - [ ] adapter state versions
  - [ ] attention/access history
  - [ ] managed storage
- [ ] Keep compatibility aliases:
  - [ ] `map`
  - [ ] `memory_compile`
  - [ ] old FFI payloads where practical
- [ ] Add migration tests from older stores.
- [ ] Add backup-before-migration behavior.
- [ ] Add store integrity check command.

## Phase 16: Definition Of Done For The Complete Vision

The complete local-first vision is reached when:

- [ ] A user imports or captures material and it becomes searchable immediately.
- [ ] The source artifact is durable, inspectable, and recoverable.
- [ ] The cortex index updates deterministically after library changes.
- [ ] A personal adapter trains automatically, passes eval, and becomes active without blocking source recall.
- [ ] The active model can route to the right source family from latent familiarity before retrieval.
- [ ] Exact answers cite source anchors and can expand to original context.
- [ ] Deleted or excluded sources are removed from future training and make affected adapters stale.
- [ ] Recursive planner/critic/retriever/solver behavior improves routing and evidence sufficiency beyond the text/tool baseline.
- [ ] Human UI supports import, browse, inspect, search, collect, source opening, provenance, attention, and adapter status.
- [ ] MCP exposes the same memory substrate to agents with read/write/capture/attention/cortex tools.
- [ ] Evaluation shows adapted cortex routing beats vector-only and base-model routing.
- [ ] The system remains local-first, recoverable, and honest about what is latent memory versus source truth.

## Suggested Next Three Milestones

1. **Real Adapter Training And Activation**
   - Queue/run MLX LoRA from Rust.
   - Track training/eval/active states.
   - Load active adapter into planner/response path.

2. **Source Artifact And Deep-Link Foundation**
   - Make original sources first-class.
   - Improve anchors for PDFs/Markdown/text.
   - Add provenance/open-original UI.

3. **RecursiveMAS Research Slice**
   - Build isolated latent RecursiveLink prototype.
   - Compare against current text/tool recursion.
   - Keep fallback and only promote after eval.
