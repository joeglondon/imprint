# imprint TODO

This is the remaining work to reach the full imprint vision: a personal memory filesystem where the model has useful latent familiarity with the user's world, while exact claims remain grounded in source recall.

Last repo-grounded status check: 2026-05-08.

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
- [x] Attention marks, access history, source metadata, and active chat/project/workspace/collection/task context influence recall ranking.
- [x] Source artifacts are persisted as first-class provenance records with stable file-hash IDs for local imports.
- [x] Local imports keep a hidden managed source copy and reconcile moves/renames by content hash.
- [x] Surf/open/expand results carry durable source open targets for local offsets, Markdown headings, PDF pages, and web URLs.
- [x] Source anchors carry source artifact IDs, explicit byte/character offsets, section hierarchy, and paragraph index metadata.
- [x] Rust tests pass as of 2026-05-08.
- [x] macOS build passes as of 2026-05-08.

Important gaps:

- [x] Actual MLX LoRA training is automatic after import/rebuild when a compiler/response/chat model is configured.
- [x] Trained adapters are loaded into planner/response model selection when active and fresh.
- [x] Adapter freshness is split from trained/active selection and source recall fallback.
- [ ] True latent RecursiveMAS/RecursiveLink is not implemented.
- [ ] CLI/MCP/UI still expose legacy map language in places.
- [ ] Human library UI and exact deep-link polish are still early.

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
- [x] Define minimum activation gates:
  - [x] route-region accuracy above baseline
  - [x] source-expansion behavior present
  - [x] critique-evidence behavior present
  - [x] no regression on "answer only from source" tests
- [x] Mark adapter `active` only after eval passes.
- [x] Surface adapter lifecycle in Swift:
  - [x] queued/training progress
  - [x] fresh/stale/trained/active distinction
  - [x] last successful training time
  - [x] failure reason and retry action
- [x] Add CLI commands:
  - [x] `cortex status`
  - [x] `cortex compile`
  - [x] `cortex train`
  - [x] `cortex eval`
  - [x] `cortex activate`
- [x] Keep old `compile`/`map` commands as compatibility aliases.

## Phase 2: Load The Personal Adapter Into The Model Path

Goal: planner/response behavior should actually use the trained personal cortex adapter.

- [x] Extend model configuration with adapter selection:
  - [x] planner adapter path
  - [x] response adapter path
  - [x] shared cortex adapter path
  - [x] active adapter hash
  - [x] activation policy
- [x] Decide how LFM2.5 is served with LoRA in the local runtime:
  - [x] MLX-LM adapter loading path
  - [x] fallback OpenAI-compatible endpoint behavior
  - [x] Ollama/llama.cpp compatibility story
- [x] Add health check that confirms the active adapter is loadable.
- [x] Add a route-only probe that tests whether the adapted model names the expected source family before retrieval.
- [x] Make planner prompt/context shorter when a fresh adapter is active.
- [x] Keep full source-grounded retrieval regardless of adapter freshness.
- [x] Add model selection rules:
  - [x] use active adapter when fresh
  - [x] warn when stale
  - [x] fall back when missing/failed
  - [x] never block exact source recall on adapter state
- [x] Add regression tests with a fake local model endpoint that verifies adapter path selection.
- [x] Add UI controls for:
  - [x] train now
  - [x] activate last trained adapter
  - [x] disable adapter
  - [x] compare base vs adapted routing

## Phase 3: Improve Cortex Training Data

Goal: teach semantic addressability and source discipline, not accidental memorization of long private text.

- [x] Expand training tasks beyond current artifact-body examples:
  - [x] `query_to_region`
  - [x] `query_to_source_family`
  - [x] `query_to_tool_plan`
  - [x] `chunk_to_semantic_address`
  - [x] `weak_evidence_to_next_action`
  - [x] `snippet_set_to_citation_boundary`
  - [x] `deleted_or_stale_memory_to_caution`
  - [x] `web_needed_or_not`
- [x] Generate route examples from:
  - [x] document titles
  - [x] headings/sections
  - [x] source paths
  - [x] entity overlaps
  - [x] citation links
  - [x] chat decisions/tasks
  - [x] web finding queries
  - [x] successful retrieval traces
- [x] Add negative examples:
  - [x] misleading source family
  - [x] weak evidence
  - [x] stale web finding
  - [x] derived memory without source anchor
  - [x] deleted/suppressed source
- [x] Keep long exact source text out of default adapter training.
- [x] Add explicit source refs and anchor IDs to every generated example when available.
- [x] Add deterministic train/eval/test split by source ID.
- [x] Prevent leakage between train/eval splits.
- [x] Version the training schema and provide migrations.
- [x] Add dataset inspection UI or CLI summary:
  - [x] task counts
  - [x] source counts
  - [x] private vs shared memory counts
  - [x] source types
  - [x] stale/deleted exclusions
- [x] Add export redaction policies for secrets and sensitive docs.

## Phase 4: True Latent RecursiveMAS

Goal: move from text/tool-trace recursion to latent hidden-state recursion inspired by RecursiveMAS.

- [x] Create an isolated research module before wiring into production chat.
- [x] Define roles:
  - [x] planner
  - [x] retriever
  - [x] critic
  - [x] solver
  - [x] memory steward
- [x] Define role interfaces at two levels:
  - [x] text/tool fallback contract
  - [x] latent hidden-state contract
- [x] Study local LFM2.5/MLX internals needed to access hidden states.
- [x] Implement a minimal RecursiveLink-style module:
  - [x] inner recursion within one role
  - [x] outer transfer between roles
  - [x] frozen base model weights
  - [x] small trainable bridge modules
- [x] Build training data for latent recursion from existing traces:
  - [x] planner actions
  - [x] retrieved snippets
  - [x] critic sufficiency
  - [x] final source-grounded answer boundary
- [x] Add eval tasks:
  - [x] fewer tool calls for same answer quality
  - [x] better region/source selection
  - [x] lower hallucination rate
  - [x] better weak-evidence refusal
  - [x] lower token usage vs text recursion
- [x] Keep text-mediated recursion as fallback.
- [x] Add trace observability without leaking hidden states:
  - [x] role sequence
  - [x] sufficiency outcome
  - [x] source refs selected
  - [x] reason codes
  - [x] confidence/caution state
- [x] Do not ship latent recursion as default until eval beats the text/tool baseline.

## Phase 5: Source Recall And Provenance

Goal: make exact recall trustworthy, deep-linkable, and reversible.

- [x] Make source artifacts first-class:
  - [x] original file identity
  - [x] managed copy or in-place reference policy
  - [x] file hash
  - [x] parser version
  - [x] import timestamp
  - [x] user-visible provenance
- [x] Decide storage modes:
  - [x] reference in place
  - [x] copy into managed local store
  - [x] both, with reconciliation
- [x] Add source artifact migration for existing documents.
- [x] Preserve stable IDs across moves/renames when content hash matches.
- [x] Add durable file opening:
  - [x] text offset
  - [x] Markdown heading
  - [x] PDF page
  - [x] PDF text selection or bounding box when possible
  - [x] email/thread location
  - [x] browser URL/archive location
- [x] Improve `SourceAnchor` precision:
  - [x] rendered page metadata
  - [x] section hierarchy
  - [x] paragraph index
  - [x] byte offsets plus character offsets
  - [x] source artifact ID
- [x] Add provenance inspector UI for every hit/chunk/document.
- [x] Add source trust policy:
  - [x] local source
  - [x] user-authored note
  - [x] imported document
  - [x] web finding
  - [x] generated summary
  - [x] compiler artifact
- [x] Make derived artifacts visibly derived, never source truth.
- [x] Add deletion/correction semantics:
  - [x] deleted source removed from search
  - [x] deleted source excluded from future adapter training
  - [x] stale adapters marked stale when deleted content was included
  - [x] re-train/revoke workflow for personal adapters

## Phase 6: Vector Index And Graph Quality

Goal: scale beyond in-memory approximate region search while making memory surfable.

- [x] Add a real ANN index:
  - [x] HNSW or equivalent local ANN
  - [x] incremental update support
  - [x] rebuild path
  - [x] persistence
  - [x] compatibility with embedding dimension changes
- [x] Add index health metadata:
  - [x] embedding model
  - [x] dimension
  - [x] corpus hash
  - [x] index version
  - [x] last rebuild time
- [x] Improve region derivation:
  - [x] stable clustering
  - [x] better labels
  - [x] hierarchical regions
  - [x] source-type-aware regions
  - [x] active project/session overlays
- [x] Improve graph links:
  - [x] citation/reference parser quality
  - [x] entity extraction
  - [x] same-source ordering
  - [x] semantic neighbor confidence
  - [x] explicit user/agent links
  - [x] link provenance and reversibility
- [x] Add graph inspection tools:
  - [x] why linked
  - [x] source evidence
  - [x] confidence
  - [x] hide/suppress link
  - [x] pin/promote link

## Phase 7: Attention, Hotness, And Ranking

Goal: make attention inspectable, reversible, and useful instead of hidden scoring magic.

- [x] Add user-facing attention controls:
  - [x] pin
  - [x] promote
  - [x] suppress
  - [x] mark hot/warm/cold
  - [x] explain why ranked
- [x] Add attention inspector:
  - [x] who set mark
  - [x] why
  - [x] when
  - [x] source target
  - [x] revert action
- [x] Add active context models:
  - [x] active chat
  - [x] active project
  - [x] active workspace/folder
  - [x] active collection
  - [x] current task/session
- [x] Add source freshness policy:
  - [x] retrieved_at
  - [x] freshness_expires_at
  - [x] stale warning
  - [x] refresh needed
  - [x] web recapture workflow
- [x] Add learned ranking experiments after deterministic baseline is stable.
- [x] Evaluate ranking with held-out retrieval traces and user feedback.

## Phase 8: Import, Watchers, And Library Management

Goal: make the library feel like a real local memory filesystem.

- [x] Add import queue:
  - [x] pending/running/succeeded/failed/cancelled states
  - [x] resumability
  - [x] per-file progress
  - [x] batch progress
  - [x] retry failed imports
- [x] Add file watcher support.
- [x] Add update detection:
  - [x] changed file
  - [x] moved file
  - [x] deleted file
  - [x] duplicate file
  - [x] replaced file
- [x] Add dedupe UX:
  - [x] same hash
  - [x] same path
  - [x] similar title/content
  - [x] choose keep/merge/replace
- [x] Add delete workflows:
  - [x] delete document
  - [x] delete source artifact
  - [x] delete derived artifacts
  - [x] mark adapter stale when needed
- [x] Add collections and saved views.
- [x] Add source type filters.
- [x] Add saved trails from surf sessions.
- [x] Add durable "open original" actions.
- [x] Add backup/export/import for the whole local library.

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

- [x] Define local managed storage layout.
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

1. **Training Data Quality And Adapter Eval**
   - Expand semantic-address tasks.
   - Add negative/stale/deleted-source examples.
   - Measure adapted routing against base/vector baselines.

2. **Source Artifact And Deep-Link Foundation**
   - Make original sources first-class.
   - Improve anchors for PDFs/Markdown/text.
   - Add provenance/open-original UI.

3. **RecursiveMAS Research Slice**
   - Build isolated latent RecursiveLink prototype.
   - Compare against current text/tool recursion.
   - Keep fallback and only promote after eval.
