# Agent Guide

This repo is building **imprint**, a shared memory filesystem and cortex for humans and AI agents.

Do not treat imprint as only "Obsidian for an LLM." That phrase is a useful wedge, but the broader goal is a personal and eventually cloud-capable repository for documents, notes, recipes, books, chat histories, web findings, project material, and source files. The same memory should be useful to a human browsing their library and to an LLM that has learned the library's semantic shape while still grounding exact claims in source context.

The long-horizon ambition is a semantic filesystem: first a local macOS app, later a web/cloud-backed memory library, and eventually the kind of baseline memory layer that could belong in an agent-native operating system.

## North Star

imprint should be a place where a person can toss all of their meaningful files and knowledge, while also making their LLMs smarter in a more brain-like way.

The fundamental architecture has two memory modes:

1. **Latent familiarity**
   - A personal cortex adapter teaches the model the shape of the user's library: projects, people, source families, recurring concepts, durable preferences, and likely retrieval addresses.
   - This is the model's blurry but useful memory. It should know that a topic exists and where inside imprint it should look, even before a tool search.
   - V1 optimizes for semantic addressability, not photographic recall.

2. **Source recall**
   - imprint remains the exact source of truth: chunks, anchors, paths, content hashes, parser versions, links, provenance, and citations.
   - When the model needs more than fuzzy familiarity, it must retrieve and expand source-grounded context from imprint.
   - Latent memory alone is not citation-safe. Exact quotes, dates, page-level claims, mutable facts, and deleted/forgotten content must resolve through source anchors or carry an explicit caveat.

The app has two first-class users:

1. **The human**
   - Imports, browses, searches, organizes, inspects, and trusts the library.
   - Can recover original sources, provenance, and context.
   - Should not need to manually create every link or connection.

2. **The AI agent**
   - Uses the cortex to recognize the shape of the library and choose where to retrieve exact evidence.
   - Searches, opens, follows links, expands source context, cites, and backtracks.
   - Can eventually save new findings, web-search results, and derived notes back into the repository with provenance.
   - Can use recursive cortex traces, critic notes, compiled cortex artifacts, and personal adapters to make local models better at semantic addressing without treating generated artifacts as source truth.

The core design should serve both users over one shared memory substrate. Avoid building a human app with AI bolted on, or an AI retrieval backend with a decorative UI bolted on.

For AI navigation, the canonical flow is:

1. Let the cortex form a latent guess about the relevant project, source family, region, or node.
2. Search or open nearby vector chunks using that semantic address.
3. Follow semantic, source, citation, or explicit links.
4. Expand from a chunk to a window, page, section, or full document.
5. Cite the source anchor and provenance.
6. If evidence is weak, recurse through planner, critic, retriever, and solver roles before answering.
7. Jump back to vector space and continue surfing.

For questions that might otherwise trigger web search, use a memory-before-web policy:

1. Check the cortex and library overview for plausible existing knowledge.
2. Search imprint before searching the web.
3. Open, expand, and cite memory hits when they are relevant enough.
4. Search the web only when imprint has no plausible semantic address, returns weak hits, contains stale information for the task, the user asks for current/live information, or external verification is required.
5. Save useful web findings back into imprint with query, source URLs, retrieval date, summary, citations, confidence, and freshness/expiration metadata.

Agents should also manage attention explicitly:

- Recently accessed, captured, edited, cited, or agent-promoted memory can become hot.
- Agents may mark results as important or unimportant for future routing, with a short reason and actor/provenance metadata.
- Importance marks, hot/warm/cold state, and pins should influence ranking and default attention, not mutate the underlying source or make old material unreachable.
- Hotness is not only for web findings. It applies to anything in imprint: local files, notes, chats, recipes, books, source docs, generated findings, and prior search results.

Do not optimize for dumping whole corpora into prompt context. Optimize for surgical latent familiarity, durable storage, anchored retrieval, local exploration, reversible navigation, source-grounded expansion, and human-inspectable organization.

## Layering Principles

**Layer 1: Library**

- imprint is a repository, not just an index.
- Imported items should have stable identity, source type, provenance, metadata, and browseable human-facing structure.
- Long term, the system may reference files in place, copy files into managed local storage, sync through cloud storage, or support a web app. Keep storage decisions explicit and migration-friendly.
- Preserve local-first behavior unless a feature explicitly opts into network or cloud behavior.

**Layer 2: Cortex**

- `CortexIndex` is the compact, inspectable semantic-address layer generated from the library.
- It should contain routing sketches, region/source-family hints, artifact IDs, source refs, corpus hashes, and examples that teach where to look without copying whole sources.
- The legacy map payload should remain only as a compatibility projection under `CortexIndex`, not the primary agent contract.
- Personal cortex adapters are local/private and should learn semantic addressability: what exists, what it resembles, and where exact recall should begin.
- Adapter refresh is part of import/rebuild. Search and chat must remain usable while the adapter is stale, training, or failed.

**Layer 3: Vector/graph source recall**

- Chunks are the primary semantic units.
- Regions are coarse entry points for cortex-guided retrieval.
- Links should make nearby memory surfable: same document, semantic neighbor, citation/reference, entity overlap, and region membership.
- The model should be able to enter at a relevant point, not traverse from a fake root every time.
- Links may be generated automatically by embeddings, parsers, citations, user actions, or agent-written findings, but they must remain inspectable.
- Retrieval should eventually include an attention/ranking layer over vector similarity: freshness, recency, access frequency, agent importance marks, hot/warm/cold state, active session/project, source trust, explicit pins, and decay.
- Decay should make memory less likely by default, not unreachable. If the user directly asks for old material, it should still be findable.
- Newly captured or recently accessed material may be hot immediately so an agent can search/grep it from memory instead of carrying bulky context around.

**Layer 4: Source context**

- Every chunk should preserve a `SourceAnchor` whenever possible.
- Anchors must include path or managed artifact identity, content hash, offsets, parser version, and page/section when available.
- Expanding context should use anchors and original document text, not just the embedded chunk.
- Never treat an embedding as a substitute for source truth.
- Web-search findings, AI summaries, and generated notes must carry provenance just like imported files.

**Layer 5: Recursive latent reasoning**

- Recursive cortex traces are execution records, not proof of truth. They should show planner actions, gathered snippets, critique, stop reasons, and source grounding.
- `BrainArtifact` records are being reframed as compiled cortex artifacts: persistent cognitive maps for semantic addresses, tool choice, critique patterns, and collaboration patterns. They are derived artifacts with source refs, content hashes, confidence, and provenance.
- Training/eval JSONL exports should teach semantic addressing, source expansion, tool choice, critique, and collaboration behavior.
- Inspired by RecursiveMAS, imprint should move toward true latent planner/critic/retriever/solver loops using small trainable RecursiveLink-style modules around frozen local models. Text-mediated recursion is a fallback, not the end state.
- The authoritative answer path still needs source anchors or explicit caveats, even when latent recursion and personal adapters are active.

## Implementation Map

- `src/types.rs`: shared memory schema. Preserve compatibility carefully.
- `src/ingest.rs`: extraction, chunking, source anchors, embedding reuse, region derivation.
- `src/index.rs`: embedder abstraction and vector search.
- `src/query.rs`: cortex-guided routing plus search inside selected regions.
- `src/graph.rs`: link generation.
- `src/surf.rs`: surf/open/neighbor/expand/jump/session behavior.
- `src/compiler.rs`: `CortexIndex`, compiled cortex artifacts, semantic-address examples, and compatibility map projection.
- `src/training.rs`: train/eval JSONL export generation for semantic addressing, tool choice, critique, source expansion, and collaboration tasks.
- `src/store.rs`: SQLite persistence.
- `src/app.rs`: app orchestration used by CLI, FFI, and Swift.
- `src/mcp.rs`: AI-facing tool server.
- `src/ffi.rs`: Swift bridge.
- `training/`: Python harness for validating exports, baseline eval, MLX cortex adapter training, and future RecursiveLink experiments.
- `MemoryApp/Sources/`: macOS UI.

## Engineering Rules

- Keep Rust data structures and Swift mirror models in sync.
- If you add or rename serialized fields, update FFI, Swift models, tests, and persistence together.
- Preserve existing imported documents and anchors during rebuilds.
- Preserve original source identity and provenance when changing ingestion, storage, or rebuild behavior.
- Do not break embedding reuse without a migration reason.
- Prefer deterministic IDs, stable ordering, and repeatable tests.
- Use real parsers and structured metadata over string hacks when possible.
- Keep local-first behavior. Network/model calls should be explicit, configurable, and recoverable.
- Design storage so future managed local libraries, cloud sync, and web access remain possible.
- Treat AI-generated links/summaries/findings as derived artifacts with provenance, not source truth.
- Treat compiled cortex artifacts and training JSONL outputs as compiler artifacts. They may improve semantic addressing and planning, but citations must still resolve to original source anchors when possible.
- Treat personal cortex adapter refresh as part of import/rebuild, while keeping search and chat usable when adapter state is stale, training, or failed.
- Missing Python ML dependencies or MLX must not corrupt the library or block source-grounded recall. They should leave adapter state inspectably stale or failed.
- When changing compiler outputs, keep deterministic IDs, schema versions, source refs, and train/eval split behavior stable or migrate them deliberately.
- When changing cortex adapter behavior, record corpus hash, dataset hash, base model, adapter path, freshness state, eval score, and failure reason where applicable.
- Do not train long exact source text into adapters by default. Train semantic addresses, source-family sketches, routing examples, critique behavior, and bounded summaries with source refs.
- Treat web search as a memory expansion path. If web findings are useful, they should become indexed, source-grounded memory rather than disposable prompt context.
- Do not make agent behavior depend on holding large web results in context when those results can be written to and searched from imprint.
- When adding importance, hotness, or suppression metadata, record who/what set it and why. Agent attention marks should be reversible and inspectable.
- Avoid UI copy that claims production-grade abilities before the Rust core actually supports them.

## Current Known Gaps

- `CortexIndex` is not yet a first-class persisted schema; current compiler artifacts are scaffolding.
- Adapter freshness state is now reported by `memory_compile`/CLI/MCP JSON, persisted in SQLite, exposed through FFI, and shown in the Swift model inspector after a cortex compile. It is not yet refreshed automatically after import/rebuild or used for model selection/loading.
- Cortex compile now automatically prepares an MLX-compatible adapter dataset when the current training exports have no fresh manifest. This writes `train.jsonl`, `valid.jsonl`, `test.jsonl`, and an inspectable `adapter_manifest.json` under `store/adapters/prepared-<source-hash>/`, using the configured compiler model. Remaining drawback: this is preparation only; real MLX LoRA training is still manual through `training/train_mlx_lora.py`, and import/rebuild do not yet trigger compile/prepare.
- MLX cortex adapter training is not yet a core import/rebuild step.
- MLX adapter preparation manifests include base model, source dataset hash, prepared dataset hash, record counts, status, and iteration target. Freshness detection expects manifests under the store's `adapters/` directory.
- Compiler-generated brain artifacts remain searchable as derived memories, but the compiler now filters its own prior `memory-compiler` artifacts out of the next source training set to avoid self-feedback. Remaining drawback: this is an in-process filtered compile view, not a persisted source/derived corpus boundary.
- True latent RecursiveLink-style planner/critic/retriever/solver loops are not implemented; current recursion is trace/text/tool mediated.
- Routing now returns a structured `RoutePlan` with candidate regions, scores, matched terms, and suggested next tool steps. It is still lexical over `MemoryMap`; it needs embedding-aware routing and a stronger recursive-agent policy contract.
- Vector search is in-memory and approximate, not yet a scalable ANN store.
- Source anchors target extracted text offsets, not exact rendered PDF or original-file deep links.
- Original source files are referenced by path but not managed as durable artifacts.
- There is not yet a clear source-artifact policy for indexing in place vs copying into managed storage.
- Model settings support local OpenAI-compatible chat/planner/response endpoints and local embeddings, but runtime coverage is still early and must stay recoverable.
- Import is synchronous from the app perspective and needs queueing, resumability, deletion, dedupe, update detection, and watchers.
- Web findings can be written into memory, but full browser capture, refresh policy, source trust, and freshness expiry are still early.
- Attention marks exist, but ranking does not yet fully use freshness, access history, importance marks, hot/warm/cold state, pinning, source trust, suppression, or decay.
- Brain artifacts and training exports are generated locally, but there is not yet a measured cortex adapter loop wired into model selection.
- Human library features are still early: collections, source types, saved views, durable file opening, and cloud/web workflows are not implemented.

## Recent Incremental Improvements

- 2026-05-05: Added a unit-tested MLX adapter preparation manifest in `training/train_mlx_lora.py`. Dry runs and completed training now share one manifest path, making prepared datasets inspectable before optional MLX dependencies are used for real training.
- 2026-05-06: Fixed `training/train_mlx_lora.py --dry-run` so adapter dataset preparation and manifest writing work even when `mlx_lm` is not installed. Added a regression test that patches `mlx_lm` discovery absent and verifies the prepared manifest.
- 2026-05-06: `compile_memory_brain` now reports `CortexAdapterState` with missing/fresh/stale/unknown freshness by comparing the current semantic training export hash against the newest `adapter_manifest.json` under `store/adapters/`. The MLX manifest now records `source_dataset_hash`, and dry runs remain available without `mlx_lm`.
- 2026-05-06: Cortex compilation now uses a source-only memory view that excludes previous `memory-compiler` derived memories and `brain_artifact` documents before generating artifacts and training exports. This keeps adapter freshness stable across repeated compile -> prepare adapter -> compile cycles while preserving generated artifacts for search.
- 2026-05-06: Cortex adapter freshness is now persisted in SQLite as a current `cortex_adapter_state`, exposed through a Swift/FFI snapshot call, and displayed in the Model inspector. The app also has a Compile Cortex action that writes artifacts and refreshes the visible adapter state. Remaining drawback: import/rebuild still do not automatically trigger adapter preparation/training, and chat/model selection does not yet load or gate on a fresh adapter.
- 2026-05-06: Cortex compile now has a core Rust-side adapter preparation hook. When exports are missing a fresh manifest, it creates MLX-ready train/valid/test JSONL under `adapters/prepared-<source-hash>/`, writes a fresh `adapter_manifest.json`, and persists the resulting fresh prepared state. Remaining drawback: this does not run MLX-LM training or load adapters into chat/model selection.

## Recent Improvements

- 2026-05-06: `RoutedQuery` now carries a typed route plan for LLM clients: ranked region candidates, lexical matched terms, per-candidate reasons, and next tool steps that preserve the route -> search -> open/surf -> expand/cite workflow. Swift mirror models were updated to keep the FFI/UI contract aligned. Remaining drawback: candidate scoring is still lexical and string-contained, so semantically related regions without shared words can be missed.

## Validation

Before handing off meaningful changes, run:

```bash
cargo test
```

For Swift/macOS changes, also run:

```bash
./scripts/build-macos-app.sh
```

If you change FFI signatures, run both and inspect `MemoryApp/Sources/RustBridge.swift`, `MemoryApp/Sources/Models.swift`, and `.ffi/module.modulemap`.

If you change cortex compiler/training exports, also run a small temp-store compile and validate the training harness, for example:

```bash
cargo run -- --store /tmp/imprint-check-store compile
python3 training/export_dataset.py --store /tmp/imprint-check-store
python3 training/eval_router.py --dataset /tmp/imprint-check-store/training
python3 training/train_mlx_lora.py --model <mlx-model> --dataset /tmp/imprint-check-store/training --output /tmp/imprint-check-store/adapters/check --dry-run
```

`training/train_mlx_lora.py` requires `mlx_lm` for adapter training and should prepare MLX-compatible `train.jsonl`, `valid.jsonl`, and `test.jsonl` files under the adapter output directory before invoking MLX-LM.

When a real cortex adapter training path is wired into import/rebuild, also verify stale/training/fresh/failed state transitions and confirm source-grounded search/chat remain usable while training is incomplete.

## Product Taste

The UI should feel like a professional memory instrument and personal library, not a marketing page. Prefer dense, inspectable, calm interfaces that make the memory easy to navigate.

The core human interaction is: import, browse, inspect, search, collect, open source, and understand provenance.

The core agent interaction is: consult cortex first, route, search, open, surf, expand, cite, recurse when evidence is weak, backtrack, and write back useful findings with provenance.
