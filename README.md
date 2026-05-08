# imprint

imprint is a shared memory filesystem and personal cortex for humans and AI agents: a place to keep documents, notes, recipes, books, chat histories, web findings, project material, and source files in one navigable repository.

The goal is larger than "Obsidian for an LLM." imprint should be useful as a personal knowledge and file library for a human, while also making an LLM smarter by putting a semantic sketch of that library inside the model and backing it with source-grounded recall. You can toss files into it like a local-first Drive or Obsidian replacement; the system extracts structure, keeps provenance, builds semantic links automatically, trains addressability into the local model, and lets agents retrieve exact context without requiring you to hand-create every connection.

Long term, imprint should be able to grow from a local macOS app into a web/cloud-backed memory library and, eventually, a candidate substrate for a more agent-native operating system: a filesystem where files, extracted text, semantic neighborhoods, citations, activity history, and AI-written findings are all part of the same navigable memory layer.

## Core Idea

The central product idea is that humans and AI agents should share the same durable memory substrate, while the AI also gains a small amount of personal memory inside its own weights.

For a human, imprint should feel like a calm, powerful repository: import a folder, drop in a PDF, save a recipe, preserve a chat export, capture web-search findings, browse what is there, and recover the original source when needed.

For an AI agent, imprint should feel less like a chatbot hunting through an external database and more like a mind with a library attached. The model should have latent familiarity with the user's world: projects, people, source families, recurring concepts, and likely retrieval addresses. When it needs precision, it should search nearby chunks, follow semantic/source/citation links, expand into source context, cite the origin, and write useful discoveries back into memory.

This creates two memory modes:

1. **Latent familiarity**
   A personal cortex adapter teaches LFM2.5 the shape of the user's imprint library. It should know what exists and where to look, like remembering the story of a book without knowing the exact words on a particular page.

2. **Source recall**
   imprint retrieves exact chunks, anchors, source paths, provenance, and citations when the model needs more than fuzzy memory. Latent memory alone is not citation-safe.

The first implementation target is semantic addressability, not full photographic memory. The model should know where inside imprint to go; imprint remains the source of truth for exact claims, quotes, dates, page-level recall, mutable facts, and anything the user later deletes or corrects.

Large context windows are useful, but they rot when everything is stuffed into them. imprint keeps memory layered:

1. **Layer 1: Library**
   The durable repository: documents, notes, chats, web findings, project files, source identity, provenance, metadata, and human-facing organization.

2. **Layer 2: Cortex**
   `CortexIndex` is the compact semantic-address layer generated from the library. It contains routing sketches, region/source-family hints, artifact IDs, source refs, corpus hashes, and examples that teach the model where to retrieve exact context. The legacy map payload remains a compatibility projection under this layer.

3. **Layer 3: Vector/graph recall**
   Chunk embeddings and graph links make the library surfable. Documents are grouped into regions, connected by semantic similarity, source order, citations/references, entities, and explicit links.

4. **Layer 4: Source context**
   Every imported item should preserve source truth: original path or managed artifact identity, content hash, extracted text, offsets, page/section metadata, parser version, and provenance. A hit should always be expandable back toward the source.

5. **Layer 5: Human library**
   The same memory must be browsable by a person: files, collections, regions, documents, passages, trails, saved findings, and source previews should be inspectable without needing to think like a retrieval engine.

6. **Layer 6: Recursive latent reasoning**
   Inspired by [Recursive Multi-Agent Systems](https://arxiv.org/abs/2604.25917) and the [RecursiveMAS project](https://recursivemas.github.io/), imprint should move toward true latent planner/critic/retriever/solver loops. Small trainable RecursiveLink-style modules should pass hidden states across roles and recursion rounds, decoding text only for final tool decisions or answers.

The intended interaction is not "ask a database once." It is cognition plus stewardship: toss material in, let imprint organize and connect it, train the cortex to know where things live, open sources, follow meaning, expand context, cite origins, backtrack, and save new findings.

## Current App Audit

This repo already has the bones of that system.

**Implemented**

- Rust core library and CLI in `src/`.
- SwiftUI macOS shell in `MemoryApp/Sources/`.
- Drag-and-drop or picker-based file import.
- Local SQLite persistence at the app support store path.
- Plain text, Markdown, and PDF text extraction.
- Chunking with overlap, content hashes, parser versions, and source anchors.
- Embedding through local Ollama by default, with a hash embedder fallback for tests/offline use.
- Embedding reuse on rebuild or unchanged imports.
- Vector-derived regions, legacy map generation, graph links, and visualization snapshots.
- Surf APIs for opening nodes, neighbors, chunk expansion, anchor jumps, session stepping, and backtracking.
- Chat memory with persisted turns, hot context traces, derived memories, web findings, attention marks, and recursive cortex trace metadata.
- Brain artifact compilation for library sketches, region cards, and routing rules, plus local train/eval JSONL exports for semantic addressing, tool choice, critique, and collaboration tasks.
- Python training harness under `training/` for export validation, baseline eval, and MLX LoRA experiments.
- MCP stdio server exposing read tools, writeback tools, attention marks, chat traces, and `memory_compile`.
- FFI bridge from Swift to Rust.
- Unit coverage for import, SQLite round trips, PDF skips, embedding reuse, anchors, surf behavior, query dedupe, navigation, visualization, chat/web writebacks, cortex traces, and brain compilation.

**Important limitations**

- `CortexIndex` is not yet a first-class persisted schema; current compiler artifacts are scaffolding.
- Adapter freshness state is not yet wired into import/rebuild, Swift UI, CLI, or MCP surfaces.
- MLX cortex adapter training is not yet a core import/rebuild step.
- True latent RecursiveLink-style planner/critic/retriever/solver loops are not implemented; current recursion is trace/text/tool mediated.
- The vector index is in-memory and approximate by local region hill climbing, not a production ANN engine like HNSW.
- Source anchors point to extracted text offsets and stored paths. They do not yet open a rendered PDF page or exact original-file viewport.
- Full documents are stored as extracted text inside SQLite. Original source files are not yet managed as durable library artifacts.
- The graph builder creates semantic, same-document, citation-style, entity-overlap, and region-membership links, but link quality is heuristic.
- Brain artifacts can be generated and exported locally, but there is not yet a measured cortex adapter loop wired into model selection.
- There is no background file watcher, managed inbox, deletion/update UI, import queue recovery, full browser capture, cloud sync, or multi-device library yet.

## Product Direction

imprint should evolve as a memory filesystem, not just a search layer.

**Human-first repository**

- A place to put personal and work material: notes, PDFs, Markdown, recipes, books, chats, emails, web pages, code docs, and exports.
- Local-first by default, with a path toward managed storage, cloud backup/sync, and a web app.
- Lightweight organization that does not require manual linking: folders, collections, source types, tags, saved views, and semantic regions can coexist.
- Strong source handling: preserve originals or durable references, detect duplicates, track updates, and make every derived memory reversible back to source.

**Agent-first memory**

- MCP tools and future agent APIs should expose the same repository the human sees.
- Agents should use cortex, regions, chunks, links, anchors, and expansion instead of dumping whole libraries into context.
- Agent findings, web-search results, citations, summaries, and trails should be saveable back into the library with provenance.
- Agents should be able to update attention metadata: mark memory as important or unimportant, promote useful items to hot/warm, cool down noisy items, and pin durable facts with an auditable reason.
- AI-created links and summaries should remain inspectable, editable, and replaceable rather than becoming hidden magic.
- Compiled cortex artifacts should teach local models how to form semantic addresses, retrieve, critique, and collaborate. They should not become the only place a fact lives.
- Personal cortex adapter refresh should be part of import/rebuild, while search and chat remain usable when adapter state is stale, training, or failed.

**Memory-before-web**

- imprint should be the agent's first memory. The web should be an expansion tool, not the default brain.
- When a user asks a question, an agent should first consult the cortex and library overview, then search imprint for prior knowledge before reaching for live web search.
- The cortex should hold semantic addresses, not full answers. It should help the agent know that topics like "bookshelf building" or "woodworking plans" already exist and where exact recall should begin.
- Web search is appropriate when memory is missing, weak, stale, explicitly asks for current information, or needs external verification.
- Web findings should be saved back into imprint with query, source URLs, retrieval date, summary, citations, confidence, and freshness/expiration metadata.
- Recently captured, accessed, edited, cited, or agent-promoted items can be "hot" so agents can grep/search them immediately without carrying bulky material in context.
- Agents should be able to mark results as important or unimportant for their own future routing. Those marks should affect ranking and attention, not rewrite source truth.
- Over time, hot items can cool down unless reused, cited, pinned, edited, or promoted again.
- Decay should change ranking and default attention, not destroy recall. Old memory should become less likely by default, but still reachable when the user asks for it directly.

**Long-horizon system**

- Start as a local macOS app with explicit model/network configuration.
- Grow into a cross-device memory library with optional cloud storage and a web interface.
- Keep the architecture compatible with a future where the memory layer acts less like an app database and more like a semantic filesystem for an agent-native OS.
- Let local models improve through private cortex adapters trained on source-linked cognitive maps, semantic-address examples, and recursive traces, while keeping source truth recoverable outside the model weights.
- Explore a research path toward more photographic latent memory, but keep semantic addressability as the product contract until measured otherwise.

## Architecture

Key Rust modules:

- `src/types.rs`: core data model: documents, chunks, regions, links, source anchors, cortex state, legacy map payloads, sessions, query results.
- `src/ingest.rs`: file collection, text/PDF extraction, chunking, source anchor creation, embedding reuse, region derivation.
- `src/index.rs`: embedder trait, hash embedder, Ollama embedder, region-level ANN index.
- `src/query.rs`: cortex-guided routing and chunk search inside chosen regions.
- `src/graph.rs`: memory graph construction and semantic/document/citation/entity links.
- `src/surf.rs`: surfable navigation primitives: open, neighbors, expand, jump to anchor, session step.
- `src/compiler.rs`: `CortexIndex`, compiled cortex artifacts, semantic-address examples, and compatibility map projection.
- `src/training.rs`: versioned train/eval JSONL export generation for semantic addressing, source expansion, tool choice, critique, and collaboration.
- `src/store.rs`: SQLite persistence and legacy JSON migration.
- `src/app.rs`: app-facing orchestration used by FFI and Swift.
- `src/mcp.rs`: MCP server for AI clients.
- `src/ffi.rs`: C ABI exported to Swift.
- `training/`: Python scripts for dataset validation, baseline eval, MLX cortex adapter training, and future RecursiveLink experiments.

Key Swift files:

- `MemoryApp/Sources/AppState.swift`: app state, import/rebuild/search/surf actions, progress polling.
- `MemoryApp/Sources/ContentView.swift`: main macOS interface.
- `MemoryApp/Sources/SemanticCloudView.swift`: visual cortex/source-recall cloud.
- `MemoryApp/Sources/RustBridge.swift`: FFI calls and JSON decoding.
- `MemoryApp/Sources/Models.swift`: Swift mirrors of Rust payload types.

## Build and Run

Requirements:

- Rust toolchain.
- Swift 6 / macOS 14 or newer.
- Optional: Ollama running locally for real embeddings.
- Python 3.9+ for training export validation and local cortex adapter experiments.
- `mlx-lm` for MLX LoRA training when adapter refresh is enabled.

Default local embedding model:

```bash
ollama pull embeddinggemma:300m
ollama serve
```

Build the Rust core:

```bash
cargo build
```

Run tests:

```bash
cargo test
```

Build the Swift app and Rust FFI library:

```bash
./scripts/build-macos-app.sh
```

Package a local app bundle:

```bash
./scripts/package-macos-app.sh
```

## CLI Examples

Use a local memory store:

```bash
cargo run -- --store .imprint ingest ~/Documents/Papers
cargo run -- --store .imprint compile
cargo run -- --store .imprint query "parts of the foot" --max-regions 3 --max-chunks 8
cargo run -- --store .imprint map # legacy alias/projection until cortex CLI lands
```

Surf a node:

```bash
cargo run -- --store .imprint surf-open chunk:some-document:chunk:0
cargo run -- --store .imprint surf-neighbors chunk:some-document:chunk:0
cargo run -- --store .imprint surf-expand some-document:chunk:0 --mode section
```

Run as an MCP server:

```bash
cargo run -- --store .imprint mcp
```

Validate cortex training exports:

```bash
python3 training/export_dataset.py --store .imprint
python3 training/eval_router.py --dataset .imprint/training
cargo run -- --store .imprint cortex eval-harness
python3 training/train_mlx_lora.py --model mlx-community/LFM2.5-1.2B-Instruct-8bit --dataset .imprint/training --output .imprint/adapters/lfm-cortex
```

Use `--dry-run` with `training/train_mlx_lora.py` to prepare the MLX-compatible dataset and print the exact training command without running fine-tuning.

## Memory Contract

The central invariant is that embeddings are not enough. Every memory item must remain connected to:

- Its original source or managed artifact.
- Its extracted text and structured metadata.
- Its source anchors and provenance.
- Its semantic regions, neighboring chunks, and graph links.
- Its attention state: freshness, access history, importance marks, hot/warm/cold state, pinning, decay, and whether it is durable or ephemeral.
- Its cortex artifacts when present: artifact ID, kind, source refs, confidence, schema version, corpus hash, adapter freshness, and train/eval export provenance.
- Its human-facing library identity: title, type, collection/folder, and browse location when available.

When a human or model cites, reasons from, or expands a memory hit, it should be able to explain where the hit came from and request more context without re-ingesting the corpus or dumping unrelated files into context. Latent familiarity can guide retrieval, but exact claims must remain anchored to source recall.

## Near-Term Roadmap

- Treat source artifacts as first-class library objects: decide when to reference files in place, when to copy into managed storage, and how to preserve stable IDs across moves.
- Add an import inbox/queue with resumability, deletion, update detection, duplicate handling, and file-watcher support.
- Add a first-class persisted `CortexIndex` and make the legacy map a compatibility projection.
- Add cortex adapter freshness state across import/rebuild, Swift UI, CLI, and MCP.
- Make MLX cortex adapter training part of import/rebuild without blocking source-grounded search or chat.
- Implement true latent RecursiveLink-style planner/critic/retriever/solver loops, with text/tool recursion as fallback.
- Measure cortex adapters against exported eval sets before surfacing them as recommended model configuration.
- Add an attention/ranking layer for freshness, recency, frequency, importance marks, hot/warm/cold state, explicit pinning, source trust, and decay.
- Add a real vector index suitable for large corpora.
- Add stable original-file opening: PDF page, text offset, Markdown heading, email/message thread, and browser source URL where applicable.
- Make MCP tools a first-class AI interface for both reading and writing memory, including captured web findings.
- Add richer document parsers for chat exports, email, HTML/web archives, and structured scientific PDFs.
- Build human library surfaces for browsing documents, collections, source types, semantic regions, saved trails, and provenance.
- Distinguish private user memory, shared project memory, and global/reference memory.
