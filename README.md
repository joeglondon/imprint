# imprint

imprint is an AI-native memory filesystem: a local, surfable knowledge base where documents are imported once, embedded into navigable vector space, and kept anchored back to their complete source text.

The product goal is "Obsidian for AI." A human can drag in papers, notes, chats, emails, exports, and folders. An AI can keep a tiny map in context, route into the relevant region of memory, search nearby chunks, expand into the original document when it needs full context, then jump back out and keep surfing.

## Core Idea

Large context windows are useful, but they rot when everything is stuffed into them. imprint keeps memory layered:

1. **Layer 1: The map**
   A compact `MemoryMap` designed to fit permanently in an LLM context window. It names the major regions of the memory and gives enough routing hints for the model to choose where to enter.

2. **Layer 2: Vector space**
   Chunk embeddings live in a local SQLite-backed store and are grouped into regions. The AI can route into a region, search chunks by embedding similarity, inspect neighboring chunks, and follow graph links without loading the whole corpus.

3. **Layer 3: Source context**
   Every imported document keeps its full extracted text, and chunks carry `SourceAnchor` metadata with path, content hash, offsets, page, section, and parser version where available. The AI can expand a chunk into a wider window, page, section, or whole document.

The intended interaction is not "ask a database once." It is surfing: enter at the best memory point, inspect local context, follow semantic or document links, expand to source, backtrack, and re-enter vector space somewhere adjacent.

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
- Vector-derived regions, compact memory map generation, graph links, and visualization snapshots.
- Surf APIs for opening nodes, neighbors, chunk expansion, anchor jumps, session stepping, and backtracking.
- MCP stdio server exposing `memory_search`, `memory_open`, `memory_neighbors`, `memory_expand`, and `memory_jump_to_anchor`.
- FFI bridge from Swift to Rust.
- Unit coverage for import, SQLite round trips, PDF skips, embedding reuse, anchors, surf behavior, query dedupe, navigation, and visualization.

**Important limitations**

- Region routing is still mostly lexical over the compact map. It is useful, but not yet an LLM-planned router.
- The vector index is in-memory and approximate by local region hill climbing, not a production ANN engine like HNSW.
- Source anchors point to extracted text offsets and stored paths. They do not yet open a rendered PDF page or exact original-file viewport.
- Full documents are stored as extracted text inside SQLite. The app should eventually treat originals as first-class source artifacts with stable open/deep-link behavior.
- The graph builder creates semantic, same-document, citation-style, entity-overlap, and region-membership links, but link quality is heuristic.
- The model settings UI stores API config, but OpenAI-compatible API embeddings are not fully implemented in the Rust embedder yet.
- There is no background file watcher, inbox folder, deletion/update UI, import queue recovery, or multi-user sync.

## Architecture

Key Rust modules:

- `src/types.rs`: core data model: documents, chunks, regions, links, source anchors, map entries, sessions, query results.
- `src/ingest.rs`: file collection, text/PDF extraction, chunking, source anchor creation, embedding reuse, region derivation.
- `src/index.rs`: embedder trait, hash embedder, Ollama embedder, region-level ANN index.
- `src/query.rs`: map routing and chunk search inside chosen regions.
- `src/graph.rs`: memory graph construction and semantic/document/citation/entity links.
- `src/surf.rs`: surfable navigation primitives: open, neighbors, expand, jump to anchor, session step.
- `src/store.rs`: SQLite persistence and legacy JSON migration.
- `src/app.rs`: app-facing orchestration used by FFI and Swift.
- `src/mcp.rs`: MCP server for AI clients.
- `src/ffi.rs`: C ABI exported to Swift.

Key Swift files:

- `MemoryApp/Sources/AppState.swift`: app state, import/rebuild/search/surf actions, progress polling.
- `MemoryApp/Sources/ContentView.swift`: main macOS interface.
- `MemoryApp/Sources/SemanticCloudView.swift`: visual memory map/cloud.
- `MemoryApp/Sources/RustBridge.swift`: FFI calls and JSON decoding.
- `MemoryApp/Sources/Models.swift`: Swift mirrors of Rust payload types.

## Build and Run

Requirements:

- Rust toolchain.
- Swift 6 / macOS 14 or newer.
- Optional: Ollama running locally for real embeddings.

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
cargo run -- --store .memory ingest ~/Documents/Papers
cargo run -- --store .memory map
cargo run -- --store .memory query "parts of the foot" --max-regions 3 --max-chunks 8
```

Surf a node:

```bash
cargo run -- --store .memory surf-open chunk:some-document:chunk:0
cargo run -- --store .memory surf-neighbors chunk:some-document:chunk:0
cargo run -- --store .memory surf-expand some-document:chunk:0 --mode section
```

Run as an MCP server:

```bash
cargo run -- --store .memory mcp
```

## Memory Contract

The central invariant is that embeddings are not enough. A chunk must remain connected to:

- Its document.
- Its region.
- Its source anchor.
- Its neighboring chunks and links.
- The full extracted source context.

When a model cites, reasons from, or expands a memory hit, it should be able to explain where the hit came from and request more context without re-ingesting the corpus or dumping unrelated files into context.

## Near-Term Roadmap

- Replace heuristic map routing with an agent-facing routing contract that can use both map entries and embedding probes.
- Add a real vector index suitable for large corpora.
- Add stable original-file opening: PDF page, text offset, Markdown heading, email/message thread, and browser source URL where applicable.
- Make the MCP tools the first-class AI interface and document their schemas for client setup.
- Add an import queue with resumability, deletion, and file-watcher updates.
- Add richer document parsers for chat exports, email, HTML/web archives, and structured scientific PDFs.
- Distinguish private user memory, shared project memory, and global/reference memory.
