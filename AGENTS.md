# Agent Guide

This repo is building **imprint**, an AI-native memory filesystem. Treat it as "Obsidian for AI": documents become a layered, navigable memory that an LLM can route through, surf across, and expand back into source context.

## North Star

The app should let an AI avoid context rot by keeping only a tiny map in context, then using tools to enter the right memory region on demand.

The canonical flow is:

1. Read the compact memory map.
2. Choose a likely region or node.
3. Search or open nearby vector chunks.
4. Follow semantic, source, citation, or explicit links.
5. Expand from a chunk to a window, page, section, or full document.
6. Jump back to vector space and continue surfing.

Do not optimize for dumping whole corpora into prompt context. Optimize for anchored retrieval, local exploration, reversible navigation, and source-grounded expansion.

## Layering Principles

**Layer 1: Map**

- `MemoryMap` must stay small enough to live in the model context.
- Map entries should be routing hints, not full summaries.
- Keep the map stable, compact, and inspectable.

**Layer 2: Vector/graph memory**

- Chunks are the primary semantic units.
- Regions are coarse entry points for routing.
- Links should make nearby memory surfable: same document, semantic neighbor, citation/reference, entity overlap, and region membership.
- The model should be able to enter at a relevant point, not traverse from a fake root every time.

**Layer 3: Source context**

- Every chunk should preserve a `SourceAnchor` whenever possible.
- Anchors must include path, content hash, offsets, parser version, and page/section when available.
- Expanding context should use anchors and original document text, not just the embedded chunk.
- Never treat an embedding as a substitute for source truth.

## Implementation Map

- `src/types.rs`: shared memory schema. Preserve compatibility carefully.
- `src/ingest.rs`: extraction, chunking, source anchors, embedding reuse, region derivation.
- `src/index.rs`: embedder abstraction and vector search.
- `src/query.rs`: map routing plus search inside selected regions.
- `src/graph.rs`: link generation.
- `src/surf.rs`: surf/open/neighbor/expand/jump/session behavior.
- `src/store.rs`: SQLite persistence.
- `src/app.rs`: app orchestration used by CLI, FFI, and Swift.
- `src/mcp.rs`: AI-facing tool server.
- `src/ffi.rs`: Swift bridge.
- `MemoryApp/Sources/`: macOS UI.

## Engineering Rules

- Keep Rust data structures and Swift mirror models in sync.
- If you add or rename serialized fields, update FFI, Swift models, tests, and persistence together.
- Preserve existing imported documents and anchors during rebuilds.
- Do not break embedding reuse without a migration reason.
- Prefer deterministic IDs, stable ordering, and repeatable tests.
- Use real parsers and structured metadata over string hacks when possible.
- Keep local-first behavior. Network/model calls should be explicit, configurable, and recoverable.
- Avoid UI copy that claims production-grade abilities before the Rust core actually supports them.

## Current Known Gaps

- Routing is still lexical over `MemoryMap`; it needs a stronger agent/tool contract.
- Vector search is in-memory and approximate, not yet a scalable ANN store.
- Source anchors target extracted text offsets, not exact rendered PDF or original-file deep links.
- Original source files are referenced by path but not managed as durable artifacts.
- API model settings exist in UI, but Rust embedding support is currently local Ollama plus hash fallback.
- Import is synchronous from the app perspective and needs queueing, resumability, deletion, and watchers.

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

## Product Taste

The UI should feel like a professional memory instrument, not a marketing page. Prefer dense, inspectable, calm interfaces that make the memory easy to navigate. The core interaction should always be: route, inspect, surf, expand, cite, backtrack.
