use crate::extract::{Extractor, MemoryExtractor};
use crate::index::{HashEmbedder, Indexer, RegionIndexer};
use crate::navigation::{MemoryNavigator, Navigator};
use crate::query::MemoryQueryEngine;
use crate::store::{FileMemoryStore, MemoryStore};
use crate::surf::{ExpandMode, SurfAction};
use crate::types::*;
use anyhow::{anyhow, Context};
use clap::{Parser, Subcommand};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

const MEMORY_ACCESS_HORIZON_MILLIS: u64 = 30 * 24 * 60 * 60 * 1000;

#[derive(Debug, Parser)]
#[command(name = "ai-memory")]
#[command(about = "Layered AI memory with surfable, greppable navigation")]
pub struct Cli {
    #[arg(long, default_value = ".memory")]
    store: PathBuf,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Ingest {
        input: PathBuf,
    },
    Rebuild,
    Cortex {
        #[command(subcommand)]
        command: CortexCommand,
    },
    Compile,
    Mcp,
    Map,
    Trace {
        query: String,
        #[arg(long, default_value_t = 3)]
        max_regions: usize,
    },
    Query {
        query: String,
        #[arg(long, default_value_t = 3)]
        max_regions: usize,
        #[arg(long, default_value_t = 5)]
        max_chunks: usize,
    },
    GrepRegion {
        region_id: String,
        needle: String,
        #[arg(long, default_value_t = 80)]
        window: usize,
    },
    OpenDocument {
        chunk_id: String,
    },
    GrepDocument {
        document_id: String,
        needle: String,
        #[arg(long, default_value_t = 80)]
        window: usize,
    },
    SemanticDocument {
        document_id: String,
        query: String,
        #[arg(long, default_value_t = 80)]
        window: usize,
    },
    Excerpt {
        hit_id: String,
        #[arg(long, default_value_t = 80)]
        window: usize,
    },
    SurfOpen {
        node: String,
    },
    SurfNeighbors {
        node: String,
        #[arg(long, default_value_t = 8)]
        max_results: usize,
    },
    SurfExpand {
        chunk_id: String,
        #[arg(long, default_value = "window")]
        mode: String,
        #[arg(long, default_value_t = 360)]
        window: usize,
    },
    SurfJump {
        anchor_id: String,
        #[arg(long, default_value_t = 360)]
        window: usize,
    },
    SurfStep {
        session: String,
        action: String,
    },
    Links {
        node: String,
    },
    Step {
        node: String,
        link_id: String,
    },
    Backtrack {
        node: String,
    },
}

#[derive(Debug, Subcommand)]
enum CortexCommand {
    Status {
        #[arg(long, default_value_t = 10)]
        max_jobs: usize,
    },
    Compile,
    Train {
        #[arg(long)]
        dry_run: bool,
        #[arg(long, default_value_t = 100)]
        iters: usize,
        #[arg(long, default_value_t = 30 * 60 * 1000)]
        timeout_millis: u64,
        #[arg(long)]
        python: Option<String>,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        log: Option<PathBuf>,
    },
}

pub fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let store = FileMemoryStore::new(cli.store);
    let embedder = HashEmbedder::default();
    let indexer = RegionIndexer;
    let engine = MemoryQueryEngine;
    let extractor = MemoryExtractor;
    let navigator = MemoryNavigator;

    match cli.command {
        Command::Ingest { input } => {
            let result = crate::app::ingest_paths(store.root(), &[input])?;
            println!("{}", serde_json::to_string_pretty(&result.summary)?);
        }
        Command::Rebuild => {
            let result = crate::app::rebuild_memory(store.root())?;
            println!("{}", serde_json::to_string_pretty(&result.summary)?);
        }
        Command::Cortex { command } => match command {
            CortexCommand::Status { max_jobs } => {
                let snapshot = crate::app::load_cortex_adapter_snapshot(store.root())?;
                let cortex_index = store.load_current_cortex_index()?;
                let jobs = store
                    .list_cortex_adapter_jobs(None)?
                    .into_iter()
                    .take(max_jobs)
                    .collect::<Vec<_>>();
                let payload = serde_json::json!({
                    "adapter_state": snapshot.adapter_state,
                    "cortex_index": cortex_index.as_ref().map(|index| serde_json::json!({
                        "id": index.id,
                        "schema_version": index.schema_version,
                        "created_at": index.created_at,
                        "corpus_hash": index.corpus_hash,
                        "source_refs": index.source_refs.len(),
                        "artifact_ids": index.artifact_ids.len(),
                        "regions": index.regions.len(),
                        "route_examples": index
                            .regions
                            .iter()
                            .map(|region| region.route_examples.len())
                            .sum::<usize>(),
                    })),
                    "recent_jobs": jobs,
                });
                println!("{}", serde_json::to_string_pretty(&payload)?);
            }
            CortexCommand::Compile => {
                let result = crate::app::compile_memory_brain(store.root())?;
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            CortexCommand::Train {
                dry_run,
                iters,
                timeout_millis,
                python,
                output,
                log,
            } => {
                let compile = crate::app::compile_memory_brain(store.root())?;
                let adapter_state = compile
                    .adapter_state
                    .as_ref()
                    .context("compile did not return adapter state")?;
                let config = crate::app::load_model_config(store.root())?;
                let base_model = adapter_state
                    .base_model
                    .clone()
                    .or(config.compiler_model)
                    .or(config.response_model)
                    .or(config.chat_model)
                    .context(
                        "no compiler, response, or chat model configured for cortex training",
                    )?;
                let job = crate::training::run_cortex_adapter_training_job(
                    store.root(),
                    &base_model,
                    &adapter_state.current_source_dataset_hash,
                    crate::training::CortexAdapterTrainingOptions {
                        dry_run,
                        timeout_millis,
                        iters,
                        python,
                        output_dir: output,
                        log_path: log,
                        ..Default::default()
                    },
                )?;
                println!("{}", serde_json::to_string_pretty(&job)?);
            }
        },
        Command::Compile => {
            let result = crate::app::compile_memory_brain(store.root())?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        Command::Mcp => {
            crate::mcp::run_stdio(store.root())?;
        }
        Command::Map => {
            let memory = load_ready_memory(&store)?;
            println!(
                "{}",
                memory.memory_map.context("memory map missing")?.serialized
            );
        }
        Command::Trace { query, max_regions } => {
            let memory = load_ready_memory(&store)?;
            let now = now_millis();
            let attention_marks = store.list_attention_marks(None).unwrap_or_default();
            let memory_accesses = store
                .list_memory_accesses(None, Some(now.saturating_sub(MEMORY_ACCESS_HORIZON_MILLIS)))
                .unwrap_or_default();
            let cortex_index = store.load_current_cortex_index().unwrap_or_default();
            let trace = engine.trace_with_signals(
                &embedder,
                &memory,
                &query,
                max_regions,
                &attention_marks,
                &memory_accesses,
                now,
                cortex_index.as_ref(),
            )?;
            println!("{}", serde_json::to_string_pretty(&trace)?);
        }
        Command::Query {
            query,
            max_regions,
            max_chunks,
        } => {
            let memory = load_ready_memory(&store)?;
            let ann = indexer.rebuild(&memory.chunks, &memory.regions);
            let result = engine.execute(
                &embedder,
                &memory,
                &ann,
                QueryRequest {
                    text: query,
                    filters: BTreeMap::new(),
                    max_regions,
                    max_chunks,
                },
            )?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        Command::GrepRegion {
            region_id,
            needle,
            window,
        } => {
            let memory = load_ready_memory(&store)?;
            let hits = extractor.grep_region(&memory, &region_id, &needle, window);
            println!("{}", serde_json::to_string_pretty(&hits)?);
        }
        Command::OpenDocument { chunk_id } => {
            let memory = load_ready_memory(&store)?;
            let chunk = memory
                .chunks
                .iter()
                .find(|chunk| chunk.id == chunk_id)
                .context("chunk not found")?;
            let document = memory
                .documents
                .iter()
                .find(|doc| doc.id == chunk.document_id)
                .context("document not found")?;
            let payload = serde_json::json!({
                "document_id": document.id,
                "title": document.title,
                "chunk_id": chunk.id,
                "start": chunk.start,
                "end": chunk.end,
                "source_anchor": chunk.source_anchor,
                "excerpt": excerpt_from_doc(&document.text, chunk.start, chunk.end),
            });
            println!("{}", serde_json::to_string_pretty(&payload)?);
        }
        Command::GrepDocument {
            document_id,
            needle,
            window,
        } => {
            let memory = load_ready_memory(&store)?;
            let hits = extractor.grep_document(&memory, &document_id, &needle, window);
            println!("{}", serde_json::to_string_pretty(&hits)?);
        }
        Command::SemanticDocument {
            document_id,
            query,
            window,
        } => {
            let memory = load_ready_memory(&store)?;
            let hits =
                extractor.semantic_document(&embedder, &memory, &document_id, &query, window)?;
            println!("{}", serde_json::to_string_pretty(&hits)?);
        }
        Command::Excerpt { hit_id, window } => {
            let memory = load_ready_memory(&store)?;
            let excerpt = extractor
                .excerpt_for_hit(&memory, &hit_id, window)
                .context("hit not found")?;
            println!("{excerpt}");
        }
        Command::SurfOpen { node } => {
            let opened = crate::app::surf_open(store.root(), &parse_node(&node)?)?;
            println!("{}", serde_json::to_string_pretty(&opened)?);
        }
        Command::SurfNeighbors { node, max_results } => {
            let neighbors =
                crate::app::surf_neighbors(store.root(), &parse_node(&node)?, max_results)?;
            println!("{}", serde_json::to_string_pretty(&neighbors)?);
        }
        Command::SurfExpand {
            chunk_id,
            mode,
            window,
        } => {
            let expansion = crate::app::surf_expand(
                store.root(),
                &chunk_id,
                parse_expand_mode(&mode)?,
                window,
            )?;
            println!("{}", serde_json::to_string_pretty(&expansion)?);
        }
        Command::SurfJump { anchor_id, window } => {
            let expansion = crate::app::surf_jump_to_anchor(store.root(), &anchor_id, window)?;
            println!("{}", serde_json::to_string_pretty(&expansion)?);
        }
        Command::SurfStep { session, action } => {
            let session: SessionState = serde_json::from_str(&session)?;
            let action: SurfAction = serde_json::from_str(&action)?;
            let result = crate::app::surf_session_step(store.root(), session, action)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        Command::Links { node } => {
            let memory = load_ready_memory(&store)?;
            let node = parse_node(&node)?;
            let links = navigator.list_links(&memory, &node);
            println!("{}", serde_json::to_string_pretty(&links)?);
        }
        Command::Step { node, link_id } => {
            let memory = load_ready_memory(&store)?;
            let node = parse_node(&node)?;
            let mut session = navigator.start_session(Some(node));
            let next = navigator
                .step(&memory, &mut session, &link_id)
                .context("link not found")?;
            let payload = serde_json::json!({
                "current": next,
                "history": session.history,
                "visited": session.visited,
            });
            println!("{}", serde_json::to_string_pretty(&payload)?);
        }
        Command::Backtrack { node } => {
            let parsed = parse_node(&node)?;
            let mut session = navigator.start_session(None);
            navigator.open(&mut session, NodeRef::Region("root".into()));
            navigator.open(&mut session, parsed);
            let previous = navigator.backtrack(&mut session).context("no history")?;
            println!("{}", serde_json::to_string_pretty(&previous)?);
        }
    }
    Ok(())
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or_default()
}

fn parse_expand_mode(raw: &str) -> anyhow::Result<ExpandMode> {
    match raw.to_ascii_lowercase().as_str() {
        "window" => Ok(ExpandMode::Window),
        "page" => Ok(ExpandMode::Page),
        "section" => Ok(ExpandMode::Section),
        "document" => Ok(ExpandMode::Document),
        _ => Err(anyhow!(
            "unknown expand mode {raw}; expected window, page, section, or document"
        )),
    }
}

fn load_ready_memory(store: &FileMemoryStore) -> anyhow::Result<PersistedMemory> {
    let memory = store.load()?;
    if memory.documents.is_empty() {
        return Err(anyhow!("store {} is empty", store.root().display()));
    }
    Ok(memory)
}

fn parse_node(raw: &str) -> anyhow::Result<NodeRef> {
    let (kind, value) = raw
        .split_once(':')
        .ok_or_else(|| anyhow!("expected node in form kind:id"))?;
    match kind {
        "document" => Ok(NodeRef::Document(value.into())),
        "chunk" => Ok(NodeRef::Chunk(value.into())),
        "region" => Ok(NodeRef::Region(value.into())),
        _ => Err(anyhow!("unknown node kind {kind}")),
    }
}

fn excerpt_from_doc(text: &str, start: usize, end: usize) -> String {
    let chars = text.chars().collect::<Vec<_>>();
    let left = start.saturating_sub(120);
    let right = (end + 120).min(chars.len());
    chars[left..right].iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;
    use crate::index::HashEmbedder;
    use crate::ingest::Ingester;
    use crate::map::MapBuilder;

    fn sample_memory() -> PersistedMemory {
        let embedder = HashEmbedder::default();
        let ingester = Ingester::new(embedder);
        let docs = vec![
            Document {
                id: "foot-anatomy".into(),
                title: "Foot Anatomy".into(),
                text: "The foot contains the heel, arch, toes, talus, calcaneus, and plantar fascia. cite:gait-study".into(),
                metadata: BTreeMap::new(),
                source_anchor: None,
                content_hash: None,
                parser_version: None,
            },
            Document {
                id: "gait-study".into(),
                title: "Gait Study".into(),
                text: "Detailed gait work explains how the arch and plantar fascia distribute force during walking.".into(),
                metadata: BTreeMap::new(),
                source_anchor: None,
                content_hash: None,
                parser_version: None,
            },
        ];
        let mut memory = ingester.ingest_documents(docs).expect("ingest documents");
        GraphBuilder.build(&mut memory);
        memory.memory_map = Some(MapBuilder::default().build(&memory.regions));
        memory
    }

    #[test]
    fn map_stays_under_budget() {
        let memory = sample_memory();
        let map = memory.memory_map.expect("map");
        assert!(map.serialized.len() <= map.budget_bytes);
        assert!(!map.entries.is_empty());
    }

    #[test]
    fn document_grep_returns_excerpt() {
        let memory = sample_memory();
        let hits = MemoryExtractor.grep_document(&memory, "foot-anatomy", "plantar fascia", 30);
        assert!(!hits.is_empty());
        assert!(hits[0].excerpt.contains("plantar fascia"));
    }

    #[test]
    fn region_query_routes_and_finds_hits() {
        let memory = sample_memory();
        let ann = RegionIndexer.rebuild(&memory.chunks, &memory.regions);
        let result = MemoryQueryEngine
            .execute(
                &HashEmbedder::default(),
                &memory,
                &ann,
                QueryRequest {
                    text: "parts of the foot".into(),
                    filters: BTreeMap::new(),
                    max_regions: 2,
                    max_chunks: 3,
                },
            )
            .expect("query");
        assert!(!result.routed.region_ids.is_empty());
        assert!(!result.hits.is_empty());
    }

    #[test]
    fn navigation_can_follow_and_backtrack() {
        let memory = sample_memory();
        let navigator = MemoryNavigator;
        let mut session =
            navigator.start_session(Some(NodeRef::Chunk("foot-anatomy:chunk:0".into())));
        let links = navigator.list_links(&memory, session.current.as_ref().expect("current"));
        let target = links
            .iter()
            .find(|link| matches!(link.target, NodeRef::Document(_)))
            .expect("doc link")
            .id
            .clone();
        let next = navigator
            .step(&memory, &mut session, &target)
            .expect("step");
        assert!(matches!(next, NodeRef::Document(_)));
        let previous = navigator.backtrack(&mut session).expect("backtrack");
        assert!(matches!(previous, NodeRef::Chunk(_)));
    }
}
