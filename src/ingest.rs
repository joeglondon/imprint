use crate::index::{cosine_similarity, tokenize, Embedder, EmbedderIdentity};
use crate::types::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const CHUNKING_VERSION: u32 = 2;
pub const PARSER_VERSION: u32 = 1;
const CHUNK_SIZE: usize = 600;
const CHUNK_OVERLAP: usize = 120;
const MAX_FILE_BYTES: u64 = 2_000_000;
const MAX_PDF_BYTES: u64 = 25_000_000;
const MAX_DOCUMENT_CHARS: usize = 250_000;
const MAX_CHUNKS_PER_DOCUMENT: usize = 600;
const EMBEDDING_BATCH_SIZE: usize = 32;
const STOPWORDS: &[&str] = &[
    "about", "after", "again", "being", "could", "every", "from", "have", "into", "just", "more",
    "most", "only", "other", "should", "some", "than", "that", "their", "there", "these", "they",
    "this", "what", "when", "where", "which", "while", "with", "would", "also", "because",
    "between", "those", "through", "under", "using", "within", "without", "and", "the", "for",
    "our", "com", "http", "https", "www",
];
const PAGE_SPANS_KEY: &str = "_page_spans";
const SECTION_SPANS_KEY: &str = "_section_spans";
const SECTION_HIERARCHY_SPANS_KEY: &str = "_section_hierarchy_spans";
const PARAGRAPH_SPANS_KEY: &str = "_paragraph_spans";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImportSkip {
    pub path: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DocumentImportBatch {
    pub documents: Vec<Document>,
    pub imported_paths: Vec<String>,
    pub skipped_paths: Vec<ImportSkip>,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct EmbeddingReuseStats {
    pub embedded_count: usize,
    pub reused_embedding_count: usize,
}

#[derive(Debug, Clone)]
pub struct Ingester<E> {
    embedder: E,
}

impl<E: Embedder> Ingester<E> {
    pub fn new(embedder: E) -> Self {
        Self { embedder }
    }

    pub fn ingest_path(&self, input: &Path) -> anyhow::Result<PersistedMemory> {
        self.ingest_paths(&[input.to_path_buf()])
    }

    pub fn ingest_paths(&self, inputs: &[PathBuf]) -> anyhow::Result<PersistedMemory> {
        self.ingest_documents(extract_documents(inputs)?.documents)
    }

    pub fn extract_documents(&self, inputs: &[PathBuf]) -> anyhow::Result<DocumentImportBatch> {
        extract_documents(inputs)
    }

    pub fn ingest_documents(&self, documents: Vec<Document>) -> anyhow::Result<PersistedMemory> {
        self.ingest_documents_with_progress(documents, |_, _| {})
    }

    pub fn ingest_documents_with_progress(
        &self,
        documents: Vec<Document>,
        progress: impl FnMut(usize, usize),
    ) -> anyhow::Result<PersistedMemory> {
        let (memory, _) = self.ingest_documents_reusing_embeddings(documents, &[], progress)?;
        Ok(memory)
    }

    pub fn ingest_documents_reusing_embeddings(
        &self,
        documents: Vec<Document>,
        reusable_chunks: &[Chunk],
        mut progress: impl FnMut(usize, usize),
    ) -> anyhow::Result<(PersistedMemory, EmbeddingReuseStats)> {
        let region_specs = derive_region_specs(&documents);
        let mut chunk_inputs = Vec::new();
        for document in &documents {
            for (ordinal, (start, end, text)) in chunk_text(&document.text).into_iter().enumerate()
            {
                let region_id = assign_region(&text, &region_specs);
                let chunk_id = format!("{}:chunk:{ordinal}", document.id);
                let mut metadata = document.metadata.clone();
                metadata.insert("document_title".into(), document.title.clone());
                let source_anchor = source_anchor_for_chunk(document, &chunk_id, start, end);
                if let Some(anchor) = &source_anchor {
                    metadata.insert("anchor_id".into(), anchor.id.clone());
                    metadata.insert("source_path".into(), anchor.path.clone());
                    metadata.insert("content_hash".into(), anchor.content_hash.clone());
                    if let Some(page) = anchor.page {
                        metadata.insert("page".into(), page.to_string());
                    }
                    if let Some(section) = &anchor.section {
                        metadata.insert("section".into(), section.clone());
                    }
                    if !anchor.section_hierarchy.is_empty() {
                        metadata.insert(
                            "section_hierarchy".into(),
                            anchor.section_hierarchy.join(" > "),
                        );
                    }
                    if let Some(paragraph_index) = anchor.paragraph_index {
                        metadata.insert("paragraph_index".into(), paragraph_index.to_string());
                    }
                }
                chunk_inputs.push(ChunkInput {
                    id: chunk_id,
                    document_id: document.id.clone(),
                    region_id,
                    ordinal,
                    start,
                    end,
                    text,
                    metadata,
                    source_anchor,
                });
            }
        }
        let identity = self.embedder.identity();
        let cache = reusable_chunks
            .iter()
            .map(|chunk| {
                (
                    (
                        chunk.document_id.clone(),
                        chunk.ordinal,
                        chunk_hash(chunk.text.as_str()),
                    ),
                    chunk,
                )
            })
            .collect::<HashMap<_, _>>();
        let mut embeddings = vec![None; chunk_inputs.len()];
        let mut reused = 0;
        for (index, chunk) in chunk_inputs.iter().enumerate() {
            let hash = chunk_hash(chunk.text.as_str());
            if let Some(cached) =
                cache.get(&(chunk.document_id.clone(), chunk.ordinal, hash.clone()))
            {
                if can_reuse_embedding(cached, &identity, &hash) {
                    embeddings[index] = Some(cached.embedding.clone());
                    reused += 1;
                }
            }
        }
        progress(reused, chunk_inputs.len());

        let mut to_embed = chunk_inputs
            .iter()
            .enumerate()
            .filter(|(index, _)| embeddings[*index].is_none())
            .map(|(index, chunk)| (index, chunk.text.clone()))
            .collect::<Vec<_>>();
        let mut embedded = 0;
        for batch in to_embed.chunks_mut(EMBEDDING_BATCH_SIZE) {
            let batch_texts = batch
                .iter()
                .map(|(_, text)| text.clone())
                .collect::<Vec<_>>();
            let batch_embeddings = self.embedder.embed_many(&batch_texts)?;
            for ((index, _), embedding) in batch.iter().zip(batch_embeddings) {
                embeddings[*index] = Some(embedding);
                embedded += 1;
            }
            progress(reused + embedded, chunk_inputs.len());
        }

        let mut chunks = chunk_inputs
            .into_iter()
            .zip(embeddings)
            .map(|(chunk, embedding)| {
                let embedding = embedding.expect("embedding filled");
                let dimension = embedding.len();
                let hash = chunk_hash(chunk.text.as_str());
                Chunk {
                    id: chunk.id,
                    document_id: chunk.document_id,
                    region_id: chunk.region_id,
                    ordinal: chunk.ordinal,
                    start: chunk.start,
                    end: chunk.end,
                    text: chunk.text,
                    metadata: chunk.metadata,
                    source_anchor: chunk.source_anchor,
                    embedding,
                    embedding_text_hash: Some(hash),
                    embedding_provider: Some(identity.provider.clone()),
                    embedding_model: Some(identity.model.clone()),
                    embedding_endpoint: Some(identity.endpoint.clone()),
                    embedding_dimension: Some(dimension),
                    chunking_version: Some(CHUNKING_VERSION),
                }
            })
            .collect::<Vec<_>>();

        let regions = derive_vector_regions(&mut chunks);

        Ok((
            PersistedMemory {
                documents,
                chunks,
                regions,
                links: Vec::new(),
                memory_map: None,
            },
            EmbeddingReuseStats {
                embedded_count: embedded,
                reused_embedding_count: reused,
            },
        ))
    }
}

fn can_reuse_embedding(chunk: &Chunk, identity: &EmbedderIdentity, text_hash: &str) -> bool {
    chunk.embedding_text_hash.as_deref() == Some(text_hash)
        && chunk.embedding_provider.as_deref() == Some(identity.provider.as_str())
        && chunk.embedding_model.as_deref() == Some(identity.model.as_str())
        && chunk.embedding_endpoint.as_deref() == Some(identity.endpoint.as_str())
        && chunk.embedding_dimension == Some(chunk.embedding.len())
        && chunk.chunking_version == Some(CHUNKING_VERSION)
        && !chunk.embedding.is_empty()
}

pub fn chunk_hash(text: &str) -> String {
    let mut hash = 14695981039346656037u64;
    for byte in text.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    format!("{hash:016x}")
}

pub fn extract_documents(inputs: &[PathBuf]) -> anyhow::Result<DocumentImportBatch> {
    let mut documents = Vec::new();
    let mut imported_paths = Vec::new();
    let mut skipped_paths = Vec::new();
    for input in inputs {
        for path in collect_files(input)? {
            if !should_consider_file(&path) {
                skipped_paths.push(skip(&path, "unsupported file type"));
                continue;
            }
            match read_document_text(&path)? {
                ReadOutcome::Text(parsed) => {
                    let text = parsed.text;
                    if text.is_empty() {
                        skipped_paths.push(skip(&path, "no extractable text"));
                        continue;
                    }
                    imported_paths.push(path.display().to_string());
                    documents.push(document_from_parsed(
                        &path,
                        text,
                        parsed.content_hash,
                        parsed.page_spans,
                        parsed.section_spans,
                        parsed.section_hierarchy_spans,
                        parsed.paragraph_spans,
                    ));
                }
                ReadOutcome::Skip(reason) => skipped_paths.push(skip(&path, reason)),
            }
        }
    }
    Ok(DocumentImportBatch {
        documents,
        imported_paths,
        skipped_paths,
    })
}

#[derive(Debug, Clone)]
struct RegionSpec {
    id: String,
    key_terms: Vec<String>,
}

struct ChunkInput {
    id: String,
    document_id: String,
    region_id: String,
    ordinal: usize,
    start: usize,
    end: usize,
    text: String,
    metadata: BTreeMap<String, String>,
    source_anchor: Option<SourceAnchor>,
}

enum ReadOutcome {
    Text(ParsedText),
    Skip(String),
}

#[derive(Debug, Clone)]
struct ParsedText {
    text: String,
    content_hash: String,
    page_spans: Vec<TextSpan>,
    section_spans: Vec<TextSpan>,
    section_hierarchy_spans: Vec<TextSpan>,
    paragraph_spans: Vec<TextSpan>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TextSpan {
    start: usize,
    end: usize,
    value: String,
}

fn collect_files(input: &Path) -> anyhow::Result<Vec<std::path::PathBuf>> {
    if input.is_file() {
        return Ok(vec![input.to_path_buf()]);
    }
    let mut files = Vec::new();
    collect_files_recursive(input, &mut files)?;
    files.sort();
    Ok(files)
}

fn collect_files_recursive(input: &Path, files: &mut Vec<PathBuf>) -> anyhow::Result<()> {
    for entry in fs::read_dir(input)? {
        let entry = entry?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if name.starts_with('.') {
            continue;
        }
        if path.is_dir() {
            collect_files_recursive(&path, files)?;
        } else if path.is_file() && should_consider_file(&path) {
            files.push(path);
        }
    }
    Ok(())
}

fn document_from_parsed(
    path: &Path,
    text: String,
    content_hash: String,
    page_spans: Vec<TextSpan>,
    section_spans: Vec<TextSpan>,
    section_hierarchy_spans: Vec<TextSpan>,
    paragraph_spans: Vec<TextSpan>,
) -> Document {
    let title = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("untitled")
        .to_string();
    let parent = path
        .parent()
        .and_then(|value| value.file_name())
        .and_then(|value| value.to_str())
        .unwrap_or("root");
    let doc_id = slugify(&format!("{parent}-{}", path.display()));
    let mut metadata = BTreeMap::new();
    metadata.insert("path".into(), path.display().to_string());
    metadata.insert("source".into(), "local".into());
    metadata.insert("content_hash".into(), content_hash.clone());
    metadata.insert("file_hash".into(), content_hash.clone());
    metadata.insert("parser_version".into(), PARSER_VERSION.to_string());
    let source_artifact_id = source_artifact_id(&content_hash);
    metadata.insert("source_artifact_id".into(), source_artifact_id.clone());
    metadata.insert("source_type".into(), "local_file".into());
    apply_source_trust_metadata(&mut metadata, path);
    metadata.insert("storage_mode".into(), "reference_in_place".into());
    metadata.insert("original_path".into(), path.display().to_string());
    metadata.insert("current_path".into(), path.display().to_string());
    metadata.insert("imported_at".into(), now_millis().to_string());
    metadata.insert(
        PAGE_SPANS_KEY.into(),
        serde_json::to_string(&page_spans).unwrap_or_else(|_| "[]".into()),
    );
    metadata.insert(
        SECTION_SPANS_KEY.into(),
        serde_json::to_string(&section_spans).unwrap_or_else(|_| "[]".into()),
    );
    metadata.insert(
        SECTION_HIERARCHY_SPANS_KEY.into(),
        serde_json::to_string(&section_hierarchy_spans).unwrap_or_else(|_| "[]".into()),
    );
    metadata.insert(
        PARAGRAPH_SPANS_KEY.into(),
        serde_json::to_string(&paragraph_spans).unwrap_or_else(|_| "[]".into()),
    );
    let text_len = text.chars().count();
    let anchor = SourceAnchor {
        id: format!("{doc_id}:document"),
        document_id: doc_id.clone(),
        chunk_id: None,
        source_artifact_id: Some(source_artifact_id),
        path: path.display().to_string(),
        content_hash: content_hash.clone(),
        start: 0,
        end: text_len,
        byte_start: Some(0),
        byte_end: Some(text.len()),
        char_start: Some(0),
        char_end: Some(text_len),
        page: page_spans
            .first()
            .and_then(|span| span.value.parse::<usize>().ok()),
        rendered_page: page_spans.first().and_then(|span| {
            span.value
                .parse::<usize>()
                .ok()
                .map(|page| RenderedPageMetadata {
                    page,
                    label: Some(page.to_string()),
                    width: None,
                    height: None,
                    unit: None,
                })
        }),
        pdf_selection: pdf_selection_for_path(path, 0, text_len, page_spans.first()),
        email_location: None,
        section: section_spans.first().map(|span| span.value.clone()),
        section_hierarchy: section_hierarchy_spans
            .first()
            .map(|span| split_hierarchy(&span.value))
            .unwrap_or_default(),
        paragraph_index: paragraph_spans
            .first()
            .and_then(|span| span.value.parse::<usize>().ok()),
        parser_version: PARSER_VERSION,
    };
    Document {
        id: doc_id,
        title,
        text,
        metadata,
        source_anchor: Some(anchor),
        content_hash: Some(content_hash),
        parser_version: Some(PARSER_VERSION),
    }
}

fn chunk_text(text: &str) -> Vec<(usize, usize, String)> {
    if text.is_empty() {
        return Vec::new();
    }
    let chars: Vec<char> = text.chars().collect();
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        if chunks.len() >= MAX_CHUNKS_PER_DOCUMENT {
            break;
        }
        let end = (start + CHUNK_SIZE).min(chars.len());
        let excerpt = chars[start..end].iter().collect::<String>();
        chunks.push((start, end, excerpt));
        if end == chars.len() {
            break;
        }
        start = end.saturating_sub(CHUNK_OVERLAP);
    }
    chunks
}

fn source_anchor_for_chunk(
    document: &Document,
    chunk_id: &str,
    start: usize,
    end: usize,
) -> Option<SourceAnchor> {
    let path = document.metadata.get("path")?.clone();
    let content_hash = document
        .content_hash
        .clone()
        .or_else(|| document.metadata.get("content_hash").cloned())
        .unwrap_or_else(|| chunk_hash(&document.text));
    let parser_version = document
        .parser_version
        .or_else(|| {
            document
                .metadata
                .get("parser_version")
                .and_then(|value| value.parse().ok())
        })
        .unwrap_or(PARSER_VERSION);
    let source_artifact_id = document.metadata.get("source_artifact_id").cloned();
    let (byte_start, byte_end) = byte_offsets_for_char_span(&document.text, start, end);
    Some(SourceAnchor {
        id: format!("{chunk_id}:anchor"),
        document_id: document.id.clone(),
        chunk_id: Some(chunk_id.to_string()),
        source_artifact_id,
        path,
        content_hash,
        start,
        end,
        byte_start: Some(byte_start),
        byte_end: Some(byte_end),
        char_start: Some(start),
        char_end: Some(end),
        page: span_value_at::<usize>(&document.metadata, PAGE_SPANS_KEY, start),
        rendered_page: span_value_at::<usize>(&document.metadata, PAGE_SPANS_KEY, start).map(
            |page| RenderedPageMetadata {
                page,
                label: Some(page.to_string()),
                width: None,
                height: None,
                unit: None,
            },
        ),
        pdf_selection: pdf_selection_for_document(document, start, end),
        email_location: email_location_for_document(document),
        section: span_value_at::<String>(&document.metadata, SECTION_SPANS_KEY, start),
        section_hierarchy: span_value_at::<String>(
            &document.metadata,
            SECTION_HIERARCHY_SPANS_KEY,
            start,
        )
        .map(|value| split_hierarchy(&value))
        .unwrap_or_default(),
        paragraph_index: span_value_at::<usize>(&document.metadata, PARAGRAPH_SPANS_KEY, start),
        parser_version,
    })
}

fn byte_offsets_for_char_span(text: &str, start: usize, end: usize) -> (usize, usize) {
    (
        byte_offset_for_char(text, start),
        byte_offset_for_char(text, end),
    )
}

fn byte_offset_for_char(text: &str, offset: usize) -> usize {
    if offset == 0 {
        return 0;
    }
    text.char_indices()
        .nth(offset)
        .map(|(byte, _)| byte)
        .unwrap_or(text.len())
}

fn split_hierarchy(value: &str) -> Vec<String> {
    value
        .split(" > ")
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn apply_source_trust_metadata(metadata: &mut BTreeMap<String, String>, path: &Path) {
    if matches!(
        path.extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| extension.to_ascii_lowercase())
            .as_deref(),
        Some("md" | "markdown" | "mdown")
    ) {
        metadata.insert("source_trust_kind".into(), "user_authored_note".into());
        metadata.insert("source_trust".into(), "0.90".into());
        metadata.insert(
            "source_trust_label".into(),
            "User-authored local note".into(),
        );
        metadata.insert(
            "source_caveat".into(),
            "Local note: cite exact anchors for quoted or mutable claims.".into(),
        );
    } else {
        metadata.insert("source_trust_kind".into(), "imported_document".into());
        metadata.insert("source_trust".into(), "0.82".into());
        metadata.insert(
            "source_trust_label".into(),
            "Imported local document".into(),
        );
        metadata.insert(
            "source_caveat".into(),
            "Imported document: use source anchors for exact recall.".into(),
        );
    }
}

fn pdf_selection_for_path(
    path: &Path,
    start: usize,
    end: usize,
    page_span: Option<&TextSpan>,
) -> Option<PdfTextSelection> {
    if !path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("pdf"))
    {
        return None;
    }
    let page = page_span
        .and_then(|span| span.value.parse::<usize>().ok())
        .unwrap_or(1);
    Some(PdfTextSelection {
        page,
        text_start: start,
        text_end: end,
        selected_text: None,
        bounding_box: None,
    })
}

fn pdf_selection_for_document(
    document: &Document,
    start: usize,
    end: usize,
) -> Option<PdfTextSelection> {
    let path = document.metadata.get("path")?;
    let path = Path::new(path);
    let page = span_value_at::<usize>(&document.metadata, PAGE_SPANS_KEY, start).unwrap_or(1);
    pdf_selection_for_path(
        path,
        start,
        end,
        Some(&TextSpan {
            start,
            end,
            value: page.to_string(),
        }),
    )
}

fn email_location_for_document(document: &Document) -> Option<EmailThreadLocation> {
    let thread_id = document.metadata.get("email_thread_id")?.clone();
    Some(EmailThreadLocation {
        thread_id,
        message_id: document.metadata.get("email_message_id").cloned(),
        mailbox: document.metadata.get("email_mailbox").cloned(),
        subject: document
            .metadata
            .get("email_subject")
            .cloned()
            .or_else(|| Some(document.title.clone())),
    })
}

fn source_artifact_id(file_hash: &str) -> String {
    format!("source-artifact:{file_hash}")
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn span_value_at<T>(metadata: &BTreeMap<String, String>, key: &str, offset: usize) -> Option<T>
where
    T: std::str::FromStr,
{
    let spans = metadata.get(key)?;
    let spans = serde_json::from_str::<Vec<TextSpan>>(spans).ok()?;
    spans
        .into_iter()
        .find(|span| offset >= span.start && offset < span.end)
        .and_then(|span| span.value.parse::<T>().ok())
}

fn derive_region_specs(documents: &[Document]) -> Vec<RegionSpec> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for document in documents {
        let unique = tokenize(&document.text).into_iter().collect::<HashSet<_>>();
        for token in unique {
            if STOPWORDS.contains(&token.as_str()) {
                continue;
            }
            *counts.entry(token).or_default() += 1;
        }
    }
    let mut ranked = counts.into_iter().collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    let mut specs = Vec::new();
    for (index, (term, _)) in ranked.into_iter().take(5).enumerate() {
        specs.push(RegionSpec {
            id: format!("region-{}", index + 1),
            key_terms: expand_terms(documents, &term),
        });
    }
    if specs.is_empty() {
        specs.push(RegionSpec {
            id: "region-1".into(),
            key_terms: vec!["general".into()],
        });
    }
    specs
}

fn expand_terms(documents: &[Document], seed: &str) -> Vec<String> {
    let mut related = HashMap::<String, usize>::new();
    for document in documents {
        let tokens = tokenize(&document.text);
        if !tokens.iter().any(|token| token == seed) {
            continue;
        }
        for token in tokens {
            if token != seed && !STOPWORDS.contains(&token.as_str()) {
                *related.entry(token).or_default() += 1;
            }
        }
    }
    let mut ranked = related.into_iter().collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    let mut terms = vec![seed.to_string()];
    terms.extend(ranked.into_iter().take(3).map(|(token, _)| token));
    terms
}

fn assign_region(text: &str, specs: &[RegionSpec]) -> String {
    let tokens = tokenize(text).into_iter().collect::<HashSet<_>>();
    specs
        .iter()
        .map(|spec| {
            let score = spec
                .key_terms
                .iter()
                .filter(|term| tokens.contains(*term))
                .count();
            (spec.id.clone(), score)
        })
        .max_by(|left, right| left.1.cmp(&right.1).then_with(|| left.0.cmp(&right.0)))
        .map(|(id, _)| id)
        .unwrap_or_else(|| specs[0].id.clone())
}

fn derive_vector_regions(chunks: &mut [Chunk]) -> Vec<Region> {
    if chunks.is_empty() {
        return Vec::new();
    }
    let cluster_count = ((chunks.len() as f64).sqrt().ceil() as usize).clamp(1, 8);
    let seeds = choose_region_seeds(chunks, cluster_count);
    let mut assignments = vec![0usize; chunks.len()];
    for (index, chunk) in chunks.iter().enumerate() {
        assignments[index] = seeds
            .iter()
            .enumerate()
            .map(|(seed_index, chunk_index)| {
                (
                    seed_index,
                    cosine_similarity(&chunk.embedding, &chunks[*chunk_index].embedding),
                )
            })
            .max_by(|left, right| {
                left.1
                    .partial_cmp(&right.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(seed_index, _)| seed_index)
            .unwrap_or(0);
    }

    let mut regions = Vec::new();
    for cluster in 0..cluster_count {
        let member_indexes = assignments
            .iter()
            .enumerate()
            .filter_map(|(index, assigned)| (*assigned == cluster).then_some(index))
            .collect::<Vec<_>>();
        if member_indexes.is_empty() {
            continue;
        }
        let region_id = format!("region-{}", regions.len() + 1);
        for index in &member_indexes {
            chunks[*index].region_id = region_id.clone();
        }
        let chunk_ids = member_indexes
            .iter()
            .map(|index| chunks[*index].id.clone())
            .collect::<Vec<_>>();
        let centroid = centroid_for_indexes(chunks, &member_indexes);
        let terms = top_terms_for_indexes(chunks, &member_indexes, 4);
        let label = format_region_label(&terms);
        let mut filters = BTreeMap::new();
        filters.insert("topic".into(), label.clone());
        let summary = if terms.is_empty() {
            "Vector region around related chunks".into()
        } else {
            format!("Vector region around {}", terms.join(", "))
        };
        regions.push(Region {
            id: region_id,
            label,
            summary,
            filters,
            chunk_ids,
            centroid,
            neighbors: Vec::new(),
        });
    }
    connect_vector_regions(&mut regions);
    regions
}

fn choose_region_seeds(chunks: &[Chunk], cluster_count: usize) -> Vec<usize> {
    let mut seeds = vec![0usize];
    while seeds.len() < cluster_count && seeds.len() < chunks.len() {
        let next = chunks
            .iter()
            .enumerate()
            .filter(|(index, _)| !seeds.contains(index))
            .map(|(index, chunk)| {
                let nearest = seeds
                    .iter()
                    .map(|seed| cosine_similarity(&chunk.embedding, &chunks[*seed].embedding))
                    .fold(f32::NEG_INFINITY, f32::max);
                (index, nearest)
            })
            .min_by(|left, right| {
                left.1
                    .partial_cmp(&right.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(index, _)| index);
        if let Some(index) = next {
            seeds.push(index);
        } else {
            break;
        }
    }
    seeds
}

fn centroid_for_indexes(chunks: &[Chunk], indexes: &[usize]) -> Vec<f32> {
    let dimensions = indexes
        .iter()
        .filter_map(|index| chunks.get(*index))
        .map(|chunk| chunk.embedding.len())
        .find(|length| *length > 0)
        .unwrap_or(0);
    let mut centroid = vec![0.0; dimensions];
    if dimensions == 0 {
        return centroid;
    }
    for index in indexes {
        if let Some(chunk) = chunks.get(*index) {
            for (slot, value) in centroid.iter_mut().zip(&chunk.embedding) {
                *slot += *value;
            }
        }
    }
    for slot in &mut centroid {
        *slot /= indexes.len().max(1) as f32;
    }
    centroid
}

fn top_terms_for_indexes(chunks: &[Chunk], indexes: &[usize], limit: usize) -> Vec<String> {
    meaningful_terms_for_chunks(indexes.iter().filter_map(|index| chunks.get(*index)), limit)
}

pub fn meaningful_terms_for_chunks<'a>(
    chunks: impl IntoIterator<Item = &'a Chunk>,
    limit: usize,
) -> Vec<String> {
    let mut counts = HashMap::<String, usize>::new();
    for chunk in chunks {
        for token in tokenize(&chunk.text) {
            if STOPWORDS.contains(&token.as_str()) {
                continue;
            }
            *counts.entry(token).or_default() += 1;
        }
    }
    let mut ranked = counts.into_iter().collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    ranked
        .into_iter()
        .take(limit)
        .map(|(term, _)| term)
        .collect()
}

pub fn format_region_label(terms: &[String]) -> String {
    let label_terms = terms.iter().take(3).cloned().collect::<Vec<_>>();
    if label_terms.is_empty() {
        "general".into()
    } else {
        label_terms.join(" / ")
    }
}

fn connect_vector_regions(regions: &mut [Region]) {
    for index in 0..regions.len() {
        let mut neighbors = regions
            .iter()
            .enumerate()
            .filter(|(other_index, _)| *other_index != index)
            .map(|(_, other)| {
                (
                    other.id.clone(),
                    cosine_similarity(&regions[index].centroid, &other.centroid),
                )
            })
            .collect::<Vec<_>>();
        neighbors.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.0.cmp(&right.0))
        });
        regions[index].neighbors = neighbors.into_iter().take(3).map(|(id, _)| id).collect();
    }
}

fn slugify(input: &str) -> String {
    let mut slug = String::new();
    let mut last_dash = false;
    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
            last_dash = false;
        } else if !last_dash {
            slug.push('-');
            last_dash = true;
        }
    }
    slug.trim_matches('-').to_string()
}

fn should_consider_file(path: &Path) -> bool {
    let ext = path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_ascii_lowercase());
    match ext.as_deref() {
        Some(
            "txt" | "md" | "markdown" | "json" | "csv" | "tsv" | "log" | "rst" | "html" | "htm"
            | "xml" | "yaml" | "yml" | "toml" | "ini" | "swift" | "rs" | "py" | "js" | "ts" | "tsx"
            | "jsx" | "css" | "scss" | "pdf",
        ) => true,
        Some(_) => false,
        None => true,
    }
}

fn parse_text(raw: &str, content_hash: String, page_aware: bool) -> ParsedText {
    let mut text = String::new();
    let mut page_spans = Vec::new();
    let segments = if page_aware {
        raw.split('\u{c}').collect::<Vec<_>>()
    } else {
        vec![raw]
    };
    for (index, segment) in segments.into_iter().enumerate() {
        let normalized = normalize_segment(segment);
        if normalized.is_empty() {
            continue;
        }
        if !text.is_empty() {
            text.push_str("\n\n");
        }
        let start = text.chars().count();
        append_capped(&mut text, &normalized);
        let end = text.chars().count();
        if end > start {
            page_spans.push(TextSpan {
                start,
                end,
                value: (index + 1).to_string(),
            });
        }
        if text.chars().count() >= MAX_DOCUMENT_CHARS {
            break;
        }
    }
    if page_spans.is_empty() && !text.is_empty() {
        page_spans.push(TextSpan {
            start: 0,
            end: text.chars().count(),
            value: "1".into(),
        });
    }
    let (section_spans, section_hierarchy_spans) = derive_section_spans(&text);
    let paragraph_spans = derive_paragraph_spans(&text);
    ParsedText {
        text,
        content_hash,
        page_spans,
        section_spans,
        section_hierarchy_spans,
        paragraph_spans,
    }
}

fn normalize_segment(raw: &str) -> String {
    raw.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn append_capped(target: &mut String, value: &str) {
    let remaining = MAX_DOCUMENT_CHARS.saturating_sub(target.chars().count());
    target.extend(value.chars().take(remaining));
}

fn derive_section_spans(text: &str) -> (Vec<TextSpan>, Vec<TextSpan>) {
    let mut headings = Vec::<(usize, String, Vec<String>)>::new();
    let mut hierarchy = Vec::<String>::new();
    let mut offset = 0usize;
    for line in text.lines() {
        if let Some((level, label)) = heading_label(line) {
            let level = level.max(1);
            if hierarchy.len() >= level {
                hierarchy.truncate(level - 1);
            }
            hierarchy.push(label.clone());
            headings.push((offset, label, hierarchy.clone()));
        }
        offset += line.chars().count() + 1;
    }
    if headings.is_empty() && !text.is_empty() {
        let span = TextSpan {
            start: 0,
            end: text.chars().count(),
            value: "document".into(),
        };
        return (vec![span.clone()], vec![span]);
    }
    let end_of_text = text.chars().count();
    let mut section_spans = Vec::new();
    let mut hierarchy_spans = Vec::new();
    for (index, (start, label, hierarchy)) in headings.iter().enumerate() {
        let end = headings
            .get(index + 1)
            .map(|(next, _, _)| *next)
            .unwrap_or(end_of_text);
        section_spans.push(TextSpan {
            start: *start,
            end,
            value: label.clone(),
        });
        hierarchy_spans.push(TextSpan {
            start: *start,
            end,
            value: hierarchy.join(" > "),
        });
    }
    (section_spans, hierarchy_spans)
}

fn derive_paragraph_spans(text: &str) -> Vec<TextSpan> {
    let mut spans = Vec::new();
    let mut offset = 0usize;
    for line in text.lines() {
        let line_len = line.chars().count();
        if !line.trim().is_empty() {
            spans.push(TextSpan {
                start: offset,
                end: offset + line_len,
                value: (spans.len() + 1).to_string(),
            });
        }
        offset += line_len + 1;
    }
    if spans.is_empty() && !text.is_empty() {
        spans.push(TextSpan {
            start: 0,
            end: text.chars().count(),
            value: "1".into(),
        });
    }
    spans
}

fn heading_label(line: &str) -> Option<(usize, String)> {
    let trimmed = line.trim();
    let hashes = trimmed.chars().take_while(|char| *char == '#').count();
    if hashes > 0 && trimmed.chars().nth(hashes).is_some_and(char::is_whitespace) {
        let label = trimmed[hashes..].trim();
        if !label.is_empty() {
            return Some((hashes, label.chars().take(96).collect()));
        }
    }
    for prefix in [
        "fn ",
        "struct ",
        "enum ",
        "impl ",
        "class ",
        "def ",
        "function ",
        "interface ",
    ] {
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            let name = rest
                .split(|ch: char| ch == '(' || ch == '{' || ch == ':' || ch.is_whitespace())
                .next()
                .unwrap_or(rest)
                .trim();
            if !name.is_empty() {
                return Some((1, format!("{} {}", prefix.trim_end(), name)));
            }
        }
    }
    None
}

fn read_document_text(path: &Path) -> anyhow::Result<ReadOutcome> {
    if path
        .extension()
        .and_then(|value| value.to_str())
        .is_some_and(|value| value.eq_ignore_ascii_case("pdf"))
    {
        return read_pdf_file(path);
    }
    read_text_file(path)
}

fn read_text_file(path: &Path) -> anyhow::Result<ReadOutcome> {
    let metadata = fs::metadata(path)?;
    if metadata.len() > MAX_FILE_BYTES {
        return Ok(ReadOutcome::Skip(
            "file is larger than the text import limit".into(),
        ));
    }
    let bytes = fs::read(path)?;
    let content_hash = content_hash(&bytes);
    if bytes.contains(&0) {
        return Ok(ReadOutcome::Skip("binary file".into()));
    }
    let Ok(text) = String::from_utf8(bytes) else {
        return Ok(ReadOutcome::Skip("file is not valid UTF-8".into()));
    };
    Ok(ReadOutcome::Text(parse_text(&text, content_hash, false)))
}

fn read_pdf_file(path: &Path) -> anyhow::Result<ReadOutcome> {
    let metadata = fs::metadata(path)?;
    if metadata.len() > MAX_PDF_BYTES {
        return Ok(ReadOutcome::Skip(
            "PDF is larger than the import limit".into(),
        ));
    }
    let bytes = fs::read(path)?;
    let content_hash = content_hash(&bytes);
    match pdf_extract::extract_text(path) {
        Ok(text) => Ok(ReadOutcome::Text(parse_text(&text, content_hash, true))),
        Err(error) => Ok(ReadOutcome::Skip(format!(
            "PDF text extraction failed: {error}"
        ))),
    }
}

fn content_hash(bytes: &[u8]) -> String {
    let mut hash = 14695981039346656037u64;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    format!("{hash:016x}")
}

fn skip(path: &Path, reason: impl Into<String>) -> ImportSkip {
    ImportSkip {
        path: path.display().to_string(),
        reason: reason.into(),
    }
}
