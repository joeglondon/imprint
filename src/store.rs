use crate::types::*;
use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use std::fs;
use std::path::{Path, PathBuf};

pub trait MemoryStore {
    fn load(&self) -> anyhow::Result<PersistedMemory>;
    fn save(&self, memory: &PersistedMemory) -> anyhow::Result<()>;
}

#[derive(Debug, Clone)]
pub struct FileMemoryStore {
    root: PathBuf,
}

impl FileMemoryStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn data_path(&self) -> PathBuf {
        self.root.join("memory.sqlite")
    }

    pub fn legacy_json_path(&self) -> PathBuf {
        self.root.join("memory.json")
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn connection(&self) -> Result<Connection> {
        fs::create_dir_all(&self.root)
            .with_context(|| format!("creating store root {}", self.root.display()))?;
        let connection = Connection::open(self.data_path())
            .with_context(|| format!("opening store {}", self.data_path().display()))?;
        migrate_schema(&connection)?;
        Ok(connection)
    }
}

impl MemoryStore for FileMemoryStore {
    fn load(&self) -> anyhow::Result<PersistedMemory> {
        if !self.data_path().exists() && self.legacy_json_path().exists() {
            let raw = fs::read_to_string(self.legacy_json_path())
                .with_context(|| format!("reading legacy store {}", self.legacy_json_path().display()))?;
            let memory = serde_json::from_str::<PersistedMemory>(&raw)
                .with_context(|| format!("parsing legacy store {}", self.legacy_json_path().display()))?;
            self.save(&memory)?;
            return Ok(memory);
        }
        let connection = self.connection()?;
        load_memory(&connection)
    }

    fn save(&self, memory: &PersistedMemory) -> anyhow::Result<()> {
        let mut connection = self.connection()?;
        save_memory(&mut connection, memory)
    }
}

fn migrate_schema(connection: &Connection) -> Result<()> {
    connection.execute_batch(
        r#"
        PRAGMA foreign_keys = ON;
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            text TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            source_anchor_json TEXT,
            content_hash TEXT,
            parser_version INTEGER
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            region_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            start_offset INTEGER NOT NULL,
            end_offset INTEGER NOT NULL,
            text TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            source_anchor_json TEXT,
            embedding_json TEXT NOT NULL,
            embedding_text_hash TEXT,
            embedding_provider TEXT,
            embedding_model TEXT,
            embedding_endpoint TEXT,
            embedding_dimension INTEGER,
            chunking_version INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id, ordinal);
        CREATE INDEX IF NOT EXISTS idx_chunks_region ON chunks(region_id);
        CREATE TABLE IF NOT EXISTS source_anchors (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            chunk_id TEXT,
            path TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            start_offset INTEGER NOT NULL,
            end_offset INTEGER NOT NULL,
            page INTEGER,
            section TEXT,
            parser_version INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_anchors_document ON source_anchors(document_id);
        CREATE INDEX IF NOT EXISTS idx_anchors_chunk ON source_anchors(chunk_id);
        CREATE TABLE IF NOT EXISTS regions (
            id TEXT PRIMARY KEY,
            label TEXT NOT NULL,
            summary TEXT NOT NULL,
            filters_json TEXT NOT NULL,
            chunk_ids_json TEXT NOT NULL,
            centroid_json TEXT NOT NULL,
            neighbors_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS links (
            id TEXT PRIMARY KEY,
            source_json TEXT NOT NULL,
            target_json TEXT NOT NULL,
            link_type_json TEXT NOT NULL,
            score REAL NOT NULL,
            label TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS memory_map (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            budget_bytes INTEGER NOT NULL,
            serialized TEXT NOT NULL,
            entries_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            state_json TEXT NOT NULL,
            updated_at INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            status TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            updated_at INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS model_metadata (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL
        );
        "#,
    )?;
    Ok(())
}

fn load_memory(connection: &Connection) -> Result<PersistedMemory> {
    let documents = load_documents(connection)?;
    let chunks = load_chunks(connection)?;
    let regions = load_regions(connection)?;
    let links = load_links(connection)?;
    let memory_map = load_memory_map(connection)?;
    Ok(PersistedMemory {
        documents,
        chunks,
        regions,
        links,
        memory_map,
    })
}

fn save_memory(connection: &mut Connection, memory: &PersistedMemory) -> Result<()> {
    let tx = connection.transaction()?;
    tx.execute_batch(
        r#"
        DELETE FROM source_anchors;
        DELETE FROM chunks;
        DELETE FROM documents;
        DELETE FROM regions;
        DELETE FROM links;
        DELETE FROM memory_map;
        "#,
    )?;

    for document in &memory.documents {
        tx.execute(
            "INSERT INTO documents (id, title, text, metadata_json, source_anchor_json, content_hash, parser_version)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                document.id,
                document.title,
                document.text,
                to_json(&document.metadata)?,
                to_optional_json(&document.source_anchor)?,
                document.content_hash,
                document.parser_version.map(|value| value as i64),
            ],
        )?;
        if let Some(anchor) = &document.source_anchor {
            insert_anchor(&tx, anchor)?;
        }
    }

    for chunk in &memory.chunks {
        tx.execute(
            "INSERT INTO chunks (
                id, document_id, region_id, ordinal, start_offset, end_offset, text,
                metadata_json, source_anchor_json, embedding_json, embedding_text_hash,
                embedding_provider, embedding_model, embedding_endpoint, embedding_dimension,
                chunking_version
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)",
            params![
                chunk.id,
                chunk.document_id,
                chunk.region_id,
                chunk.ordinal as i64,
                chunk.start as i64,
                chunk.end as i64,
                chunk.text,
                to_json(&chunk.metadata)?,
                to_optional_json(&chunk.source_anchor)?,
                to_json(&chunk.embedding)?,
                chunk.embedding_text_hash,
                chunk.embedding_provider,
                chunk.embedding_model,
                chunk.embedding_endpoint,
                chunk.embedding_dimension.map(|value| value as i64),
                chunk.chunking_version.map(|value| value as i64),
            ],
        )?;
        if let Some(anchor) = &chunk.source_anchor {
            insert_anchor(&tx, anchor)?;
        }
    }

    for region in &memory.regions {
        tx.execute(
            "INSERT INTO regions (id, label, summary, filters_json, chunk_ids_json, centroid_json, neighbors_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                region.id,
                region.label,
                region.summary,
                to_json(&region.filters)?,
                to_json(&region.chunk_ids)?,
                to_json(&region.centroid)?,
                to_json(&region.neighbors)?,
            ],
        )?;
    }

    for link in &memory.links {
        tx.execute(
            "INSERT INTO links (id, source_json, target_json, link_type_json, score, label)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                link.id,
                to_json(&link.source)?,
                to_json(&link.target)?,
                to_json(&link.link_type)?,
                link.score,
                link.label,
            ],
        )?;
    }

    if let Some(map) = &memory.memory_map {
        tx.execute(
            "INSERT INTO memory_map (id, budget_bytes, serialized, entries_json)
             VALUES (1, ?1, ?2, ?3)",
            params![map.budget_bytes as i64, map.serialized, to_json(&map.entries)?],
        )?;
    }

    tx.commit()?;
    Ok(())
}

fn insert_anchor(connection: &Connection, anchor: &SourceAnchor) -> Result<()> {
    connection.execute(
        "INSERT OR REPLACE INTO source_anchors (
            id, document_id, chunk_id, path, content_hash, start_offset, end_offset, page, section, parser_version
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        params![
            anchor.id,
            anchor.document_id,
            anchor.chunk_id,
            anchor.path,
            anchor.content_hash,
            anchor.start as i64,
            anchor.end as i64,
            anchor.page.map(|value| value as i64),
            anchor.section,
            anchor.parser_version as i64,
        ],
    )?;
    Ok(())
}

fn load_documents(connection: &Connection) -> Result<Vec<Document>> {
    let mut statement = connection.prepare(
        "SELECT id, title, text, metadata_json, source_anchor_json, content_hash, parser_version
         FROM documents ORDER BY id",
    )?;
    let rows = statement.query_map([], |row| {
        Ok(Document {
            id: row.get(0)?,
            title: row.get(1)?,
            text: row.get(2)?,
            metadata: from_json(row.get::<_, String>(3)?)?,
            source_anchor: from_optional_json(row.get::<_, Option<String>>(4)?)?,
            content_hash: row.get(5)?,
            parser_version: row.get::<_, Option<i64>>(6)?.map(|value| value as u32),
        })
    })?;
    collect_rows(rows)
}

fn load_chunks(connection: &Connection) -> Result<Vec<Chunk>> {
    let mut statement = connection.prepare(
        "SELECT id, document_id, region_id, ordinal, start_offset, end_offset, text,
            metadata_json, source_anchor_json, embedding_json, embedding_text_hash,
            embedding_provider, embedding_model, embedding_endpoint, embedding_dimension,
            chunking_version
         FROM chunks ORDER BY document_id, ordinal",
    )?;
    let rows = statement.query_map([], |row| {
        Ok(Chunk {
            id: row.get(0)?,
            document_id: row.get(1)?,
            region_id: row.get(2)?,
            ordinal: row.get::<_, i64>(3)? as usize,
            start: row.get::<_, i64>(4)? as usize,
            end: row.get::<_, i64>(5)? as usize,
            text: row.get(6)?,
            metadata: from_json(row.get::<_, String>(7)?)?,
            source_anchor: from_optional_json(row.get::<_, Option<String>>(8)?)?,
            embedding: from_json(row.get::<_, String>(9)?)?,
            embedding_text_hash: row.get(10)?,
            embedding_provider: row.get(11)?,
            embedding_model: row.get(12)?,
            embedding_endpoint: row.get(13)?,
            embedding_dimension: row.get::<_, Option<i64>>(14)?.map(|value| value as usize),
            chunking_version: row.get::<_, Option<i64>>(15)?.map(|value| value as u32),
        })
    })?;
    collect_rows(rows)
}

fn load_regions(connection: &Connection) -> Result<Vec<Region>> {
    let mut statement = connection.prepare(
        "SELECT id, label, summary, filters_json, chunk_ids_json, centroid_json, neighbors_json
         FROM regions ORDER BY id",
    )?;
    let rows = statement.query_map([], |row| {
        Ok(Region {
            id: row.get(0)?,
            label: row.get(1)?,
            summary: row.get(2)?,
            filters: from_json(row.get::<_, String>(3)?)?,
            chunk_ids: from_json(row.get::<_, String>(4)?)?,
            centroid: from_json(row.get::<_, String>(5)?)?,
            neighbors: from_json(row.get::<_, String>(6)?)?,
        })
    })?;
    collect_rows(rows)
}

fn load_links(connection: &Connection) -> Result<Vec<Link>> {
    let mut statement = connection.prepare(
        "SELECT id, source_json, target_json, link_type_json, score, label
         FROM links ORDER BY id",
    )?;
    let rows = statement.query_map([], |row| {
        Ok(Link {
            id: row.get(0)?,
            source: from_json(row.get::<_, String>(1)?)?,
            target: from_json(row.get::<_, String>(2)?)?,
            link_type: from_json(row.get::<_, String>(3)?)?,
            score: row.get(4)?,
            label: row.get(5)?,
        })
    })?;
    collect_rows(rows)
}

fn load_memory_map(connection: &Connection) -> Result<Option<MemoryMap>> {
    connection
        .query_row(
            "SELECT budget_bytes, serialized, entries_json FROM memory_map WHERE id = 1",
            [],
            |row| {
                Ok(MemoryMap {
                    budget_bytes: row.get::<_, i64>(0)? as usize,
                    serialized: row.get(1)?,
                    entries: from_json(row.get::<_, String>(2)?)?,
                })
            },
        )
        .optional()
        .map_err(Into::into)
}

fn collect_rows<T>(rows: rusqlite::MappedRows<'_, impl FnMut(&rusqlite::Row<'_>) -> rusqlite::Result<T>>) -> Result<Vec<T>> {
    let mut values = Vec::new();
    for row in rows {
        values.push(row?);
    }
    Ok(values)
}

fn to_json<T: serde::Serialize>(value: &T) -> Result<String> {
    Ok(serde_json::to_string(value)?)
}

fn to_optional_json<T: serde::Serialize>(value: &Option<T>) -> Result<Option<String>> {
    value.as_ref().map(to_json).transpose()
}

fn from_json<T: serde::de::DeserializeOwned>(raw: String) -> rusqlite::Result<T> {
    serde_json::from_str(&raw).map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))
}

fn from_optional_json<T: serde::de::DeserializeOwned>(raw: Option<String>) -> rusqlite::Result<Option<T>> {
    raw.map(from_json).transpose()
}
