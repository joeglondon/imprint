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

    pub fn insert_chat_session(&self, session: &ChatSession) -> Result<()> {
        let connection = self.connection()?;
        connection.execute(
            "INSERT OR REPLACE INTO chat_sessions (id, title, created_at, updated_at, hotness)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                session.id,
                session.title,
                session.created_at as i64,
                session.updated_at as i64,
                session.hotness,
            ],
        )?;
        Ok(())
    }

    pub fn update_chat_session_timestamp(
        &self,
        session_id: &str,
        updated_at: u64,
        hotness: f32,
    ) -> Result<()> {
        let connection = self.connection()?;
        connection.execute(
            "UPDATE chat_sessions SET updated_at = ?2, hotness = ?3 WHERE id = ?1",
            params![session_id, updated_at as i64, hotness],
        )?;
        Ok(())
    }

    pub fn load_chat_session(&self, session_id: &str) -> Result<Option<ChatSession>> {
        let connection = self.connection()?;
        connection
            .query_row(
                "SELECT id, title, created_at, updated_at, hotness FROM chat_sessions WHERE id = ?1",
                params![session_id],
                chat_session_from_row,
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn list_chat_sessions(&self) -> Result<Vec<ChatSession>> {
        let connection = self.connection()?;
        let mut statement = connection.prepare(
            "SELECT id, title, created_at, updated_at, hotness FROM chat_sessions ORDER BY updated_at DESC, id",
        )?;
        let rows = statement.query_map([], chat_session_from_row)?;
        collect_rows(rows)
    }

    pub fn insert_chat_message(&self, message: &ChatMessage) -> Result<()> {
        let connection = self.connection()?;
        connection.execute(
            "INSERT OR REPLACE INTO chat_messages (
                id, session_id, role_json, content, created_at, token_estimate, source_anchor_json
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                message.id,
                message.session_id,
                to_json(&message.role)?,
                message.content,
                message.created_at as i64,
                message.token_estimate as i64,
                to_optional_json(&message.source_anchor)?,
            ],
        )?;
        Ok(())
    }

    pub fn list_chat_messages(&self, session_id: &str) -> Result<Vec<ChatMessage>> {
        let connection = self.connection()?;
        let mut statement = connection.prepare(
            "SELECT id, session_id, role_json, content, created_at, token_estimate, source_anchor_json
             FROM chat_messages WHERE session_id = ?1 ORDER BY created_at, id",
        )?;
        let rows = statement.query_map(params![session_id], chat_message_from_row)?;
        collect_rows(rows)
    }

    pub fn insert_transcript_chunk(&self, chunk: &TranscriptChunk) -> Result<()> {
        let connection = self.connection()?;
        connection.execute(
            "INSERT OR REPLACE INTO transcript_chunks (
                id, session_id, message_id, ordinal, text, embedding_json, embedding_provider,
                embedding_model, embedding_endpoint, source_anchor_json, attention_state_json,
                hotness, created_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
            params![
                chunk.id,
                chunk.session_id,
                chunk.message_id,
                chunk.ordinal as i64,
                chunk.text,
                to_json(&chunk.embedding)?,
                chunk.embedding_provider,
                chunk.embedding_model,
                chunk.embedding_endpoint,
                to_json(&chunk.source_anchor)?,
                to_json(&chunk.attention_state)?,
                chunk.hotness,
                chunk.created_at as i64,
            ],
        )?;
        Ok(())
    }

    pub fn list_transcript_chunks(&self, session_id: Option<&str>) -> Result<Vec<TranscriptChunk>> {
        let connection = self.connection()?;
        let sql =
            "SELECT id, session_id, message_id, ordinal, text, embedding_json, embedding_provider,
                embedding_model, embedding_endpoint, source_anchor_json, attention_state_json,
                hotness, created_at FROM transcript_chunks";
        if let Some(session_id) = session_id {
            let mut statement = connection.prepare(&format!(
                "{sql} WHERE session_id = ?1 ORDER BY created_at, id"
            ))?;
            let rows = statement.query_map(params![session_id], transcript_chunk_from_row)?;
            collect_rows(rows)
        } else {
            let mut statement =
                connection.prepare(&format!("{sql} ORDER BY created_at DESC, id"))?;
            let rows = statement.query_map([], transcript_chunk_from_row)?;
            collect_rows(rows)
        }
    }

    pub fn delete_transcript_chunks(&self, session_id: Option<&str>) -> Result<()> {
        let connection = self.connection()?;
        if let Some(session_id) = session_id {
            connection.execute(
                "DELETE FROM transcript_chunks WHERE session_id = ?1",
                params![session_id],
            )?;
        } else {
            connection.execute("DELETE FROM transcript_chunks", [])?;
        }
        Ok(())
    }

    pub fn insert_derived_memory(&self, memory: &DerivedMemory) -> Result<()> {
        let connection = self.connection()?;
        connection.execute(
            "INSERT OR REPLACE INTO derived_memories (
                id, session_id, kind_json, text, source_message_ids_json, actor, confidence,
                created_at, provenance_json
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                memory.id,
                memory.session_id,
                to_json(&memory.kind)?,
                memory.text,
                to_json(&memory.source_message_ids)?,
                memory.actor,
                memory.confidence,
                memory.created_at as i64,
                to_json(&memory.provenance)?,
            ],
        )?;
        Ok(())
    }

    pub fn list_derived_memories(&self, session_id: Option<&str>) -> Result<Vec<DerivedMemory>> {
        let connection = self.connection()?;
        let sql =
            "SELECT id, session_id, kind_json, text, source_message_ids_json, actor, confidence,
                created_at, provenance_json FROM derived_memories";
        if let Some(session_id) = session_id {
            let mut statement = connection.prepare(&format!(
                "{sql} WHERE session_id = ?1 ORDER BY created_at DESC, id"
            ))?;
            let rows = statement.query_map(params![session_id], derived_memory_from_row)?;
            collect_rows(rows)
        } else {
            let mut statement =
                connection.prepare(&format!("{sql} ORDER BY created_at DESC, id"))?;
            let rows = statement.query_map([], derived_memory_from_row)?;
            collect_rows(rows)
        }
    }

    pub fn insert_brain_artifact(&self, artifact: &BrainArtifact) -> Result<()> {
        let connection = self.connection()?;
        connection.execute(
            "INSERT OR REPLACE INTO brain_artifacts (
                id, kind_json, title, body, source_refs_json, content_hash, provenance_json,
                confidence, created_at, updated_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                artifact.id,
                to_json(&artifact.kind)?,
                artifact.title,
                artifact.body,
                to_json(&artifact.source_refs)?,
                artifact.content_hash,
                to_json(&artifact.provenance)?,
                artifact.confidence as i64,
                artifact.created_at as i64,
                artifact.updated_at as i64,
            ],
        )?;
        Ok(())
    }

    pub fn list_brain_artifacts(
        &self,
        kind: Option<BrainArtifactKind>,
    ) -> Result<Vec<BrainArtifact>> {
        let connection = self.connection()?;
        let sql =
            "SELECT id, kind_json, title, body, source_refs_json, content_hash, provenance_json,
                confidence, created_at, updated_at FROM brain_artifacts";
        if let Some(kind) = kind {
            let mut statement =
                connection.prepare(&format!("{sql} WHERE kind_json = ?1 ORDER BY id"))?;
            let rows = statement.query_map(params![to_json(&kind)?], brain_artifact_from_row)?;
            collect_rows(rows)
        } else {
            let mut statement = connection.prepare(&format!("{sql} ORDER BY id"))?;
            let rows = statement.query_map([], brain_artifact_from_row)?;
            collect_rows(rows)
        }
    }

    pub fn save_cortex_index(&self, index: &CortexIndex) -> Result<()> {
        let connection = self.connection()?;
        connection.execute(
            "INSERT OR REPLACE INTO cortex_indexes (
                id, schema_version, corpus_hash, created_at, compiler, index_json
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                index.id,
                index.schema_version as i64,
                index.corpus_hash,
                index.created_at as i64,
                index.compiler,
                to_json(index)?,
            ],
        )?;
        Ok(())
    }

    pub fn load_current_cortex_index(&self) -> Result<Option<CortexIndex>> {
        let connection = self.connection()?;
        connection
            .query_row(
                "SELECT index_json FROM cortex_indexes ORDER BY created_at DESC, id LIMIT 1",
                [],
                |row| from_json(row.get::<_, String>(0)?),
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn save_cortex_adapter_state(&self, state: &CortexAdapterState) -> Result<()> {
        let connection = self.connection()?;
        connection.execute(
            "INSERT OR REPLACE INTO cortex_adapter_state (
                id, freshness, status, checked_at, state_json
             ) VALUES (1, ?1, ?2, ?3, ?4)",
            params![
                state.freshness,
                state.status,
                state.checked_at as i64,
                to_json(state)?,
            ],
        )?;
        Ok(())
    }

    pub fn load_cortex_adapter_state(&self) -> Result<Option<CortexAdapterState>> {
        let connection = self.connection()?;
        connection
            .query_row(
                "SELECT state_json FROM cortex_adapter_state WHERE id = 1",
                [],
                |row| from_json(row.get::<_, String>(0)?),
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn upsert_cortex_adapter_job(&self, job: &CortexAdapterJob) -> Result<()> {
        let connection = self.connection()?;
        connection.execute(
            "INSERT OR REPLACE INTO cortex_adapter_jobs (
                id, status, source_dataset_hash, prepared_dataset_hash, base_model,
                adapter_output_path, manifest_path, train_records, valid_records, test_records,
                iters, command_json, log_path, failure_reason, payload_json,
                created_at, updated_at, started_at, finished_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19)",
            params![
                job.id,
                job.status,
                job.source_dataset_hash,
                job.prepared_dataset_hash,
                job.base_model,
                job.adapter_output_path,
                job.manifest_path,
                job.train_records.map(|value| value as i64),
                job.valid_records.map(|value| value as i64),
                job.test_records.map(|value| value as i64),
                job.iters.map(|value| value as i64),
                to_json(&job.command)?,
                job.log_path,
                job.failure_reason,
                to_json(&job.payload)?,
                job.created_at as i64,
                job.updated_at as i64,
                job.started_at.map(|value| value as i64),
                job.finished_at.map(|value| value as i64),
            ],
        )?;
        Ok(())
    }

    pub fn load_cortex_adapter_job(&self, id: &str) -> Result<Option<CortexAdapterJob>> {
        let connection = self.connection()?;
        connection
            .query_row(
                "SELECT id, status, source_dataset_hash, prepared_dataset_hash, base_model,
                    adapter_output_path, manifest_path, train_records, valid_records, test_records,
                    iters, command_json, log_path, failure_reason, payload_json,
                    created_at, updated_at, started_at, finished_at
                 FROM cortex_adapter_jobs WHERE id = ?1",
                params![id],
                cortex_adapter_job_from_row,
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn list_cortex_adapter_jobs(&self, status: Option<&str>) -> Result<Vec<CortexAdapterJob>> {
        let connection = self.connection()?;
        let sql = "SELECT id, status, source_dataset_hash, prepared_dataset_hash, base_model,
                adapter_output_path, manifest_path, train_records, valid_records, test_records,
                iters, command_json, log_path, failure_reason, payload_json,
                created_at, updated_at, started_at, finished_at FROM cortex_adapter_jobs";
        if let Some(status) = status {
            let mut statement = connection.prepare(&format!(
                "{sql} WHERE status = ?1 ORDER BY updated_at DESC, id"
            ))?;
            let rows = statement.query_map(params![status], cortex_adapter_job_from_row)?;
            collect_rows(rows)
        } else {
            let mut statement =
                connection.prepare(&format!("{sql} ORDER BY updated_at DESC, id"))?;
            let rows = statement.query_map([], cortex_adapter_job_from_row)?;
            collect_rows(rows)
        }
    }

    pub fn insert_web_finding(&self, finding: &WebFinding) -> Result<()> {
        let connection = self.connection()?;
        connection.execute(
            "INSERT OR REPLACE INTO web_findings (
                id, session_id, query, url, title, summary, retrieved_at, confidence,
                actor, created_at, provenance_json
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                finding.id,
                finding.session_id,
                finding.query,
                finding.url,
                finding.title,
                finding.summary,
                finding.retrieved_at as i64,
                finding.confidence,
                finding.actor,
                finding.created_at as i64,
                to_json(&finding.provenance)?,
            ],
        )?;
        Ok(())
    }

    pub fn list_web_findings(&self, session_id: Option<&str>) -> Result<Vec<WebFinding>> {
        let connection = self.connection()?;
        let sql = "SELECT id, session_id, query, url, title, summary, retrieved_at, confidence,
                actor, created_at, provenance_json FROM web_findings";
        if let Some(session_id) = session_id {
            let mut statement = connection.prepare(&format!(
                "{sql} WHERE session_id = ?1 ORDER BY created_at DESC, id"
            ))?;
            let rows = statement.query_map(params![session_id], web_finding_from_row)?;
            collect_rows(rows)
        } else {
            let mut statement =
                connection.prepare(&format!("{sql} ORDER BY created_at DESC, id"))?;
            let rows = statement.query_map([], web_finding_from_row)?;
            collect_rows(rows)
        }
    }

    pub fn insert_agent_link(&self, link: &AgentLinkMemory) -> Result<()> {
        let connection = self.connection()?;
        connection.execute(
            "INSERT OR REPLACE INTO agent_links (
                id, source_id, target_id, label, actor, created_at, provenance_json
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                link.id,
                link.source_id,
                link.target_id,
                link.label,
                link.actor,
                link.created_at as i64,
                to_json(&link.provenance)?,
            ],
        )?;
        Ok(())
    }

    pub fn insert_attention_mark(&self, mark: &AttentionMark) -> Result<()> {
        let connection = self.connection()?;
        connection.execute(
            "INSERT OR REPLACE INTO attention_marks (
                id, target_id, target_kind_json, action_json, reason, actor, created_at, reverted_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                mark.id,
                mark.target_id,
                to_json(&mark.target_kind)?,
                to_json(&mark.action)?,
                mark.reason,
                mark.actor,
                mark.created_at as i64,
                mark.reverted_at.map(|value| value as i64),
            ],
        )?;
        Ok(())
    }

    pub fn list_attention_marks(&self, target_id: Option<&str>) -> Result<Vec<AttentionMark>> {
        let connection = self.connection()?;
        let sql = "SELECT id, target_id, target_kind_json, action_json, reason, actor, created_at, reverted_at FROM attention_marks";
        if let Some(target_id) = target_id {
            let mut statement = connection.prepare(&format!(
                "{sql} WHERE target_id = ?1 ORDER BY created_at DESC, id"
            ))?;
            let rows = statement.query_map(params![target_id], attention_mark_from_row)?;
            collect_rows(rows)
        } else {
            let mut statement =
                connection.prepare(&format!("{sql} ORDER BY created_at DESC, id"))?;
            let rows = statement.query_map([], attention_mark_from_row)?;
            collect_rows(rows)
        }
    }

    pub fn insert_memory_access(&self, access: &MemoryAccess) -> Result<()> {
        let connection = self.connection()?;
        connection.execute(
            "INSERT OR REPLACE INTO memory_accesses (
                id, target_id, target_kind_json, access_kind_json, reason, actor, accessed_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                access.id,
                access.target_id,
                to_json(&access.target_kind)?,
                to_json(&access.access_kind)?,
                access.reason,
                access.actor,
                access.accessed_at as i64,
            ],
        )?;
        Ok(())
    }

    pub fn list_memory_accesses(
        &self,
        target_id: Option<&str>,
        since: Option<u64>,
    ) -> Result<Vec<MemoryAccess>> {
        let connection = self.connection()?;
        let sql = "SELECT id, target_id, target_kind_json, access_kind_json, reason, actor, accessed_at FROM memory_accesses";
        match (target_id, since) {
            (Some(target_id), Some(since)) => {
                let mut statement = connection.prepare(&format!(
                    "{sql} WHERE target_id = ?1 AND accessed_at >= ?2 ORDER BY accessed_at DESC, id"
                ))?;
                let rows = statement
                    .query_map(params![target_id, since as i64], memory_access_from_row)?;
                collect_rows(rows)
            }
            (Some(target_id), None) => {
                let mut statement = connection.prepare(&format!(
                    "{sql} WHERE target_id = ?1 ORDER BY accessed_at DESC, id"
                ))?;
                let rows = statement.query_map(params![target_id], memory_access_from_row)?;
                collect_rows(rows)
            }
            (None, Some(since)) => {
                let mut statement = connection.prepare(&format!(
                    "{sql} WHERE accessed_at >= ?1 ORDER BY accessed_at DESC, id"
                ))?;
                let rows = statement.query_map(params![since as i64], memory_access_from_row)?;
                collect_rows(rows)
            }
            (None, None) => {
                let mut statement =
                    connection.prepare(&format!("{sql} ORDER BY accessed_at DESC, id"))?;
                let rows = statement.query_map([], memory_access_from_row)?;
                collect_rows(rows)
            }
        }
    }

    pub fn insert_audit_event(&self, event: &AuditEvent) -> Result<()> {
        let connection = self.connection()?;
        connection.execute(
            "INSERT OR REPLACE INTO audit_events (
                id, session_id, event_type, target_id, actor, payload_json, created_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                event.id,
                event.session_id,
                event.event_type,
                event.target_id,
                event.actor,
                event.payload_json,
                event.created_at as i64,
            ],
        )?;
        Ok(())
    }

    pub fn list_audit_events(&self, session_id: Option<&str>) -> Result<Vec<AuditEvent>> {
        let connection = self.connection()?;
        let sql = "SELECT id, session_id, event_type, target_id, actor, payload_json, created_at FROM audit_events";
        if let Some(session_id) = session_id {
            let mut statement = connection.prepare(&format!(
                "{sql} WHERE session_id = ?1 ORDER BY created_at DESC, id"
            ))?;
            let rows = statement.query_map(params![session_id], audit_event_from_row)?;
            collect_rows(rows)
        } else {
            let mut statement =
                connection.prepare(&format!("{sql} ORDER BY created_at DESC, id"))?;
            let rows = statement.query_map([], audit_event_from_row)?;
            collect_rows(rows)
        }
    }

    pub fn insert_chat_context_trace(&self, trace: &ChatContextTrace) -> Result<()> {
        let connection = self.connection()?;
        connection.execute(
            "INSERT OR REPLACE INTO chat_context_traces (
                id, session_id, user_message_id, snippets_json, tool_trace_json, created_at, cortex_trace_json
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                trace.id,
                trace.session_id,
                trace.user_message_id,
                to_json(&trace.snippets)?,
                to_json(&trace.tool_trace)?,
                trace.created_at as i64,
                to_optional_json(&trace.cortex_trace)?,
            ],
        )?;
        Ok(())
    }

    pub fn list_chat_context_traces(&self, session_id: &str) -> Result<Vec<ChatContextTrace>> {
        let connection = self.connection()?;
        let mut statement = connection.prepare(
            "SELECT id, session_id, user_message_id, snippets_json, tool_trace_json, created_at, cortex_trace_json
             FROM chat_context_traces WHERE session_id = ?1 ORDER BY created_at DESC, id",
        )?;
        let rows = statement.query_map(params![session_id], chat_context_trace_from_row)?;
        collect_rows(rows)
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
            let raw = fs::read_to_string(self.legacy_json_path()).with_context(|| {
                format!("reading legacy store {}", self.legacy_json_path().display())
            })?;
            let memory = serde_json::from_str::<PersistedMemory>(&raw).with_context(|| {
                format!("parsing legacy store {}", self.legacy_json_path().display())
            })?;
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
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            hotness REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated ON chat_sessions(updated_at);
        CREATE TABLE IF NOT EXISTS chat_messages (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role_json TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            token_estimate INTEGER NOT NULL,
            source_anchor_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, created_at);
        CREATE TABLE IF NOT EXISTS transcript_chunks (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding_json TEXT NOT NULL,
            embedding_provider TEXT NOT NULL,
            embedding_model TEXT NOT NULL,
            embedding_endpoint TEXT NOT NULL,
            source_anchor_json TEXT NOT NULL,
            attention_state_json TEXT NOT NULL,
            hotness REAL NOT NULL,
            created_at INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_transcript_chunks_session ON transcript_chunks(session_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_transcript_chunks_message ON transcript_chunks(message_id);
        CREATE TABLE IF NOT EXISTS derived_memories (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            kind_json TEXT NOT NULL,
            text TEXT NOT NULL,
            source_message_ids_json TEXT NOT NULL,
            actor TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at INTEGER NOT NULL,
            provenance_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_derived_memories_session ON derived_memories(session_id, created_at);
        CREATE TABLE IF NOT EXISTS web_findings (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            query TEXT NOT NULL,
            url TEXT NOT NULL,
            title TEXT NOT NULL,
            summary TEXT NOT NULL,
            retrieved_at INTEGER NOT NULL,
            confidence REAL NOT NULL,
            actor TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            provenance_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_web_findings_session ON web_findings(session_id, created_at);
        CREATE TABLE IF NOT EXISTS agent_links (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            label TEXT NOT NULL,
            actor TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            provenance_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS brain_artifacts (
            id TEXT PRIMARY KEY,
            kind_json TEXT NOT NULL,
            title TEXT NOT NULL,
            body TEXT NOT NULL,
            source_refs_json TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            provenance_json TEXT NOT NULL,
            confidence INTEGER NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_brain_artifacts_kind ON brain_artifacts(kind_json);
        CREATE TABLE IF NOT EXISTS cortex_adapter_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            freshness TEXT NOT NULL,
            status TEXT NOT NULL,
            checked_at INTEGER NOT NULL,
            state_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_cortex_adapter_state_checked ON cortex_adapter_state(checked_at);
        CREATE TABLE IF NOT EXISTS cortex_indexes (
            id TEXT PRIMARY KEY,
            schema_version INTEGER NOT NULL,
            corpus_hash TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            compiler TEXT NOT NULL,
            index_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_cortex_indexes_created ON cortex_indexes(created_at);
        CREATE TABLE IF NOT EXISTS cortex_adapter_jobs (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            source_dataset_hash TEXT NOT NULL,
            prepared_dataset_hash TEXT,
            base_model TEXT,
            adapter_output_path TEXT NOT NULL,
            manifest_path TEXT,
            train_records INTEGER,
            valid_records INTEGER,
            test_records INTEGER,
            iters INTEGER,
            command_json TEXT NOT NULL,
            log_path TEXT,
            failure_reason TEXT,
            payload_json TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            started_at INTEGER,
            finished_at INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_cortex_adapter_jobs_status ON cortex_adapter_jobs(status, updated_at);
        CREATE INDEX IF NOT EXISTS idx_cortex_adapter_jobs_source ON cortex_adapter_jobs(source_dataset_hash, updated_at);
        CREATE TABLE IF NOT EXISTS attention_marks (
            id TEXT PRIMARY KEY,
            target_id TEXT NOT NULL,
            target_kind_json TEXT NOT NULL,
            action_json TEXT NOT NULL,
            reason TEXT NOT NULL,
            actor TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            reverted_at INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_attention_marks_target ON attention_marks(target_id, created_at);
        CREATE TABLE IF NOT EXISTS memory_accesses (
            id TEXT PRIMARY KEY,
            target_id TEXT NOT NULL,
            target_kind_json TEXT NOT NULL,
            access_kind_json TEXT NOT NULL,
            reason TEXT NOT NULL,
            actor TEXT NOT NULL,
            accessed_at INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_memory_accesses_target ON memory_accesses(target_id, accessed_at);
        CREATE INDEX IF NOT EXISTS idx_memory_accesses_accessed ON memory_accesses(accessed_at);
        CREATE TABLE IF NOT EXISTS audit_events (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            event_type TEXT NOT NULL,
            target_id TEXT NOT NULL,
            actor TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_audit_events_session ON audit_events(session_id, created_at);
        CREATE TABLE IF NOT EXISTS chat_context_traces (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            user_message_id TEXT NOT NULL,
            snippets_json TEXT NOT NULL,
            tool_trace_json TEXT NOT NULL,
            created_at INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_chat_context_traces_session ON chat_context_traces(session_id, created_at);
        "#,
    )?;
    if !table_has_column(connection, "chat_context_traces", "cortex_trace_json")? {
        connection.execute(
            "ALTER TABLE chat_context_traces ADD COLUMN cortex_trace_json TEXT",
            [],
        )?;
    }
    Ok(())
}

fn table_has_column(connection: &Connection, table: &str, column: &str) -> Result<bool> {
    let mut statement = connection.prepare(&format!("PRAGMA table_info({table})"))?;
    let rows = statement.query_map([], |row| row.get::<_, String>(1))?;
    for row in rows {
        if row? == column {
            return Ok(true);
        }
    }
    Ok(false)
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
            params![
                map.budget_bytes as i64,
                map.serialized,
                to_json(&map.entries)?
            ],
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

fn chat_session_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ChatSession> {
    Ok(ChatSession {
        id: row.get(0)?,
        title: row.get(1)?,
        created_at: row.get::<_, i64>(2)? as u64,
        updated_at: row.get::<_, i64>(3)? as u64,
        hotness: row.get(4)?,
    })
}

fn chat_message_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ChatMessage> {
    Ok(ChatMessage {
        id: row.get(0)?,
        session_id: row.get(1)?,
        role: from_json(row.get::<_, String>(2)?)?,
        content: row.get(3)?,
        created_at: row.get::<_, i64>(4)? as u64,
        token_estimate: row.get::<_, i64>(5)? as usize,
        source_anchor: from_optional_json(row.get::<_, Option<String>>(6)?)?,
    })
}

fn transcript_chunk_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<TranscriptChunk> {
    Ok(TranscriptChunk {
        id: row.get(0)?,
        session_id: row.get(1)?,
        message_id: row.get(2)?,
        ordinal: row.get::<_, i64>(3)? as usize,
        text: row.get(4)?,
        embedding: from_json(row.get::<_, String>(5)?)?,
        embedding_provider: row.get(6)?,
        embedding_model: row.get(7)?,
        embedding_endpoint: row.get(8)?,
        source_anchor: from_json(row.get::<_, String>(9)?)?,
        attention_state: from_json(row.get::<_, String>(10)?)?,
        hotness: row.get(11)?,
        created_at: row.get::<_, i64>(12)? as u64,
    })
}

fn derived_memory_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<DerivedMemory> {
    Ok(DerivedMemory {
        id: row.get(0)?,
        session_id: row.get(1)?,
        kind: from_json(row.get::<_, String>(2)?)?,
        text: row.get(3)?,
        source_message_ids: from_json(row.get::<_, String>(4)?)?,
        actor: row.get(5)?,
        confidence: row.get(6)?,
        created_at: row.get::<_, i64>(7)? as u64,
        provenance: from_json(row.get::<_, String>(8)?)?,
    })
}

fn brain_artifact_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<BrainArtifact> {
    Ok(BrainArtifact {
        id: row.get(0)?,
        kind: from_json(row.get::<_, String>(1)?)?,
        title: row.get(2)?,
        body: row.get(3)?,
        source_refs: from_json(row.get::<_, String>(4)?)?,
        content_hash: row.get(5)?,
        provenance: from_json(row.get::<_, String>(6)?)?,
        confidence: row.get::<_, i64>(7)? as u8,
        created_at: row.get::<_, i64>(8)? as u64,
        updated_at: row.get::<_, i64>(9)? as u64,
    })
}

fn web_finding_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<WebFinding> {
    Ok(WebFinding {
        id: row.get(0)?,
        session_id: row.get(1)?,
        query: row.get(2)?,
        url: row.get(3)?,
        title: row.get(4)?,
        summary: row.get(5)?,
        retrieved_at: row.get::<_, i64>(6)? as u64,
        confidence: row.get(7)?,
        actor: row.get(8)?,
        created_at: row.get::<_, i64>(9)? as u64,
        provenance: from_json(row.get::<_, String>(10)?)?,
    })
}

fn attention_mark_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<AttentionMark> {
    Ok(AttentionMark {
        id: row.get(0)?,
        target_id: row.get(1)?,
        target_kind: from_json(row.get::<_, String>(2)?)?,
        action: from_json(row.get::<_, String>(3)?)?,
        reason: row.get(4)?,
        actor: row.get(5)?,
        created_at: row.get::<_, i64>(6)? as u64,
        reverted_at: row.get::<_, Option<i64>>(7)?.map(|value| value as u64),
    })
}

fn memory_access_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<MemoryAccess> {
    Ok(MemoryAccess {
        id: row.get(0)?,
        target_id: row.get(1)?,
        target_kind: from_json(row.get::<_, String>(2)?)?,
        access_kind: from_json(row.get::<_, String>(3)?)?,
        reason: row.get(4)?,
        actor: row.get(5)?,
        accessed_at: row.get::<_, i64>(6)? as u64,
    })
}

fn cortex_adapter_job_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<CortexAdapterJob> {
    Ok(CortexAdapterJob {
        id: row.get(0)?,
        status: row.get(1)?,
        source_dataset_hash: row.get(2)?,
        prepared_dataset_hash: row.get(3)?,
        base_model: row.get(4)?,
        adapter_output_path: row.get(5)?,
        manifest_path: row.get(6)?,
        train_records: row.get::<_, Option<i64>>(7)?.map(|value| value as usize),
        valid_records: row.get::<_, Option<i64>>(8)?.map(|value| value as usize),
        test_records: row.get::<_, Option<i64>>(9)?.map(|value| value as usize),
        iters: row.get::<_, Option<i64>>(10)?.map(|value| value as usize),
        command: from_json(row.get::<_, String>(11)?)?,
        log_path: row.get(12)?,
        failure_reason: row.get(13)?,
        payload: from_json(row.get::<_, String>(14)?)?,
        created_at: row.get::<_, i64>(15)? as u64,
        updated_at: row.get::<_, i64>(16)? as u64,
        started_at: row.get::<_, Option<i64>>(17)?.map(|value| value as u64),
        finished_at: row.get::<_, Option<i64>>(18)?.map(|value| value as u64),
    })
}

fn audit_event_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<AuditEvent> {
    Ok(AuditEvent {
        id: row.get(0)?,
        session_id: row.get(1)?,
        event_type: row.get(2)?,
        target_id: row.get(3)?,
        actor: row.get(4)?,
        payload_json: row.get(5)?,
        created_at: row.get::<_, i64>(6)? as u64,
    })
}

fn chat_context_trace_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ChatContextTrace> {
    Ok(ChatContextTrace {
        id: row.get(0)?,
        session_id: row.get(1)?,
        user_message_id: row.get(2)?,
        snippets: from_json(row.get::<_, String>(3)?)?,
        tool_trace: from_json(row.get::<_, String>(4)?)?,
        created_at: row.get::<_, i64>(5)? as u64,
        cortex_trace: from_optional_json(row.get::<_, Option<String>>(6)?)?,
    })
}

fn collect_rows<T>(
    rows: rusqlite::MappedRows<'_, impl FnMut(&rusqlite::Row<'_>) -> rusqlite::Result<T>>,
) -> Result<Vec<T>> {
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
    serde_json::from_str(&raw)
        .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))
}

fn from_optional_json<T: serde::de::DeserializeOwned>(
    raw: Option<String>,
) -> rusqlite::Result<Option<T>> {
    raw.map(from_json).transpose()
}
