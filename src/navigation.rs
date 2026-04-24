use crate::types::*;
use std::collections::HashMap;

pub trait Navigator {
    fn start_session(&self, seed: Option<NodeRef>) -> SessionState;
    fn open(&self, session: &mut SessionState, node: NodeRef);
    fn backtrack(&self, session: &mut SessionState) -> Option<NodeRef>;
    fn list_links(&self, memory: &PersistedMemory, node: &NodeRef) -> Vec<Link>;
    fn step(&self, memory: &PersistedMemory, session: &mut SessionState, link_id: &str) -> Option<NodeRef>;
}

#[derive(Debug, Clone, Default)]
pub struct MemoryNavigator;

impl Navigator for MemoryNavigator {
    fn start_session(&self, seed: Option<NodeRef>) -> SessionState {
        let mut state = SessionState {
            id: "session-1".into(),
            current: None,
            history: Vec::new(),
            visited: Vec::new(),
        };
        if let Some(node) = seed {
            self.open(&mut state, node);
        }
        state
    }

    fn open(&self, session: &mut SessionState, node: NodeRef) {
        if let Some(current) = session.current.take() {
            session.history.push(current);
        }
        session.visited.push(node.clone());
        session.current = Some(node);
    }

    fn backtrack(&self, session: &mut SessionState) -> Option<NodeRef> {
        let previous = session.history.pop()?;
        if let Some(current) = session.current.take() {
            session.visited.push(current);
        }
        session.current = Some(previous.clone());
        Some(previous)
    }

    fn list_links(&self, memory: &PersistedMemory, node: &NodeRef) -> Vec<Link> {
        let mut links = memory
            .links
            .iter()
            .filter(|link| &link.source == node || &link.target == node)
            .cloned()
            .collect::<Vec<_>>();

        if let NodeRef::Chunk(chunk_id) = node {
            if let Some(chunk) = memory.chunks.iter().find(|chunk| &chunk.id == chunk_id) {
                links.push(Link {
                    id: format!("open-document:{chunk_id}"),
                    source: node.clone(),
                    target: NodeRef::Document(chunk.document_id.clone()),
                    link_type: LinkType::SameDocument,
                    score: 1.0,
                    label: "open full document".into(),
                });
                links.push(Link {
                    id: format!("open-region:{chunk_id}"),
                    source: node.clone(),
                    target: NodeRef::Region(chunk.region_id.clone()),
                    link_type: LinkType::RegionMembership,
                    score: 1.0,
                    label: "return to region".into(),
                });
            }
        }

        if let NodeRef::Document(document_id) = node {
            let chunks = memory
                .chunks
                .iter()
                .filter(|chunk| &chunk.document_id == document_id)
                .take(3)
                .collect::<Vec<_>>();
            for chunk in chunks {
                links.push(Link {
                    id: format!("doc-first:{}:{}", document_id, chunk.id),
                    source: node.clone(),
                    target: NodeRef::Chunk(chunk.id.clone()),
                    link_type: LinkType::SameDocument,
                    score: 0.95,
                    label: "document chunk".into(),
                });
            }
        }

        let mut deduped = HashMap::new();
        for link in links {
            deduped.entry(link.id.clone()).or_insert(link);
        }
        deduped.into_values().collect()
    }

    fn step(&self, memory: &PersistedMemory, session: &mut SessionState, link_id: &str) -> Option<NodeRef> {
        let links = self.list_links(memory, session.current.as_ref()?);
        let link = links.into_iter().find(|link| link.id == link_id)?;
        let next = if link.source == session.current.clone()? {
            link.target
        } else {
            link.source
        };
        self.open(session, next.clone());
        Some(next)
    }
}

