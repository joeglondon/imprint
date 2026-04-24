use crate::types::*;
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct MapBuilder {
    budget_bytes: usize,
}

impl MapBuilder {
    pub fn new(budget_bytes: usize) -> Self {
        Self { budget_bytes }
    }

    pub fn build(&self, regions: &[Region]) -> MemoryMap {
        let mut entries = regions
            .iter()
            .map(|region| MapEntry {
                region_id: region.id.clone(),
                label: truncate(&region.label, 18),
                summary: truncate(&region.summary, 64),
                filters: region.filters.clone(),
            })
            .collect::<Vec<_>>();
        loop {
            let serialized = compact_json(&entries);
            if serialized.len() <= self.budget_bytes || entries.is_empty() {
                return MemoryMap {
                    budget_bytes: self.budget_bytes,
                    serialized,
                    entries,
                };
            }
            if entries.len() > 3 {
                entries.pop();
                continue;
            }
            for entry in &mut entries {
                entry.summary = truncate(&entry.summary, entry.summary.len().saturating_sub(8));
                trim_filters(&mut entry.filters);
            }
        }
    }
}

impl Default for MapBuilder {
    fn default() -> Self {
        Self::new(2_048)
    }
}

fn compact_json(entries: &[MapEntry]) -> String {
    #[derive(Serialize)]
    struct Wire<'a> {
        entries: &'a [MapEntry],
    }
    serde_json::to_string(&Wire { entries }).unwrap_or_else(|_| "{\"entries\":[]}".into())
}

fn trim_filters(filters: &mut BTreeMap<String, String>) {
    if let Some((key, value)) = filters.iter().next().map(|(k, v)| (k.clone(), v.clone())) {
        filters.clear();
        filters.insert(truncate(&key, 8), truncate(&value, 12));
    }
}

fn truncate(value: &str, max: usize) -> String {
    value.chars().take(max.max(1)).collect()
}

