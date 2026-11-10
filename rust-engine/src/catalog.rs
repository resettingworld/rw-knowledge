use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use rusqlite::Connection;
use crate::models::RagManifest;

#[derive(Debug, Clone)]
pub struct RagCatalog {
    pub base_path: PathBuf,
    pub rags: HashMap<String, RagManifest>,
}

impl RagCatalog {
    pub fn load_from_dir<P: AsRef<Path>>(base_path: P) -> Self {
        let base = base_path.as_ref().to_path_buf();
        let mut rags = HashMap::new();

        if let Ok(entries) = fs::read_dir(&base) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    let manifest_file = path.join("manifest.json");
                    let db_file = path.join("index.db");

                    if manifest_file.exists() && db_file.exists() {
                        if let Ok(content) = fs::read_to_string(&manifest_file) {
                            if let Ok(mut manifest) = serde_json::from_str::<RagManifest>(&content) {
                                // If manifest has 0 counts, check SQLite directly
                                if manifest.total_chunks == 0 {
                                    if let Ok(conn) = Connection::open(&db_file) {
                                        if let Ok(count) = conn.query_row(
                                            "SELECT count(*) FROM chunks",
                                            [],
                                            |row| row.get::<_, usize>(0)
                                        ) {
                                            manifest.total_chunks = count;
                                        }
                                    }
                                }
                                rags.insert(manifest.rag_id.clone(), manifest);
                            }
                        }
                    }
                }
            }
        }

        Self {
            base_path: base,
            rags,
        }
    }

    pub fn list_rags(&self) -> Vec<RagManifest> {
        let mut list: Vec<RagManifest> = self.rags.values().cloned().collect();
        list.sort_by(|a, b| a.rag_id.cmp(&b.rag_id));
        list
    }

    pub fn get_rag(&self, rag_id: &str) -> Option<&RagManifest> {
        self.rags.get(rag_id)
    }

    pub fn get_db_path(&self, rag_id: &str) -> Option<PathBuf> {
        if self.rags.contains_key(rag_id) {
            Some(self.base_path.join(rag_id).join("index.db"))
        } else {
            None
        }
    }

    #[allow(dead_code)]
    pub fn get_assets_path(&self, rag_id: &str) -> Option<PathBuf> {
        if self.rags.contains_key(rag_id) {
            Some(self.base_path.join(rag_id).join("assets"))
        } else {
            None
        }
    }
}
