use std::path::Path;
use rusqlite::Connection;
use crate::models::RetrievedChunk;

pub struct LowLevelRetriever;

impl LowLevelRetriever {
    /// Computes dot product between two float slices
    #[inline(always)]
    fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let mut sum = 0.0f32;
        for i in 0..len {
            sum += a[i] * b[i];
        }
        sum
    }

    /// Normalizes a float slice in-place
    fn normalize(vec: &mut [f32]) {
        let sum_sq: f32 = vec.iter().map(|v| v * v).sum();
        let norm = sum_sq.sqrt();
        if norm > 0.0 {
            for v in vec.iter_mut() {
                *v /= norm;
            }
        }
    }

    pub fn search_rag(
        db_path: &Path,
        rag_id: &str,
        query_vec: &[f32],
        query_text: &str,
        top_k: usize,
        dim: usize,
    ) -> Result<Vec<RetrievedChunk>, Box<dyn std::error::Error>> {
        let conn = Connection::open(db_path)?;

        let mut stmt = conn.prepare(
            "SELECT c.id, c.document_id, c.rag_id, c.modality, c.content, c.page_number, c.asset_rel_path,
                    c.embedding_blob, d.filename
             FROM chunks c
             LEFT JOIN documents d ON c.document_id = d.id
             WHERE c.embedding_blob IS NOT NULL"
        )?;

        let query_lower = query_text.to_lowercase();
        let query_keywords: Vec<&str> = query_lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 2)
            .collect();

        let mut scored_chunks: Vec<RetrievedChunk> = Vec::new();

        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let chunk_id: String = row.get(0)?;
            let _doc_id: String = row.get::<_, Option<String>>(1)?.unwrap_or_default();
            let row_rag_id: String = row.get::<_, Option<String>>(2)?.unwrap_or_else(|| rag_id.to_string());
            let modality: String = row.get::<_, Option<String>>(3)?.unwrap_or_else(|| "text".to_string());
            let content: String = row.get(4)?;
            let page_number: Option<i64> = row.get(5)?;
            let asset_rel_path: Option<String> = row.get(6)?;
            let blob: Vec<u8> = row.get(7)?;
            let filename: String = row.get::<_, Option<String>>(8)?.unwrap_or_else(|| "manual.pdf".to_string());

            if blob.len() < dim * 4 {
                continue;
            }

            // Decode f32 elements from bytes (Little-Endian IEEE 754)
            let mut chunk_vec = Vec::with_capacity(dim);
            for i in 0..dim {
                let byte_idx = i * 4;
                let bytes = [
                    blob[byte_idx],
                    blob[byte_idx + 1],
                    blob[byte_idx + 2],
                    blob[byte_idx + 3],
                ];
                chunk_vec.push(f32::from_le_bytes(bytes));
            }

            // Normalize for Matryoshka sub-dimensions
            Self::normalize(&mut chunk_vec);

            let v_score = Self::dot_product(query_vec, &chunk_vec);

            // Fast keyword match bonus
            let mut kw_score = 0.0f32;
            if !query_keywords.is_empty() {
                let content_lower = content.to_lowercase();
                let file_lower = filename.to_lowercase();
                let mut hits = 0;
                for kw in &query_keywords {
                    if content_lower.contains(kw) || file_lower.contains(kw) {
                        hits += 1;
                    }
                }
                kw_score = (hits as f32) / (query_keywords.len() as f32);
            }

            // Combined scoring: 70% vector semantic + 30% exact keyword presence
            let final_score = (v_score * 0.70) + (kw_score * 0.30);

            let page_str = match page_number {
                Some(p) => format!(", Pàg. {}", p),
                None => "".to_string(),
            };
            let citation = format!("[📖 Manual: {}{}]", filename, page_str);

            scored_chunks.push(RetrievedChunk {
                chunk_id,
                rag_id: row_rag_id,
                modality,
                content,
                page_number,
                source_path: filename,
                asset_rel_path,
                score: final_score,
                citation,
            });
        }

        // Sort descending by score
        scored_chunks.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored_chunks.truncate(top_k);

        Ok(scored_chunks)
    }
}
