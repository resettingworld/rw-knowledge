use std::time::Instant;
use crate::catalog::RagCatalog;
use crate::models::{ScoredRag, TriageResult};

pub struct FastTriager;

impl FastTriager {
    pub fn triage(catalog: &RagCatalog, query: &str) -> TriageResult {
        let start = Instant::now();
        let query_lower = query.to_lowercase();
        let stop_words = [
            "com", "que", "què", "los", "las", "les", "els", "per", "por", "para",
            "con", "amb", "del", "dels", "una", "uno", "unes", "unos", "puc", "tinc",
            "estic", "estoy", "como", "how", "what", "can", "the", "and", "sobre",
            "entre", "quin", "quina", "quins", "quines", "aixo", "esto", "aquest",
            "miral", "mirall", "algun", "alguna", "trobar", "buscar", "obtener", "obtenir"
        ];

        let query_tokens: Vec<&str> = query_lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 2 && !stop_words.contains(w))
            .collect();

        // 1. Urgency & Category Detection (Sistema 1 - Rule-based fast heuristic)
        let medical_critical = [
            "hemorragia", "arterial", "sangrado", "torniquete", "torniquet",
            "rcp", "paro", "asfixia", "fractura", "veneno", "mordedura", "serpiente",
            "quemadura", "infeccion", "shock", "herida", "ahogamiento", "auxilio", "auxilis"
        ];
        let survival_urgent = [
            "agua", "aigua", "sed", "potable", "purificar", "hongo", "bolet", "seta",
            "planta", "comestible", "refugio", "frio", "hipotermia", "fuego", "foc",
            "selva", "jungla", "bosque", "bosc", "desierto", "desert", "mar", "naufragio",
            "supervivencia", "supervivència", "bushcraft", "mochila", "evacuacion", "sismo", "terremoto"
        ];
        let algorithm_keywords = [
            "algoritmo", "algoritme", "algoritmos", "algoritmes", "arbol", "arboles", "arbre", "arbres",
            "grafo", "grafos", "graf", "grafs", "busqueda", "cerca", "ordenamiento", "ordenacio",
            "complejidad", "complexitat", "big-o", "dijkstra", "dinamica", "recursivo", "recursivitat",
            "programacion", "programacio", "python", "rust", "estructures"
        ];

        let mut urgency = "NORMAL".to_string();
        let mut category = "GENERAL".to_string();

        for word in &medical_critical {
            if query_lower.contains(word) {
                urgency = "CRITICAL".to_string();
                category = "MEDICINA_PRIMEROS_AUXILIOS".to_string();
                break;
            }
        }

        if urgency == "NORMAL" {
            for word in &survival_urgent {
                if query_lower.contains(word) {
                    urgency = "HIGH".to_string();
                    category = "SUPERVIVENCIA_RECURSOS".to_string();
                    break;
                }
            }
        }

        if category == "GENERAL" {
            for word in &algorithm_keywords {
                if query_lower.contains(word) {
                    urgency = "LOW".to_string();
                    category = "ALGORITMOS_CIENCIAS".to_string();
                    break;
                }
            }
        }

        // 2. Score Available RAGs
        let mut scored_rags = Vec::new();

        for manifest in catalog.list_rags() {
            let mut score = 0.0f32;
            let mut matched_tags = Vec::new();

            // Tag exact/substring matches (requiring length > 3 for substring)
            for tag in &manifest.tags {
                let tag_lower = tag.to_lowercase();
                for token in &query_tokens {
                    let is_exact = tag_lower == *token;
                    let is_part = tag_lower.split('-').any(|part| part == *token);
                    let is_substr = token.len() >= 4 && tag_lower.contains(token);
                    if is_exact || is_part || is_substr {
                        score += if is_exact { 5.0 } else { 3.0 };
                        if !matched_tags.contains(tag) {
                            matched_tags.push(tag.clone());
                        }
                    }
                }
            }

            // Description and name matches
            let desc_lower = manifest.description.to_lowercase();
            let name_lower = manifest.name.to_lowercase();

            for token in &query_tokens {
                if desc_lower.contains(token) {
                    score += 1.5;
                }
                if name_lower.contains(token) {
                    score += 2.5;
                }
            }

            // Domain bonuses based on detected category
            match category.as_str() {
                "MEDICINA_PRIMEROS_AUXILIOS" => {
                    if manifest.rag_id == "supervivencia-es" || manifest.rag_id.contains("health") {
                        score += 20.0;
                    }
                }
                "SUPERVIVENCIA_RECURSOS" => {
                    if manifest.rag_id == "supervivencia-es" {
                        score += 20.0;
                    } else if manifest.rag_id == "ciencias-naturales" {
                        score += 5.0;
                    }
                }
                "ALGORITMOS_CIENCIAS" => {
                    if manifest.rag_id == "algoritmo" || manifest.rag_id.contains("comput") {
                        score += 20.0;
                    }
                }
                _ => {}
            }

            // Base threshold or fallback
            if score > 0.0 {
                scored_rags.push(ScoredRag {
                    rag_id: manifest.rag_id.clone(),
                    name: manifest.name.clone(),
                    score,
                    matched_tags,
                });
            }
        }

        // Sort descending by score
        scored_rags.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // If no RAG scored above 0, fallback to default RAGs (e.g. supervivencia-es)
        if scored_rags.is_empty() {
            if let Some(def) = catalog.get_rag("supervivencia-es") {
                scored_rags.push(ScoredRag {
                    rag_id: def.rag_id.clone(),
                    name: def.name.clone(),
                    score: 1.0,
                    matched_tags: vec!["fallback".to_string()],
                });
            } else if let Some(first) = catalog.list_rags().first() {
                scored_rags.push(ScoredRag {
                    rag_id: first.rag_id.clone(),
                    name: first.name.clone(),
                    score: 0.5,
                    matched_tags: vec!["default".to_string()],
                });
            }
        }

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        TriageResult {
            query: query.to_string(),
            urgency,
            category,
            selected_rags: scored_rags,
            decision_latency_ms: elapsed,
        }
    }
}
