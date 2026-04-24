// ========== DOSYA: sentinel-intelligence/src/main.rs ==========
use anyhow::{Context, Result};
use futures_util::StreamExt;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;
use phf::phf_map;
use prost::Message;
use std::sync::Arc;
use std::time::Instant;
use tokenizers::Tokenizer;
use tokio::sync::RwLock;
use tokio::time::{timeout, Duration};
use tonic::{transport::Server, Request, Response, Status};
use tracing::{error, info, warn};

pub mod sentinel_protos {
    pub mod intelligence {
        include!(concat!(env!("OUT_DIR"), "/sentinel.intelligence.v1.rs"));
    }
    pub mod market {
        include!(concat!(env!("OUT_DIR"), "/sentinel.market.v1.rs"));
    }
}

use sentinel_protos::intelligence::sentiment_analyzer_service_server::{
    SentimentAnalyzerService, SentimentAnalyzerServiceServer,
};
use sentinel_protos::intelligence::{AnalyzeTextRequest, AnalyzeTextResponse, SemanticVector};
use sentinel_protos::market::RawNewsEvent;

// ==============================================================================
// 1. TIER 1: ONNX TENSOR ENGINE (C++ / RUST ZERO-LATENCY BINDING)
// ==============================================================================

struct OnnxEngine {
    session: RwLock<Session>,
    tokenizer: Tokenizer,
    pos_id: usize,
    neg_id: usize,
}

impl OnnxEngine {
    fn load() -> Result<Self> {
        let _ = ort::init().with_name("vq_capital_onnx").commit();

        let model_path =
            std::env::var("MODEL_PATH").unwrap_or_else(|_| "/opt/models/model.onnx".to_string());
        let tokenizer_path = std::env::var("TOKENIZER_PATH")
            .unwrap_or_else(|_| "/opt/models/tokenizer.json".to_string());

        info!(
            "⏳ [TIER-1] ONNX Modeli yükleniyor (Local Baked): {}",
            model_path
        );

        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow::anyhow!(e))?;

        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("Session Builder Error: {}", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("Optimization Error: {}", e))?
            .with_intra_threads(1)
            .map_err(|e| anyhow::anyhow!("Thread Config Error: {}", e))?
            .commit_from_file(&model_path)
            .map_err(|e| anyhow::anyhow!("Model Load Error: {}", e))?;

        let pos_id = 2;
        let neg_id = 0;

        info!("✅ [TIER-1] ONNX Tensor Engine Başarıyla Yüklendi!");

        Ok(Self {
            session: RwLock::new(session),
            tokenizer,
            pos_id,
            neg_id,
        })
    }

    // FIX: &String yerine &str (HFT Best Practice)
    async fn predict(&self, text: &str) -> Result<f64> {
        let tokens = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!(e))?;
        let token_ids = tokens.get_ids();
        let seq_len = std::cmp::min(token_ids.len(), 128);

        let token_ids_i64: Vec<i64> = token_ids[..seq_len].iter().map(|&x| x as i64).collect();
        let attention_mask_i64: Vec<i64> = vec![1i64; seq_len];
        let token_type_ids_i64: Vec<i64> = vec![0i64; seq_len];

        let shape = [1_usize, seq_len];

        let input_ids_val = Value::from_array((shape, token_ids_i64))
            .map_err(|e| anyhow::anyhow!("Input IDs Tensor Error: {}", e))?;
        let attention_mask_val = Value::from_array((shape, attention_mask_i64))
            .map_err(|e| anyhow::anyhow!("Mask Tensor Error: {}", e))?;
        let token_type_ids_val = Value::from_array((shape, token_type_ids_i64))
            .map_err(|e| anyhow::anyhow!("Token Type Tensor Error: {}", e))?;

        let inputs_vec = ort::inputs![
            "input_ids" => input_ids_val,
            "attention_mask" => attention_mask_val,
            "token_type_ids" => token_type_ids_val,
        ];

        let mut session_guard = self.session.write().await;
        let outputs = session_guard
            .run(inputs_vec)
            .map_err(|e| anyhow::anyhow!("ONNX Runtime Execution Error: {}", e))?;

        let extracted = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("Logits Extraction Error: {}", e))?;

        let logits_slice = extracted.1;

        if logits_slice.len() >= 3 {
            let neg_logit = logits_slice[self.neg_id];
            let pos_logit = logits_slice[self.pos_id];
            let neutral_logit = logits_slice[1];

            let max_val = neg_logit.max(pos_logit).max(neutral_logit);
            let sum_exp = (neg_logit - max_val).exp()
                + (pos_logit - max_val).exp()
                + (neutral_logit - max_val).exp();
            let pos_prob = (pos_logit - max_val).exp() / sum_exp;
            let neg_prob = (neg_logit - max_val).exp() / sum_exp;

            Ok((pos_prob - neg_prob) as f64)
        } else {
            Err(anyhow::anyhow!(
                "ONNX Model çıkışı beklenenden farklı boyutta!"
            ))
        }
    }
}

// ==============================================================================
// 2. TIER 0: ULTRA-FAST PATH LEXICON (O(1) Hashmap)
// ==============================================================================

static EXTREME_SIGNALS: phf::Map<&'static str, f64> = phf_map! {
    "hack" => -1.0, "hacked" => -1.0, "exploit" => -1.0, "exploiter" => -1.0,
    "bankruptcy" => -1.0, "bankrupt" => -1.0, "insolvent" => -1.0,
    "sec" => -0.8, "lawsuit" => -0.9, "probe" => -0.8, "investigation" => -0.8,
    "delist" => -1.0, "delisted" => -1.0, "arrest" => -1.0, "arrested" => -1.0,
    "scam" => -1.0, "rugpull" => -1.0, "rug" => -1.0,
    "approval" => 1.0, "approved" => 1.0, "partnership" => 0.8, "partners" => 0.8,
};

static STANDARD_LEXICON: phf::Map<&'static str, f64> = phf_map! {
    "bullish" => 0.8, "moon" => 0.9, "breakout" => 0.7, "surge" => 0.8, "surges" => 0.8,
    "rally" => 0.7, "adoption" => 0.6, "inflows" => 0.7, "accumulate" => 0.7,
    "bearish" => -0.8, "crash" => -0.9, "crashes" => -0.9, "dump" => -0.8, "dumps" => -0.8,
    "resistance" => -0.5, "slumps" => -0.7, "plunge" => -0.8, "plunges" => -0.8,
};

// ==============================================================================
// 3. MULTI-TIER ORCHESTRATOR & SLA WATCHDOG
// ==============================================================================

pub struct VQIntelligenceCore {
    onnx_engine: Arc<Option<OnnxEngine>>,
}

impl Default for VQIntelligenceCore {
    fn default() -> Self {
        Self::build()
    }
}

impl VQIntelligenceCore {
    pub fn build() -> Self {
        info!("🤖 VQ-Capital Multi-Tier Engine Başlatılıyor.");

        let onnx_engine = match OnnxEngine::load() {
            Ok(model) => Some(model),
            Err(e) => {
                error!("🚨 [ONNX HATASI] Model yüklenemedi: {}", e);
                warn!("⚠️ TIER-1 Devre Dışı! Sadece TIER-0 (O(1) Nanosecond Lexicon) çalışacak.");
                None
            }
        };

        Self {
            onnx_engine: Arc::new(onnx_engine),
        }
    }

    pub async fn analyze_async(&self, text: &str) -> (f64, &'static str) {
        let start_time = Instant::now();
        let lower_text = text.to_lowercase();

        for word in lower_text.split_whitespace() {
            let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric());
            if let Some(&score) = EXTREME_SIGNALS.get(clean_word) {
                return (score, "TIER-0 (EXTREME)");
            }
        }

        if let Some(engine) = self.onnx_engine.as_ref() {
            let result = timeout(Duration::from_millis(4), engine.predict(text)).await;

            match result {
                Ok(Ok(ml_score)) => {
                    let _eval_micros = start_time.elapsed().as_micros();
                    return (ml_score, "TIER-1 (ONNX)");
                }
                Ok(Err(e)) => warn!("⚠️ ONNX Çıkarım Hatası: {}. Tier-0'a Düşülüyor.", e),
                Err(_) => warn!("⏳ [SLA İHLALİ] Model 4ms'yi aştı. İşlem Timeout oldu (Aborted)."),
            }
        }

        let mut lexicon_score = 0.0;
        let mut match_count = 0;
        for word in lower_text.split_whitespace() {
            let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric());
            if let Some(&score) = STANDARD_LEXICON.get(clean_word) {
                lexicon_score += score;
                match_count += 1;
            }
        }

        let final_score = if match_count > 0 {
            (lexicon_score / match_count as f64).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        (final_score, "TIER-0 (LEXICON)")
    }

    pub fn extract_symbol(text_upper: &str) -> Option<&'static str> {
        if text_upper.contains("BITCOIN")
            || text_upper.contains("BTC ")
            || text_upper.contains(" BTC")
        {
            return Some("BTCUSDT");
        }
        if text_upper.contains("ETHEREUM")
            || text_upper.contains("ETHER")
            || text_upper.contains("ETH ")
        {
            return Some("ETHUSDT");
        }
        if text_upper.contains("SOLANA")
            || text_upper.contains("SOL ")
            || text_upper.contains(" SOL")
        {
            return Some("SOLUSDT");
        }
        if text_upper.contains("BINANCE")
            || text_upper.contains("BNB ")
            || text_upper.contains(" BNB")
        {
            return Some("BNBUSDT");
        }
        None
    }
}

#[tonic::async_trait]
impl SentimentAnalyzerService for VQIntelligenceCore {
    async fn analyze_text(
        &self,
        request: Request<AnalyzeTextRequest>,
    ) -> Result<Response<AnalyzeTextResponse>, Status> {
        let (final_score, _) = self.analyze_async(&request.into_inner().text).await;
        Ok(Response::new(AnalyzeTextResponse { score: final_score }))
    }
}

// ==============================================================================
// 4. MAIN RUNTIME
// ==============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let nats_url =
        std::env::var("NATS_URL").unwrap_or_else(|_| "nats://localhost:4222".to_string());
    let nats_client = async_nats::connect(&nats_url)
        .await
        .context("CRITICAL: NATS bağlanılamadı")?;

    let ai_service = tokio::task::spawn_blocking(VQIntelligenceCore::build)
        .await
        .context("AI Builder Panic Hatası")?;

    let ai_arc = Arc::new(ai_service);
    let grpc_ai = ai_arc.clone();
    let nats_pub = nats_client.clone();

    tokio::spawn(async move {
        if let Ok(mut sub) = nats_client.subscribe("news.raw.>").await {
            info!("📡 Multi-Tier AI Worker: Haber Akışına Bağlandı.");

            while let Some(msg) = sub.next().await {
                if let Ok(raw_news) = RawNewsEvent::decode(msg.payload) {
                    let text_upper = raw_news.headline.to_uppercase();

                    let symbol = match VQIntelligenceCore::extract_symbol(&text_upper) {
                        Some(s) => s,
                        None => continue,
                    };

                    let start_eval = Instant::now();
                    let (score, tier_used) = ai_arc.analyze_async(&raw_news.headline).await;
                    let eval_time = start_eval.elapsed().as_micros();

                    if score.abs() < 0.05 {
                        continue;
                    }

                    let vector = SemanticVector {
                        symbol: symbol.to_string(),
                        sentiment_score: score,
                        source: raw_news.source,
                        original_headline: raw_news.headline,
                        timestamp: chrono::Utc::now().timestamp_millis(),
                    };

                    let mut buf = Vec::new();
                    if vector.encode(&mut buf).is_ok() {
                        let _ = nats_pub
                            .publish("intelligence.news.vector".to_string(), buf.into())
                            .await;
                        let direction = if score > 0.0 {
                            "🟢 BOĞA"
                        } else {
                            "🔴 AYI"
                        };

                        info!(
                            "⚡ [{}] {} {} (Skor: {:.2}, Hız: {}µs) | {}",
                            tier_used,
                            symbol,
                            direction,
                            score,
                            eval_time,
                            vector.original_headline
                        );
                    }
                }
            }
        }
    });

    let addr = "0.0.0.0:50051".parse()?;
    info!("⚡ Sentinel-Intelligence gRPC dinliyor: {}", addr);

    let service_clone = VQIntelligenceCore {
        onnx_engine: grpc_ai.onnx_engine.clone(),
    };

    Server::builder()
        .add_service(SentimentAnalyzerServiceServer::new(service_clone))
        .serve(addr)
        .await?;

    Ok(())
}
