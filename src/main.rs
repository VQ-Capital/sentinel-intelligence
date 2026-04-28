// ========== DOSYA: sentinel-intelligence/src/main.rs ==========
use anyhow::{Context, Result};
use futures_util::StreamExt;
use phf::phf_map;
use prost::bytes::BytesMut;
use prost::Message;
use std::sync::Arc;
use std::time::Instant;
use tokenizers::Tokenizer;
use tokio::time::{timeout, Duration};
use tonic::{transport::Server, Request, Response, Status};
use tracing::{error, info, warn};
use tract_onnx::prelude::*;

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

// -----------------------------------------------------------------------------
// ⚡ TIER-0: ULTRA-FAST LEXICON
// -----------------------------------------------------------------------------
static FAST_PATH_WORDS: phf::Map<&'static str, f64> = phf_map! {
    "hack" => -1.0, "hacked" => -1.0, "exploit" => -1.0, "exploiter" => -1.0,
    "bankruptcy" => -1.0, "bankrupt" => -1.0, "freeze" => -0.9, "frozen" => -0.9,
    "insolvent" => -0.9, "liquidated" => -0.8, "scam" => -1.0, "rugpull" => -1.0,
    "ban" => -1.0, "banned" => -1.0, "arrest" => -0.9, "arrested" => -0.9,
    "sec" => -0.4, "lawsuit" => -0.6, "sued" => -0.6, "probe" => -0.5,
    "approval" => 1.0, "approved" => 1.0, "etf" => 0.7, "listing" => 0.5,
    "partnership" => 0.8, "integrated" => 0.6, "breakout" => 0.4,
};

// -----------------------------------------------------------------------------
// 🤖 TIER-1: PURE RUST NEURAL ENGINE (Tract / Lock-Free)
// -----------------------------------------------------------------------------
struct NeuralBrain {
    // 🔥 CERRAHİ: TypedSimplePlan Send+Sync'tir. Mutex'e gerek yoktur!
    model: TypedSimplePlan<TypedModel>,
    tokenizer: Tokenizer,
}

impl NeuralBrain {
    fn predict_sync(&self, text: &str) -> Result<f64> {
        let tokens = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;

        let token_ids: Vec<i64> = tokens.get_ids().iter().map(|&x| x as i64).collect();
        let seq_len = token_ids.len();

        // Tract için Tensörleri Hazırla (Shape: [1, seq_len])
        let input_ids = tract_ndarray::Array2::from_shape_vec((1, seq_len), token_ids.clone())
            .context("Input_ids array shape error")?
            .into_tensor();

        let mask = tract_ndarray::Array2::from_shape_vec((1, seq_len), vec![1i64; seq_len])
            .context("Mask array shape error")?
            .into_tensor();

        let type_ids = tract_ndarray::Array2::from_shape_vec((1, seq_len), vec![0i64; seq_len])
            .context("Type_ids array shape error")?
            .into_tensor();

        // Optimum ile export edilen modelin giriş sırası: input_ids, attention_mask, token_type_ids
        let inputs = tvec![input_ids.into(), mask.into(), type_ids.into()];

        // Kilit (Mutex) olmadan paralel run() çalıştırılıyor!
        let outputs = self.model.run(inputs).context("Tract Execution Error")?;

        let logits_view = outputs[0].to_array_view::<f32>()?;
        let slice = logits_view.as_slice().context("Logits extraction error")?;

        if slice.len() >= 3 {
            // [Negative, Neutral, Positive]
            Ok((slice[2] - slice[0]) as f64)
        } else {
            Ok(0.0)
        }
    }
}

// -----------------------------------------------------------------------------
// 🏢 SERVICE IMPLEMENTATION
// -----------------------------------------------------------------------------
pub struct VQIntelligence {
    brain: Arc<NeuralBrain>,
    sla_timeout_ms: u64,
}

#[tonic::async_trait]
impl SentimentAnalyzerService for VQIntelligence {
    async fn analyze_text(
        &self,
        request: Request<AnalyzeTextRequest>,
    ) -> Result<Response<AnalyzeTextResponse>, Status> {
        let text = request.into_inner().text;
        let (score, _) = self.process_full_stack(text).await;
        Ok(Response::new(AnalyzeTextResponse { score }))
    }
}

impl VQIntelligence {
    async fn process_full_stack(&self, text: String) -> (f64, &'static str) {
        // TIER-0: O(1) Lexicon Fast-Path
        for word in text.to_lowercase().split_whitespace() {
            let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
            if let Some(&score) = FAST_PATH_WORDS.get(clean) {
                return (score, "TIER-0 (LEXICON)");
            }
        }

        // TIER-1: Pure Rust Neural Predict
        let brain_clone = self.brain.clone();
        let ai_result = timeout(
            Duration::from_millis(self.sla_timeout_ms),
            tokio::task::spawn_blocking(move || brain_clone.predict_sync(&text)),
        )
        .await;

        match ai_result {
            Ok(Ok(Ok(score))) => (score, "TIER-1 (RUST-NEURAL)"),
            Ok(Ok(Err(e))) => {
                error!("Neural Prediction Error: {}", e);
                (0.0, "NEURAL-ERROR")
            }
            Ok(Err(e)) => {
                error!("Thread Spawn Error: {}", e);
                (0.0, "THREAD-ERROR")
            }
            Err(_) => {
                warn!(
                    "⏳ [SLA-VIOLATION] AI took > {}ms! GRACEFUL DEGRADATION ACTIVATED.",
                    self.sla_timeout_ms
                );
                (0.0, "SLA-TIMEOUT")
            }
        }
    }
}

fn extract_target_symbol(text: &str) -> Option<&'static str> {
    let t = text.to_uppercase();
    if t.contains("BITCOIN") || t.contains("BTC") {
        return Some("BTCUSDT");
    }
    if t.contains("ETHEREUM") || t.contains("ETH") {
        return Some("ETHUSDT");
    }
    if t.contains("SOLANA") || t.contains("SOL") {
        return Some("SOLUSDT");
    }
    if t.contains("BINANCE") || t.contains("BNB") {
        return Some("BNBUSDT");
    }
    None
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    info!(
        "📡 Service: {} | Version: {} (V5 PURE RUST AI)",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION")
    );

    let nats_url =
        std::env::var("NATS_URL").unwrap_or_else(|_| "nats://localhost:4222".to_string());
    let sla_timeout_ms = std::env::var("SLA_TIMEOUT_MS")
        .unwrap_or_else(|_| "25".to_string())
        .parse::<u64>()
        .unwrap_or(25);

    info!(
        "⚙️ Pure Rust AI Engine SLA Timeout set to: {}ms",
        sla_timeout_ms
    );

    let nats_client = async_nats::connect(&nats_url)
        .await
        .context("NATS Connection Failed")?;

    let model_path =
        std::env::var("MODEL_PATH").unwrap_or_else(|_| "/opt/models/model.onnx".to_string());
    let tokenizer_path = std::env::var("TOKENIZER_PATH")
        .unwrap_or_else(|_| "/opt/models/tokenizer.json".to_string());

    info!("🧠 Tract Modeli Yükleniyor... Bu işlem birkaç saniye sürebilir.");

    let model = tract_onnx::onnx()
        .model_for_path(&model_path)?
        .into_optimized()?
        .into_runnable()?;

    let brain = Arc::new(NeuralBrain {
        model,
        tokenizer: Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?,
    });
    info!("✅ PURE RUST AI ENGINE BAŞARIYLA BAŞLATILDI!");

    let intel_service = Arc::new(VQIntelligence {
        brain,
        sla_timeout_ms,
    });

    // Haberleri dinleyen Asenkron Task
    let nats_clone = nats_client.clone();
    let service_clone = intel_service.clone();
    tokio::spawn(async move {
        if let Ok(mut sub) = nats_clone.subscribe("news.raw.>").await {
            while let Some(msg) = sub.next().await {
                if let Ok(news) = RawNewsEvent::decode(msg.payload) {
                    let start = Instant::now();
                    let (score, method) = service_clone
                        .process_full_stack(news.headline.clone())
                        .await;

                    if let Some(symbol) = extract_target_symbol(&news.headline) {
                        if score.abs() > 0.10 {
                            let vector = SemanticVector {
                                symbol: symbol.to_string(),
                                sentiment_score: score,
                                source: news.source,
                                original_headline: news.headline,
                                timestamp: chrono::Utc::now().timestamp_millis(),
                            };

                            let mut buf = BytesMut::with_capacity(256);
                            if vector.encode(&mut buf).is_ok() {
                                let _ = nats_clone
                                    .publish("intelligence.news.vector".to_string(), buf.into())
                                    .await;

                                info!(
                                    "🧠 [ALPHA-DETECTED] {} | Score: {:.2} ({}) | Symbol: {} | Latency: {:?}",
                                    vector.original_headline, score, method, symbol, start.elapsed()
                                );
                            }
                        }
                    }
                }
            }
        }
    });

    let addr = "0.0.0.0:50051".parse()?;
    info!("📡 gRPC Endpoint Hazır: {}", addr);

    Server::builder()
        .add_service(SentimentAnalyzerServiceServer::new(VQIntelligence {
            brain: intel_service.brain.clone(),
            sla_timeout_ms,
        }))
        .serve(addr)
        .await?;

    Ok(())
}
