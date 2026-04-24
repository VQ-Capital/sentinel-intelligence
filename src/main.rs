// ========== DOSYA: sentinel-intelligence/src/main.rs ==========
use anyhow::{Context, Result};
use futures_util::StreamExt;
use ort::{session::Session, value::Value};
use phf::phf_map;
use prost::Message;
use std::sync::Arc;
use std::time::Instant;
use tokenizers::Tokenizer;
use tokio::sync::RwLock; // FIX: Interior Mutability için eklendi
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
// 🤖 TIER-1: NEURAL ENGINE
// -----------------------------------------------------------------------------
struct NeuralBrain {
    session: RwLock<Session>, // FIX: Mutability hatasını çözmek için RwLock eklendi
    tokenizer: Tokenizer,
}

impl NeuralBrain {
    async fn predict(&self, text: &str) -> Result<f64> {
        let tokens = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!(e))?;
        let token_ids: Vec<i64> = tokens.get_ids().iter().map(|&x| x as i64).collect();
        let seq_len = token_ids.len();

        let input_ids = Value::from_array(([1, seq_len], token_ids))?;
        let mask = Value::from_array(([1, seq_len], vec![1i64; seq_len]))?;
        let type_ids = Value::from_array(([1, seq_len], vec![0i64; seq_len]))?;

        // FIX: Mutable borrow hatasını RwLock yazma kilidi alarak çözüyoruz
        let mut session_guard = self.session.write().await;
        let outputs = session_guard.run(ort::inputs![
            "input_ids" => input_ids,
            "attention_mask" => mask,
            "token_type_ids" => type_ids,
        ])?;

        let logits = outputs["logits"].try_extract_tensor::<f32>()?;
        let slice = logits.1;

        if slice.len() >= 3 {
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
}

#[tonic::async_trait]
impl SentimentAnalyzerService for VQIntelligence {
    async fn analyze_text(
        &self,
        request: Request<AnalyzeTextRequest>,
    ) -> Result<Response<AnalyzeTextResponse>, Status> {
        let text = request.into_inner().text;
        let (score, _) = self.process_full_stack(&text).await;
        Ok(Response::new(AnalyzeTextResponse { score }))
    }
}

impl VQIntelligence {
    async fn process_full_stack(&self, text: &str) -> (f64, &'static str) {
        // Tier-0: Lexicon
        for word in text.to_lowercase().split_whitespace() {
            let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
            if let Some(&score) = FAST_PATH_WORDS.get(clean) {
                return (score, "TIER-0 (LEXICON)");
            }
        }

        // Tier-1: Neural with 4ms Timeout
        let brain_clone = self.brain.clone();
        let text_owned = text.to_string();
        let ai_result = timeout(Duration::from_millis(4), async move {
            brain_clone.predict(&text_owned).await
        })
        .await;

        match ai_result {
            Ok(Ok(score)) => (score, "TIER-1 (NEURAL)"),
            Ok(Err(e)) => {
                error!("Neural Prediction Error: {}", e);
                (0.0, "NEURAL-ERROR")
            }
            Err(_) => {
                warn!("⏳ [SLA-VIOLATION] AI took > 4ms for: {}", text);
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
    info!("🧠 VQ-Intelligence v4.0: Starting Multi-Tier Brain (Resilient Mode)...");

    let nats_url =
        std::env::var("NATS_URL").unwrap_or_else(|_| "nats://localhost:4222".to_string());
    let nats_client = async_nats::connect(&nats_url)
        .await
        .context("NATS Connection Failed")?;

    let model_path =
        std::env::var("MODEL_PATH").unwrap_or_else(|_| "/opt/models/model.onnx".to_string());
    let tokenizer_path = std::env::var("TOKENIZER_PATH")
        .unwrap_or_else(|_| "/opt/models/tokenizer.json".to_string());

    let brain = Arc::new(NeuralBrain {
        session: RwLock::new(Session::builder()?.commit_from_file(model_path)?),
        tokenizer: Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!(e))?,
    });

    let intel_service = Arc::new(VQIntelligence { brain });

    // 1. NATS WORKER LOOP
    let nats_clone = nats_client.clone();
    let service_clone = intel_service.clone();
    tokio::spawn(async move {
        if let Ok(mut sub) = nats_clone.subscribe("news.raw.>").await {
            while let Some(msg) = sub.next().await {
                if let Ok(news) = RawNewsEvent::decode(msg.payload) {
                    let start = Instant::now();
                    let (score, method) = service_clone.process_full_stack(&news.headline).await;

                    if let Some(symbol) = extract_target_symbol(&news.headline) {
                        if score.abs() > 0.05 {
                            let vector = SemanticVector {
                                symbol: symbol.to_string(),
                                sentiment_score: score,
                                source: news.source,
                                original_headline: news.headline,
                                timestamp: chrono::Utc::now().timestamp_millis(),
                            };

                            let mut buf = Vec::new();
                            if vector.encode(&mut buf).is_ok() {
                                let _ = nats_clone
                                    .publish("intelligence.news.vector".to_string(), buf.into())
                                    .await;
                                info!(
                                    "⚡ [INTEL] {} | Score: {:.2} | Method: {} | Latency: {:?} | Symbol: {}",
                                    vector.original_headline, score, method, start.elapsed(), symbol
                                );
                            }
                        }
                    }
                }
            }
        }
    });

    // 2. gRPC SERVER
    let addr = "0.0.0.0:50051".parse()?;
    info!("📡 Sentinel-Intelligence gRPC online at {}", addr);

    Server::builder()
        .add_service(SentimentAnalyzerServiceServer::new(VQIntelligence {
            brain: intel_service.brain.clone(),
        }))
        .serve(addr)
        .await?;

    Ok(())
}
