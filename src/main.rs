// ========== DOSYA: sentinel-intelligence/src/main.rs ==========
use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::{Linear, VarBuilder};
use candle_transformers::models::bert::{BertModel, Config};
use futures_util::StreamExt;
use hf_hub::api::sync::Api;
use phf::phf_map;
use prost::Message;
use std::sync::Arc;
use std::time::Instant;
use tokenizers::Tokenizer;
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
// 1. TIER 1: TENSOR ENGINE (CANDLE)
// ==============================================================================

struct TensorEngine {
    bert: BertModel,
    classifier: Linear,
    tokenizer: Tokenizer,
    device: Device,
    pos_id: usize,
    neg_id: usize,
}

impl TensorEngine {
    fn load(device: &Device) -> Result<Self> {
        // DİNAMİK YAPILANDIRMA (Hard-code bitti)
        let repo_id = std::env::var("MODEL_REPO_ID").unwrap_or_else(|_| {
            "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis".to_string()
        });

        let pos_id: usize = std::env::var("MODEL_POS_ID")
            .unwrap_or_else(|_| "2".to_string())
            .parse()
            .unwrap_or(2);
        let neg_id: usize = std::env::var("MODEL_NEG_ID")
            .unwrap_or_else(|_| "0".to_string())
            .parse()
            .unwrap_or(0);

        info!(
            "⏳ [TIER-1] ENV üzerinden '{}' ağırlıkları aranıyor...",
            repo_id
        );

        let api = Api::new().context("HuggingFace API Error")?;
        let repo = api.model(repo_id.clone());

        let tokenizer_filename = repo
            .get("tokenizer.json")
            .context("Tokenizer.json bulunamadı")?;
        let weights_filename = repo
            .get("model.safetensors")
            .context("model.safetensors bulunamadı")?;
        let config_filename = repo.get("config.json").context("Config bulunamadı")?;

        let config_str = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config_str)?;
        let config_json: serde_json::Value = serde_json::from_str(&config_str)?;
        let hidden_size = config_json["hidden_size"].as_u64().unwrap_or(768) as usize;

        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(|e| anyhow::anyhow!(e))?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_filename], DType::F32, device)
                .context("Safetensors mmap hatası")?
        };

        let bert = BertModel::load(vb.clone(), &config).context("Bert Modeli yüklenemedi")?;
        let classifier = candle_nn::linear(hidden_size, 3, vb.pp("classifier"))
            .context("Classifier yüklenemedi")?;

        info!(
            "✅ [TIER-1] Tensor Engine Başarıyla Yüklendi! Model: {} | Pos ID: {} | Neg ID: {}",
            repo_id, pos_id, neg_id
        );

        Ok(Self {
            bert,
            classifier,
            tokenizer,
            device: device.clone(),
            pos_id,
            neg_id,
        })
    }

    fn predict(&self, text: &str) -> Result<f64> {
        let tokens = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!(e))?;
        let mut token_ids = tokens.get_ids();

        if token_ids.len() > 128 {
            token_ids = &token_ids[..128];
        }

        let token_type_ids = vec![0u32; token_ids.len()];
        let input_tensor = Tensor::new(token_ids, &self.device)?.unsqueeze(0)?;
        let token_type_tensor =
            Tensor::new(token_type_ids.as_slice(), &self.device)?.unsqueeze(0)?;

        let embeddings = self.bert.forward(&input_tensor, &token_type_tensor)?;
        let cls_embedding = embeddings.i((.., 0, ..))?;
        let logits = self.classifier.forward(&cls_embedding)?;

        let probs = candle_nn::ops::softmax(&logits, candle_core::D::Minus1)?;
        let probs_vec = probs.squeeze(0)?.to_vec1::<f32>()?;

        // Güvenli Erişim
        let pos = *probs_vec.get(self.pos_id).unwrap_or(&0.0) as f64;
        let neg = *probs_vec.get(self.neg_id).unwrap_or(&0.0) as f64;

        Ok(pos - neg)
    }
}

// ==============================================================================
// 2. TIER 0: ULTRA-FAST PATH LEXICON (Nanosecond Resolution)
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
// 3. MULTI-TIER ORCHESTRATOR
// ==============================================================================

pub struct VQIntelligenceCore {
    tensor_engine: Option<TensorEngine>,
    device: Device,
}

impl Default for VQIntelligenceCore {
    fn default() -> Self {
        Self::build()
    }
}

impl VQIntelligenceCore {
    pub fn build() -> Self {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        info!(
            "🤖 VQ-Capital Multi-Tier Engine Başlatılıyor. Donanım: {:?}",
            device
        );

        let tensor_engine = match TensorEngine::load(&device) {
            Ok(model) => Some(model),
            Err(e) => {
                error!("🚨 [TENSOR HATASI] Model yüklenemedi: {}", e);
                warn!("⚠️ TIER-1 Devre Dışı! Sadece TIER-0 (O(1) Nanosecond Lexicon) çalışacak.");
                None
            }
        };

        Self {
            tensor_engine,
            device,
        }
    }

    pub fn analyze(&self, text: &str) -> (f64, &'static str) {
        let start_time = Instant::now();
        let lower_text = text.to_lowercase();

        for word in lower_text.split_whitespace() {
            let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric());
            if let Some(&score) = EXTREME_SIGNALS.get(clean_word) {
                return (score, "TIER-0 (EXTREME)");
            }
        }

        if let Some(ref model) = self.tensor_engine {
            match model.predict(text) {
                Ok(ml_score) => {
                    let elapsed = start_time.elapsed().as_micros();
                    if elapsed > 10000 {
                        warn!("⚠️ [SLA İHLALİ] Tensor süresi çok uzun: {}µs", elapsed);
                    }
                    return (ml_score, "TIER-1 (TENSOR)");
                }
                Err(e) => warn!("⚠️ Tensor Hatası: {}. Tier-0'a Düşülüyor.", e),
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
        let (final_score, _) = self.analyze(&request.into_inner().text);
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
                    let (score, tier_used) = ai_arc.analyze(&raw_news.headline);
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
        tensor_engine: None,
        device: grpc_ai.device.clone(),
    };

    Server::builder()
        .add_service(SentimentAnalyzerServiceServer::new(service_clone))
        .serve(addr)
        .await?;

    Ok(())
}
