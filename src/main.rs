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
// 1. GERÇEK YAPAY ZEKA (DistilRoBERTa) - CANDLE HFT BACKEND
// ==============================================================================

struct FinBertSlm {
    bert: BertModel,
    classifier: Linear,
    tokenizer: Tokenizer,
    device: Device,
    pos_id: usize,
    neg_id: usize,
}

impl FinBertSlm {
    fn load(device: &Device) -> Result<Self> {
        // HFT ve Rust (Candle) ile en uyumlu, en hızlı model seçildi.
        let repo_id = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis";
        info!("⏳ [HF-HUB] {} ağırlıkları aranıyor...", repo_id);

        let api = Api::new().context("HuggingFace API Error")?;
        let repo = api.model(repo_id.to_string());

        // Hata fırlatmak yerine dosyaları dikkatlice çekiyoruz.
        let tokenizer_filename = repo.get("tokenizer.json").context("Tokenizer bulunamadı")?;
        let weights_filename = repo
            .get("model.safetensors")
            .context("Safetensors bulunamadı")?;
        let config_filename = repo.get("config.json").context("Config bulunamadı")?;

        let config_str = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config_str)?;
        let config_json: serde_json::Value = serde_json::from_str(&config_str)?;
        let hidden_size = config_json["hidden_size"].as_u64().unwrap_or(768) as usize;

        // Dinamik Sınıf Eşleme (0, 1, 2)
        let mut pos_id = 2;
        let mut neg_id = 0;

        if let Some(id2label) = config_json["id2label"].as_object() {
            for (id_str, label_val) in id2label {
                if let (Ok(idx), Some(label_str)) = (id_str.parse::<usize>(), label_val.as_str()) {
                    let lower = label_str.to_lowercase();
                    if lower.contains("positive") {
                        pos_id = idx;
                    } else if lower.contains("negative") {
                        neg_id = idx;
                    }
                }
            }
        }

        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(|e| anyhow::anyhow!(e))?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_filename], DType::F32, device)
                .context("Safetensors mmap hatası")?
        };

        let bert = BertModel::load(vb.clone(), &config).context("Bert Modeli yüklenemedi")?;
        let classifier = candle_nn::linear(hidden_size, 3, vb.pp("classifier"))
            .context("Classifier yüklenemedi")?;

        info!(
            "✅ [CANDLE] Model başarıyla yüklendi. (Pos: {}, Neg: {})",
            pos_id, neg_id
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

        let pos = probs_vec[self.pos_id] as f64;
        let neg = probs_vec[self.neg_id] as f64;

        Ok(pos - neg)
    }
}

// ==============================================================================
// 2. FALLBACK (YEDEK) LEXICON ENGINE
// ==============================================================================

static FINANCIAL_LEXICON: phf::Map<&'static str, f64> = phf_map! {
    "bullish" => 0.8, "moon" => 0.9, "breakout" => 0.7, "surge" => 0.8, "surges" => 0.8,
    "rally" => 0.7, "adoption" => 0.6, "inflows" => 0.7, "accumulate" => 0.7, "accumulation" => 0.7,
    "bearish" => -0.8, "crash" => -0.9, "crashes" => -0.9, "dump" => -0.8, "dumps" => -0.8,
    "resistance" => -0.5, "partnership" => 0.6, "partners" => 0.6, "lawsuit" => -0.9,
    "hack" => -0.9, "hacked" => -0.9, "exploiter" => -0.8, "freeze" => -0.6, "freezes" => -0.6,
    "slumps" => -0.7, "sec" => -0.5, "probe" => -0.6, "investigation" => -0.6,
};

// ==============================================================================
// 3. CORE AI YAPISI & ENTITY RECOGNITION
// ==============================================================================

pub struct NativeRustAI {
    slm: Option<FinBertSlm>,
    device: Device,
}

impl Default for NativeRustAI {
    fn default() -> Self {
        Self::build()
    }
}

impl NativeRustAI {
    pub fn build() -> Self {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        info!(
            "🤖 VQ-Capital AI Motoru Başlatılıyor. Donanım: {:?}",
            device
        );

        // Hata durumunda sistemi çökertmez.
        let slm = match FinBertSlm::load(&device) {
            Ok(model) => Some(model),
            Err(e) => {
                error!("🚨 HF-Hub İndirme/Yükleme Hatası (Ağ veya Format): {}", e);
                warn!(
                    "⚠️ HFT Kesintiye Uğramayacak! Sistem 'Lexicon (Sözlük)' modunda devam ediyor."
                );
                None
            }
        };

        Self { slm, device }
    }

    pub fn analyze(&self, text: &str) -> f64 {
        if let Some(ref model) = self.slm {
            match model.predict(text) {
                Ok(ml_score) => return ml_score,
                Err(e) => warn!("⚠️ NLP Inference Hatası: {}", e),
            }
        }

        let mut lexicon_score = 0.0;
        let mut match_count = 0;

        for word in text.to_lowercase().split_whitespace() {
            let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric());
            if let Some(&score) = FINANCIAL_LEXICON.get(clean_word) {
                lexicon_score += score;
                match_count += 1;
            }
        }
        if match_count > 0 {
            (lexicon_score / match_count as f64).clamp(-1.0, 1.0)
        } else {
            0.0
        }
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
impl SentimentAnalyzerService for NativeRustAI {
    async fn analyze_text(
        &self,
        request: Request<AnalyzeTextRequest>,
    ) -> Result<Response<AnalyzeTextResponse>, Status> {
        let final_score = self.analyze(&request.into_inner().text);
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

    let ai_service = tokio::task::spawn_blocking(NativeRustAI::build)
        .await
        .context("AI Builder Panic Hatası")?;

    let ai_arc = Arc::new(ai_service);
    let grpc_ai = ai_arc.clone();
    let nats_pub = nats_client.clone();

    tokio::spawn(async move {
        if let Ok(mut sub) = nats_client.subscribe("news.raw.>").await {
            info!("📡 AI Tensor Worker: Haber Akışına Bağlandı.");

            while let Some(msg) = sub.next().await {
                if let Ok(raw_news) = RawNewsEvent::decode(msg.payload) {
                    let text_upper = raw_news.headline.to_uppercase();

                    let symbol = match NativeRustAI::extract_symbol(&text_upper) {
                        Some(s) => s,
                        None => continue,
                    };

                    let score = ai_arc.analyze(&raw_news.headline);

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
                            "🧠 [NLP] {} {} (Skor: {:.2}) | {}",
                            symbol, direction, score, vector.original_headline
                        );
                    }
                }
            }
        }
    });

    let addr = "0.0.0.0:50051".parse()?;
    info!("⚡ Sentinel-Intelligence gRPC dinliyor: {}", addr);

    let service_clone = NativeRustAI {
        slm: None,
        device: grpc_ai.device.clone(),
    };

    Server::builder()
        .add_service(SentimentAnalyzerServiceServer::new(service_clone))
        .serve(addr)
        .await?;

    Ok(())
}
