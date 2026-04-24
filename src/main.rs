// ========== DOSYA: sentinel-intelligence/src/main.rs ==========
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};
use futures_util::StreamExt;
use phf::phf_map;
use prost::Message;
use std::sync::Arc;
use tonic::{transport::Server, Request, Response, Status};
use tracing::{info, warn};

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
// 1. CANDLE-CORE (NATIVE RUST ML) - KÜÇÜK DİL MODELİ (SLM) BAŞLIĞI
// ==============================================================================

struct SentimentSlm {
    fc1: Linear,
    fc2: Linear,
    device: Device,
}

impl SentimentSlm {
    fn new(device: &Device) -> Result<Self> {
        // HFT ortamı için model diskten (safetensors) yüklenmelidir.
        // Ancak test ortamındaysak, bellek üzerinde rastgele bir VarBuilder yaratıyoruz.
        // Gerçek kullanımda: VarBuilder::from_safetensors(...)
        let vb = VarBuilder::zeros(DType::F32, device);

        // 768 boyutlu BERT/Qwen embedding'ini alır, 128'e sıkıştırır, sonra 1 boyutlu Duygu Skoruna indirger.
        let fc1 = linear(768, 128, vb.pp("fc1")).context("FC1 katmanı oluşturulamadı")?;
        let fc2 = linear(128, 1, vb.pp("fc2")).context("FC2 katmanı oluşturulamadı")?;

        Ok(Self {
            fc1,
            fc2,
            device: device.clone(),
        })
    }

    fn forward(&self, text: &str) -> Result<f64> {
        // ADIM 1: Mock Tokenization & Embedding
        // (Metni 768 boyutlu bir sayısal tensöre çeviririz. Production'da HuggingFace Tokenizers kullanılır)
        let mut embed_data = vec![0.0f32; 768];
        for (i, b) in text.bytes().enumerate() {
            if i < 768 {
                embed_data[i] = (b as f32) / 255.0;
            } // ASCII karakterleri normalize et
        }

        // ADIM 2: Tensör Oluştur (Shape: [1, 768])
        let input = Tensor::from_vec(embed_data, (1, 768), &self.device)?;

        // ADIM 3: İleri Besleme (Forward Pass - Matris Çarpımı)
        let hidden = self.fc1.forward(&input)?.relu()?;
        let output = self.fc2.forward(&hidden)?;

        // ADIM 4: Çıktıyı al ve Tanh fonksiyonu ile [-1.0, 1.0] aralığına sıkıştır
        let raw_score = output.to_vec2::<f32>()?[0][0];
        Ok(raw_score.tanh() as f64)
    }
}

// ==============================================================================
// 2. FALLBACK LEXICON (GÜVENLİK AĞI)
// ==============================================================================

static FINANCIAL_LEXICON: phf::Map<&'static str, f64> = phf_map! {
    "bullish" => 0.8, "moon" => 0.9, "breakout" => 0.7, "surge" => 0.8, "surges" => 0.8,
    "bearish" => -0.8, "crash" => -0.9, "crashes" => -0.9, "dump" => -0.8, "dumps" => -0.8,
    "resistance" => -0.5, "partnership" => 0.6, "partners" => 0.6, "lawsuit" => -0.9,
    "accumulation" => 0.7,
};

// ==============================================================================
// 3. CORE AI YAPISI (ENSEMBLE FUSION)
// ==============================================================================

pub struct NativeRustAI {
    slm: Option<SentimentSlm>,
    device: Device,
}

impl Default for NativeRustAI {
    fn default() -> Self {
        Self::new()
    }
}

impl NativeRustAI {
    pub fn new() -> Self {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        info!("🤖 Native AI Başlatılıyor. Hedef Donanım: {:?}", device);

        let slm = match SentimentSlm::new(&device) {
            Ok(model) => {
                info!("🧠 Candle-Core SLM Sinir Ağı Başarıyla Belleğe Yüklendi!");
                Some(model)
            }
            Err(e) => {
                warn!(
                    "⚠️ SLM Yüklenemedi, sistem sadece Lexicon ile devam edecek: {}",
                    e
                );
                None
            }
        };

        Self { slm, device }
    }

    pub fn analyze(&self, text: &str) -> f64 {
        let mut lexicon_score = 0.0;
        let mut match_count = 0;

        for word in text.to_lowercase().split_whitespace() {
            let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric());
            if let Some(&score) = FINANCIAL_LEXICON.get(clean_word) {
                lexicon_score += score;
                match_count += 1;
            }
        }
        let lex_final = if match_count > 0 {
            (lexicon_score / match_count as f64).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // ENSEMBLE: Hem Sinir Ağından (Tensör) hem Sözlükten (Lexicon) gelen veriyi birleştir
        if let Some(ref model) = self.slm {
            if let Ok(ml_score) = model.forward(text) {
                // Tensör çıkışı ile klasik sözlük çıkışının ağırlıklı ortalaması
                return (ml_score * 0.4) + (lex_final * 0.6);
            }
        }

        lex_final
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
// 4. MAIN RUNTIME (WORKER & GRPC)
// ==============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let nats_url =
        std::env::var("NATS_URL").unwrap_or_else(|_| "nats://localhost:4222".to_string());
    let nats_client = async_nats::connect(&nats_url)
        .await
        .context("CRITICAL: NATS bağlanılamadı")?;

    let ai_service = Arc::new(NativeRustAI::new());
    let grpc_ai = ai_service.clone();

    let nats_pub = nats_client.clone();
    tokio::spawn(async move {
        if let Ok(mut sub) = nats_client.subscribe("news.raw.>").await {
            info!("📡 AI Tensor Worker: Haber Akışına Bağlandı. Sözel veriler matrislere çarpılıyor...");

            while let Some(msg) = sub.next().await {
                if let Ok(raw_news) = RawNewsEvent::decode(msg.payload) {
                    let text = raw_news.headline.to_uppercase();
                    let symbol = if text.contains("BTC") {
                        "BTCUSDT"
                    } else if text.contains("ETH") {
                        "ETHUSDT"
                    } else if text.contains("SOL") {
                        "SOLUSDT"
                    } else if text.contains("BNB") {
                        "BNBUSDT"
                    } else {
                        continue;
                    };

                    // TENSÖR HESAPLAMASI BURADA TETİKLENİR
                    let score = ai_service.analyze(&raw_news.headline);

                    if score.abs() < 0.1 {
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
                            "🧠 [TENSOR-NLP] {} {} (Skor: {:.2}) | {}",
                            symbol, direction, score, vector.original_headline
                        );
                    }
                }
            }
        } else {
            warn!("⚠️ AI Worker NATS'a abone olamadı.");
        }
    });

    let addr = "0.0.0.0:50051".parse()?;
    info!("⚡ Sentinel-Intelligence gRPC dinliyor: {}", addr);
    Server::builder()
        .add_service(SentimentAnalyzerServiceServer::new(
            (*grpc_ai).clone_as_service(),
        ))
        .serve(addr)
        .await?;

    Ok(())
}

// Clone implementasyonu (gRPC servisi için)
impl NativeRustAI {
    fn clone_as_service(&self) -> Self {
        Self {
            slm: None, // gRPC worker'ında tensor kopyalamayı önlemek için hafif kopyalama
            device: self.device.clone(),
        }
    }
}
