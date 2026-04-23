// ========== DOSYA: sentinel-intelligence/src/main.rs ==========
use anyhow::{Context, Result};
use candle_core::Device;
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

// FALLBACK FİNANSAL SÖZLÜK (LLM Yoksa Diye)
static FINANCIAL_LEXICON: phf::Map<&'static str, f64> = phf_map! {
    "bullish" => 0.8, "moon" => 0.9, "breakout" => 0.7, "surge" => 0.8, "surges" => 0.8,
    "bearish" => -0.8, "crash" => -0.9, "crashes" => -0.9, "dump" => -0.8, "dumps" => -0.8,
    "resistance" => -0.5, "partnership" => 0.6, "partners" => 0.6, "lawsuit" => -0.9,
    "accumulation" => 0.7,
};

pub struct NativeRustAI {
    use_ml: bool,
    #[allow(dead_code)]
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
        Self {
            use_ml: false,
            device,
        }
    }

    pub fn fallback_analysis(&self, text: &str) -> f64 {
        let mut total_score = 0.0;
        let mut match_count = 0;
        for word in text.to_lowercase().split_whitespace() {
            let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric());
            if let Some(&score) = FINANCIAL_LEXICON.get(clean_word) {
                total_score += score;
                match_count += 1;
            }
        }
        if match_count > 0 {
            (total_score / match_count as f64).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }
}

#[tonic::async_trait]
impl SentimentAnalyzerService for NativeRustAI {
    async fn analyze_text(
        &self,
        request: Request<AnalyzeTextRequest>,
    ) -> Result<Response<AnalyzeTextResponse>, Status> {
        let text = request.into_inner().text;
        let final_score = if self.use_ml {
            0.0
        } else {
            self.fallback_analysis(&text)
        };
        Ok(Response::new(AnalyzeTextResponse { score: final_score }))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let nats_url =
        std::env::var("NATS_URL").unwrap_or_else(|_| "nats://localhost:4222".to_string());
    let nats_client = async_nats::connect(&nats_url)
        .await
        .context("CRITICAL: NATS bağlanılamadı")?;

    let ai_service = Arc::new(NativeRustAI::new());
    let _grpc_ai = ai_service.clone();

    // NATS EVENT-DRIVEN WORKER (Ham Haberleri Dinleyip Anlam (Sentiment) Üreten Kısım)
    let nats_pub = nats_client.clone();
    tokio::spawn(async move {
        if let Ok(mut sub) = nats_client.subscribe("news.raw.>").await {
            info!("📡 AI Worker: Haber Akışına Bağlandı. Sözel veriler analiz ediliyor...");

            while let Some(msg) = sub.next().await {
                if let Ok(raw_news) = RawNewsEvent::decode(msg.payload) {
                    let text = raw_news.headline.to_uppercase();

                    // Basit Entity Extraction (Hangi coin ile ilgili?)
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
                    }; // Bilinmeyen coini yoksay

                    let score = ai_service.fallback_analysis(&raw_news.headline);

                    // Sıfır Etki (Nötr) haberleri NATS'ı yormamak için filtrele
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
                            "🧠 [NLP] {} {} (Skor: {:.2}) | {}",
                            symbol, direction, score, vector.original_headline
                        );
                    }
                }
            }
        } else {
            warn!("⚠️ AI Worker NATS'a abone olamadı.");
        }
    });

    // Klasik gRPC Sunucusunu da dış sorgular için açık tutuyoruz
    let addr = "0.0.0.0:50051".parse()?;
    info!("⚡ Sentinel-Intelligence gRPC dinliyor: {}", addr);
    Server::builder()
        .add_service(SentimentAnalyzerServiceServer::new(NativeRustAI::new()))
        .serve(addr)
        .await?;

    Ok(())
}
