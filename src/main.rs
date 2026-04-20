// ========== DOSYA: sentinel-intelligence/src/main.rs ==========
use tonic::{transport::Server, Request, Response, Status};
use tracing::{info, error};

// Protobuf tarafından üretilen kodları bu modül altına alıyoruz
pub mod intelligence {
    tonic::include_proto!("sentinel.intelligence");
}

use intelligence::sentiment_analyzer_server::{SentimentAnalyzer, SentimentAnalyzerServer};
use intelligence::{SentimentRequest, SentimentResponse};

#[derive(Debug, Default)]
pub struct MySentimentAnalyzer {}

#[tonic::async_trait]
impl SentimentAnalyzer for MySentimentAnalyzer {
    async fn analyze_text(
        &self,
        request: Request<SentimentRequest>,
    ) -> Result<Response<SentimentResponse>, Status> {
        let text = request.into_inner().text;
        
        // HFT Zekası: Şimdilik kural tabanlı, ileride Candle/CUDA entegrasyonu buraya gelecek.
        let mut score = 0.0;
        let lower_text = text.to_lowercase();
        
        if lower_text.contains("bullish") || lower_text.contains("moon") || lower_text.contains("recovery") {
            score = 0.85;
        } else if lower_text.contains("crash") || lower_text.contains("dump") || lower_text.contains("resistance") {
            score = -0.90;
        }

        info!("🤖 [AI-RUST] İşlenen: '{}' | Skor: {}", text, score);

        Ok(Response::new(SentimentResponse { score }))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Loglamayı başlat
    tracing_subscriber::fmt::init();

    let addr = "0.0.0.0:50051".parse()?;
    let analyzer = MySentimentAnalyzer::default();

    info!("⚡ Sentinel-Intelligence (Rust Server) dinliyor: {}", addr);

    Server::builder()
        .add_service(SentimentAnalyzerServer::new(analyzer))
        .serve(addr)
        .await?;

    Ok(())
}