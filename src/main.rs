use tonic::{transport::Server, Request, Response, Status};
use intelligence::sentiment_analyzer_server::{SentimentAnalyzer, SentimentAnalyzerServer};
use intelligence::{SentimentRequest, SentimentResponse};

pub mod intelligence {
    tonic::include_proto!("sentinel.intelligence");
}

#[derive(Debug, Default)]
pub struct MySentimentAnalyzer {}

#[tonic::async_trait]
impl SentimentAnalyzer for MySentimentAnalyzer {
    async fn analyze_text(
        &self,
        request: Request<SentimentRequest>,
    ) -> Result<Response<SentimentResponse>, Status> {
        let text = request.into_inner().text;
        
        // --- BURASI KRİTİK: GPU/AI MANTIĞI ---
        // Şimdilik kural tabanlı, ileride Candle/CUDA entegre edilecek
        let mut score = 0.0;
        if text.contains("bullish") || text.contains("moon") { score = 0.85; }
        if text.contains("crash") || text.contains("dump") { score = -0.90; }

        println!("🤖 [AI-RUST] İşlenen: '{}' | Skor: {}", text, score);

        Ok(Response::new(SentimentResponse { score }))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let addr = "0.0.0.0:50051".parse()?;
    let analyzer = MySentimentAnalyzer::default();

    println!("⚡ Sentinel-Intelligence (Rust/GPU-Ready) dinliyor: {}", addr);

    Server::builder()
        .add_service(SentimentAnalyzerServer::new(analyzer))
        .serve(addr)
        .await?;

    Ok(())
}