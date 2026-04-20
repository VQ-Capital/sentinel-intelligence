// ========== DOSYA: sentinel-intelligence/src/main.rs ==========
use phf::phf_map;
use tonic::{transport::Server, Request, Response, Status};
use tracing::{debug, info};

pub mod intelligence {
    tonic::include_proto!("sentinel.intelligence.v1");
}

use intelligence::sentiment_analyzer_service_server::{
    SentimentAnalyzerService, SentimentAnalyzerServiceServer,
};
use intelligence::{AnalyzeTextRequest, AnalyzeTextResponse};

// DERLEME ZAMANLI FİNANSAL SÖZLÜK
static FINANCIAL_LEXICON: phf::Map<&'static str, f64> = phf_map! {
    "bullish" => 0.8, "moon" => 0.9, "breakout" => 0.7, "surge" => 0.8, "accumulation" => 0.5,
    "support" => 0.4, "pump" => 0.6, "adoption" => 0.7, "upgrade" => 0.5, "profit" => 0.6,
    "bearish" => -0.8, "crash" => -0.9, "dump" => -0.8, "resistance" => -0.5, "hack" => -1.0,
    "lawsuit" => -0.9, "inflation" => -0.6, "ban" => -0.9, "collapse" => -1.0, "selloff" => -0.8,
};

#[derive(Debug, Default)]
pub struct NativeRustAI {}

#[tonic::async_trait]
impl SentimentAnalyzerService for NativeRustAI {
    async fn analyze_text(
        &self,
        request: Request<AnalyzeTextRequest>,
    ) -> Result<Response<AnalyzeTextResponse>, Status> {
        let text = request.into_inner().text;
        let mut total_score = 0.0;
        let mut match_count = 0;

        for word in text.to_lowercase().split_whitespace() {
            let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric());
            if let Some(&score) = FINANCIAL_LEXICON.get(clean_word) {
                total_score += score;
                match_count += 1;
            }
        }

        let final_score = if match_count > 0 {
            (total_score / match_count as f64).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        debug!("🧠 [NLP] Metin: '{}' | Skor: {:.2}", text, final_score);
        Ok(Response::new(AnalyzeTextResponse { score: final_score }))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let addr = "0.0.0.0:50051".parse()?;

    info!(
        "⚡ Sentinel-Intelligence (Saf Rust NLP Motoru) dinliyor: {}",
        addr
    );

    Server::builder()
        .add_service(SentimentAnalyzerServiceServer::new(NativeRustAI::default()))
        .serve(addr)
        .await?;

    Ok(())
}
