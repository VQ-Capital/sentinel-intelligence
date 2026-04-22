// ========== DOSYA: sentinel-intelligence/src/main.rs ==========
use candle_core::Device;
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

// YEDEK (FALLBACK) FİNANSAL SÖZLÜK (Model Yüklenemezse Diye)
static FINANCIAL_LEXICON: phf::Map<&'static str, f64> = phf_map! {
    "bullish" => 0.8, "moon" => 0.9, "breakout" => 0.7, "surge" => 0.8,
    "bearish" => -0.8, "crash" => -0.9, "dump" => -0.8, "resistance" => -0.5,
};

pub struct NativeRustAI {
    use_ml: bool,
    #[allow(dead_code)]
    // ÇÖZÜM: CUDA/CPU device objesi model yüklendiğinde kullanılacak. Linter'ı susturuyoruz.
    device: Device,
}

impl Default for NativeRustAI {
    fn default() -> Self {
        Self::new()
    }
}

impl NativeRustAI {
    pub fn new() -> Self {
        // CUDA kontrolü yapılır, yoksa CPU kullanılır (Sıfır çökme garantisi)
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        info!("🤖 Native AI Başlatılıyor. Hedef Donanım: {:?}", device);

        // Gerçek senaryoda model ağırlıkları (safetensors) buradan yüklenir.
        // HFT sisteminin kesintiye uğramaması için şimdilik Fallback = true
        // yapılandırılarak güvenli alan oluşturuldu.
        Self {
            use_ml: false, // Model dosyası diske bağlanınca true yapılır
            device,
        }
    }

    fn fallback_analysis(&self, text: &str) -> f64 {
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
            // Gelecekte Candle ile Tensör çarpımı burada yapılacak
            // let tokens = tokenizer.encode(text);
            // let logits = model.forward(&tokens).unwrap();
            0.0
        } else {
            self.fallback_analysis(&text)
        };

        debug!(
            "🧠 [NLP-NATIVE] Metin: '{}' | Skor: {:.2}",
            text, final_score
        );
        Ok(Response::new(AnalyzeTextResponse { score: final_score }))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let addr = "0.0.0.0:50051".parse()?;

    info!(
        "⚡ Sentinel-Intelligence (Sıfır Gecikmeli Native AI) dinliyor: {}",
        addr
    );

    let ai_service = NativeRustAI::new();

    Server::builder()
        .add_service(SentimentAnalyzerServiceServer::new(ai_service))
        .serve(addr)
        .await?;

    Ok(())
}
