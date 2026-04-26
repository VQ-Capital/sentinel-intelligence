# 🤖 sentinel-neural-engine (Legacy: sentinel-intelligence)

**Domain:** NLP & Semantic Scoring (ONNX/CUDA)
**Rol:** Sistemin Duygu Merkezi (The Amygdala)

Bu servis, gRPC üzerinden gelen insan dili metinlerini işler. Sıfır gecikme hedeflenen "HFT Lexicon" katmanı ile acil durum kelimelerini O(1) hızında yakalarken, "ONNX Runtime (CUDA)" üzerinden yerel bir INT8/FP16 Transformer modelini çalıştırarak cümlenin bağlamsal finansal skorunu (-1.0 ile 1.0 arası) üretir. Dış LLM API'lerine bağlanmak KESİNLİKLE yasaktır.

- **NATS Çıktısı:** `intelligence.news.vector`
- **SLA Hedefi:** < 15ms (Graceful Degradation aktif)