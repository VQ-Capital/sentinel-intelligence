# ========== DOSYA: sentinel-intelligence/Dockerfile ==========

# 1. AŞAMA: MODEL BAKER (ONNX EXPORT)
# Python kullanarak modeli indirir, ONNX formatına çevirir ve donanım uyumlu hale getirir.
FROM python:3.10-slim AS model-baker
RUN pip install --no-cache-dir optimum[onnxruntime]
# Modeli HuggingFace'ten çekip ONNX'e dönüştürerek /models klasörüne donduruyoruz.
RUN optimum-cli export onnx --model ahmedrachid/FinancialBERT-Sentiment-Analysis /models/financial-bert-onnx

# 2. AŞAMA: RUST DERLEYİCİSİ (CORE ARCHITECT)
FROM rust:1.95-slim-bookworm AS builder
RUN apt-get update && apt-get install -y protobuf-compiler pkg-config libssl-dev
WORKDIR /usr/src/app
COPY . .
RUN cargo build --release

# 3. AŞAMA: ÜRETİM ORTAMI (ZERO-DEPENDENCY RUNTIME)
FROM debian:bookworm-slim
# SSL ve ONNX Runtime bağımlılıklarını kuruyoruz
RUN apt-get update && apt-get install -y libssl3 ca-certificates libomp-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Modeli ve Tokenizer'ı Python imajından (1. Aşama) kopyalıyoruz (Tamamen Çevrimdışı)
COPY --from=model-baker /models/financial-bert-onnx /opt/models

# Derlenmiş Rust çalıştırılabilir dosyasını kopyalıyoruz
COPY --from=builder /usr/src/app/target/release/sentinel-intelligence .

# ENV Değişkenleri: Sistem artık internetten model aramayacak, direkt bu dosyaları okuyacak.
ENV MODEL_PATH="/opt/models/model.onnx"
ENV TOKENIZER_PATH="/opt/models/tokenizer.json"
ENV OMP_NUM_THREADS=1

# Çalıştırma Komutu
CMD ["./sentinel-intelligence"]