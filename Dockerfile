# ========== DOSYA: sentinel-intelligence/Dockerfile ==========

# 1. AŞAMA: MODEL BAKER (Sadece İndirme ve Export - Çalışma Zamanında Yoktur)
FROM python:3.10-slim AS model-baker
RUN pip install --no-cache-dir optimum[onnxruntime]
RUN optimum-cli export onnx --model ahmedrachid/FinancialBERT-Sentiment-Analysis /models/financial-bert-onnx

# 2. AŞAMA: RUST DERLEYİCİSİ (PURE RUST BUILDER)
FROM rust:1.95-slim-bookworm AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    protobuf-compiler pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app
COPY . .
RUN cargo build --release

# 3. AŞAMA: ÜRETİM ORTAMI (ZERO C++, ZERO PYTHON, NO CUDA)
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Modeli ve Tokenizer'ı kopyalıyoruz
COPY --from=model-baker /models/financial-bert-onnx /opt/models

# Saf Rust Binary kopyalıyoruz
COPY --from=builder /usr/src/app/target/release/sentinel-intelligence .

ENV MODEL_PATH="/opt/models/model.onnx"
ENV TOKENIZER_PATH="/opt/models/tokenizer.json"
# Tract (Saf Rust CPU Inferencer) Multi-Threading ayarı
ENV TRACT_NUM_THREADS=4 

CMD ["./sentinel-intelligence"]