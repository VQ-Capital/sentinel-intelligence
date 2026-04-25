# ========== DOSYA: sentinel-intelligence/Dockerfile ==========

# 1. AŞAMA: MODEL BAKER (ONNX EXPORT)
FROM python:3.10-slim AS model-baker
RUN pip install --no-cache-dir optimum[onnxruntime]
RUN optimum-cli export onnx --model ahmedrachid/FinancialBERT-Sentiment-Analysis /models/financial-bert-onnx

# 2. AŞAMA: RUST DERLEYİCİSİ (UBUNTU 24.04 - GLIBC 2.39 İÇİN ZORUNLU)
FROM ubuntu:24.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    curl build-essential protobuf-compiler pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /usr/src/app
COPY . .
RUN cargo build --release

# 3. AŞAMA: ÜRETİM ORTAMI (CUDA 12.6 / UBUNTU 24.04)
# Makinenin maksimum desteklediği 13.0 sınırının altında güvenli liman!
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    libomp5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Modeli ve Tokenizer'ı kopyalıyoruz
COPY --from=model-baker /models/financial-bert-onnx /opt/models

# Binary'yi kopyalıyoruz
COPY --from=builder /usr/src/app/target/release/sentinel-intelligence .

# 🚀 YENİ EKLENEN SATIR: ONNX GPU (CUDA) Kütüphanelerini Kopyalıyoruz!
COPY --from=builder /usr/src/app/target/release/libonnxruntime*.so* ./

ENV MODEL_PATH="/opt/models/model.onnx"
ENV TOKENIZER_PATH="/opt/models/tokenizer.json"
ENV OMP_NUM_THREADS=1

CMD ["./sentinel-intelligence"]