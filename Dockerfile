# ========== DOSYA: sentinel-intelligence/Dockerfile ==========

# 1. AŞAMA: MODEL BAKER (ONNX EXPORT)
FROM python:3.10-slim AS model-baker
RUN pip install --no-cache-dir optimum[onnxruntime]
RUN optimum-cli export onnx --model ahmedrachid/FinancialBERT-Sentiment-Analysis /models/financial-bert-onnx

# 2. AŞAMA: RUST DERLEYİCİSİ (UBUNTU 24.04)
# Resmi Rust imajı yerine Ubuntu Noble üzerine manuel Rust kuruyoruz (glibc 2.39 garantisi)
FROM ubuntu:24.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    protobuf-compiler \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Rust Toolchain Kurulumu
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /usr/src/app
COPY . .
RUN cargo build --release

# 3. AŞAMA: ÜRETİM ORTAMI (MODERN RUNTIME)
FROM ubuntu:24.04
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

ENV MODEL_PATH="/opt/models/model.onnx"
ENV TOKENIZER_PATH="/opt/models/tokenizer.json"
ENV OMP_NUM_THREADS=1

CMD ["./sentinel-intelligence"]