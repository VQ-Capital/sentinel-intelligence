# ========== DOSYA: sentinel-intelligence/Dockerfile ==========
# 1. Derleme Aşaması
FROM rust:1.95-slim-bookworm AS builder

# Sistem bağımlılıkları (Protobuf derleyicisi için şart)
RUN apt-get update && apt-get install -y protobuf-compiler pkg-config libssl-dev

WORKDIR /usr/src/app
COPY . .

# Release derlemesi
RUN cargo build --release

# 2. Çalıştırma Aşaması (Tertemiz ve Küçük)
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y libssl3 ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Binaries ismini cargo build çıktısından kopyala
COPY --from=builder /usr/src/app/target/release/sentinel-intelligence .

CMD ["./sentinel-intelligence"]