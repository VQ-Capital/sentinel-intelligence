# ========== DOSYA: sentinel-intelligence/Dockerfile ==========
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Sistem bağımlılıkları (Rust ve Protobuf derleyicisi için)
RUN apt-get update && apt-get install -y \
    curl build-essential protobuf-compiler libssl-dev pkg-config

# Rust kurulumu
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app
COPY . .

# Release build
RUN cargo build --release

CMD ["./target/release/sentinel-intelligence"]