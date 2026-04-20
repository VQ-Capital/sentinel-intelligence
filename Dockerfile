# ========== DOSYA: sentinel-intelligence/Dockerfile ==========
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    build-essential cmake git pkg-config \
    libgrpc++-dev protobuf-compiler-grpc \
    libprotobuf-dev

WORKDIR /app
COPY . .

# Derleme
RUN mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc)

CMD ["./build/sentinel_intelligence"]