// ========== DOSYA: sentinel-intelligence/src/main.cpp ==========
#include <iostream>
#include <memory>
#include <string>
#include <grpcpp/grpcpp.h>
#include "intelligence.grpc.pb.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using sentinel::intelligence::SentimentAnalyzer;
using sentinel::intelligence::SentimentRequest;
using sentinel::intelligence::SentimentResponse;

// AI MODELİ SİMÜLASYONU (Buraya CUDA/llama.cpp gelecek)
class SentimentAnalyzerImpl final : public SentimentAnalyzer::Service {
    Status AnalyzeText(ServerContext* context, const SentimentRequest* request, SentimentResponse* reply) override {
        std::string text = request->text();
        
        // Şimdilik basit bir mantık: Kelime bazlı skor (Gerçek AI buraya bağlanacak)
        double score = 0.0;
        if (text.find("bullish") != std::string::npos || text.find("moon") != std::string::npos) score = 0.8;
        if (text.find("crash") != std::string::npos || text.find("dump") != std::string::npos) score = -0.9;
        
        std::cout << "🤖 [AI] İşlenen Metin: " << text << " | Üretilen Skor: " << score << std::endl;
        
        reply->set_score(score);
        return Status::OK;
    }
};

void RunServer() {
    std::string server_address("0.0.0.0:50051");
    SentimentAnalyzerImpl service;

    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "⚡ Sentinel-Intelligence C++ (NVIDIA Enabled) dinliyor: " << server_address << std::endl;
    server->Wait();
}

int main(int argc, char** argv) {
    RunServer();
    return 0;
}