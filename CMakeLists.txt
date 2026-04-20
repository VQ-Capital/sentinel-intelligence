# ========== DOSYA: sentinel-intelligence/CMakeLists.txt ==========
cmake_minimum_required(VERSION 3.15)
project(sentinel_intelligence CXX)

set(CMAKE_CXX_STANDARD 17)

# Bağımlılıkları bul (gRPC, Protobuf)
find_package(gRPC CONFIG REQUIRED)
find_package(Protobuf CONFIG REQUIRED)

# Protobuf & gRPC kod üretimi
set(proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/intelligence.pb.cc")
set(proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/intelligence.pb.h")
set(grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/intelligence.grpc.pb.cc")
set(grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/intelligence.grpc.pb.h")

add_custom_command(
      OUTPUT "${proto_srcs}" "${proto_hdrs}" "${grpc_srcs}" "${grpc_hdrs}"
      COMMAND ${Protobuf_PROTOC_EXECUTABLE}
      ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
           --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
           -I "${CMAKE_CURRENT_SOURCE_DIR}/proto"
           --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
           "${CMAKE_CURRENT_SOURCE_DIR}/proto/intelligence.proto"
)

include_directories("${CMAKE_CURRENT_BINARY_DIR}")

add_executable(sentinel_intelligence src/main.cpp ${proto_srcs} ${grpc_srcs})
target_link_libraries(sentinel_intelligence 
    gRPC::grpc++ 
    protobuf::libprotobuf
)