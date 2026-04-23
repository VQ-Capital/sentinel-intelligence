// ========== DOSYA: sentinel-intelligence/build.rs ==========
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure().build_server(true).compile(
        &[
            "sentinel-spec/proto/sentinel/intelligence/v1/intelligence.proto",
            "sentinel-spec/proto/sentinel/market/v1/market_data.proto", // YENİ
        ],
        &["sentinel-spec/proto/"],
    )?;
    Ok(())
}
