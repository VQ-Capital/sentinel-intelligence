// ========== DOSYA: sentinel-intelligence/build.rs ==========
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_client(false) // Bu repo sunucu (server) olacak
        .compile(
            &["proto/intelligence.proto"],
            &["proto/"]
        )?;
    Ok(())
}