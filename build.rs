fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure().build_client(false).compile(
        &["sentinel-spec/proto/sentinel/intelligence/v1/intelligence.proto"],
        &["sentinel-spec/proto/"],
    )?;
    Ok(())
}
