use std::path::PathBuf;

use anyhow::{Result, ensure};
use clap::{Parser, Subcommand};
use squash_engine::checkpoint::Checkpoint;
use squash_engine::loader::{self, LoadOptions};
use squash_engine::selection;

#[derive(Parser)]
#[command(name = "squash", about = "Qwen3.8 Flash-Next inference engine")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Upload the checkpoint to the GPU, report throughput, and optionally verify tensors.
    Load {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        mtp: bool,
        #[arg(long)]
        vision: bool,
        #[arg(long, default_value_t = 8)]
        threads: usize,
        #[arg(long, default_value_t = 32)]
        chunk_mib: usize,
        /// Read through the page cache instead of O_DIRECT.
        #[arg(long)]
        buffered: bool,
        /// Verify this many tensors (spread across the checkpoint) against the file bytes.
        #[arg(long, default_value_t = 0)]
        verify: usize,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Load { model, mtp, vision, threads, chunk_mib, buffered, verify } => {
            squash_cuda::set_device(0)?;
            let t = std::time::Instant::now();
            let ckpt = Checkpoint::open(&model)?;
            println!("indexed {} tensors in {} files in {:.2}s", ckpt.tensors.len(), ckpt.files.len(), t.elapsed().as_secs_f64());

            let select = |name: &str| selection::text(name) || (mtp && selection::mtp(name)) || (vision && selection::vision(name));
            let opts = LoadOptions { threads, chunk_bytes: chunk_mib << 20, direct: !buffered };
            let (weights, stats) = loader::upload(&ckpt, select, opts)?;
            let gib = stats.bytes as f64 / (1u64 << 30) as f64;
            println!(
                "uploaded {} tensors, {:.2} GiB in {} segments, {:.2}s, {:.2} GiB/s",
                weights.tensors.len(),
                gib,
                stats.segments,
                stats.seconds,
                gib / stats.seconds
            );
            let (free, total) = squash_cuda::mem_info()?;
            println!("device memory: {:.2} GiB free of {:.2} GiB", free as f64 / 1073741824.0, total as f64 / 1073741824.0);

            if verify > 0 {
                let mut names: Vec<&String> = weights.tensors.keys().collect();
                names.sort();
                let stream = squash_cuda::Stream::new()?;
                let step = (names.len() / verify).max(1);
                let mut checked = 0;
                for name in names.iter().step_by(step) {
                    let tensor = weights.get(name)?;
                    let host = squash_cuda::host_checksum(&ckpt.read(name)?);
                    let device = squash_cuda::device_checksum(tensor.addr, tensor.bytes as usize, &stream)?;
                    ensure!(host == device, "checksum mismatch for {name}: host {host:#x}, device {device:#x}");
                    checked += 1;
                }
                println!("verified {checked} tensors byte-exact");
            }
        }
    }
    Ok(())
}
