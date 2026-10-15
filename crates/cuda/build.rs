//! Compiles every `kernels/*.cu` file with nvcc for sm_120a into one static library.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn cuda_home() -> PathBuf {
    for var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Ok(path) = env::var(var) {
            return PathBuf::from(path);
        }
    }
    PathBuf::from("/opt/cuda")
}

fn main() {
    let cuda = cuda_home();
    let nvcc = cuda.join("bin/nvcc");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let kernels = Path::new("kernels");
    let profile = env::var("PROFILE").unwrap();

    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-changed=kernels");

    let mut sources: Vec<PathBuf> = std::fs::read_dir(kernels)
        .expect("kernels directory")
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "cu"))
        .collect();
    sources.sort();

    let mut objects = Vec::new();
    for source in &sources {
        println!("cargo:rerun-if-changed={}", source.display());
        let object = out.join(source.file_stem().unwrap()).with_extension("o");
        let mut cmd = Command::new(&nvcc);
        cmd.args(["-c", "-std=c++20", "-arch=sm_120a", "-Xcompiler=-fPIC", "--expt-relaxed-constexpr"])
            .arg("-I")
            .arg(kernels)
            .arg(if profile == "release" { "-O3" } else { "-O2" })
            .arg("-lineinfo")
            .arg(source)
            .arg("-o")
            .arg(&object);
        let status = cmd.status().expect("failed to run nvcc");
        assert!(status.success(), "nvcc failed on {}", source.display());
        objects.push(object);
    }
    for header in std::fs::read_dir(kernels).unwrap() {
        let path = header.unwrap().path();
        if path.extension().is_some_and(|ext| ext == "cuh") {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }

    let library = out.join("libsquash_kernels.a");
    let _ = std::fs::remove_file(&library);
    let status = Command::new("ar").arg("crs").arg(&library).args(&objects).status().expect("failed to run ar");
    assert!(status.success(), "ar failed");

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=squash_kernels");
    println!("cargo:rustc-link-search=native={}", cuda.join("lib64").display());
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
