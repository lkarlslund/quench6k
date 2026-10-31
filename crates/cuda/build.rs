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

fn exe(name: &str) -> String {
    if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    }
}

fn main() {
    let cuda = cuda_home();
    let nvcc = cuda.join("bin").join(exe("nvcc"));
    let windows = env::var("CARGO_CFG_TARGET_OS").unwrap() == "windows";
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
        let object = out
            .join(source.file_stem().unwrap())
            .with_extension(if windows { "obj" } else { "o" });
        let mut cmd = Command::new(&nvcc);
        cmd.args([
            "-c",
            "-std=c++20",
            "-gencode=arch=compute_120a,code=sm_120a",
            "--expt-relaxed-constexpr",
        ]);
        if windows {
            // The dynamic CRT (/MD), as Rust's MSVC targets use; the conforming preprocessor for C++20.
            cmd.args(["-Xcompiler=/MD", "-Xcompiler=/Zc:preprocessor"]);
        } else {
            cmd.arg("-Xcompiler=-fPIC");
        }
        cmd.arg("-I")
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

    let library = out.join(if windows {
        "quench6k_kernels.lib"
    } else {
        "libquench6k_kernels.a"
    });
    let _ = std::fs::remove_file(&library);
    let status = Command::new(&nvcc)
        .arg("--lib")
        .arg("-o")
        .arg(&library)
        .args(&objects)
        .status()
        .expect("failed to run nvcc --lib");
    assert!(status.success(), "nvcc --lib failed");

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=quench6k_kernels");
    // The static CUDA runtime: binaries need only the NVIDIA driver.
    if windows {
        println!(
            "cargo:rustc-link-search=native={}",
            cuda.join("lib").join("x64").display()
        );
        println!("cargo:rustc-link-lib=static=cudart_static");
    } else {
        for dir in [
            "lib64",
            "lib",
            "targets/sbsa-linux/lib",
            "targets/x86_64-linux/lib",
        ] {
            println!(
                "cargo:rustc-link-search=native={}",
                cuda.join(dir).display()
            );
        }
        println!("cargo:rustc-link-lib=static=cudart_static");
        // The driver API, for growable buffers (part of the NVIDIA driver).
        for lib in ["cuda", "stdc++", "dl", "pthread", "rt"] {
            println!("cargo:rustc-link-lib=dylib={lib}");
        }
    }
}
