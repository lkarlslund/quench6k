//! What nvcc actually emitted (dev/KBENCH.md "SASS check"): the machine code of selected kernels in
//! this binary, read with the CUDA toolkit's cuobjdump. A tuning hint the compiler ignored (a load that
//! was not hoisted, a cache hint that was dropped, registers that spill) changes no output and may
//! change no timing either; checking for the instructions an optimization depends on tells "it did not
//! help" apart from "it is not there".

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail, ensure};
use regex::Regex;

fn tool(name: &str) -> PathBuf {
    let home = ["CUDA_HOME", "CUDA_PATH"]
        .iter()
        .find_map(|v| std::env::var(v).ok())
        .unwrap_or_else(|| "/opt/cuda".into());
    let path = Path::new(&home).join("bin").join(if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    });
    if path.exists() {
        path
    } else {
        PathBuf::from(name)
    }
}

/// One kernel: its demangled name and its instructions (opcode and operands, predicate included).
pub struct Function {
    pub name: String,
    pub code: Vec<String>,
}

impl Function {
    /// Local-memory loads and stores: register spills (or a local array the compiler could not keep in
    /// registers).
    pub fn spills(&self) -> (usize, usize) {
        let count = |p: &str| {
            self.code
                .iter()
                .filter(|i| opcode(i).starts_with(p))
                .count()
        };
        (count("LDL"), count("STL"))
    }
}

/// The opcode of an instruction: its first word after the predicate ("@P0", "@!PT").
pub fn opcode(instr: &str) -> &str {
    instr
        .split_whitespace()
        .find(|w| !w.starts_with('@'))
        .unwrap_or("")
}

/// Kernels from the cubins whose file name contains `file` (the kernel source name, "gemv" for
/// kernels/gemv.cu; empty for all, which takes about 20 s), optionally only those whose demangled name
/// matches `kernel`.
pub fn functions(binary: &Path, file: &str, kernel: Option<&Regex>) -> Result<Vec<Function>> {
    let dir = std::env::temp_dir().join(format!("quench6k-sass-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let result = (|| {
        let out = Command::new(tool("cuobjdump"))
            .arg("-xelf")
            .arg(if file.is_empty() { "all" } else { file })
            .arg(binary)
            .current_dir(&dir)
            .output()
            .context("running cuobjdump (CUDA toolkit)")?;
        ensure!(
            out.status.success(),
            "cuobjdump -xelf: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let mut cubins: Vec<PathBuf> = std::fs::read_dir(&dir)?
            .map(|e| e.map(|e| e.path()))
            .collect::<Result<_, _>>()?;
        cubins.sort();
        ensure!(
            !cubins.is_empty(),
            "no cubin in {} matches {file:?}",
            binary.display()
        );
        let mut funcs = Vec::new();
        for cubin in &cubins {
            let out = Command::new(tool("cuobjdump"))
                .arg("-sass")
                .arg(cubin)
                .output()?;
            ensure!(
                out.status.success(),
                "cuobjdump -sass {}: {}",
                cubin.display(),
                String::from_utf8_lossy(&out.stderr)
            );
            parse(&String::from_utf8_lossy(&out.stdout), &mut funcs);
        }
        demangle(&mut funcs)?;
        if let Some(re) = kernel {
            funcs.retain(|f| re.is_match(&f.name));
        }
        Ok(funcs)
    })();
    let _ = std::fs::remove_dir_all(&dir);
    result
}

/// cuobjdump's listing: "Function : <mangled>" headers, then lines "/*0090*/  OPCODE operands ; /* encoding */".
fn parse(text: &str, funcs: &mut Vec<Function>) {
    for line in text.lines() {
        let t = line.trim();
        if let Some(name) = t.strip_prefix("Function : ") {
            funcs.push(Function {
                name: name.trim().to_string(),
                code: Vec::new(),
            });
        } else if let (Some(f), Some(rest)) = (funcs.last_mut(), t.strip_prefix("/*")) {
            // "/*0090*/ INSTR ;" and the encoding-only continuation lines "/* 0x... */".
            if let Some((addr, instr)) = rest.split_once("*/")
                && addr.chars().all(|c| c.is_ascii_hexdigit())
            {
                let instr = instr.split(';').next().unwrap_or("").trim();
                if !instr.is_empty() {
                    f.code.push(instr.to_string());
                }
            }
        }
    }
}

fn demangle(funcs: &mut [Function]) -> Result<()> {
    let mut child = match Command::new(tool("cu++filt"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(_) => return Ok(()), // keep the mangled names
    };
    let names: String = funcs.iter().map(|f| format!("{}\n", f.name)).collect();
    let mut stdin = child.stdin.take().unwrap();
    let writer = std::thread::spawn(move || stdin.write_all(names.as_bytes()));
    let out = child.wait_with_output()?;
    writer.join().unwrap()?;
    let lines: Vec<&str> = std::str::from_utf8(&out.stdout)?.lines().collect();
    if lines.len() == funcs.len() {
        for (f, l) in funcs.iter_mut().zip(lines) {
            f.name = l.to_string();
        }
    }
    Ok(())
}

/// One check: kernels of `file` matching `kernel` must contain an instruction matching `expect`.
/// Written FILE/KERNEL=INSTR, for example `gemv/gemv_kernel<\(WFmt\)4, 16=LDG.E.NA`.
pub struct Check {
    pub spec: String,
    pub file: String,
    pub kernel: Regex,
    pub expect: Regex,
}

impl Check {
    pub fn parse(spec: &str) -> Result<Self> {
        let (file, rest) = spec.split_once('/').context("expected FILE/KERNEL=INSTR")?;
        let (kernel, expect) = rest
            .rsplit_once('=')
            .context("expected FILE/KERNEL=INSTR")?;
        Ok(Self {
            spec: spec.to_string(),
            file: file.to_string(),
            kernel: Regex::new(kernel)?,
            expect: Regex::new(expect)?,
        })
    }

    /// One result per matching kernel.
    pub fn run(&self, binary: &Path) -> Result<Vec<Found>> {
        let funcs = functions(binary, &self.file, Some(&self.kernel))?;
        if funcs.is_empty() {
            bail!(
                "{}: no kernel in {} matches {}",
                self.spec,
                self.file,
                self.kernel
            );
        }
        Ok(funcs
            .iter()
            .map(|f| {
                let hits: Vec<usize> = f
                    .code
                    .iter()
                    .enumerate()
                    .filter(|(_, i)| self.expect.is_match(i))
                    .map(|(n, _)| n)
                    .collect();
                Found {
                    name: f.name.clone(),
                    instrs: f.code.len(),
                    hits: hits.len(),
                    first: hits.first().copied(),
                }
            })
            .collect())
    }
}

/// A kernel a check matched: its instruction count, the instructions matching INSTR, and the index of
/// the first.
pub struct Found {
    pub name: String,
    pub instrs: usize,
    pub hits: usize,
    pub first: Option<usize>,
}

/// `quench6k sass`: list kernels (instructions, spills), opcode counts, and FILE/KERNEL=INSTR checks.
pub fn run(
    binary: Option<&Path>,
    file: &str,
    kernel: Option<&str>,
    top: usize,
    checks: &[String],
) -> Result<()> {
    let exe = std::env::current_exe()?;
    let binary = binary.unwrap_or(&exe);
    let mut failed = 0;
    if checks.is_empty() {
        let re = kernel.map(Regex::new).transpose()?;
        let funcs = functions(binary, file, re.as_ref())?;
        println!("{:>7} {:>5} {:>5}  kernel", "instrs", "ldl", "stl");
        for f in &funcs {
            let (l, s) = f.spills();
            println!("{:>7} {l:>5} {s:>5}  {}", f.code.len(), f.name);
        }
        if top > 0 {
            let mut counts = std::collections::HashMap::<String, usize>::new();
            for f in &funcs {
                for i in &f.code {
                    *counts.entry(opcode(i).to_string()).or_default() += 1;
                }
            }
            let mut v: Vec<_> = counts.into_iter().collect();
            v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            println!("\nopcodes over {} kernel(s):", funcs.len());
            for (op, n) in v.into_iter().take(top) {
                println!("{n:>9}  {op}");
            }
        }
    }
    for spec in checks {
        let check = Check::parse(spec)?;
        println!("{spec}");
        for k in check.run(binary)? {
            let at = k
                .first
                .map_or(String::new(), |i| format!(", first at {i} of {}", k.instrs));
            println!(
                "  {:<8} {:>4} match(es){at}  {}",
                if k.hits > 0 { "present" } else { "ABSENT" },
                k.hits,
                k.name
            );
            failed += (k.hits == 0) as usize;
        }
    }
    ensure!(
        failed == 0,
        "{failed} kernel(s) lack an expected instruction"
    );
    Ok(())
}
