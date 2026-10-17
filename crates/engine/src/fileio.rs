//! Positional file reads and direct-I/O opens on Unix and Windows.

use std::fs::File;
use std::io;
use std::path::Path;

/// Reads into `buf` from `offset`; returns the bytes read (0 at the end of the file).
pub fn read_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    #[cfg(unix)]
    {
        std::os::unix::fs::FileExt::read_at(file, buf, offset)
    }
    #[cfg(windows)]
    {
        std::os::windows::fs::FileExt::seek_read(file, buf, offset)
    }
}

/// Fills `buf` from `offset`.
pub fn read_exact_at(file: &File, mut buf: &mut [u8], mut offset: u64) -> io::Result<()> {
    while !buf.is_empty() {
        match read_at(file, buf, offset) {
            Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
            Ok(n) => {
                buf = &mut buf[n..];
                offset += n as u64;
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Opens `path` for reads that bypass the page cache (Linux `O_DIRECT`); None where unsupported.
pub fn open_direct(path: &Path) -> Option<File> {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // O_DIRECT differs between architectures.
        const O_DIRECT: i32 = if cfg!(any(target_arch = "aarch64", target_arch = "arm")) { 0o200000 } else { 0o40000 };
        std::fs::OpenOptions::new().read(true).custom_flags(O_DIRECT).open(path).ok()
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = path;
        None
    }
}
