//! Calibration-window helpers for `nn-rs` ESP-DL export.
//!
//! The `.npy` reader is intentionally local to `nn-rs`: it exists for
//! native-vs-legacy acceptance parity, while normal export collects
//! calibration windows from the manifest/audio/mel pipeline.

use std::path::Path;

pub fn collect_calibration_windows_from_npy(
    dir: &Path,
    n_mels: usize,
    n_frames: usize,
    limit: usize,
) -> std::io::Result<Vec<Vec<f32>>> {
    let mut files = std::fs::read_dir(dir)?
        .map(|res| res.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()?;
    files.retain(|p| p.extension().is_some_and(|ext| ext == "npy"));
    files.sort();
    if files.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("no .npy calibration files found in {}", dir.display()),
        ));
    }

    let mut windows = Vec::with_capacity(limit.min(files.len()));
    for path in files.into_iter().take(limit) {
        let window = read_f32_npy_2d(&path, n_mels, n_frames)?;
        windows.push(window);
    }
    Ok(windows)
}

fn read_f32_npy_2d(path: &Path, n_mels: usize, n_frames: usize) -> std::io::Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    const MAGIC: &[u8; 6] = b"\x93NUMPY";
    if bytes.len() < 10 || &bytes[..6] != MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{} is not a .npy file", path.display()),
        ));
    }
    let major = bytes[6];
    let header_start;
    let header_len;
    match major {
        1 => {
            header_start = 10;
            header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        }
        2 | 3 => {
            if bytes.len() < 12 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("{} has a truncated .npy header", path.display()),
                ));
            }
            header_start = 12;
            header_len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        }
        other => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unsupported .npy version {other} in {}", path.display()),
            ));
        }
    }
    let data_start = header_start + header_len;
    if bytes.len() < data_start {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{} has a truncated .npy header", path.display()),
        ));
    }
    let header = std::str::from_utf8(&bytes[header_start..data_start]).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{} has a non-utf8 .npy header: {e}", path.display()),
        )
    })?;
    if !(header.contains("'descr': '<f4'") || header.contains("\"descr\": \"<f4\"")) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{} is not little-endian float32", path.display()),
        ));
    }
    if header.contains("True") {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{} is Fortran-ordered; expected C-order", path.display()),
        ));
    }
    let shape = parse_npy_shape(header).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{} has an unsupported .npy shape header", path.display()),
        )
    })?;
    if shape != vec![n_mels, n_frames] {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "{} shape {:?}, expected [{n_mels}, {n_frames}]",
                path.display(),
                shape
            ),
        ));
    }

    let expected_values = n_mels * n_frames;
    let expected_bytes = expected_values * std::mem::size_of::<f32>();
    if bytes.len() < data_start + expected_bytes {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{} has truncated float32 payload", path.display()),
        ));
    }
    let mut out = Vec::with_capacity(expected_values);
    for chunk in bytes[data_start..data_start + expected_bytes].chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn parse_npy_shape(header: &str) -> Option<Vec<usize>> {
    let shape_key = header.find("shape")?;
    let after_shape = &header[shape_key..];
    let open = after_shape.find('(')?;
    let close = after_shape[open + 1..].find(')')? + open + 1;
    let inner = &after_shape[open + 1..close];
    let mut out = Vec::new();
    for part in inner.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        out.push(part.parse().ok()?);
    }
    Some(out)
}
