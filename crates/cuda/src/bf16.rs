//! Host-side BF16 conversions (round to nearest even), for tests and small uploads.

pub fn from_f32(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return ((bits >> 16) | 0x40) as u16;
    }
    let round = 0x7fff + ((bits >> 16) & 1);
    ((bits + round) >> 16) as u16
}

pub fn to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

pub fn round(x: f32) -> f32 {
    to_f32(from_f32(x))
}
