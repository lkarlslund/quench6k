//! Engram n-gram hashing of Hy4-flash (`HYV4FlashEngramLayer.hash_ngrams`): each position's token and
//! the two before it are mapped through the layer's frozen lookup table, multiplied by per-layer odd
//! multipliers, XOR-mixed per n-gram order and reduced modulo each hash head's prime vocabulary.
//!
//! The multipliers come from NumPy's default generator (PCG64 seeded through SeedSequence), drawn with
//! `integers(0, bound, size=3, dtype=int64)`; this module reproduces that stream.

/// NumPy SeedSequence (pool size 4) for an integer entropy below 2^64, without spawn keys.
fn seed_sequence_state(entropy: u64, words64: usize) -> Vec<u64> {
    const INIT_A: u32 = 0x43b0_d7e5;
    const MULT_A: u32 = 0x931e_8875;
    const INIT_B: u32 = 0x8b51_f9dd;
    const MULT_B: u32 = 0x58f3_8ded;
    const MIX_MULT_L: u32 = 0xca01_f9dd;
    const MIX_MULT_R: u32 = 0x4973_f715;
    const XSHIFT: u32 = 16;
    // _coerce_to_uint32_array: little-endian 32-bit words, [0] for zero.
    let mut entropy_words = Vec::new();
    let mut e = entropy;
    loop {
        entropy_words.push(e as u32);
        e >>= 32;
        if e == 0 {
            break;
        }
    }
    let mut hash_const = INIT_A;
    let hashmix = |value: u32, hc: &mut u32| -> u32 {
        let mut v = value ^ *hc;
        *hc = hc.wrapping_mul(MULT_A);
        v = v.wrapping_mul(*hc);
        v ^ (v >> XSHIFT)
    };
    let mix = |x: u32, y: u32| -> u32 {
        let r = MIX_MULT_L.wrapping_mul(x).wrapping_sub(MIX_MULT_R.wrapping_mul(y));
        r ^ (r >> XSHIFT)
    };
    let mut pool = [0u32; 4];
    for (i, slot) in pool.iter_mut().enumerate() {
        let v = entropy_words.get(i).copied().unwrap_or(0);
        *slot = hashmix(v, &mut hash_const);
    }
    for src in 0..4 {
        for dst in 0..4 {
            if src != dst {
                let h = hashmix(pool[src], &mut hash_const);
                pool[dst] = mix(pool[dst], h);
            }
        }
    }
    for &w in entropy_words.iter().skip(4) {
        for dst in 0..4 {
            let h = hashmix(w, &mut hash_const);
            pool[dst] = mix(pool[dst], h);
        }
    }
    // generate_state(words64, uint64): 32-bit words cycling over the pool.
    let mut hc = INIT_B;
    let mut words = Vec::with_capacity(2 * words64);
    for i in 0..2 * words64 {
        let mut v = pool[i % 4] ^ hc;
        hc = hc.wrapping_mul(MULT_B);
        v = v.wrapping_mul(hc);
        v ^= v >> XSHIFT;
        words.push(v);
    }
    words
        .chunks_exact(2)
        .map(|w| w[0] as u64 | (w[1] as u64) << 32)
        .collect()
}

/// NumPy's PCG64 (XSL-RR 128/64).
struct Pcg64 {
    state: u128,
    inc: u128,
}

const PCG_MULT: u128 = 0x2360_ed05_1fc6_5da4_4385_df64_9fcc_f645;

impl Pcg64 {
    fn from_seed(seed: u64) -> Self {
        let s = seed_sequence_state(seed, 4);
        let init_state = (s[0] as u128) << 64 | s[1] as u128;
        let init_seq = (s[2] as u128) << 64 | s[3] as u128;
        let mut rng = Self {
            state: 0,
            inc: (init_seq << 1) | 1,
        };
        rng.step();
        rng.state = rng.state.wrapping_add(init_state);
        rng.step();
        rng
    }

    fn step(&mut self) {
        self.state = self.state.wrapping_mul(PCG_MULT).wrapping_add(self.inc);
    }

    fn next_u64(&mut self) -> u64 {
        self.step();
        let hi = (self.state >> 64) as u64;
        let lo = self.state as u64;
        (hi ^ lo).rotate_right((hi >> 58) as u32)
    }

    /// `Generator.integers(0, high)` for int64 with high - 1 > 2^32 - 1 (Lemire, 64-bit).
    fn below(&mut self, high: u64) -> u64 {
        let rng = high - 1;
        if rng == 0 {
            return 0;
        }
        assert!(rng > u32::MAX as u64, "32-bit bounded draws are not implemented");
        let excl = rng.wrapping_add(1);
        let mut m = self.next_u64() as u128 * excl as u128;
        let mut leftover = m as u64;
        if leftover < excl {
            let threshold = (u64::MAX - rng) % excl;
            while leftover < threshold {
                m = self.next_u64() as u128 * excl as u128;
                leftover = m as u64;
            }
        }
        (m >> 64) as u64
    }
}

/// The odd multipliers of an Engram layer: `default_rng(seed + 10007 * layer_id).integers(0, bound,
/// size=n)` * 2 + 1 with bound = max(1, i64::MAX / compressed_vocab / 2).
pub fn multipliers(seed: u64, layer_id: u64, compressed_vocab: u64, n: usize) -> Vec<i64> {
    let bound = (i64::MAX as u64 / compressed_vocab / 2).max(1);
    let mut rng = Pcg64::from_seed(seed + 10007 * layer_id);
    (0..n).map(|_| (rng.below(bound) * 2 + 1) as i64).collect()
}

/// One layer's hash: lookup table, multipliers and per-head moduli (n-gram-major: 8 bigram heads, then
/// 8 trigram heads) with each head's row offset in the shared table.
pub struct EngramHash {
    pub lookup: Vec<i32>,
    pub mult: Vec<i64>,
    pub moduli: Vec<i64>,
    pub offsets: Vec<i64>,
    pub pad: i64,
    pub heads_per_order: usize,
}

impl EngramHash {
    pub fn new(lookup: Vec<i32>, seed: u64, layer_id: u64, max_ngram: usize, moduli: Vec<i64>, pad: i64) -> Self {
        // The bound comes from the compressed vocabulary the map uses (as vLLM's Hy4-flash derives it).
        let cv = lookup.iter().copied().max().unwrap_or(0) as u64 + 1;
        let mult = multipliers(seed, layer_id, cv, max_ngram);
        let mut offsets = Vec::with_capacity(moduli.len());
        let mut acc = 0i64;
        for m in &moduli {
            offsets.push(acc);
            acc += m;
        }
        let heads_per_order = moduli.len() / (max_ngram - 1);
        Self {
            lookup,
            mult,
            moduli,
            offsets,
            pad,
            heads_per_order,
        }
    }

    fn compress(&self, tok: i64) -> i64 {
        let n = self.lookup.len() as i64;
        if (0..n).contains(&tok) {
            self.lookup[tok as usize] as i64
        } else {
            self.lookup[self.pad as usize] as i64
        }
    }

    /// Table rows (one per head) for the n-gram ending at a position: `ngram[0]` is its token,
    /// `ngram[1]` the one before, ... (the pad id where the sequence has no such token).
    pub fn rows(&self, ngram: &[i64], out: &mut Vec<i64>) {
        let mixed: Vec<i64> = ngram
            .iter()
            .zip(&self.mult)
            .map(|(&t, &m)| self.compress(t).wrapping_mul(m))
            .collect();
        for order in 2..=self.mult.len() {
            let mut mix = mixed[0];
            for v in &mixed[1..order] {
                mix ^= v;
            }
            let base = (order - 2) * self.heads_per_order;
            for h in base..base + self.heads_per_order {
                out.push(mix.rem_euclid(self.moduli[h]) + self.offsets[h]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numpy_stream() {
        // np.random.default_rng(20014).bit_generator.random_raw(3)
        let mut r = Pcg64::from_seed(20014);
        assert_eq!(
            [r.next_u64(), r.next_u64(), r.next_u64()],
            [0xd26c38b1a43ef95a, 0x66083b28e7b16e3d, 0x3e49d20d3dcf0fd9]
        );
    }

    #[test]
    fn numpy_multipliers() {
        // integers(0, i64max // cv // 2, size=3, dtype=int64) * 2 + 1 under default_rng(seed + 10007 * id)
        assert_eq!(
            multipliers(0, 2, 120818, 3),
            [62749575316035, 30426722171175, 18574837598557]
        );
        assert_eq!(
            multipliers(0, 12, 120818, 3),
            [14916611332703, 34290019069965, 64992498366675]
        );
        assert_eq!(
            multipliers(0, 32, 120000, 3),
            [22482354630879, 12924468900031, 17346636404993]
        );
        assert_eq!(
            multipliers(7, 1, 1000, 3),
            [1156537163146049, 8016615858685391, 3279014256592297]
        );
        assert_eq!(
            multipliers(0, 22, 5, 3),
            [1351774637581581107, 398832873693494025, 1037308012371131693]
        );
    }
}
