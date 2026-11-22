//! Canonical prefill check shared by `hy4 canon-check` and `flint canon-check`: multi-turn conversations
//! (a shared system prompt, user text, greedy replies) computed cold (a fresh sequence per turn), warm (one
//! sequence per conversation reusing its previous turn, decoding in verification steps with a rejected
//! draft, a long unaligned generation before the last turn so that a boundary snapshot is restored) and
//! cross (two sequences taking the conversations' turns in turn, so they reuse prefixes cached by other
//! conversations, with prefill chunks interleaved with a third sequence's steps and decode steps batched
//! with it). Every turn's logits rows (the prompt's last and every reply token's) and tokens must be
//! bitwise those of cold. Then whether a prefill row's bits depend on its chunk's length.

use anyhow::{Result, ensure};

/// A logits row's hash and its greedy token.
pub type Row = (u64, i32);

/// What the check needs of an engine.
pub trait Engine {
    type St;
    fn name(&self) -> &'static str;
    /// The engine's CANON_CHUNK.
    fn chunk(&self) -> usize;
    /// Rows of the sliding rings (the unaligned filler before the warm mode's last turn exceeds it).
    fn ring(&self) -> usize;
    fn new_state(&mut self, capacity: usize) -> Result<Self::St>;
    fn history<'a>(&self, st: &'a Self::St) -> &'a [i32];
    fn frontier(&self, st: &Self::St) -> usize;
    fn snapshots(&self, st: &Self::St) -> Vec<usize>;
    fn rewind(&mut self, st: &mut Self::St, prompt: &[i32]) -> Result<usize>;
    /// Canonical prefill of `tokens` at the state's position: the last token's row.
    fn prefill(&mut self, st: &mut Self::St, tokens: &[i32]) -> Result<Row>;
    /// One forward step of up to `chunk()` tokens as given (no alignment), without logits.
    fn feed(&mut self, st: &mut Self::St, tokens: &[i32]) -> Result<()>;
    /// One batched step (1..=8 tokens per state): every fed token's row, state by state.
    fn step(&mut self, states: &mut [&mut Self::St], tokens: &[&[i32]]) -> Result<Vec<Row>>;
    fn truncate(&mut self, st: &mut Self::St, pos: usize);
    /// Waits for the queued work.
    fn sync(&self) -> Result<()>;
}

#[derive(Clone, PartialEq)]
struct Turn {
    rows: Vec<u64>,
    reply: Vec<i32>,
}

/// A sequence doing other work between and alongside a turn's steps.
struct Other<S> {
    st: S,
    pending: i32,
}

impl<S> Other<S> {
    fn poke<E: Engine<St = S>>(&mut self, e: &mut E) -> Result<()> {
        let r = e.step(&mut [&mut self.st], &[&[self.pending]])?;
        self.pending = r[0].1;
        Ok(())
    }
}

enum Decode<'a> {
    /// One token per step.
    Plain,
    /// Steps of up to 3 tokens: the pending token, the next reference token and a wrong one.
    Verify(&'a [i32]),
    /// One token per step, batched with the other sequence.
    Batched,
}

struct Info {
    reused: usize,
    /// What reuse of the whole cached common prefix would have kept (at most all but the last token).
    common: usize,
    frontier: usize,
    snapshots: Vec<usize>,
}

/// A turn on `st`: rewind, canonical prefill (each chunk followed by a step of `other`), then `n` reply
/// tokens (greedy).
fn turn<E: Engine>(
    e: &mut E,
    st: &mut E::St,
    prompt: &[i32],
    n: usize,
    decode: Decode,
    mut other: Option<&mut Other<E::St>>,
) -> Result<(Turn, Info)> {
    let c = e.chunk();
    let lcp = e.history(st).iter().zip(prompt).take_while(|(a, b)| a == b).count();
    let (frontier, snapshots) = (e.frontier(st), e.snapshots(st));
    let reused = e.rewind(st, prompt)?;
    ensure!(reused % c == 0 && reused < prompt.len(), "{}: rewind to {reused}", e.name());
    let mut at = reused;
    let mut last = None;
    while at < prompt.len() {
        let end = ((at / c + 1) * c).min(prompt.len());
        last = Some(e.prefill(st, &prompt[at..end])?);
        if let Some(o) = other.as_deref_mut() {
            o.poke(e)?;
        }
        at = end;
    }
    let first = last.unwrap();
    let mut t = Turn { rows: vec![first.0], reply: vec![first.1] };
    while t.reply.len() < n {
        let pending = *t.reply.last().unwrap();
        match decode {
            Decode::Plain => {
                let r = e.step(&mut [&mut *st], &[&[pending]])?;
                t.rows.push(r[0].0);
                t.reply.push(r[0].1);
            }
            Decode::Batched => {
                let o = other.as_deref_mut().expect("a batched turn needs another sequence");
                let r = e.step(&mut [&mut *st, &mut o.st], &[&[pending], &[o.pending]])?;
                t.rows.push(r[0].0);
                t.reply.push(r[0].1);
                o.pending = r[1].1;
            }
            Decode::Verify(hint) => {
                let j = t.reply.len();
                let w = 3.min(n - j);
                let mut feed = vec![pending];
                feed.extend_from_slice(&hint[j..j + w - 1]);
                if w > 1 {
                    let x = feed[w - 1];
                    feed[w - 1] = if x > 0 { x - 1 } else { 1 };
                }
                let before = e.history(st).len();
                let r = e.step(&mut [&mut *st], &[&feed])?;
                let mut k = 0;
                loop {
                    t.rows.push(r[k].0);
                    t.reply.push(r[k].1);
                    if k + 1 < feed.len() && r[k].1 == feed[k + 1] && t.reply.len() < n {
                        k += 1;
                        continue;
                    }
                    break;
                }
                e.truncate(st, before + k + 1);
            }
        }
    }
    let common = lcp.min(prompt.len() - 1);
    Ok((t, Info { reused, common, frontier, snapshots }))
}

/// `ids`: the text's tokens; `system`: shared system prompt tokens; `convs` conversations of `turns`
/// turns with `n` reply tokens each.
pub fn check<E: Engine>(e: &mut E, ids: &[i32], system: usize, convs: usize, turns: usize, n: usize) -> Result<()> {
    let c = e.chunk();
    let user_len = |cv: usize, k: usize| 500 + 97 * k + 131 * cv;
    let stride = user_len(convs, turns) + 64;
    let need = system + stride * convs * turns + 1000;
    ensure!(ids.len() >= need, "the text has {} tokens, {need} needed", ids.len());
    let user = |cv: usize, k: usize| {
        let at = system + stride * (cv * turns + k);
        &ids[at..at + user_len(cv, k)]
    };
    let other_prompt = &ids[need - 1000..need - 300];
    let filler: Vec<i32> = ids.iter().copied().cycle().skip(system).take(e.ring() + c).collect();
    let cap = system + turns * (stride + n) + filler.len() + 2 * c;
    let name = e.name();

    // Cold: every turn on a fresh sequence; its replies build the later prompts.
    let mut prompts: Vec<Vec<Vec<i32>>> = Vec::new();
    let mut cold: Vec<Vec<Turn>> = Vec::new();
    for cv in 0..convs {
        let (mut ps, mut ts) = (Vec::new(), Vec::new());
        let mut hist = ids[..system].to_vec();
        for k in 0..turns {
            hist.extend_from_slice(user(cv, k));
            let mut st = e.new_state(cap)?;
            let (t, _) = turn(e, &mut st, &hist, n, Decode::Plain, None)?;
            ps.push(hist.clone());
            hist.extend_from_slice(&t.reply);
            ts.push(t);
        }
        prompts.push(ps);
        cold.push(ts);
    }
    let mut failures = 0usize;
    let mut extra_total = 0usize;
    let mut report = |mode: &str, cv: usize, k: usize, slot: Option<usize>, t: &Turn, info: &Info| {
        let r = &cold[cv][k];
        let same = t == r;
        let p = &prompts[cv][k];
        let extra = info.common.saturating_sub(info.reused);
        extra_total += extra;
        println!(
            "{name} canon {mode:<5} conv {cv} turn {k}{}: prompt {:>5}, cached common {:>5} (frontier {:>5}, snapshots {:?}), reused {:>5}, prefilled {:>5} = {} new + {} extra: {}",
            slot.map_or(String::new(), |s| format!(" (sequence {s})")),
            p.len(),
            info.common,
            info.frontier,
            info.snapshots,
            info.reused,
            p.len() - info.reused,
            p.len() - info.common.max(info.reused),
            extra,
            if same {
                "bit-identical to cold".to_string()
            } else {
                format!(
                    "DIFFERS ({} of {} rows, tokens {})",
                    t.rows.iter().zip(&r.rows).filter(|(a, b)| a != b).count(),
                    r.rows.len(),
                    if t.reply == r.reply { "equal" } else { "differ" }
                )
            }
        );
        failures += !same as usize;
    };

    // Warm: one sequence per conversation, verification steps; before the last turn a long generation
    // the conversation does not keep (unaligned chunks) moves the rings beyond the frontier's window.
    for cv in 0..convs {
        let mut st = e.new_state(cap)?;
        for k in 0..turns {
            if k + 1 == turns && k > 0 {
                for piece in filler.chunks(c - 1) {
                    e.feed(&mut st, piece)?;
                }
            }
            let (t, info) = turn(e, &mut st, &prompts[cv][k], n, Decode::Verify(&cold[cv][k].reply), None)?;
            report("warm", cv, k, None, &t, &info);
        }
    }

    // Cross: two sequences take the turns in order (turn by turn, conversation by conversation), so most
    // turns land on a sequence that last held another conversation; a third sequence steps between the
    // prefill chunks and decodes batched with each turn.
    let mut seqs = vec![e.new_state(cap)?, e.new_state(cap)?];
    let mut other = {
        let mut st = e.new_state(cap)?;
        let r = e.prefill(&mut st, other_prompt)?;
        Other { st, pending: r.1 }
    };
    let mut j = 0;
    for k in 0..turns {
        for cv in 0..convs {
            let slot = j % 2;
            let (t, info) = turn(e, &mut seqs[slot], &prompts[cv][k], n, Decode::Batched, Some(&mut other))?;
            report("cross", cv, k, Some(slot), &t, &info);
            j += 1;
        }
    }
    drop(seqs);
    drop(other);
    println!(
        "{name} canon: {} turns per mode; {} extra prefill tokens over all warm and cross turns",
        convs * turns,
        extra_total
    );

    // Chunk length: rows of a short chunk vs the same rows of a full chunk (then truncated), compared
    // through the logits of the next token's step (which reads every cache row before it).
    let m = 776.min(c / 2);
    for start in [0, c] {
        let mut rows = Vec::new();
        for len in [m, c] {
            let mut st = e.new_state(start + c + 16)?;
            if start > 0 {
                e.prefill(&mut st, &ids[..start])?;
            }
            e.feed(&mut st, &ids[start..start + len])?;
            e.truncate(&mut st, start + m);
            rows.push(e.step(&mut [&mut st], &[&ids[start + m..start + m + 1]])?[0]);
        }
        let same = rows[0] == rows[1];
        println!(
            "{name} canon chunk length: positions {start}..{} computed as a {m}-token chunk and as the start of a {c}-token chunk: next row {}",
            start + m,
            if same { "bit-identical" } else { "differs (a row's bits depend on its chunk's length)" }
        );
    }
    // What the chunk size costs: 2 * CANON_CHUNK tokens prefilled in chunks of CANON_CHUNK and of half
    // that (fresh sequences, twice each, alternating).
    let mut secs = [0f64; 2];
    for _ in 0..2 {
        for (i, piece) in [c, c / 2].into_iter().enumerate() {
            let mut st = e.new_state(2 * c + 16)?;
            e.sync()?;
            let t0 = std::time::Instant::now();
            for p in ids[..2 * c].chunks(piece) {
                e.feed(&mut st, p)?;
            }
            e.sync()?;
            secs[i] += t0.elapsed().as_secs_f64();
        }
    }
    println!(
        "{name} canon chunk cost: {} tokens in chunks of {c}: {:.0} tokens/s; in chunks of {}: {:.0} tokens/s",
        2 * c,
        4.0 * c as f64 / secs[0],
        c / 2,
        4.0 * c as f64 / secs[1]
    );
    ensure!(failures == 0, "{failures} turns differ from cold");
    println!("{name} canon check passed: every turn bit-identical to cold");
    Ok(())
}
