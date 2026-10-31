//! Real-traffic numbers from the server's request log (dev/KBENCH.md "Real traffic"): the "done:" line
//! every request prints, read from `journalctl -o short-iso` on stdin, summarized per day or week and
//! context range. Benchmarks measure fixed prompts at 512 to 128K tokens; this measures what the service
//! actually did, at the contexts and cache states it actually saw.

use std::collections::BTreeMap;
use std::io::BufRead;

use anyhow::Result;

struct Request {
    date: String,
    prompt: usize,
    reused: usize,
    prefill_s: f64,
    generated: usize,
    decode_s: f64,
    per_round: Option<f64>,
}

/// "2026-10-27T14:49:45+01:00 host quench6k[1]: done: slot 0, prompt 167323 (166611 reused) prefill 0.24s
/// (2927 tok/s), 710 generated in 2.39s (297.2 tok/s), MTP 2.90 tokens/round, Stop"
fn parse(line: &str) -> Option<Request> {
    let date = line
        .get(..10)
        .filter(|d| d.as_bytes()[4] == b'-' && d.as_bytes()[7] == b'-')?
        .to_string();
    let rest = &line[line.find("done: ")? + 6..];
    let word_after = |key: &str| -> Option<&str> {
        let i = rest.find(key)? + key.len();
        rest[i..]
            .split([' ', ',', '(', ')'])
            .find(|w| !w.is_empty())
    };
    let secs = |key: &str| word_after(key)?.trim_end_matches('s').parse::<f64>().ok();
    Some(Request {
        date,
        prompt: word_after("prompt ")?.parse().ok()?,
        reused: rest[rest.find('(')? + 1..]
            .split(' ')
            .next()?
            .parse()
            .ok()?,
        prefill_s: secs("prefill ")?,
        generated: rest[rest.find("), ")? + 3..]
            .split(' ')
            .next()?
            .parse()
            .ok()?,
        decode_s: secs("generated in ")?,
        per_round: word_after("MTP ").and_then(|w| w.parse().ok()),
    })
}

const BUCKETS: [(usize, &str); 5] = [
    (16_384, "<16K"),
    (65_536, "16-64K"),
    (131_072, "64-128K"),
    (196_608, "128-192K"),
    (usize::MAX, ">=192K"),
];

#[derive(Default)]
struct Acc {
    n: usize,
    generated: usize,
    decode_s: f64,
    prefill_s: f64,
    prompt: usize,
    reused: usize,
    /// Requests that prefilled at least COLD new tokens: (new tokens, seconds).
    cold: (usize, f64),
    rounds: (f64, usize),
}

/// Prefill rates of short suffixes measure the fixed per-request cost, not prefill throughput.
const COLD: usize = 4096;

fn period(date: &str, weekly: bool) -> String {
    if !weekly {
        return date.to_string();
    }
    // Days since 1970-01-01 of the civil date (H. Hinnant's algorithm), then back from that week's Monday.
    let (y, m, d): (i64, i64, i64) = (
        date[..4].parse().unwrap_or(0),
        date[5..7].parse().unwrap_or(1),
        date[8..10].parse().unwrap_or(1),
    );
    let y = y - (m <= 2) as i64;
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + d - 1;
    let days = era * 146_097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719_468;
    format!(
        "{}+",
        crate::bench::civil_date(days - (days + 3).rem_euclid(7))
    )
}

pub fn run(weekly: bool, since: Option<&str>) -> Result<()> {
    let mut groups: BTreeMap<(String, usize), Acc> = BTreeMap::new();
    for line in std::io::stdin().lock().lines() {
        let Some(r) = parse(&line?) else { continue };
        if since.is_some_and(|s| r.date.as_str() < s) {
            continue;
        }
        let b = BUCKETS.iter().position(|&(hi, _)| r.prompt < hi).unwrap();
        let a = groups.entry((period(&r.date, weekly), b)).or_default();
        a.n += 1;
        a.generated += r.generated;
        a.decode_s += r.decode_s;
        a.prefill_s += r.prefill_s;
        a.prompt += r.prompt;
        a.reused += r.reused;
        if r.prompt - r.reused >= COLD {
            a.cold.0 += r.prompt - r.reused;
            a.cold.1 += r.prefill_s;
        }
        if let Some(p) = r.per_round {
            a.rounds.0 += p;
            a.rounds.1 += 1;
        }
    }
    println!(
        "{:<12} {:>9} {:>6} {:>9} {:>9} {:>10} {:>7} {:>8} {:>13}",
        "period",
        "context",
        "reqs",
        "generated",
        "decode/s",
        "tok/round",
        "reused",
        "prefill",
        "cold prefill/s"
    );
    for ((p, b), a) in &groups {
        let rate = |tokens: usize, s: f64| {
            if s > 0.0 {
                format!("{:.0}", tokens as f64 / s)
            } else {
                "-".into()
            }
        };
        let rounds = if a.rounds.1 > 0 {
            format!("{:.2}", a.rounds.0 / a.rounds.1 as f64)
        } else {
            "-".into()
        };
        println!(
            "{p:<12} {:>9} {:>6} {:>9} {:>9} {rounds:>10} {:>6.1}% {:>7.0}% {:>13}",
            BUCKETS[*b].1,
            a.n,
            a.generated,
            rate(a.generated, a.decode_s),
            100.0 * a.reused as f64 / a.prompt.max(1) as f64,
            100.0 * a.prefill_s / (a.prefill_s + a.decode_s).max(1e-9),
            rate(a.cold.0, a.cold.1)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_done_lines() {
        let r = parse("2026-10-27T14:49:45+01:00 halo quench6k[500662]: done: slot 0, prompt 167323 (166611 reused) prefill 0.24s (2927 tok/s), 710 generated in 2.39s (297.2 tok/s), MTP 2.90 tokens/round, Stop").unwrap();
        assert_eq!(
            (r.date.as_str(), r.prompt, r.reused, r.generated),
            ("2026-10-27", 167323, 166611, 710)
        );
        assert_eq!(
            (r.prefill_s, r.decode_s, r.per_round),
            (0.24, 2.39, Some(2.90))
        );
        let r = parse("2026-10-17T22:54:34+02:00 halo quench6k[1]: done: prompt 24 (0 reused) prefill 0.34s (70 tok/s), 300 generated in 3.05s (98.1 tok/s), Length").unwrap();
        assert_eq!(
            (r.prompt, r.reused, r.generated, r.per_round),
            (24, 0, 300, None)
        );
        assert!(parse("okt 17 22:54:34 halo quench6k[1]: done: slot 0, prompt 24 (0 reused) prefill 0.34s (70 tok/s), 300 generated in 3.05s (98.1 tok/s), Length").is_none());
    }

    #[test]
    fn weeks_start_on_monday() {
        assert_eq!(period("2026-10-27", true), "2026-10-26+");
        assert_eq!(period("2026-10-26", true), "2026-10-26+");
        assert_eq!(period("2026-10-25", true), "2026-10-19+");
    }
}
