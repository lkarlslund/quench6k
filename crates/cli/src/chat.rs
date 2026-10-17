//! Tokenizer and chat template of the checkpoint (`tokenizer.json`, `chat_template.jinja`).

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use serde_json::Value;

pub struct Chat {
    tokenizer: tokenizers::Tokenizer,
    env: minijinja::Environment<'static>,
    /// Generation stops at these tokens (`generation_config.json` eos ids and `<|im_end|>`).
    pub stop_tokens: Vec<i32>,
    im_start: Option<i32>,
    /// Encodings of `<|im_start|>`-delimited prompt segments (see [`Chat::encode`]) and their total
    /// text bytes.
    segments: std::sync::Mutex<(std::collections::HashMap<String, Vec<i32>>, usize)>,
}

/// Cached segment text beyond which the segment cache starts over.
const SEGMENT_CACHE_BYTES: usize = 64 << 20;

impl Chat {
    /// `template`: a chat template file overriding the model's `chat_template.jinja`.
    pub fn load(model_dir: &Path, template: Option<&Path>) -> Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(|e| anyhow!("tokenizer.json: {e}"))?;
        let template_path = template.map(Path::to_path_buf).unwrap_or_else(|| model_dir.join("chat_template.jinja"));
        let source = std::fs::read_to_string(&template_path).with_context(|| format!("chat template {}", template_path.display()))?;
        let mut env = minijinja::Environment::new();
        minijinja_contrib::add_to_environment(&mut env);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |msg: String| -> Result<String, minijinja::Error> {
            Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, msg))
        });
        env.add_template_owned("chat", source)?;
        let mut stop_tokens = Vec::new();
        if let Ok(text) = std::fs::read_to_string(model_dir.join("generation_config.json")) {
            let cfg: Value = serde_json::from_str(&text)?;
            match &cfg["eos_token_id"] {
                Value::Array(a) => stop_tokens.extend(a.iter().filter_map(|v| v.as_i64()).map(|v| v as i32)),
                Value::Number(n) => stop_tokens.extend(n.as_i64().map(|v| v as i32)),
                _ => {}
            }
        }
        if let Some(id) = tokenizer.token_to_id("<|im_end|>") {
            if !stop_tokens.contains(&(id as i32)) {
                stop_tokens.push(id as i32);
            }
        }
        let im_start = tokenizer.token_to_id("<|im_start|>").map(|i| i as i32);
        Ok(Self { tokenizer, env, stop_tokens, im_start, segments: Default::default() })
    }

    /// Renders `messages` (OpenAI-style JSON) with an open assistant turn.
    pub fn render(&self, messages: &Value, tools: Option<&Value>, enable_thinking: Option<bool>, reasoning_effort: Option<&str>) -> Result<String> {
        let tmpl = self.env.get_template("chat")?;
        // OpenAI sends tool-call arguments as a JSON string; the template iterates them as a mapping.
        let mut messages = messages.clone();
        for m in messages.as_array_mut().into_iter().flatten() {
            for call in m.get_mut("tool_calls").and_then(Value::as_array_mut).into_iter().flatten() {
                if let Some(args) = call.pointer_mut("/function/arguments") {
                    if let Some(parsed) = args.as_str().and_then(|text| serde_json::from_str::<Value>(text).ok()) {
                        *args = parsed;
                    }
                }
            }
        }
        let mut ctx = serde_json::Map::new();
        ctx.insert("messages".into(), messages);
        ctx.insert("add_generation_prompt".into(), Value::Bool(true));
        if let Some(t) = tools.filter(|t| !t.is_null()) {
            ctx.insert("tools".into(), t.clone());
        }
        if let Some(e) = enable_thinking {
            ctx.insert("enable_thinking".into(), Value::Bool(e));
        }
        if let Some(r) = reasoning_effort {
            ctx.insert("reasoning_effort".into(), Value::String(r.into()));
        }
        Ok(tmpl.render(minijinja::Value::from_serialize(&ctx))?)
    }

    /// Snapshot positions for prefix reuse: the start of the prompt's final turn (its last
    /// `<|im_start|>`), which the next request of the same conversation shares.
    /// Expands each `<|image_pad|>` placeholder of `ids` (in order) into one token per merged image
    /// token, and each `<|video_pad|>` into, per temporal group,
    /// `<{t:.1} seconds><|vision_start|>` + one pad per merged token + `<|vision_end|>` (the HF
    /// Qwen3-VL processor's expansion).
    pub fn expand_media(&self, ids: &[i32], images: Vec<quench6k_engine::vision::VisionInput>, videos: Vec<quench6k_engine::vision::VideoClip>) -> Result<(Vec<i32>, Vec<std::sync::Arc<quench6k_engine::session::ImageInput>>)> {
        use quench6k_engine::session::ImageInput;
        let token = |t: &str| self.token_id(t).ok_or_else(|| anyhow::anyhow!("tokenizer has no {t}"));
        let (pad, vpad, vstart, vend) = (token("<|image_pad|>")?, token("<|video_pad|>")?, token("<|vision_start|>")?, token("<|vision_end|>")?);
        let found = ids.iter().filter(|&&t| t == pad).count();
        anyhow::ensure!(found == images.len(), "{} images for {found} image placeholders", images.len());
        let vfound = ids.iter().filter(|&&t| t == vpad).count();
        anyhow::ensure!(vfound == videos.len(), "{} videos for {vfound} video placeholders", videos.len());
        let mut out = Vec::with_capacity(ids.len());
        let mut spans = Vec::with_capacity(images.len() + videos.len());
        let (mut it, mut vt) = (images.into_iter(), videos.into_iter());
        for &t in ids {
            if t == pad {
                let input = it.next().unwrap();
                let start = out.len();
                out.extend(std::iter::repeat_n(pad, input.tokens()));
                spans.push(std::sync::Arc::new(ImageInput { starts: vec![start], input }));
            } else if t == vpad {
                let clip = vt.next().unwrap();
                let per = clip.input.tokens() / clip.input.grid[0];
                let mut starts = Vec::with_capacity(clip.timestamps.len());
                for ts in &clip.timestamps {
                    out.extend(self.encode(&format!("<{ts:.1} seconds>"))?);
                    out.push(vstart);
                    starts.push(out.len());
                    out.extend(std::iter::repeat_n(vpad, per));
                    out.push(vend);
                }
                spans.push(std::sync::Arc::new(ImageInput { starts, input: clip.input }));
            } else {
                out.push(t);
            }
        }
        Ok((out, spans))
    }

    pub fn boundaries(&self, ids: &[i32]) -> Vec<usize> {
        self.im_start.and_then(|t| ids.iter().rposition(|&i| i == t)).into_iter().filter(|&p| p > 0).collect()
    }

    pub fn token_id(&self, text: &str) -> Option<i32> {
        self.tokenizer.token_to_id(text).map(|i| i as i32)
    }

    /// Token ids of `text`. The tokenizer splits at the special token `<|im_start|>`, so the text
    /// between two of them encodes independently of the rest; those segments' encodings are cached,
    /// and a conversation re-sent with a new turn (as agents do) only tokenizes its new messages.
    pub fn encode(&self, text: &str) -> Result<Vec<i32>> {
        const MARK: &str = "<|im_start|>";
        if self.im_start.is_none() || text.len() < 4096 {
            return self.encode_plain(text);
        }
        let mut cuts: Vec<usize> = text.match_indices(MARK).map(|(i, _)| i).collect();
        if cuts.first() != Some(&0) {
            cuts.insert(0, 0);
        }
        cuts.push(text.len());
        let mut ids = Vec::new();
        for w in cuts.windows(2) {
            let seg = &text[w[0]..w[1]];
            if let Some(v) = self.segments.lock().unwrap().0.get(seg) {
                ids.extend_from_slice(v);
                continue;
            }
            let v = self.encode_plain(seg)?;
            ids.extend_from_slice(&v);
            let mut cache = self.segments.lock().unwrap();
            if cache.1 + seg.len() > SEGMENT_CACHE_BYTES {
                *cache = Default::default();
            }
            cache.1 += seg.len();
            cache.0.insert(seg.to_string(), v);
        }
        if std::env::var_os("QUENCH6K_CHECK_ENCODE").is_some() {
            anyhow::ensure!(ids == self.encode_plain(text)?, "segmented encoding differs from the whole text's");
        }
        Ok(ids)
    }

    fn encode_plain(&self, text: &str) -> Result<Vec<i32>> {
        let enc = self.tokenizer.encode(text, false).map_err(|e| anyhow!("tokenize: {e}"))?;
        Ok(enc.get_ids().iter().map(|&i| i as i32).collect())
    }

    pub fn decode(&self, ids: &[i32], skip_special: bool) -> Result<String> {
        let ids: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
        self.tokenizer.decode(&ids, skip_special).map_err(|e| anyhow!("detokenize: {e}"))
    }
}

/// Incremental detokenization: the text of all ids so far, emitted as new suffixes. A trailing
/// replacement character (an incomplete UTF-8 sequence) is held back until it completes.
#[derive(Default)]
pub struct TextStream {
    ids: Vec<i32>,
    /// Token window decoded per push: ids[prefix..read] were emitted already (their text ends on a
    /// complete character), ids[read..] not yet. Decoding only the window keeps a push O(1) in the
    /// length of the generation.
    prefix: usize,
    read: usize,
}

impl TextStream {
    /// A stream continuing after `ids` (already shown).
    pub fn after(_chat: &Chat, ids: &[i32]) -> Result<Self> {
        Ok(Self { ids: ids.to_vec(), prefix: ids.len().saturating_sub(6), read: ids.len() })
    }

    pub fn ids(&self) -> &[i32] {
        &self.ids
    }

    /// Appends `id`; returns the text it completes (empty while a character is still incomplete).
    pub fn push(&mut self, chat: &Chat, id: i32) -> Result<String> {
        self.ids.push(id);
        let shown = chat.decode(&self.ids[self.prefix..self.read], true)?;
        let text = chat.decode(&self.ids[self.prefix..], true)?;
        if text.len() <= shown.len() || text.ends_with('\u{FFFD}') || !text.is_char_boundary(shown.len()) || !text.starts_with(&shown) {
            return Ok(String::new());
        }
        self.prefix = self.read;
        self.read = self.ids.len();
        Ok(text[shown.len()..].to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Streaming a generation token by token yields exactly its full decode (multi-byte characters
    /// split across tokens included).
    #[test]
    fn text_stream_matches_full_decode() {
        let dir = std::path::Path::new("/storage/ai/huggingface/main/ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4");
        if !dir.join("tokenizer.json").exists() {
            return;
        }
        let chat = Chat::load(dir, None).unwrap();
        let text = "Hello, wörld! 你好，世界。 🦀🚀 emoji ZWJ 👩‍💻 and code:\n```rust\nfn main() { println!(\"ok\"); }\n```\n".repeat(20);
        let ids = chat.encode_plain(&text).unwrap();
        let mut stream = TextStream::default();
        let mut out = String::new();
        for &id in &ids {
            out += &stream.push(&chat, id).unwrap();
        }
        assert_eq!(out, chat.decode(&ids, true).unwrap());
        // Continuing after a prefix.
        let half = ids.len() / 2;
        let mut stream = TextStream::after(&chat, &ids[..half]).unwrap();
        let mut rest = String::new();
        for &id in &ids[half..] {
            rest += &stream.push(&chat, id).unwrap();
        }
        assert_eq!(chat.decode(&ids[..half], true).unwrap() + &rest, chat.decode(&ids, true).unwrap());
    }
}
