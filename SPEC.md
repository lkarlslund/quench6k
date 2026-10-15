# Qwen3.8 Flash-Next 125B-A6B: model specification

The exact mathematics squash implements. Sources, in order of authority:

1. HF transformers `models/qwen4_exp/modeling_qwen4_exp.py` (text model, vision tower, PLE, QSA);
2. llama.cpp `src/models/qwen4exp.cpp` and `conversion/qwen4exp.py` (MTP block, which HF does not
   implement, and an independent check of everything else);
3. the checkpoint itself: `ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4` (`model_type: qwen4_exp`).

Notation: `B` tokens in a step, `D = 2560` hidden, `S = 4` HyperConnection streams,
`H = S·D = 10240`. Every `RMSNorm` below is `y = x · rsqrt(mean(x²) + 1e-6) · (1 + w)`, computed in
FP32, unless it is marked **plain** (`y = … · w`). "Grouped" RMSNorm over `H` normalizes each
2560-wide stream separately with its own slice of the 10240-wide weight.

## 1. Dimensions

| Quantity | Value |
|---|---:|
| hidden `D` / layers | 2560 / 48 |
| vocabulary | 248,320 (embedding and LM head untied) |
| layer types | layer `i` is QSA attention if `i % 4 == 3` (12 layers), otherwise GDN (36 layers) |
| QSA query heads / KV heads / head dim | 24 / 2 / 256 (GQA group 12) |
| QSA rotary dims | 64 of 256 (`partial_rotary_factor 0.25`), θ = 10,000,000, interleaved MRoPE sections [11, 11, 10] |
| QSA indexer | 4 query heads + 1 key head, dim 128; group 4 tokens; budget 2048 tokens (512 groups) |
| GDN key heads / value heads / head dim | 16 / 48 / 128; conv channels 10240, kernel 4 |
| routed experts / top-k / width | 512 / 10 / 640, plus one shared expert of width 640 |
| HyperConnection streams / low rank | 4 / 320 |
| PLE | layer 1 only (`ple_layer_ids = [2]` is 1-based); 16 heads × 160 = 2560; conv kernel 4, dilation 3 |
| MTP | 1 layer, QSA attention |
| max context | 262,144 |
| special tokens | EOS / BOS / PLE reset `248044`; also stop on `248046` (`<|im_end|>`); image `248056`, video `248057`, vision start/end `248053`/`248054` |

## 2. Top level

```
x0      = embed_tokens[token]                         # [B, D] BF16
h       = concat(x0, x0, x0, x0)                      # [B, H]  four identical streams
for i in 0..47:  h = layer_i(h)
out     = Mixer(h)                                    # [B, D]  (HC mix without injection, §4)
logits  = lm_head · out                               # [B, 248320]
```

There is no final RMSNorm: the final HyperConnection mixer (`model.language_model.hyper_connection_mixer`)
normalizes the streams. The 10240-wide `h` after layer 47, before the mixer, is what MTP consumes.

## 3. Decoder layer

```
if i == 1:  h = h + PLE(h, tokens)                    # §8
m, inj      = HC_attn(h)                              # m: [B, D], inj: [B, S]
y           = GDN(m) if i % 4 != 3 else QSA(m)
h           = h + y ⊗ inj                             # h[s] += y · inj[s] for each stream s
m, inj      = HC_mlp(h)
y           = MoE(m)
h           = h + y ⊗ inj
```

## 4. HyperConnection (`attn_hyper_connection`, `mlp_hyper_connection`, mixer)

Weights: `hc_norm [H]`, `input_mix_weight_down [320, H]`, `input_mix_weight_up [H, 320]`,
`block_inject_weight [4, H]` (absent in the mixers).

```
n    = GroupedRMSNorm(h)                              # [B, H]
g    = sigmoid( up · silu( (down · n) / 4 ) )         # [B, H]
m    = mean over s of ( g[s] ⊙ n[s] )                 # [B, D]
inj  = 2 · sigmoid( (block_inject · n) / 4 )          # [B, S]
```

The returned residual is the *unnormalized* `h`. The mix weights are per feature (10240 gates), not
four scalars.

**Steering hook.** Activation steering (§11) modifies `h` immediately before `n` in `HC_attn`.

## 5. Gated DeltaNet (36 layers)

Weights: `in_proj_qkv [10240, D]`, `in_proj_z [6144, D]`, `in_proj_a [48, D]`, `in_proj_b [48, D]`,
`conv1d [10240, 1, 4]` (no bias), `A_log [48]`, `dt_bias [48]`, `norm [128]` (**plain** weight),
`out_proj [D, 6144]`.

```
qkv   = in_proj_qkv · m                               # [B, 10240] = q 2048 | k 2048 | v 6144
qkv   = silu( causal depthwise conv1d(qkv, kernel 4) ) # with 3 columns of conv state
q, k  = split into 16 heads × 128;  v into 48 heads × 128
q, k  = repeat_interleave ×3 → 48 heads              # head j uses key head j / 3
q, k  = l2norm(q), l2norm(k)                          # x · rsqrt(Σx² + 1e-6), FP32
q     = q / sqrt(128)
beta  = sigmoid(in_proj_b · m)                        # [B, 48]
g     = -exp(A_log) · softplus(in_proj_a · m + dt_bias)   # [B, 48], FP32, ≤ 0
per head, per token t (S is [128 k × 128 v], FP32):
    S      = exp(g_t) · S
    delta  = beta_t · (v_t − Sᵀ k_t)
    S      = S + k_t ⊗ delta
    o_t    = Sᵀ q_t
o     = RMSNorm_plain(o per head, w = norm) ⊙ sigmoid(z per head),  z = in_proj_z · m
y     = out_proj · o
```

The gated norm uses **sigmoid** (`output_gate_type: sigmoid`), not silu. Prefill may use the
chunked (WY/UT-transform) form; it must equal the recurrence above.

State per layer and sequence: conv `[10240, 3]` (BF16 in HF) and recurrent `[48, 128, 128]` FP32.

## 6. QSA attention (12 layers)

Weights: `q_proj [12288, D]` (per head: 256 query then 256 gate), `k_proj [512, D]`,
`v_proj [512, D]`, `q_norm [256]`, `k_norm [256]`, `o_proj [D, 6144]`, and the indexer:
`index_qk_proj [640, D]` (4×128 query rows then 128 key rows), `q_layernorm [128]`,
`k_layernorm [128]`.

### 6.1 Main projections
```
q, gate = split each head of q_proj · m               # 24 heads × (256, 256)
q       = RMSNorm(q per head, q_norm);  k = RMSNorm(k_proj · m per head, k_norm);  v = v_proj · m
q, k    = RoPE on dims [0, 64) (rotate-half form), MRoPE positions (§6.4)
```

### 6.2 Indexer: which keys a query may see
For a query at position `t` whose visible (causal) keys are positions `0..t`:
```
iq          = index_qk_proj · m → 4 heads × 128 query  |  1 × 128 raw key
iq          = RoPE( RMSNorm(iq, q_layernorm) )        # position t
raw_key[t]  = raw indexer key (BF16), stored per position (persistent state)
groups      = ⌊(t+1)/4⌋ complete groups [4j, 4j+3];  tail = the (t+1) mod 4 newest tokens
pooled_j    = RoPE( RMSNorm( mean(raw_key[4j..4j+3]) in FP32 → BF16, k_layernorm ), position 4j )
score_j     = Σ over 4 heads of relu(iq_h · pooled_j) / sqrt(128)      # FP32
selected    = top-min(512, groups) groups by score  ∪  tail
```
Exact attention then runs over the selected positions only (at most 2048 + 3 keys). Below 2048
visible tokens every key is selected and QSA is ordinary causal attention. A complete group's
pooled key is a pure function of its four raw keys and position, so it can be computed once when
the group completes and cached.

### 6.3 Attention
Softmax attention of each query head over the selected K/V of its KV head (head `h` uses KV head
`h / 12`), scale `1/16`, softmax in FP32. Output `o = attn ⊙ sigmoid(gate)`, `y = o_proj · o`.

State per layer: K and V `[2, 256]` per position, raw indexer key `[128]` per position (and
optionally cached pooled group keys).

### 6.4 Positions
Text positions are `(t, t, t)` for all three MRoPE axes. The 32 rotary frequency pairs
`inv_freq[i] = θ^(−2i/64)` are assigned to axes interleaved: pair `i` uses the H position if
`i % 3 == 1` and `i < 33`, the W position if `i % 3 == 2` and `i < 30`, otherwise T. With vision
input, image/video tokens carry (t, h, w) grid positions (§12).

## 7. Sparse MoE

Weights: `gate [512, D]`, 512 experts each with `gate_proj [640, D]`, `up_proj [640, D]`,
`down_proj [D, 640]` (NVFP4, §9), shared expert `gate_proj`, `up_proj [640, D]`,
`down_proj [D, 640]` (BF16), `shared_expert_gate [1, D]`.

```
p        = softmax(gate · m) in FP32                   # [B, 512]
top      = top-10 of p;  w = p[top] / Σ p[top]          # renormalized
routed   = Σ_e w_e · down_e( silu(gate_e · m) ⊙ up_e · m )
shared   = sigmoid(shared_expert_gate · m) · down_s( silu(gate_s · m) ⊙ up_s · m )
y        = routed + shared
```

## 8. PLE (layer 1)

Weights: `ngram_embedding` table, 128 shards × `[2,500,012, 160]` BF16 (320,001,536 rows; shard
`r / 2500012` holds row `r`); `layer_multipliers [3]` I64, `ngram_heads_vocab_sizes [16]` I64,
`ngram_heads_offsets [16]` I64 (all three stored in the checkpoint); `key_proj [H, 2560]`,
`value_proj [D, 2560]`, `norm_key`, `norm_query`, `norm_conv [H]` (grouped), `conv1d [H, 1, 4]`.

### 8.1 Hashed n-gram rows
For each token position, `s_k` is the token `k` positions back (`k = 0, 1, 2`) within the current
EOS-delimited segment. A token *after* an EOS starts a new segment; positions before the segment
start, or before the sequence start, read as EOS (`248044`). The history carries over between steps
(state: the last 2 tokens).
```
bigram   = (s0·M0) xor (s1·M1)                        # I64 wrapping multiply
trigram  = (s0·M0) xor (s1·M1) xor (s2·M2)
row[j]   = (bigram  mod size[j]) + offset[j]  for heads j = 0..7    # mod is floor-mod, result ≥ 0
row[j]   = (trigram mod size[j]) + offset[j]  for heads j = 8..15
e        = concat of the 16 table rows               # [B, 2560]
```

### 8.2 Injection
```
kn    = GroupedRMSNorm(key_proj · e, norm_key)         # [B, S, D]
qn    = GroupedRMSNorm(h, norm_query)                  # [B, S, D]
v     = value_proj · e                                 # [B, D]
a_s   = (kn[s] · qn[s]) / sqrt(2560);  a_s = sign(a_s) · sqrt(max(|a_s|, 1e-6))
gv[s] = sigmoid(a_s) · v                               # [B, S, D] → flatten to [B, H]
c     = silu( depthwise conv1d(GroupedRMSNorm(gv, norm_conv), kernel 4, dilation 3) )
PLE   = gv + c                                         # added to h
```
The conv runs over the normalized `gv` with 9 columns of history (`(4−1)·3`), state `[H, 9]`.
Tokens masked out of the sequence are replaced by EOS before hashing.

## 9. Weight formats in the checkpoint

- Routed experts of the 48 main layers: NVFP4 (ModelOpt, W4A4, block 16).
  `weight U8 [N, K/2]` (two E2M1 codes per byte, low nibble first), `weight_scale F8_E4M3 [N, K/16]`,
  `weight_scale_2 F32 []`, `input_scale F32 []`.
  `W[n,k] = e2m1(code) · weight_scale[n, k/16] · weight_scale_2`.
  Activations are quantized the same way: per 16-element block, scale = FP8(amax / 6 / input_scale),
  value = E2M1(x / (scale · input_scale)). The product accumulates in FP32.
- Everything else is BF16: attention, GDN, HyperConnection, router, shared experts, PLE, embedding,
  LM head, vision, and the MTP layer (whose experts are BF16 `gate_up_proj [512, 1280, D]` and
  `down_proj [512, D, 640]`).
- squash's 8-bit option converts non-expert BF16 projections to FP8 on the GPU at load time. It is
  an engine choice and must pass the correctness gate like everything else.

## 10. MTP (from llama.cpp; HF ignores `mtp.*`)

Weights: `fc_embedding [D, D]`, `fc_hidden [D, D]`, `pre_fc_norm_embedding [D]`,
`pre_fc_norm_hidden [H]` (grouped), one QSA decoder layer `mtp.layers.0.*` (HC, QSA with indexer,
MoE with BF16 experts), and its own mixer `mtp.hyper_connection_mixer.*`. It shares `embed_tokens`
and `lm_head` with the main model.

To draft the token after position `t+1`, given the main model's pre-mixer residual `h_t` and the
next token `x_{t+1}`:
```
e      = RMSNorm(embed_tokens[x_{t+1}], pre_fc_norm_embedding)
hn     = GroupedRMSNorm(h_t, pre_fc_norm_hidden)
r[s]   = fc_embedding · e + fc_hidden · hn[s]          # [S, D]: one stream per s
r      = QSA decoder layer (§3 without PLE, layer type attention)
logits = lm_head · Mixer_mtp(r)
```
The MTP layer keeps its own K/V and indexer cache. Chained drafts feed the MTP output residual `r`
back as the next `h`. Position assignment for the MTP K/V is to be verified against llama.cpp at M5.

## 11. Activation steering

A pack holds, for selected layers `L`, unit-norm F32 directions `d_k ∈ R^{4×2560}`, `k < rank ≤ 32`,
and a 4-bit lane mask. In `HC_attn` of layer `L`, before normalization, for each selected stream:
```
h ← h − a · Σ_k (h · d_k) d_k            # dot products all taken against the original h
```
`a ∈ [0, 1]` is the per-request strength, defaulting to the pack's value. `a = 0` must reproduce the
unsteered computation bit for bit. An optional norm-preserving mode rescales each adjusted stream to
its original norm. The MTP drafter is never steered. Capture records `h` (after steering), `n`,
`g`, and `m` at the last prompt token and the last executed completion token.

## 12. Vision tower

Qwen3-VL-style ViT without deepstack: Conv3d patch embedding `[1152, 3, 2, 16, 16]` with bias,
learned 48×48 position embeddings resampled bilinearly (align corners) to each grid, 27 blocks of
pre-LayerNorm (eps 1e-6) attention (16 heads × 72, bias, 2D axial RoPE) and GELU-tanh MLP
(1152 → 4304 → 1152), non-causal within each image, then a merger: LayerNorm → 2×2 spatial merge
(4608) → Linear 4608 → GELU → Linear → 2560. Image/video token embeddings are replaced by merger
outputs; their MRoPE positions follow the (t, h, w) grid. Preprocessing and the multimodal position
rules are specified in detail at M10.

## 13. Persistent state per sequence

| Component | Shape per sequence | Notes |
|---|---|---|
| QSA K, V (12 layers + MTP) | per position `[2, 256]` each | BF16; quantized options must pass the gate |
| QSA raw indexer key | per position `[128]` | BF16 |
| QSA pooled group key cache | per 4 positions `[128]` | optional, derived |
| GDN conv (36 layers) | `[10240, 3]` | |
| GDN recurrent (36 layers) | `[48, 128, 128]` FP32 | 3 MiB per layer |
| PLE token history | 2 tokens | |
| PLE conv | `[10240, 9]` | |
| positions | per position, 3 axes | needed for the indexer's group RoPE |

Speculative decoding must be able to roll this state back to any accepted prefix of a draft. The
context cache (GOAL.md §3) snapshots exactly this set.
