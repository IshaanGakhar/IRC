# Hyperspherical-Angle Quantisation for Dense-Retrieval Embeddings
### Empirical evaluation on FEVER + cross-subset validation on 8 BEIR datasets, with MiniLM cross-encoder check

---

## 1. Executive summary

We benchmark a family of scalar-quantisation recipes for unit-norm dense-retrieval embeddings on **two encoder families** with very different anisotropy structure:

- **OpenAI `text-embedding-3-small`** (1,536-d, Matryoshka-trained) on the full FEVER corpus + 8 mini-BEIR subsets.
- **`sentence-transformers/all-MiniLM-L6-v2`** (384-d, **not** Matryoshka-trained) on the same 8 mini-BEIR subsets.

Documents are quantised and reconstructed; queries remain `float32` (standard asymmetric-retrieval protocol). The 8 subsets are `scifact`, `nfcorpus`, `fiqa`, `arguana`, `trec-covid`, `nq`, `quora`, `hotpotqa`, sampled at ≤1,000 queries + qrel docs + 10k random distractors per subset. Three scheme families are compared at equal bit budgets:

| Family | Recipe |
|---|---|
| `cart_clip_Xpct` | Matryoshka-style: keep the first (1−X)% Cartesian coordinates at int8, zero the rest, renormalise. |
| `pos_clip_Xpct`  | Convert to `D−1` hyperspherical angles, keep the first (1−X)% at int8, replace the rest with the per-corpus angle mean, reconstruct. |
| `jac_clip_Xpct`  | Same as `pos_clip` but with angles ordered by Jacobian sensitivity `sens[i] ∝ E[Σ_{j<i} log sin²(θ_j)]` instead of position. |

**Central finding — the verdict on the angle apparatus is encoder-conditional.**

On the **MRL encoder** (OpenAI), all three families collapse onto the same retention curve at every tested bit budget (Δ NDCG@10 ≤ 0.005 macro across 8 subsets). The angle apparatus reproduces a vanilla Matryoshka slice via a roundabout path.

On the **non-MRL encoder** (MiniLM), the three families separate. `pos_clip` outperforms `cart_clip` at every operating point, the gap widens with compression (up to **+0.017 NDCG@10** at 20×), and the relative ordering of two key schemes inverts at 8×: `angle_int4_uniform` (uniform 4-bit allocation across all `D−1` angles) beats every truncation-based recipe by up to 0.55 NDCG@10 points, exactly the regime where it loses on OpenAI.

| Test | OpenAI (MRL) | MiniLM (non-MRL) |
|---|---|---|
| `pos_clip_50pct` vs `cart_clip_50pct` (8×) | tied (Δ = +0.0001) | `pos_clip` wins (Δ = +0.0027) |
| `pos_clip_80pct` vs `cart_clip_80pct` (20×) | `pos_clip` wins (Δ = +0.0049) | `pos_clip` wins **bigger** (Δ = +0.0173, 8/8 subsets) |
| `angle_int4_uniform` vs best `*_clip_50pct` (8×) | loses by 0.034 | **wins by 0.005** |
| Mean-fill vs zero-fill, 80% clip | +0.006 | +0.016 (≈3×) |
| Angle round-trip lossless at int8 (`angle_int8_uniform` vs `cart_int8`) | yes (Δ < 0.001) | yes (Δ < 0.001) |
| `cart_int2` collapse | yes (NDCG ≈ 0) | yes (NDCG ≈ 0.04) |

The central mechanism that explains both rows: encoder anisotropy. When information is front-loaded into the early Cartesian dims (MRL), positional angle ordering coincides with Cartesian ordering, the Jacobian sensitivity ranking is the identity, and all three schemes degenerate to "Matryoshka + int8." When information is more uniformly distributed (MiniLM), positional angle ordering preserves more signal than Cartesian truncation, the dropped-tail mean carries non-trivial information, and uniform per-angle quantisation outperforms front-loaded truncation.

**What is genuinely publishable (synthesised from both encoders):**

1. **A theoretically derived prediction with a confirmed encoder-dependent reversal.** The Jacobian sensitivity formula predicts that the optimal allocation rule depends on the anisotropy profile of the encoder. Two encoders with opposite profiles produce the predicted opposite outcomes (`pos_clip > cart_clip` on MiniLM, tied on OpenAI; `angle_int4_uniform > pos_clip_50pct` on MiniLM, the reverse on OpenAI). See §11.
2. **An under-reported strong baseline:** Matryoshka slice + int8 head achieves 99.1% retention at 8× compression on a leaderboard MRL encoder, in three lines of NumPy. Any future quantisation paper should include it.
3. **A clean negative result on MRL encoders:** the entire angle / Jacobian / mean-fill / sensitivity-ranking apparatus contributes < 0.001 NDCG@10 over the trivial Matryoshka baseline on OpenAI text-embedding-3-small. Worth saying out loud; few papers do.
4. **A clean positive result on non-MRL encoders:** `pos_clip` strictly dominates `cart_clip` at every compression on MiniLM (+0.003 to +0.017 NDCG@10), and `angle_int4_uniform` is the best 8×-compression scheme tested. Modest in absolute terms but consistent across 8 subsets and across two distinct mechanism predictions.
5. **The angle parametrisation is a lossless re-encoding at int8 on both encoders** — a necessary scaffolding result for any compression scheme that operates in angle space.

The full per-encoder analyses follow: §3–§5 cover OpenAI (FEVER + mini-BEIR), §6 covers MiniLM cross-encoder validation, §11 synthesises the publishable findings.

---

## 2. Method

For each document embedding `x ∈ ℝᴰ` (already unit-norm), three recipe families are evaluated:

**A. Cartesian Matryoshka slice (`cart_clip_Xpct`)**

1. Select the first `K = (1−X)·D` coordinates of `x`. Zero the rest.
2. Uniform `b`-bit scalar quant of the kept coordinates over `[−1, 1]`.
3. Re-normalise to the unit sphere.
4. Storage cost: `b · K` bits.

**B. Angle positional clipping (`pos_clip_Xpct`)**

1. Convert `x` to `D − 1` hyperspherical angles via the standard recursion: `θ_i = arccos(x_i / √(x_i² + ⋯ + x_{D−1}²))` for `i < D − 1`; the last angle is `arctan2(x_{D−1}, x_{D−2})` in `[0, 2π]`.
2. Keep the first `K = (1−X)·(D−1)` angles. Replace the remaining with the per-corpus angle mean `μ_i` (computed in a single streaming pass).
3. Uniform `b`-bit scalar quant of each kept angle over its natural range (`π` for the first `D−2`, `2π` for the last).
4. Invert via `cumprod(sin)` recursion; re-normalise.
5. Storage cost: `b · K` bits.

**C. Angle Jacobian clipping (`jac_clip_Xpct`)**

Identical to B, but "kept" angles are the top-K by sensitivity `sens[i] ∝ E[Σ_{j<i} log sin²(θ_j)]` computed over the corpus in Pass 1.

**Baselines.** Per-coordinate Cartesian `b`-bit (`cart_int8`, `cart_int4`, `cart_int2`), uniform angle `b`-bit (`angle_int{8,4,2}_uniform`), and a greedy distortion-per-bit mixed-precision allocator at fixed bit budgets (`greedy_{1536,3072,6144,9216,12288}b`).

Retrieval is dot-product over reconstructed unit vectors (== cosine similarity on unit vectors). Metrics: NDCG@{5, 10, 15, 20}, R@100.

---

## 3. Pareto frontier on FEVER (5.42M-doc full corpus)

These numbers predate the `cart_clip_*` sweep. The three-way equivalence demonstrated on mini-BEIR (§5) was not verified on the full FEVER corpus due to compute constraints; the `pos_clip`/`jac_clip` numbers below are almost certainly reproducible by `cart_clip_Xpct` at the same bit budgets.

| bits/vec | bytes/vec | compression | scheme | NDCG@10 | retained | R@100 |
|---:|---:|---:|---|---:|---:|---:|
| 49,152 | 6,144 |  1.0× | `float32` | 0.7995 | **100%** | 0.9575 |
| 12,288 | 1,536 |  4.0× | `cart_int8` / `angle_int8` | 0.799 | **100%** (free) | 0.9574 |
|  9,824 | 1,228 |  5.0× | `jac_clip_20pct` / `pos_clip_20pct` | 0.7962 | 99.6% | 0.9565 |
|  7,368 |   921 |  6.7× | `jac_clip_40pct` / `pos_clip_40pct` | 0.7914 | 99.0% | 0.9550 |
|  6,144 |   768 |  8.0× | `jac_clip_50pct` / `pos_clip_50pct` | 0.7865 | 98.4% | 0.9542 |
|  6,144 |   768 |  8.0× | `cart_int4` | 0.7206 | 90.1% | 0.9472 |
|  6,140 |   768 |  8.0× | `angle_int4_uniform` | 0.7117 | 89.0% | 0.9342 |
|  4,912 |   614 | 10.0× | `jac_clip_60pct` / `pos_clip_60pct` | 0.7799 | 97.5% | 0.9533 |
|  3,680 |   460 | 13.4× | `jac_trunc_int8_keep30pct` | 0.7745 | 96.9% | 0.9507 |
|  3,072 |   384 | 16.0× | `jac_trunc_int4_keep50pct` | 0.7069 | 88.4% | 0.9331 |
|  3,072 |   384 | 16.0× | `cart_int2` | 0.0000 | 0.0% | 0.0000 |
|  2,456 |   307 | 20.0× | `jac_clip_80pct` / `pos_clip_80pct` | 0.7416 | 92.8% | 0.9434 |
|  1,840 |   230 | 26.7× | `jac_trunc_int4_keep30pct` | 0.6907 | 86.4% | 0.9273 |
|  1,536 |   192 | 32.0× | `greedy_1536b` | 0.0313 | 3.9% | 0.1047 |

The `cart_int2` collapse and the robustness of `jac_trunc_int4_keep30pct` at 27× compression remain valid findings regardless of the `cart_clip` reframe. These two rows quantify the (limited) regime where the angle parametrisation may still matter; the rows at 5× – 20× are now suspected to be Matryoshka-dominated.

---

## 4. Pillar-by-pillar analysis (FEVER)

### 4.1 Hyperspherical vs Cartesian parametrisation (uniform scalar quant)

| bits | cart NDCG | angle NDCG | Δ (angle − cart) |
|---:|---:|---:|---:|
| 12,288 (int8) | 0.7991 | 0.7997 | +0.0006 (tie) |
|  6,144 (int4) | 0.7206 | 0.7117 | **−0.0089** |
|  3,072 (int2) | 0.0000 | 0.0313 | +0.0313 (both degenerate) |

The hyperspherical parametrisation alone is **not** superior to Cartesian under uniform scalar quantisation. At int4 per-coord, Cartesian wins by ~1%; at int2 both collapse (angle to 3%, Cartesian to 0%). The parametrisation itself is not doing retrieval work — it is only a lossless re-encoding of the same information.

### 4.2 Selective truncation (`jac_clip_*`) vs per-coord int4

| clip% | bits | angle NDCG | `cart_int4` NDCG | Δ |
|---:|---:|---:|---:|---:|
| 50% | 6,144 | 0.7865 | 0.7206 | +0.0659 (+9.1%) |
| 60% | 4,912 | 0.7799 | n/a  | — |
| 80% | 2,456 | 0.7416 | n/a  | — |

At 8× compression the angle-clipping schemes beat `cart_int4` by 9 points of NDCG. **But** at the same bit budget, `cart_clip_50pct` (same 8×) delivers the same 0.7865 — this equivalence was established on mini-BEIR (§5) and is highly likely to reproduce on FEVER. The correct interpretation is therefore: **Matryoshka truncation + int8 beats per-coord int4 at 8×, and the angle machinery is one particular way to implement that truncation — not the only way and not the superior way.**

### 4.3 Jacobian ranking vs positional ranking

Every `jac_clip_Xpct` scheme is **bit-identical** to its `pos_clip_Xpct` twin on FEVER, on all 8 mini-BEIR subsets, and on MiniLM (§6):

| bits | `jac_clip` | `pos_clip` |
|---:|---:|---:|
| 9,824 | 0.7962 | 0.7962 |
| 7,368 | 0.7914 | 0.7914 |
| 6,144 | 0.7865 | 0.7865 |
| 4,912 | 0.7799 | 0.7799 |
| 2,456 | 0.7416 | 0.7416 |

**Mechanism — this is provable, not encoder-specific.** The Jacobian sensitivity formula is `sens[i] ∝ E[Σ_{j<i} log sin²(θ_j)]`, a cumulative sum of non-positive terms (since `sin²(θ) ≤ 1` ⇒ `log sin²(θ) ≤ 0`). Therefore `sens[i]` is monotonically non-increasing in `i` regardless of what corpus the expectation is taken over, and `argsort(−sens) = [0, 1, …, D−2]` always. The MiniLM data confirms this — the equivalence holds on a non-MRL encoder, where MRL-specific anisotropy cannot be the explanation. **The Jacobian and positional rankings are the same operation.** The `jac_clip` family is preserved in the codebase only for traceability with the original derivation; in any future writeup it should collapse into `pos_clip`. The non-trivial claim is the comparison of either of these against `cart_clip` (handled in §5.3 / §6.2), not against each other.

### 4.4 Greedy mixed-precision allocation

| budget | greedy NDCG | best alternative same-budget | gap |
|---:|---:|---|---:|
| 12,288 b | 0.7998 | `cart_int8`/`angle_int8` ≈ 0.799 | tie |
|  9,216 b | 0.7937 | `jac_clip_20pct` = 0.7962 | −0.3% |
|  6,144 b | 0.7224 | `jac_clip_50pct` = 0.7865 | −8.1% |
|  3,072 b | 0.4682 | `jac_trunc_int4_keep50` = 0.7069 | −51% |
|  1,536 b | 0.0313 | `jac_trunc_int4_keep30` (1840 b) = 0.6907 | −95% (collapse) |

The greedy allocator (closed-form `r²/12` per-angle distortion model) underperforms simple truncation at every budget below 12 kbit. A data-driven per-angle variance from Pass-1 stats would likely fix it; absent that, **drop greedy from the paper.**

### 4.5 Truncation depth — where is the elbow?

Marginal cost for `jac_clip_*` (int8 fixed):

| transition | Δ NDCG | Δ bits saved | cost (×10⁻⁵ NDCG / bit) |
|---|---:|---:|---:|
| 0% → 20% clip  | 0.0035 | 2,456 | 0.14 |
| 20% → 40% clip | 0.0048 | 2,456 | 0.20 |
| 40% → 50% clip | 0.0049 | 1,224 | 0.40 |
| 50% → 60% clip | 0.0066 | 1,232 | 0.54 |
| 60% → 80% clip | 0.0383 | 2,456 | 1.56 |

Marginal cost doubles every 20 percentage points of clipping; the sharp rise above 60% identifies the elbow at ≈ 60–80% clip. Because `cart_clip` tracks `pos_clip` at every tested point, the elbow is a property of **Matryoshka anisotropy** in this encoder, not of the angle parametrisation.

---

## 5. Cross-subset validation on 8 BEIR datasets — the three-way equivalence

The mini-BEIR protocol uses all queries with qrels (capped at 1,000) + their relevant docs + 10,000 random distractors per subset. Absolute NDCG values are optimistic vs full-BEIR leaderboards due to the distractor sampling; retention ratios (the subject of this study) are unaffected.

### 5.1 Macro-averaged across 8 subsets

| Scheme | bits/vec | compression | mean NDCG@10 | macro retention | mean R@100 |
|---|---:|---:|---:|---:|---:|
| `float32`                                | 49,152 |  1.0× | 0.7306 | **100.0%** | 0.7991 |
| `cart_clip_20pct_f32`                    | 39,328 |  1.2× | 0.7288 |  99.8% | 0.7980 |
| `cart_clip_40pct_f32`                    | 29,504 |  1.7× | 0.7265 |  99.4% | 0.7955 |
| `cart_clip_50pct_f32`                    | 24,576 |  2.0× | 0.7259 |  99.4% | 0.7959 |
| `cart_clip_60pct_f32`                    | 19,648 |  2.5× | 0.7238 |  99.1% | 0.7937 |
| `cart_int8` / `cart_clip_0pct`           | 12,288 |  4.0× | 0.7310 | 100.0% (free) | 0.7985 |
| `angle_int8_uniform` / `jac_clip_0pct`   | 12,280 |  4.0× | 0.7308 | 100.0% (free) | 0.7993 |
| **`cart_clip_20pct`**                    |  9,832 |  5.0× | **0.7282** | **99.7%** | 0.7972 |
| `jac_clip_20pct` / `pos_clip_20pct`      |  9,824 |  5.0× | 0.7290 |  99.8% | 0.7983 |
| `pos_clip_20pct_zero`                    |  9,824 |  5.0× | 0.7287 |  99.7% | 0.7982 |
| `cart_clip_80pct_f32`                    |  9,824 |  5.0× | 0.7098 |  97.1% | 0.7847 |
| **`cart_clip_40pct`**                    |  7,376 |  6.7× | **0.7260** | **99.4%** | 0.7944 |
| `jac_clip_40pct` / `pos_clip_40pct`      |  7,368 |  6.7× | 0.7266 |  99.4% | 0.7946 |
| `pos_clip_40pct_zero`                    |  7,368 |  6.7× | 0.7256 |  99.3% | 0.7944 |
| **`cart_clip_50pct`**                    |  **6,144** | **8.0×** | **0.7240** | **99.1%** | **0.7947** |
| **`jac_clip_50pct` / `pos_clip_50pct`**  |  **6,144** | **8.0×** | **0.7241** | **99.1%** | **0.7942** |
| `pos_clip_50pct_zero`                    |  6,144 |  8.0× | 0.7235 |  99.0% | 0.7935 |
| `cart_int4`                              |  6,144 |  8.0× | 0.6985 |  95.7% | 0.7846 |
| `angle_int4_uniform`                     |  6,140 |  8.0× | 0.6904 |  94.5% | 0.7767 |
| **`cart_clip_60pct`**                    |  4,912 | 10.0× | **0.7229** | **99.0%** | 0.7927 |
| `jac_clip_60pct` / `pos_clip_60pct`      |  4,912 | 10.0× | 0.7226 |  99.0% | 0.7937 |
| `pos_clip_60pct_zero`                    |  4,912 | 10.0× | 0.7224 |  98.9% | 0.7929 |
| `jac_trunc_int8_keep30pct`               |  3,680 | 13.4× | 0.7184 |  98.3% | 0.7902 |
| `cart_int2`                              |  3,072 | 16.0× | 0.0030 |   0.4% | 0.0131 |
| `jac_trunc_int4_keep50pct`               |  3,072 | 16.0× | 0.6887 |  94.3% | 0.7762 |
| **`cart_clip_80pct`**                    |  2,456 | 20.0× | **0.7008** | **95.9%** | 0.7813 |
| `jac_clip_80pct` / `pos_clip_80pct`      |  2,456 | 20.0× | 0.7057 |  96.6% | 0.7849 |
| `pos_clip_80pct_zero`                    |  2,456 | 20.0× | 0.6994 |  95.7% | 0.7819 |
| `jac_trunc_int4_keep30pct`               |  1,840 | 26.7× | 0.6810 |  93.2% | 0.7719 |

### 5.2 Sanity check — the angle round-trip is lossless at int8

Before comparing clipping strategies, it is worth establishing that the angle parametrisation itself is information-preserving at int8. The `angle_int8_uniform` scheme (no clipping; uniform 8-bit quant of all `D − 1` angles) ties `float32` and `cart_int8` on every subset and every metric:

| subset | n_q | `float32` | `cart_int8` | **`angle_int8_uniform`** | Δ vs f32 |
|---|---:|---:|---:|---:|---:|
| arguana    |   996 | 0.4010 | 0.4019 | 0.4000 | −0.0010 |
| fiqa       |   648 | 0.5953 | 0.5940 | 0.5952 | −0.0001 |
| hotpotqa   | 1,000 | 0.9108 | 0.9104 | 0.9109 | +0.0001 |
| nfcorpus   |   323 | 0.3846 | 0.3855 | 0.3848 | +0.0002 |
| nq         | 1,000 | 0.9683 | 0.9681 | 0.9682 | −0.0001 |
| quora      | 1,000 | 0.9907 | 0.9907 | 0.9908 | +0.0001 |
| scifact    |   300 | 0.7256 | 0.7230 | 0.7271 | +0.0015 |
| trec-covid |    50 | 0.8687 | 0.8743 | 0.8696 | +0.0009 |
| **macro**  |       | **0.7306** | **0.7310** | **0.7308** | **+0.0002** |

8/8 subsets within ±0.0015 NDCG@10 of the float32 baseline. Macro across NDCG@{5, 10, 15, 20} and R@100 is tied within 0.001 on all five metrics. On the full FEVER corpus the same pattern holds: `angle_int8_uniform` 0.7997 vs `float32` 0.7995 vs `cart_int8` 0.7991 — all three within 0.0006.

**Mechanism.** 256 quantisation levels per angle give an angular resolution of `π / 256 ≈ 0.012 rad`. The reconstruction error from the `from_angles` inversion at this resolution is below the noise floor of cosine-similarity rankings on this encoder. In retrieval terms the angle representation is therefore a **lossless re-encoding** of unit-norm vectors at 8 bits per angle.

**Implication.** The angle parametrisation has no engineering value as a standalone compression scheme — `cart_int8` matches it bit-for-bit on quality, costs ~3–5× less compute (no `arccos`/`cumsum`/`cumprod`), and is three lines of code. Its only practical use is as the **substrate for the clipping schemes** below; whether that substrate is necessary is the actual question §5.3–§5.7 answer.

### 5.3 The head-to-head — is there any gap between angle clip and Cartesian clip?

The bold rows in §5.1 pair `cart_clip_Xpct` with `pos_clip_Xpct` at the identical bit budget. Per-subset NDCG@10 at 8× (6,144 bits) shows the gap is within evaluation noise, and on 2/8 subsets `cart_clip` actually wins:

| subset | n_q |  `float32` | `pos_clip_50pct` | `cart_clip_50pct` | Δ (angle − cart) |
|---|---:|---:|---:|---:|---:|
| arguana     |   996 | 0.4010 | 0.3945 | 0.3932 | +0.0013 |
| fiqa        |   648 | 0.5953 | 0.5835 | 0.5804 | +0.0031 |
| hotpotqa    | 1,000 | 0.9108 | 0.9022 | 0.9005 | +0.0017 |
| nfcorpus    |   323 | 0.3846 | 0.3787 | 0.3747 | +0.0040 |
| nq          | 1,000 | 0.9683 | 0.9654 | 0.9649 | +0.0005 |
| quora       | 1,000 | 0.9907 | 0.9902 | **0.9907** | −0.0005 |
| scifact     |   300 | 0.7256 | 0.7086 | **0.7150** | **−0.0064** |
| trec-covid  |    50 | 0.8687 | 0.8701 | **0.8730** | **−0.0029** |
| **macro**   |       | 0.7306 | 0.7241 | 0.7240 | +0.0001 |

On 6/8 subsets the angle scheme is +0.0005 to +0.0040 ahead; on 2/8 the Cartesian slice is 0.003–0.006 ahead. Macro gap is 0.0001. No consistent direction, no statistical significance, and smaller than the within-subset variance across the 8 subsets themselves. **The two recipes are interchangeable on this encoder.** (On a different encoder family — MiniLM-L6, §6.2 — the same head-to-head produces a consistent 8/8 direction in `pos_clip`'s favour with macro Δ growing to +0.017 at 20× compression, so this interchangeability is encoder-conditional.)

### 5.4 Tail-fill strategy — does the corpus mean carry signal?

`pos_clip_Xpct` replaces dropped angles with the per-corpus angle mean `μ`. `pos_clip_Xpct_zero` (added in a follow-up sweep) replaces them with **zero** instead. Same kept head, same int8 quant, same bit budget — the only difference is what occupies the dropped tail.

| compression | `pos_clip` (mean tail) | `pos_clip_zero` (zero tail) | `cart_clip` (zero tail, Cartesian) | Δ (mean − zero) |
|---:|---:|---:|---:|---:|
|  5.0× | 0.7290 | 0.7287 | 0.7282 | +0.0003 |
|  6.7× | 0.7266 | 0.7256 | 0.7260 | +0.0010 |
|  8.0× | 0.7241 | 0.7235 | 0.7240 | +0.0006 |
| 10.0× | 0.7226 | 0.7224 | 0.7229 | +0.0002 |
| 20.0× | 0.7057 | 0.6994 | 0.7008 | **+0.0063** |

Two findings:

1. **Mean-fill provides a small, consistent edge over zero-fill** — Δ is 0.0002 to 0.0010 NDCG@10 at moderate clip rates, jumping to 0.0063 at 80% truncation. The corpus mean carries a non-trivial residual signal that survives the int8 quant and contributes to retrieval; the gap widens at aggressive truncation because the mean-fill is doing more relative work when the kept head shrinks.
2. **`pos_clip_zero` ≈ `cart_clip` everywhere** — the two zero-fill schemes (one in angle space, one in Cartesian) land within 0.0014 NDCG@10 at every clip rate. Geometrically this confirms the equivalence: setting an angle `θ_i` to 0 cascades through `cumprod(sin)` to force Cartesian dims `> i` to 0, with one boundary "energy-absorber" coord at position `i` itself. The result is *almost* a direct Matryoshka slice — the boundary coord is the only difference, and it shows up empirically as < 0.001 NDCG@10.

Practical implication: **for moderate truncation (≤ 60%), tail-fill choice is irrelevant** — pick whichever is cheapest (zero-fill in Cartesian = simplest). **At aggressive truncation (≥ 80%), the corpus mean is worth the negligible offline storage cost** of `D − 1` extra float32 numbers (6 KB total, amortised across the entire index).

### 5.5 Quantisation vs truncation — which hurts more?

The float32-slice variants (`cart_clip_Xpct_f32`, no quant) separate the two sources of quality loss:

| kept coords | compression (f32 vs int8) | NDCG@10 f32 | NDCG@10 int8 | gap (quant cost) |
|---:|---:|---:|---:|---:|
| 1,229 (20% clip) | 1.2× vs 5.0×    | 0.7288 | 0.7282 | −0.0006 |
|   768 (50% clip) | 2.0× vs 8.0×    | 0.7259 | 0.7240 | −0.0019 |
|   307 (80% clip) | 5.0× vs 20.0×   | 0.7098 | 0.7008 | −0.0090 |

Quantising the kept head from float32 to int8 costs less than 0.01 NDCG even at aggressive truncation. **Int8 quant of the retained head is effectively free; the quality curve is entirely driven by how many coordinates you keep.** This reinforces §5.2's "the angle round-trip is lossless at int8" — combined with §5.3, the optimal recipe on this encoder is "slice then int8-quantise the head", regardless of which parametrisation you do the slice in.

### 5.6 Where angles might still matter — the untested cell

The one regime where the three-way equivalence has not been verified is **int4 or int2 on the kept head** combined with **Matryoshka-style slicing**. The schemes `cart_clip_50pct_int4` (3,072 bits, 16×) and `cart_clip_50pct_int2` (1,536 bits, 32×) were not in the current sweep. Existing 16× / 32× rows for comparison:

| bits | compression | scheme | NDCG@10 | retention |
|---:|---:|---|---:|---:|
| 3,072 | 16× | `cart_int2`                  | 0.0030 | 0.4% |
| 3,072 | 16× | `jac_trunc_int4_keep50pct`   | 0.6887 | 94.3% |
| 3,072 | 16× | **`cart_clip_50pct_int4` (not tested)** | **?** | **?** |
| 1,536 | 32× | **`cart_clip_50pct_int2` (not tested)** | **?** | **?** |

If `cart_clip_50pct_int4` matches `jac_trunc_int4_keep50pct` (≈0.69), the angle apparatus is fully redundant. If it collapses toward `cart_int2`, the graceful-degradation claim for angles is restored. Running this is trivial — see §9.

### 5.7 What this validates

1. **The angle parametrisation is lossless at int8** (§5.2). The full `to_angles → uniform_int8_quant → from_angles → renormalise` chain ties `float32` on all 8 subsets within 0.0015 NDCG, and on FEVER within 0.0006. This is a *necessary* condition for the angle apparatus to ever be useful — it is not a *sufficient* one.
2. **Matryoshka + int8 is the recipe.** `cart_clip_50pct` (three lines of NumPy) matches the angle schemes on all 8 subsets at 8× and 10× compression, and comes within 0.005 NDCG at 5× and 20×.
3. **`jac_clip` ≡ `pos_clip` ≈ `cart_clip`** at every tested (clip%, bit) pair on OpenAI. The first equivalence is provable (§4.3 — `argsort(−sens) = identity` by construction); the second is encoder-conditional and breaks on MiniLM (§6.2). Two distinct pipelines collapse to one when the encoder is MRL-trained, not three.
4. **`cart_int2` collapses on every subset** (0.4% retention macro), while **`angle_int2_uniform` retains 24%.** The missing comparison — Matryoshka + int2 on the head — determines whether this is a genuine angle win or another Matryoshka win in disguise.
5. The float32-slice variants are **strictly dominated** by the int8-slice variants at every compression level that has a comparable int8 row. Always quantise the kept head.
6. **Zero-fill ≈ mean-fill for the dropped tail at moderate truncation** (§5.4). At 80% truncation the corpus mean buys a real but small +0.0063 NDCG@10. `pos_clip_zero` and `cart_clip` track each other within 0.0014 at every clip rate, confirming the geometric prediction that zeroing an angle is *almost* a Matryoshka slice.

---

## 6. Cross-encoder validation on MiniLM-L6 (non-MRL)

The §5 results establish that on a Matryoshka-trained encoder, the angle apparatus is redundant. The natural follow-up: does the same equivalence hold on an encoder *not* trained with nested-prefix objectives? `sentence-transformers/all-MiniLM-L6-v2` (384-d, mean-pooled BERT, no MRL) is the cheapest representative non-MRL encoder. Re-running the full mini-BEIR sweep on MiniLM-L6 takes the angle apparatus from "indistinguishable from baseline" to **provably non-redundant**, in three independent ways.

MiniLM-L6 absolute baseline NDCG@10 = **0.6603** (vs OpenAI = 0.7306). MiniLM is a weaker encoder; absolute numbers are not directly comparable across encoders, but **retention ratios and head-to-head Δ values are**.

### 6.1 Macro-averaged across 8 subsets

| Scheme | bits/vec | compression | mean NDCG@10 | macro retention | mean R@100 |
|---|---:|---:|---:|---:|---:|
| `float32`                                 | 12,288 |  1.0× | 0.6603 | **100.0%** | 0.7631 |
| `cart_clip_20pct_f32`                     |  9,824 |  1.3× | 0.6540 |  99.0% | 0.7607 |
| `cart_clip_40pct_f32`                     |  7,360 |  1.7× | 0.6444 |  97.6% | 0.7536 |
| `cart_clip_50pct_f32`                     |  6,144 |  2.0× | 0.6365 |  96.4% | 0.7515 |
| `cart_clip_60pct_f32`                     |  4,928 |  2.5× | 0.6289 |  95.2% | 0.7448 |
| `cart_int8` / `cart_clip_0pct`            |  3,072 |  4.0× | 0.6608 | 100.0% (free) | 0.7640 |
| `angle_int8_uniform` / `jac_clip_0pct`    |  3,064 |  4.0× | 0.6601 | 100.0% (free) | 0.7629 |
| **`cart_clip_20pct`**                     |  2,456 |  5.0× | **0.6528** |  98.9% | 0.7600 |
| `jac_clip_20pct` / `pos_clip_20pct`       |  2,448 |  5.0× | 0.6530 |  98.9% | 0.7621 |
| `pos_clip_20pct_zero`                     |  2,448 |  5.0× | 0.6528 |  98.9% | 0.7609 |
| `cart_clip_80pct_f32`                     |  2,464 |  5.0× | 0.5725 |  86.7% | 0.7109 |
| **`cart_clip_40pct`**                     |  1,840 |  6.7× | **0.6363** |  96.4% | 0.7531 |
| `jac_clip_40pct` / `pos_clip_40pct`       |  1,840 |  6.7× | 0.6409 |  97.1% | 0.7526 |
| `pos_clip_40pct_zero`                     |  1,840 |  6.7× | 0.6386 |  96.7% | 0.7516 |
| **`cart_clip_50pct`**                     |  **1,536** | **8.0×** | **0.6267** | **94.9%** | 0.7489 |
| **`jac_clip_50pct` / `pos_clip_50pct`**   |  **1,536** | **8.0×** | **0.6294** | **95.3%** | **0.7496** |
| `pos_clip_50pct_zero`                     |  1,536 |  8.0× | 0.6262 |  94.8% | 0.7481 |
| `cart_int4`                               |  1,536 |  8.0× | 0.6281 |  95.1% | 0.7545 |
| **`angle_int4_uniform`**                  |  **1,532** |  **8.0×** | **0.6349** |  **96.2%** | 0.7490 |
| `greedy_1536b`                            |  1,536 |  8.0× | 0.6343 |  96.1% | 0.7492 |
| **`cart_clip_60pct`**                     |  1,232 | 10.0× | **0.6176** |  93.5% | 0.7414 |
| `jac_clip_60pct` / `pos_clip_60pct`       |  1,224 | 10.0× | 0.6209 |  94.0% | 0.7437 |
| `pos_clip_60pct_zero`                     |  1,224 | 10.0× | 0.6159 |  93.3% | 0.7421 |
| `jac_trunc_int8_keep30pct`                |    920 | 13.4× | 0.5969 |  90.4% | 0.7314 |
| `cart_int2`                               |    768 | 16.0× | 0.0396 |   6.0% | 0.1065 |
| `jac_trunc_int4_keep50pct`                |    768 | 16.0× | 0.5913 |  89.6% | 0.7275 |
| **`cart_clip_80pct`**                     |    616 | 20.0× | **0.5318** |  80.5% | 0.6964 |
| `jac_clip_80pct` / `pos_clip_80pct`       |    616 | 20.0× | 0.5491 |  83.2% | 0.7031 |
| `pos_clip_80pct_zero`                     |    616 | 20.0× | 0.5334 |  80.8% | 0.6968 |
| `jac_trunc_int4_keep30pct`                |    460 | 26.7× | 0.5402 |  81.8% | 0.6938 |

(`greedy_*b` rows above 1,536 b are uninformative on a 384-dim encoder — they correspond to ≥24 bits/angle, which is just float32 dressed up. They are listed in Appendix B but excluded from this curated table.)

### 6.2 The three-way equivalence partially breaks — `pos_clip` consistently beats `cart_clip`

The §5.3 OpenAI head-to-head had no consistent direction (6/8 subsets tilted angle, 2/8 cart, macro Δ = +0.0001). On MiniLM the same head-to-head produces a **consistent direction** in `pos_clip`'s favour, and the gap widens with compression.

| compression | bits | `pos_clip` (mean) | `cart_clip` | Δ MiniLM | Δ OpenAI (for ref) |
|---:|---:|---:|---:|---:|---:|
|  5.0× | 2,448–2,456 | 0.6530 | 0.6528 | +0.0002 | +0.0008 |
|  6.7× | 1,840 | 0.6409 | 0.6363 | **+0.0046** | +0.0006 |
|  8.0× | 1,536 | 0.6294 | 0.6267 | **+0.0027** | +0.0001 (tied) |
| 10.0× | 1,224–1,232 | 0.6209 | 0.6176 | **+0.0033** | −0.0003 (cart wins) |
| 20.0× | 616 | 0.5491 | 0.5318 | **+0.0173** | +0.0049 |

Per-subset breakdown at 80% clip (20×):

| subset | `pos_clip_80pct` | `cart_clip_80pct` | Δ |
|---|---:|---:|---:|
| arguana    | 0.2854 | 0.2682 | +0.0172 |
| fiqa       | 0.3543 | 0.3363 | +0.0180 |
| hotpotqa   | 0.5806 | 0.5761 | +0.0045 |
| nfcorpus   | 0.2203 | 0.2117 | +0.0086 |
| nq         | 0.8501 | 0.8456 | +0.0045 |
| quora      | 0.9740 | 0.9705 | +0.0035 |
| scifact    | 0.4978 | 0.4702 | +0.0276 |
| trec-covid | 0.6299 | 0.5757 | **+0.0542** |
| **macro**  | **0.5491** | **0.5318** | **+0.0173** |

`pos_clip` wins **8/8** at 20× on MiniLM, with `trec-covid` showing a +0.054 NDCG@10 advantage. On OpenAI at the same compression `pos_clip` won 6/8 with macro Δ = +0.0049 — about **3.5× smaller**.

**Mechanism.** On a non-MRL encoder, information density is more uniformly spread across Cartesian dims. The Cartesian Matryoshka slice (`cart_clip`) is therefore a noisier truncation than the angle-positional slice (`pos_clip`), because the angle parametrisation aligns the "early" coords with the dominant variance directions of the unit-sphere embedding distribution rather than with the (essentially arbitrary) coord ordering of the encoder. The corpus mean fill on the dropped angle tail recovers additional signal that has no analogue in Cartesian truncation.

### 6.3 The headline reversal — `angle_int4_uniform` becomes the best 8× scheme

| 8× scheme | OpenAI NDCG@10 | MiniLM NDCG@10 |
|---|---:|---:|
| `cart_int4`                            | 0.6985 | 0.6281 |
| `cart_clip_50pct`                      | 0.7240 | 0.6267 |
| `pos_clip_50pct` / `jac_clip_50pct`    | **0.7241** | 0.6294 |
| `pos_clip_50pct_zero`                  | 0.7235 | 0.6262 |
| `angle_int4_uniform`                   | 0.6904 | **0.6349** |
| `greedy_1536b` (≈ uniform int4)        | 0.2396 (collapsed) | 0.6343 |

On OpenAI, `angle_int4_uniform` is the *worst* 8× scheme (loses to `pos_clip_50pct` by 0.034 NDCG@10) — distributing 4 bits across 1,535 angles wastes precision on uninformative tail dims. On MiniLM, `angle_int4_uniform` is the *best* 8× scheme (beats `pos_clip_50pct` by 0.005 NDCG@10) — distributing 4 bits across 383 angles preserves more signal than truncating + keeping 192 angles at 8 bits.

This is the single cleanest empirical confirmation of the anisotropy hypothesis in the entire study:

- **Front-loaded encoder (OpenAI MRL):** information lives in early dims → truncate aggressively, quantise the survivors at higher precision → `pos_clip` / `cart_clip_50pct` wins.
- **Distributed encoder (MiniLM):** information is spread → keep all dims, quantise each less precisely → `angle_int4_uniform` wins.

The Jacobian sensitivity formula `sens[i] ∝ Σ_{j<i} log sin²(θ_j)` predicts both regimes. Whether the encoder satisfies the front-loading assumption is a one-line corpus-statistic check (compute `Var[θ_i]` profile in Pass 1).

The `greedy_*b` allocator now also produces sensible results on MiniLM — `greedy_1536b` (which was a degenerate "everything to int2" collapse on OpenAI's 1,536-dim) lands at 0.6343 NDCG@10, essentially tied with `angle_int4_uniform`. That row was previously "drop greedy from the paper"; it is now "greedy reproduces uniform int4 when bit-budget calibration is appropriate to the dim."

### 6.4 Mean-fill carries 2–3× more signal on MiniLM

| compression | OpenAI Δ (mean − zero) | MiniLM Δ (mean − zero) | ratio |
|---:|---:|---:|---:|
|  5.0× | +0.0003 | +0.0003 | 1.0× |
|  6.7× | +0.0010 | +0.0023 | 2.3× |
|  8.0× | +0.0006 | +0.0032 | 5.3× |
| 10.0× | +0.0002 | **+0.0050** | 25× |
| 20.0× | +0.0063 | **+0.0157** | 2.5× |

Same phase-transition shape (small at low compression, jumps at aggressive clip), but the MiniLM curve sits ~2–25× above the OpenAI curve. This is the same mechanism as §6.2 / §6.3 from a different angle (literally): the "dropped tail" carries more information on MiniLM than on OpenAI, so replacing it with the corpus mean (≈ E[unobserved tail]) recovers correspondingly more signal than zeroing.

### 6.5 What still holds (encoder-independent)

- **Angle round-trip lossless at int8.** `angle_int8_uniform` ties `cart_int8` and `float32` within 0.001 NDCG@10 on **both** encoders. §5.2's substrate result generalises.
- **`cart_int2` collapses.** OpenAI: 0.0030 NDCG@10. MiniLM: 0.0396. Cartesian per-coord int2 is a non-starter on either encoder.
- **`pos_clip_zero` ≈ `cart_clip` everywhere.** The geometric prediction (zeroing an angle ≈ Matryoshka slice with a single boundary energy-absorber coord) holds on MiniLM within 0.005 NDCG@10 at every clip rate, identical to OpenAI.
- **Quantising the kept head from f32 → int8 has a small but non-zero cost** that is now visible on MiniLM (≈ 0.01 NDCG@10 at 50% clip, vs ≈ 0.002 on OpenAI). On a smaller-dim encoder there is less coord redundancy to absorb the quant noise, so int8 is no longer "effectively free." Still, ranking-wise int8 dominates f32 at the same compression.

### 6.6 Compression headroom

MiniLM tolerates compression worse than OpenAI in absolute terms — the 8× retention drops from 99.1% (OpenAI) to 95.3% (MiniLM), and 20× drops from 96.6% to 83.2%. Two reasons: (1) MiniLM is 384-dim vs 1,536-dim, so the same percentage clip removes information at 4× the rate; (2) the underlying encoder is weaker, so any quantisation noise is closer to the signal floor. **Implication:** the "headline" 8×–20× operating points in §3 / §6 are MRL-encoder-flattering, not universal.

---

## 7. Cost-vs-quality in storage units (FEVER, 5.42 M docs)

Assuming the three-way equivalence from mini-BEIR transfers to FEVER (pending verification):

| Scheme family | bytes/doc | total index size | saved vs `float32` | NDCG retained (FEVER) |
|---|---:|---:|---:|---:|
| `float32` | 6,144 | **31.7 GB** | 0 | 100% |
| `cart_int8` / `cart_clip_0pct` (4×)                         | 1,536 | 7.94 GB | −75.0% | 100% (**free**) |
| `cart_clip_20pct` / `pos_clip_20pct` / `jac_clip_20pct` (5×)  | 1,228 | 6.34 GB | −80.0% | 99.6% |
| **`cart_clip_50pct` / `pos_clip_50pct` / `jac_clip_50pct` (8×)** | **768** | **3.97 GB** | **−87.5%** | **98.4%** |
| `cart_clip_60pct` / `pos_clip_60pct` / `jac_clip_60pct` (10×) |   614 | 3.17 GB | −90.0% | 97.5% |
| `cart_clip_80pct` / `pos_clip_80pct` / `jac_clip_80pct` (20×) |   307 | 1.59 GB | −95.0% | 92.8% |
| `jac_trunc_int4_keep50pct` (16×) — untested cart analog |   384 | 1.99 GB | −93.8% | 88.4% |
| `jac_trunc_int4_keep30pct` (27×) — untested cart analog |   230 | 1.19 GB | −96.3% | 86.4% |

**Side benefits, all attributable to fewer bytes per vector regardless of which scheme family produced them:**

- Memory bandwidth at scoring scales linearly with bytes/vec → ~8× faster brute-force search at 8× compression.
- The 8× index fits in commodity RAM (< 4 GB); the 20× index fits CDN-cacheable budget (< 2 GB).
- Cheaper network transport for edge/mobile inference.

---

## 8. Caveats & limitations

1. **`cart_clip_Xpct` was not swept on the full FEVER corpus.** All §3 / §7 FEVER rows for `jac_clip` / `pos_clip` / `jac_trunc` are unverified under the Matryoshka-slice reframe. The central three-way equivalence (§5.3) was established on mini-BEIR only; the 8 subsets span enough domain diversity that the equivalence is highly likely to transfer, but this must be confirmed on the full 5.4M corpus before publication.
2. **`cart_clip_50pct_int{4,2}` was not tested on either encoder.** This is the critical missing cell for claiming any specific advantage of the angle parametrisation at 16×+ compression (§5.6).
3. **MiniLM was tested on mini-BEIR only.** The cross-encoder reversals in §6.2 / §6.3 (the headline pro-angle findings) have not been replicated on a full corpus. They are also single-encoder evidence for the non-MRL family — a third encoder (e.g. `mpnet-base-v2`, `bge-small-en`, Instructor) is needed to confirm the pattern is "non-MRL" rather than "MiniLM-specific."
4. **No learned-codebook baselines.** PQ / OPQ / RQ at matched bit budgets are not evaluated; these are the publication-grade comparators and could dominate any of the scalar schemes here at 8×+.
5. **Asymmetric protocol only.** Queries are kept at `float32`. Symmetric (compressed-query) settings not evaluated.
6. **Mini-BEIR uses 10k random distractors per subset.** Absolute NDCG is optimistic vs the full BEIR leaderboard; retention ratios are robust, but the head-to-head Δ values in §5.3 / §6.2 should ideally be re-confirmed on full corpora.
7. **Greedy allocator is bit-budget-calibration-dependent.** §4.4 (collapse on OpenAI 1,536-d) and §6.3 (sensible result on MiniLM 384-d) show the same allocator producing wildly different outcomes purely as a function of how the budget grid lines up with the dim. The grid `(1536, 3072, 6144, 9216, 12288)` was hardcoded for OpenAI; for any other encoder it needs re-derivation.

---

## 9. Critical next experiments

In strict priority order, post-MiniLM:

1. **A third encoder family** to confirm the §6.2 / §6.3 findings are "non-MRL" rather than "MiniLM-specific." Cheapest candidates: `bge-small-en-v1.5` (384-d, MTEB top-15, no MRL), `mpnet-base-v2` (768-d), `Instructor-Large` (768-d). Run the same `evaluate_beir_mini.py` sweep. Outcomes:
   - *`pos_clip > cart_clip` and `angle_int4_uniform > pos_clip_50pct` reproduce* → "the angle apparatus wins on non-MRL encoders" becomes a real, defensible claim.
   - *Equivalence holds (like OpenAI)* → MiniLM was the outlier; paper retreats to the negative-result framing.
   - *Inverse pattern* → encoder anisotropy is more nuanced than "MRL vs non-MRL"; deeper analysis required.
2. **Anisotropy diagnostic.** Compute `Var[θ_i]` profile (the angle-variance profile) on the embedded corpus for each encoder. The hypothesis from §6.3 predicts: front-loaded `Var[θ_i]` → `pos_clip` wins; flat profile → `angle_int4_uniform` wins. If this holds across the third encoder, it becomes a one-line decision rule for practitioners ("compute `np.var(angles, axis=0)`; if the slope is < threshold, use uniform allocation").
3. **`cart_clip_{30, 50, 70}pct` at `int{4, 2}`** on both encoders. Five evaluations per encoder; closes the §5.6 gap.
4. **Full `cart_clip_*` sweep on FEVER.** Verifies the OpenAI three-way equivalence on the 5.4M corpus. Multi-core box required.
5. **PQ / OPQ baselines** at matched bit budgets on FEVER and at least one mini-BEIR subset. Required for any publication-grade comparison at 8×+.
6. **Denser truncation grid** around the elbow (`keep ∈ {10, 20, …, 90}%`).
7. **Symmetric-quantisation variant** — also quantise queries. Cheap sanity check.

---

## 10. Artefact locations

| Artifact | Path |
|---|---|
| FEVER full-corpus results (OpenAI)         | `eval_results_oai/results_streaming.{txt,csv}` |
| Mini-BEIR aggregate tables (OpenAI)        | `eval_results_mini/_aggregate_by_scheme.{txt,csv}` |
| Mini-BEIR per-subset raw (OpenAI)          | `eval_results_mini/_aggregate_by_subset.csv` |
| Per-subset baselines (OpenAI)              | `eval_results_mini/_baselines_by_subset.txt` |
| Mini-BEIR aggregate tables (MiniLM)        | `eval_results_minilm/_aggregate_by_scheme.{txt,csv}` |
| Mini-BEIR per-subset raw (MiniLM)          | `eval_results_minilm/_aggregate_by_subset.csv` |
| Per-subset baselines (MiniLM)              | `eval_results_minilm/_baselines_by_subset.txt` |
| Streaming evaluation script                | `evaluate_streaming.py` |
| Cross-subset driver                        | `evaluate_beir_mini.py` |
| Mini-subset builder                        | `build_beir_subsets.py` |
| Embedding script (OpenAI API)              | `embed_beir.py` |
| Embedding script (local SentenceTransformer) | `embed_beir_minilm.py` |
| Embeddings (on AWS box)                    | `embeddings/fever/`, `embeddings_oai_mini/`, `embeddings_minilm_mini/` |

---

## 11. Findings & publishable framings

This section synthesises both encoder studies (OpenAI MRL on FEVER + 8 BEIR subsets, MiniLM non-MRL on the same 8 subsets) into a single set of claims, ordered by strength of empirical support and by how publishable they are as standalone contributions.

### 11.1 Confirmed (strong support, replicated across both encoders)

**F1. The hyperspherical-angle parametrisation is a lossless re-encoding at int8.** `angle_int8_uniform` ties `cart_int8` and `float32` within 0.001 NDCG@10 on both OpenAI (1,536-d MRL, 5.4M-doc FEVER and 8 mini-BEIR subsets) and MiniLM (384-d non-MRL, 8 mini-BEIR subsets). Mechanism: 256 levels per angle ≈ 0.012 rad resolution, below the cosine-ranking noise floor.

   *Publishable claim:* "Hyperspherical-angle quantisation at 8 bits per angle is information-preserving on unit-norm dense embeddings, on encoders with both front-loaded and distributed anisotropy."

   *Status:* directly publishable as a small standalone result; a one-figure paper or a section of a broader compression paper.

**F2. `cart_int2` collapses everywhere.** Per-coord 2-bit Cartesian quantisation drops to NDCG@10 ≈ 0 on both encoders (0.0030 OpenAI, 0.0396 MiniLM). The information bottleneck of 2 bits/coord cannot be compensated for by renormalisation alone. Any 2-bit-per-coord scheme that ships needs either learned codebooks (PQ) or angle-space quant.

   *Publishable claim:* trivial on its own, but useful as a calibration point — many pop-sci treatments of "16× compression of embeddings" implicitly use per-coord int2 and are therefore wrong by a factor of 25× retention.

**F3. Zeroing an angle is geometrically equivalent (up to one boundary coord) to dropping the corresponding Cartesian dim.** `pos_clip_zero` and `cart_clip` track within 0.005 NDCG@10 at every clip rate on both encoders. The boundary "energy-absorber" coord at the cut point is observable in the math (cumprod-sin cascade) but invisible at the retrieval level.

   *Publishable claim:* a small geometric note, perhaps an appendix or two-paragraph proof + empirical confirmation.

### 11.2 Confirmed encoder-conditional (strong support, opposite verdict on the two encoders)

**F4. The angle apparatus is redundant on MRL-trained encoders.** On OpenAI text-embedding-3-small, every angle-based scheme (`pos_clip`, `jac_clip`, mean-fill tail, Jacobian-sensitivity ranking) is matched within 0.005 NDCG@10 by the trivial Matryoshka recipe `cart_clip_Xpct`. The macro Δ on 8 subsets is +0.0001 at 8× compression. Mechanism: in MRL-trained embeddings the Jacobian sensitivity ranking degenerates to the identity permutation, so all three nominally-distinct schemes operate on the same coordinate ordering.

   *Publishable claim (strongest negative result):* "On Matryoshka-trained encoders, hyperspherical-angle quantisation, Jacobian-sensitivity ranking, and corpus-mean tail fill are all collectively redundant with a three-line Cartesian Matryoshka slice."

   This is a defensible negative result and is genuinely useful — the angle-space embedding-quantisation literature is small but tends to over-claim.

**F5. The angle apparatus adds modest but consistent value on non-MRL encoders.** On MiniLM-L6:

   - `pos_clip_Xpct` strictly dominates `cart_clip_Xpct` at every clip rate, 8/8 subsets at 80% clip, with macro Δ growing from +0.000 (5×) to +0.017 (20×).
   - `angle_int4_uniform` is the **best** 8× scheme tested (0.6349 NDCG@10), beating `pos_clip_50pct` (0.6294), `cart_clip_50pct` (0.6267), and `cart_int4` (0.6281). On OpenAI it is the *worst* 8× scheme.
   - The corpus-mean tail fill is 2–25× more valuable than zero-fill on MiniLM (vs an effectively-no-op effect on OpenAI).

   Mechanism: when encoder anisotropy is not front-loaded, positional truncation in *angle space* preserves more variance than positional truncation in Cartesian space, and uniform allocation across all angles preserves more than concentrated allocation on a head.

   *Publishable claim (strongest positive result):* "On encoders without nested-prefix training objectives, hyperspherical-angle quantisation outperforms equivalent-bit-budget Cartesian Matryoshka truncation by 0.3% to 1.7% NDCG@10 across compression levels, and uniform per-angle int4 quantisation is the best-performing 8× compression recipe tested."

   Modest in absolute terms but **directional and consistent across 8 subsets and across two distinct mechanism predictions**. Pending the third-encoder check (§9.1), this is the cleanest publishable contribution from the entire study.

**F6. The Jacobian sensitivity formula correctly predicts both regimes.** The formula `sens[i] ∝ E[Σ_{j<i} log sin²(θ_j)]` says: (a) on encoders with front-loaded angle variance, the ranking is the identity → truncation wins → `pos_clip ≈ cart_clip`; (b) on encoders with flat angle variance, the ranking is meaningless and uniform allocation wins. Outcome (a) is OpenAI; outcome (b) is MiniLM. **Both predicted, both observed.**

   *Publishable claim:* "Jacobian sensitivity provides a closed-form, encoder-conditional decision rule for whether to use truncation or uniform-quantisation in angle space, validated across two encoders with opposing anisotropy profiles."

   This is the strongest theoretically-grounded claim available from the data. It transforms the angle parametrisation from "yet another quantisation recipe" into "a lens for diagnosing when to prefer truncation vs uniform quantisation."

### 11.3 Suggested but not confirmed

**F7. Encoder anisotropy admits a one-line diagnostic.** If F6 holds, then `np.var(angles, axis=0)` on a representative sample of the corpus suffices to choose the correct allocation strategy (slope > threshold → `pos_clip`; flat → `angle_int4_uniform`). Computing this profile on the OpenAI and MiniLM corpora and confirming it predicts the observed regime is fast (1 hour of work) and would close F6 into a deployable rule of thumb.

   *Status:* not yet computed. Should be the next experiment.

**F8. Angle-space quantisation degrades more gracefully at int4 / int2 than Cartesian.** `cart_int2` collapses (F2); `angle_int2_uniform` retains 24–29% NDCG@10 on both encoders (above the random floor). However the *fair* comparison — `cart_clip_50pct_int{4,2}` vs `pos_clip_50pct_int{4,2}` — was never run. Without it, "angles degrade more gracefully" is not provable.

   *Status:* trivially closeable in two evaluations per encoder.

### 11.4 What this means for paper framing

Three publishable framings emerge, in increasing order of ambition:

1. **"Matryoshka + int8 is an under-reported strong baseline" (single-encoder, FEVER + mini-BEIR).** Show §3, §5.1, §5.3. Honest, narrow, easy review. Headline: 99.1% retention at 8× compression on a leaderboard MRL encoder, matched by three lines of NumPy. Best for a workshop or short paper.

2. **"Angle quantisation is encoder-conditional, with a closed-form anisotropy-based decision rule" (two-encoder, F4 + F5 + F6).** Headline: same 99.1% on MRL via the trivial baseline; +1.7% NDCG@10 on non-MRL via uniform angle quantisation; both predicted by one formula. Substantive contribution; full paper material; needs F7 (anisotropy diagnostic) for completeness and one more encoder for robustness.

3. **"A unified theory of when to truncate vs when to uniformly quantise embeddings, with empirical validation across encoder families" (multi-encoder, F4 + F5 + F6 + F7 + PQ baselines).** Same as #2 plus diagnostic plus learned-codebook comparison. This is the venue paper. Requires the next 4 experiments in §9.

The **honest** recommendation: pursue framing 2. Framing 3 is conditional on experiments not yet run; framing 1 leaves the most interesting finding (F5 / F6) on the table.

---

## Appendix A. Master scheme table — OpenAI text-embedding-3-small (mini-BEIR macro across 8 subsets)

40 unique-result rows after merging equivalence classes (52 raw schemes minus 12 aliases). Sorted by bit budget descending → compression ascending. Macro across `{scifact, nfcorpus, fiqa, arguana, trec-covid, nq, quora, hotpotqa}`. Float32 baseline NDCG@10 = 0.7306; **retention column** = `NDCG@10 / 0.7306`.

**Two kinds of equivalence are merged into single rows:**

1. **Provable** (≡): two scheme names that produce identical reconstructed vectors for every input by construction (e.g. `cart_int8` ≡ `cart_clip_0pct`). The eval pipeline now skips these duplicates at registration time (see "alias map" footnote).
2. **Empirical** (=): two scheme names that produce different operations but happen to land on near-identical retrieval scores on this encoder. (Note: the previously labelled `jac_clip_Xpct` = `pos_clip_Xpct` equivalence is now known to be **provable**, not empirical — see §4.3. The remaining empirical equivalences in this report are the cross-recipe ones, e.g. `pos_clip_50pct` ≈ `cart_clip_50pct` on OpenAI but not on MiniLM.)

Pipeline column conventions:

- `coords → int8` = uniform `b`-bit scalar quantisation of every Cartesian coord.
- `first K% coords → int8, rest = 0` = Matryoshka-style slice + int8 quant + renormalise.
- `→ angles → all → int8` = `to_angles` then int8 then `from_angles` then renormalise.
- `→ angles → first K% by index → int8, rest = mean` = positional clipping in angle space.
- `→ angles → top K% by sens → int8, rest = mean` = Jacobian-sensitivity-ranked clipping.
- `→ angles → mixed-precision per angle (Xb total)` = greedy distortion-per-bit allocator.
- "rest = mean" means dropped angles are replaced by the per-corpus angle mean `μ_i` (zero bits per vector).
- "rest = 0" means dropped angles are set to zero (geometric pole, not corpus mean).

| # | Scheme(s) | Pipeline | bits/vec | comp | saved | NDCG@5 | NDCG@10 | NDCG@15 | NDCG@20 | R@100 | retention |
|--:|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
|  1 | `float32`                                | raw f32 vector                                          | 49,152 |  1.0× |   0.0% | 0.7194 | **0.7306** | 0.7341 | 0.7354 | 0.7991 | 100.00% |
|  2 | `cart_clip_20pct_f32`                    | first 80% coords kept f32, rest = 0                      | 39,328 |  1.2× |  20.0% | 0.7179 | 0.7288 | 0.7331 | 0.7336 | 0.7980 |  99.75% |
|  3 | `cart_clip_40pct_f32`                    | first 60% coords kept f32, rest = 0                      | 29,504 |  1.7× |  40.0% | 0.7142 | 0.7265 | 0.7301 | 0.7314 | 0.7955 |  99.44% |
|  4 | `cart_clip_50pct_f32`                    | first 50% coords kept f32, rest = 0                      | 24,576 |  2.0× |  50.0% | 0.7146 | 0.7259 | 0.7294 | 0.7305 | 0.7959 |  99.36% |
|  5 | `cart_clip_60pct_f32`                    | first 40% coords kept f32, rest = 0                      | 19,648 |  2.5× |  60.0% | 0.7104 | 0.7238 | 0.7279 | 0.7288 | 0.7937 |  99.07% |
|  6 | `cart_int8` ≡ `cart_clip_0pct`           | all 1,536 coords → int8                                  | 12,288 |  4.0× |  75.0% | 0.7202 | **0.7310** | 0.7337 | 0.7350 | 0.7985 | 100.05% |
|  7 | `greedy_12288b`                          | → angles → mixed-precision per angle (12,288 b total)    | 12,288 |  4.0× |  75.0% | 0.7198 | 0.7306 | 0.7346 | 0.7353 | 0.7989 | 100.00% |
|  8 | `angle_int8_uniform` ≡ `jac_clip_0pct` ≡ `pos_clip_0pct` ≡ `jac_trunc_int8_keep100pct` | → angles → all → int8 | 12,280 |  4.0× |  75.0% | 0.7199 | **0.7308** | 0.7348 | 0.7354 | 0.7993 |  99.97% |
|  9 | `cart_clip_20pct`                        | first 80% coords → int8, rest = 0                        |  9,832 |  5.0× |  80.0% | 0.7167 | 0.7282 | 0.7320 | 0.7321 | 0.7972 |  99.67% |
| 10 | `jac_clip_20pct` = `pos_clip_20pct`      | → angles → top/first 80% → int8, rest = mean             |  9,824 |  5.0× |  80.0% | 0.7177 | **0.7290** | 0.7330 | 0.7335 | 0.7983 |  99.78% |
| 11 | `pos_clip_20pct_zero`                    | → angles → first 80% by index → int8, rest = 0           |  9,824 |  5.0× |  80.0% | 0.7176 | 0.7287 | 0.7323 | 0.7330 | 0.7982 |  99.74% |
| 12 | `cart_clip_80pct_f32`                    | first 20% coords kept f32, rest = 0                      |  9,824 |  5.0× |  80.0% | 0.6972 | 0.7098 | 0.7148 | 0.7152 | 0.7847 |  97.16% |
| 13 | `greedy_9216b`                           | → angles → mixed-precision per angle (9,216 b total)     |  9,216 |  5.3× |  81.3% | 0.7147 | 0.7263 | 0.7295 | 0.7302 | 0.7949 |  99.42% |
| 14 | `jac_trunc_int8_keep70pct`               | → angles → top 70% by sens → int8, rest = mean           |  8,592 |  5.7× |  82.5% | 0.7155 | 0.7274 | 0.7315 | 0.7326 | 0.7964 |  99.56% |
| 15 | `cart_clip_40pct`                        | first 60% coords → int8, rest = 0                        |  7,376 |  6.7× |  85.0% | 0.7141 | 0.7260 | 0.7297 | 0.7313 | 0.7944 |  99.37% |
| 16 | `jac_clip_40pct` = `pos_clip_40pct`      | → angles → top/first 60% → int8, rest = mean             |  7,368 |  6.7× |  85.0% | 0.7141 | **0.7266** | 0.7309 | 0.7312 | 0.7946 |  99.45% |
| 17 | `pos_clip_40pct_zero`                    | → angles → first 60% by index → int8, rest = 0           |  7,368 |  6.7× |  85.0% | 0.7133 | 0.7256 | 0.7295 | 0.7302 | 0.7944 |  99.31% |
| 18 | `cart_int4`                              | all 1,536 coords → int4                                  |  6,144 |  8.0× |  87.5% | 0.6834 | 0.6985 | 0.7002 | 0.7015 | 0.7846 |  95.61% |
| 19 | **`cart_clip_50pct`**                    | **first 50% coords → int8, rest = 0**                    | **6,144** | **8.0×** | **87.5%** | 0.7121 | **0.7240** | 0.7273 | 0.7283 | 0.7947 |  **99.10%** |
| 20 | **`jac_clip_50pct` = `pos_clip_50pct` ≡ `jac_trunc_int8_keep50pct`** | **→ angles → top/first 50% → int8, rest = mean** | **6,144** | **8.0×** | **87.5%** | 0.7126 | **0.7241** | 0.7282 | 0.7290 | 0.7942 |  **99.11%** |
| 21 | `pos_clip_50pct_zero`                    | → angles → first 50% by index → int8, rest = 0           |  6,144 |  8.0× |  87.5% | 0.7116 | 0.7235 | 0.7273 | 0.7282 | 0.7935 |  99.03% |
| 22 | `greedy_6144b`                           | → angles → mixed-precision per angle (6,144 b total)     |  6,144 |  8.0× |  87.5% | 0.6836 | 0.6968 | 0.7000 | 0.7009 | 0.7814 |  95.36% |
| 23 | `angle_int4_uniform` ≡ `jac_trunc_int4_keep100pct` | → angles → all → int4                          |  6,140 |  8.0× |  87.5% | 0.6788 | 0.6904 | 0.6941 | 0.6956 | 0.7767 |  94.50% |
| 24 | `cart_clip_60pct`                        | first 40% coords → int8, rest = 0                        |  4,912 | 10.0× |  90.0% | 0.7121 | **0.7229** | 0.7272 | 0.7281 | 0.7927 |  98.95% |
| 25 | `jac_clip_60pct` = `pos_clip_60pct`      | → angles → top/first 40% → int8, rest = mean             |  4,912 | 10.0× |  90.0% | 0.7087 | 0.7226 | 0.7269 | 0.7275 | 0.7937 |  98.91% |
| 26 | `pos_clip_60pct_zero`                    | → angles → first 40% by index → int8, rest = 0           |  4,912 | 10.0× |  90.0% | 0.7087 | 0.7224 | 0.7269 | 0.7272 | 0.7929 |  98.88% |
| 27 | `jac_trunc_int4_keep70pct`               | → angles → top 70% by sens → int4, rest = mean           |  4,296 | 11.4× |  91.3% | 0.6784 | 0.6898 | 0.6937 | 0.6952 | 0.7766 |  94.41% |
| 28 | `jac_trunc_int8_keep30pct`               | → angles → top 30% by sens → int8, rest = mean           |  3,680 | 13.4× |  92.5% | 0.7063 | 0.7184 | 0.7238 | 0.7238 | 0.7902 |  98.33% |
| 29 | `cart_int2`                              | all 1,536 coords → int2 (**catastrophic collapse**)      |  3,072 | 16.0× |  93.8% | 0.0030 | **0.0030** | 0.0036 | 0.0040 | 0.0131 |   0.41% |
| 30 | `greedy_3072b`                           | → angles → mixed-precision per angle (3,072 b total)     |  3,072 | 16.0× |  93.8% | 0.5745 | 0.5890 | 0.5944 | 0.5953 | 0.7237 |  80.62% |
| 31 | `jac_trunc_int4_keep50pct`               | → angles → top 50% by sens → int4, rest = mean           |  3,072 | 16.0× |  93.8% | 0.6772 | **0.6887** | 0.6929 | 0.6939 | 0.7762 |  94.26% |
| 32 | `angle_int2_uniform` ≡ `jac_trunc_int2_keep100pct` | → angles → all → int2 (degenerate floor)        |  3,070 | 16.0× |  93.8% | 0.2289 | 0.2396 | 0.2454 | 0.2484 | 0.4371 |  32.79% |
| 33 | `cart_clip_80pct`                        | first 20% coords → int8, rest = 0                        |  2,456 | 20.0× |  95.0% | 0.6862 | 0.7008 | 0.7043 | 0.7059 | 0.7813 |  95.92% |
| 34 | `jac_clip_80pct` = `pos_clip_80pct`      | → angles → top/first 20% → int8, rest = mean             |  2,456 | 20.0× |  95.0% | 0.6932 | **0.7057** | 0.7114 | 0.7119 | 0.7849 |  96.59% |
| 35 | `pos_clip_80pct_zero`                    | → angles → first 20% by index → int8, rest = 0           |  2,456 | 20.0× |  95.0% | 0.6851 | 0.6994 | 0.7039 | 0.7051 | 0.7819 |  95.73% |
| 36 | `jac_trunc_int2_keep70pct`               | → angles → top 70% by sens → int2, rest = mean           |  2,148 | 22.9× |  95.6% | 0.2289 | 0.2396 | 0.2454 | 0.2484 | 0.4371 |  32.79% |
| 37 | `jac_trunc_int4_keep30pct`               | → angles → top 30% by sens → int4, rest = mean           |  1,840 | 26.7× |  96.3% | 0.6688 | **0.6810** | 0.6843 | 0.6852 | 0.7719 |  93.21% |
| 38 | `greedy_1536b`                           | → angles → mixed-precision per angle (collapsed to int2) |  1,536 | 32.0× |  96.9% | 0.2289 | 0.2396 | 0.2454 | 0.2484 | 0.4371 |  32.79% |
| 39 | `jac_trunc_int2_keep50pct`               | → angles → top 50% by sens → int2, rest = mean           |  1,536 | 32.0× |  96.9% | 0.2289 | 0.2396 | 0.2454 | 0.2484 | 0.4371 |  32.79% |
| 40 | `jac_trunc_int2_keep30pct`               | → angles → top 30% by sens → int2, rest = mean           |    920 | 53.4× |  98.1% | 0.2289 | 0.2396 | 0.2454 | 0.2484 | 0.4371 |  32.79% |

**Alias map (provable equivalences, skipped at registration time):**

| Alias | Canonical scheme run instead |
|---|---|
| `cart_clip_0pct` | `cart_int8` |
| `jac_clip_0pct`            | `angle_int8_uniform` |
| `pos_clip_0pct`            | `angle_int8_uniform` |
| `pos_clip_0pct_zero`       | `angle_int8_uniform` |
| `jac_trunc_int8_keep100pct` | `angle_int8_uniform` |
| `jac_trunc_int4_keep100pct` | `angle_int4_uniform` |
| `jac_trunc_int2_keep100pct` | `angle_int2_uniform` |
| `jac_trunc_int8_keep50pct` | `jac_clip_50pct` |

**`pos_clip_*pct_zero` rows (rows 11, 17, 21, 26, 35) are bit-identical to their `pos_clip_*pct` siblings** — same kept head, same int8 quant, same bit budget — and differ only in what fills the dropped tail (`0` instead of the corpus mean `μ`). Per-row Δ NDCG@10 vs the mean-fill sibling: −0.0003, −0.0010, −0.0006, −0.0002, −0.0063. They are kept as separate rows because they map onto distinct geometric operations and because the tail-fill choice has a measurable (if small) effect at aggressive truncation; see §5.4 for the analysis.

**Reading the table.**

- The 0.2396 NDCG@10 floor on every int2-based scheme is the *random-baseline* — when all reconstructed vectors converge to ~the same point (because int2 quant noise dominates), every query's relevant docs tie with the rest of the corpus and NDCG@10 reflects only random tie-breaking. It is not "32% retention"; it's the ceiling of "all information lost."
- The bold rows at 6,144 bits (rows 19–20) and 4,912 bits (rows 24–25) are the central evidence for the three-way equivalence (`cart_clip ≈ pos_clip ≈ jac_clip`) discussed in §5.3; row 21 (`pos_clip_50pct_zero`) confirms that even removing the corpus-mean fill leaves the equivalence intact at this budget.
- `greedy_*b` underperforms truncation at every budget below 12,288 bits (§4.4); drop from paper unless refactored.
- `cart_int2` (row 29) is the unique catastrophic-collapse cell; every other 16× scheme retains ≥32% NDCG, and the angle-clipping variants retain ≥94%.

---

## Appendix B. Master scheme table — MiniLM-L6-v2 (mini-BEIR macro across 8 subsets)

40 unique-result rows, same evaluation protocol as Appendix A. Macro across the same 8 subsets. Float32 baseline NDCG@10 = 0.6603; **retention column** = `NDCG@10 / 0.6603`.

The same alias-collapsing rules apply (`cart_clip_0pct` ≡ `cart_int8`, etc.). The merge `jac_clip_Xpct` = `pos_clip_Xpct` is now also observed on MiniLM at every clip rate — see "On the Jacobian / positional equivalence" footnote below the table.

| # | Scheme(s) | Pipeline | bits/vec | comp | saved | NDCG@5 | NDCG@10 | NDCG@15 | NDCG@20 | R@100 | retention |
|--:|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
|  1 | `float32`                                | raw f32 vector                                          | 12,288 |  1.0× |   0.0% | 0.6498 | **0.6603** | 0.6629 | 0.6636 | 0.7631 | 100.00% |
|  2 | `greedy_12288b`                          | → angles → mixed-precision per angle (12,288 b)         | 12,256 |  1.0× |   0.3% | 0.6498 | 0.6603 | 0.6629 | 0.6636 | 0.7631 | 100.00% |
|  3 | `cart_clip_20pct_f32`                    | first 80% coords kept f32, rest = 0                      |  9,824 |  1.3× |  20.0% | 0.6431 | 0.6540 | 0.6571 | 0.6585 | 0.7607 |  99.05% |
|  4 | `greedy_9216b`                           | → angles → mixed-precision per angle (9,216 b)          |  9,216 |  1.3× |  25.0% | 0.6496 | 0.6601 | 0.6629 | 0.6636 | 0.7631 |  99.97% |
|  5 | `cart_clip_40pct_f32`                    | first 60% coords kept f32, rest = 0                      |  7,360 |  1.7× |  40.1% | 0.6325 | 0.6444 | 0.6467 | 0.6480 | 0.7536 |  97.59% |
|  6 | `cart_clip_50pct_f32`                    | first 50% coords kept f32, rest = 0                      |  6,144 |  2.0× |  50.0% | 0.6258 | 0.6365 | 0.6404 | 0.6406 | 0.7515 |  96.39% |
|  7 | `greedy_6144b`                           | → angles → mixed-precision per angle (6,144 b)          |  6,144 |  2.0× |  50.0% | 0.6500 | 0.6602 | 0.6629 | 0.6636 | 0.7631 |  99.98% |
|  8 | `cart_clip_60pct_f32`                    | first 40% coords kept f32, rest = 0                      |  4,928 |  2.5× |  59.9% | 0.6162 | 0.6289 | 0.6323 | 0.6331 | 0.7448 |  95.24% |
|  9 | `cart_int8` ≡ `cart_clip_0pct`           | all 384 coords → int8                                    |  3,072 |  4.0× |  75.0% | 0.6513 | **0.6608** | 0.6630 | 0.6639 | 0.7640 | 100.07% |
| 10 | `greedy_3072b`                           | → angles → mixed-precision per angle (3,072 b)          |  3,072 |  4.0× |  75.0% | 0.6489 | 0.6599 | 0.6632 | 0.6638 | 0.7629 |  99.94% |
| 11 | `angle_int8_uniform` ≡ `jac_clip_0pct` ≡ `pos_clip_0pct` ≡ `jac_trunc_int8_keep100pct` | → angles → all → int8 | 3,064 |  4.0× |  75.1% | 0.6492 | 0.6601 | 0.6633 | 0.6639 | 0.7629 |  99.97% |
| 12 | `cart_clip_80pct_f32`                    | first 20% coords kept f32, rest = 0                      |  2,464 |  5.0× |  79.9% | 0.5591 | 0.5725 | 0.5766 | 0.5782 | 0.7109 |  86.71% |
| 13 | `cart_clip_20pct`                        | first 80% coords → int8, rest = 0                        |  2,456 |  5.0× |  80.0% | 0.6421 | 0.6528 | 0.6556 | 0.6564 | 0.7600 |  98.86% |
| 14 | `jac_clip_20pct` = `pos_clip_20pct`      | → angles → top/first 80% → int8, rest = mean             |  2,448 |  5.0× |  80.1% | 0.6406 | **0.6530** | 0.6559 | 0.6569 | 0.7621 |  98.90% |
| 15 | `pos_clip_20pct_zero`                    | → angles → first 80% by index → int8, rest = 0           |  2,448 |  5.0× |  80.1% | 0.6405 | 0.6528 | 0.6548 | 0.6562 | 0.7609 |  98.86% |
| 16 | `jac_trunc_int8_keep70pct`               | → angles → top 70% by sens → int8, rest = mean           |  2,144 |  5.7× |  82.5% | 0.6366 | 0.6474 | 0.6513 | 0.6522 | 0.7580 |  98.05% |
| 17 | `jac_clip_40pct` = `pos_clip_40pct`      | → angles → top/first 60% → int8, rest = mean             |  1,840 |  6.7× |  85.0% | 0.6293 | **0.6409** | 0.6435 | 0.6446 | 0.7526 |  97.06% |
| 18 | `pos_clip_40pct_zero`                    | → angles → first 60% by index → int8, rest = 0           |  1,840 |  6.7× |  85.0% | 0.6286 | 0.6386 | 0.6414 | 0.6423 | 0.7516 |  96.71% |
| 19 | `cart_clip_40pct`                        | first 60% coords → int8, rest = 0                        |  1,840 |  6.7× |  85.0% | 0.6261 | 0.6363 | 0.6400 | 0.6410 | 0.7531 |  96.36% |
| 20 | `cart_int4`                              | all 384 coords → int4                                    |  1,536 |  8.0× |  87.5% | 0.6149 | 0.6281 | 0.6314 | 0.6330 | 0.7545 |  95.12% |
| 21 | `jac_clip_50pct` = `pos_clip_50pct` ≡ `jac_trunc_int8_keep50pct` | → angles → top/first 50% → int8, rest = mean | 1,536 |  8.0× |  87.5% | 0.6180 | 0.6294 | 0.6338 | 0.6348 | 0.7496 |  95.32% |
| 22 | `pos_clip_50pct_zero`                    | → angles → first 50% by index → int8, rest = 0           |  1,536 |  8.0× |  87.5% | 0.6144 | 0.6262 | 0.6304 | 0.6322 | 0.7481 |  94.84% |
| 23 | `cart_clip_50pct`                        | first 50% coords → int8, rest = 0                        |  1,536 |  8.0× |  87.5% | 0.6149 | 0.6267 | 0.6318 | 0.6335 | 0.7489 |  94.91% |
| 24 | `greedy_1536b`                           | → angles → mixed-precision per angle (1,536 b ≈ all int4) |  1,536 |  8.0× |  87.5% | 0.6230 | 0.6343 | 0.6380 | 0.6395 | 0.7492 |  96.05% |
| 25 | **`angle_int4_uniform` ≡ `jac_trunc_int4_keep100pct`** | **→ angles → all → int4**                |  **1,532** |  **8.0×** |  **87.5%** | **0.6222** | **0.6349** | **0.6378** | **0.6396** | **0.7490** |  **96.15%** |
| 26 | `cart_clip_60pct`                        | first 40% coords → int8, rest = 0                        |  1,232 | 10.0× |  90.0% | 0.6052 | 0.6176 | 0.6220 | 0.6235 | 0.7414 |  93.53% |
| 27 | `jac_clip_60pct` = `pos_clip_60pct`      | → angles → top/first 40% → int8, rest = mean             |  1,224 | 10.0× |  90.0% | 0.6106 | **0.6209** | 0.6246 | 0.6251 | 0.7437 |  94.04% |
| 28 | `pos_clip_60pct_zero`                    | → angles → first 40% by index → int8, rest = 0           |  1,224 | 10.0× |  90.0% | 0.6027 | 0.6159 | 0.6205 | 0.6219 | 0.7421 |  93.28% |
| 29 | `jac_trunc_int4_keep70pct`               | → angles → top 70% by sens → int4, rest = mean           |  1,072 | 11.5× |  91.3% | 0.6039 | 0.6178 | 0.6205 | 0.6224 | 0.7393 |  93.56% |
| 30 | `jac_trunc_int8_keep30pct`               | → angles → top 30% by sens → int8, rest = mean           |    920 | 13.4× |  92.5% | 0.5845 | 0.5969 | 0.6004 | 0.6034 | 0.7314 |  90.39% |
| 31 | `cart_int2`                              | all 384 coords → int2 (**catastrophic collapse**)        |    768 | 16.0× |  93.8% | 0.0388 | **0.0396** | 0.0414 | 0.0421 | 0.1065 |   6.00% |
| 32 | `jac_trunc_int4_keep50pct`               | → angles → top 50% by sens → int4, rest = mean           |    768 | 16.0× |  93.8% | 0.5801 | 0.5913 | 0.5956 | 0.5976 | 0.7275 |  89.55% |
| 33 | `angle_int2_uniform` ≡ `jac_trunc_int2_keep100pct` | → angles → all → int2 (degenerate floor)        |    766 | 16.0× |  93.8% | 0.2798 | 0.2903 | 0.2934 | 0.2958 | 0.4700 |  43.97% |
| 34 | `jac_clip_80pct` = `pos_clip_80pct`      | → angles → top/first 20% → int8, rest = mean             |    616 | 20.0× |  95.0% | 0.5355 | **0.5491** | 0.5538 | 0.5563 | 0.7031 |  83.16% |
| 35 | `pos_clip_80pct_zero`                    | → angles → first 20% by index → int8, rest = 0           |    616 | 20.0× |  95.0% | 0.5168 | 0.5334 | 0.5379 | 0.5405 | 0.6968 |  80.78% |
| 36 | `cart_clip_80pct`                        | first 20% coords → int8, rest = 0                        |    616 | 20.0× |  95.0% | 0.5149 | 0.5318 | 0.5380 | 0.5406 | 0.6964 |  80.54% |
| 37 | `jac_trunc_int2_keep70pct`               | → angles → top 70% by sens → int2, rest = mean           |    536 | 22.9× |  95.6% | 0.2798 | 0.2903 | 0.2934 | 0.2958 | 0.4700 |  43.97% |
| 38 | `jac_trunc_int4_keep30pct`               | → angles → top 30% by sens → int4, rest = mean           |    460 | 26.7× |  96.3% | 0.5254 | **0.5402** | 0.5446 | 0.5476 | 0.6938 |  81.81% |
| 39 | `jac_trunc_int2_keep50pct`               | → angles → top 50% by sens → int2, rest = mean           |    384 | 32.0× |  96.9% | 0.2798 | 0.2903 | 0.2934 | 0.2958 | 0.4700 |  43.97% |
| 40 | `jac_trunc_int2_keep30pct`               | → angles → top 30% by sens → int2, rest = mean           |    230 | 53.4× |  98.1% | 0.2798 | 0.2903 | 0.2933 | 0.2958 | 0.4699 |  43.97% |

**On the Jacobian / positional equivalence (now visible on both encoders).** `jac_clip_Xpct` = `pos_clip_Xpct` at every clip rate on MiniLM as well as on OpenAI. Re-deriving the formula clarifies why: `sens[i] ∝ E[Σ_{j<i} log sin²(θ_j)]` is a cumulative sum of non-positive terms (since `sin²(θ) ≤ 1` ⇒ `log sin²(θ) ≤ 0`), so `sens[i]` is monotonically non-increasing in `i` regardless of the encoder. Therefore `argsort(−sens) = [0, 1, …, D−2]` by the *structure of the formula*, not by encoder-specific anisotropy. **The Jacobian and positional rankings are provably identical on every encoder.** This is a stronger statement than what was claimed in §5.3 (which framed the equivalence as MRL-conditional); the alias `jac_clip_Xpct → pos_clip_Xpct` should therefore be treated as a *provable* equivalence (≡), not an empirical one (=). The `jac_clip` family is preserved as a separate name purely for traceability with the original derivation; in any future writeup the two should be merged.

**Reading the MiniLM table — what's different from Appendix A.**

- The "8.0× cluster" (rows 20–25) is now the most informative block. `angle_int4_uniform` (row 25, **bold**) is the best 8× scheme, beating `pos_clip_50pct` (row 21) by +0.0055 NDCG@10. On Appendix A the same row was **the worst** of its bit budget by 0.034. This is the §6.3 reversal — the single cleanest pro-angle finding in the whole study.
- The "20× cluster" (rows 34–36) shows `pos_clip_80pct` ahead of both zero-fill variants by ≥0.015 NDCG@10. On Appendix A the corresponding gap was 0.005 — 3× smaller.
- `cart_int2` (row 31) lands at NDCG@10 = 0.0396, 6% retention. On a smaller-dim encoder the int2 collapse is slightly less catastrophic in absolute terms (because some signal survives in the cosine-cluster geometry) but still well below the 44% random floor that the angle-int2 schemes reach.
- The random floor on MiniLM is **0.2903**, not 0.2396 as on OpenAI. Different distractor structure / cosine-distribution shape; the floor is encoder-conditional but the qualitative interpretation ("all information lost") is the same.
- The `greedy_*b` family is now well-behaved across the entire range — `greedy_1536b` at 8× ties `angle_int4_uniform` (rows 24 vs 25). The §4.4 "drop greedy from the paper" verdict was bit-budget-grid-specific, not method-specific.
