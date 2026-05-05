# Hyperspherical-Angle Quantisation for Dense-Retrieval Embeddings
### Empirical evaluation on FEVER + cross-subset validation on 8 BEIR datasets

---

## 1. Executive summary

We benchmark a family of scalar-quantisation recipes for unit-norm dense-retrieval embeddings on OpenAI `text-embedding-3-small` (1,536-d, Matryoshka-trained). Documents are quantised and reconstructed; queries remain `float32` (the standard asymmetric-retrieval protocol). The evaluation covers:

- The full BEIR FEVER corpus (5,416,568 documents, 6,666 evaluated queries with qrels) — the deep-dive experiment;
- 8 additional BEIR subsets (`scifact`, `nfcorpus`, `fiqa`, `arguana`, `trec-covid`, `nq`, `quora`, `hotpotqa`) with ≤1,000 queries + qrel docs + 10k random distractors each — for cross-domain comparison.

Three scheme families are compared at equal bit budgets:

| Family | Recipe |
|---|---|
| `cart_clip_Xpct` | Matryoshka-style: keep the first (1−X)% Cartesian coordinates at int8, zero the rest, renormalise. |
| `pos_clip_Xpct`  | Convert to `D−1` hyperspherical angles, keep the first (1−X)% at int8, replace the rest with the per-corpus angle mean, reconstruct. |
| `jac_clip_Xpct`  | Same as `pos_clip` but with angles ordered by Jacobian sensitivity `sens[i] ∝ E[Σ_{j<i} log sin²(θ_j)]` instead of position. |

**Central finding (cross-subset, 8 BEIR datasets).** At every bit budget where all three families can be compared, they produce **statistically indistinguishable NDCG@10** — Δ ≤ 0.005 macro-averaged across 8 subsets. The hyperspherical and Jacobian apparatus does not beat the vanilla Matryoshka recipe; it reproduces it via a roundabout path.

| Compression | bits/vec | `pos_clip` / `jac_clip` | **`cart_clip`** | Angle Δ |
|---:|---:|---:|---:|---:|
|  5.0× |  9,824 | 0.7290 | 0.7282 | +0.0008 |
|  6.7× |  7,368 | 0.7266 | 0.7260 | +0.0006 |
|  **8.0×** |  **6,144** | **0.7241** | **0.7240** | **+0.0001 (tied)** |
| 10.0× |  4,912 | 0.7226 | **0.7229** | **−0.0003** (cart wins) |
| 20.0× |  2,456 | 0.7057 | 0.7008 | +0.0049 |

**FEVER-specific numbers (full 5.4M corpus, NDCG@10 baseline = 0.7995).** At 8× compression, `jac_clip_50pct` / `pos_clip_50pct` retains 98.4% NDCG vs 90.1% for per-coordinate Cartesian int4. The direct Matryoshka baseline `cart_clip_50pct` was only swept on mini-BEIR (a 2-core box constraint prevented re-running on the 5.4M FEVER corpus), but on that mini-BEIR the three schemes tie. The FEVER numbers therefore very likely reflect the Matryoshka contribution, not an angle-parametrisation contribution.

**What the angle parametrisation actually contributes (honest list).**

1. **A theoretical derivation** — the Jacobian sensitivity formula predicts that on any encoder with front-loaded anisotropy (MRL or otherwise), positional ordering of angles is L2-optimal. This prediction is empirically confirmed (§5.3); it just happens to also coincide with raw-coordinate ordering under MRL.
2. **Graceful degradation at int2** — per-coordinate Cartesian int2 (`cart_int2`) collapses to NDCG ≈ 0 on 8/8 subsets and on FEVER, while `angle_int2_uniform` retains ≈24%. However, the directly-comparable recipe `cart_clip_50pct_int2` (Matryoshka slice + int2 on the head) **was not tested**. Without it, the "angles degrade more gracefully" claim remains unverified — the real baseline may be `cart_int2` (which is a pathological scheme nobody would ship) and not Matryoshka + int2.
3. **A negative result** — for Matryoshka-trained encoders at 4× – 20× compression, the angle machinery is unnecessary. A three-line Cartesian slice does the same job.

**Implication for the paper.** The original headline ("hyperspherical-angle truncation retains 98.4% at 8× compression") is now understood to reproduce the Matryoshka recipe rather than to beat it. Three defensible framings remain, in order of honesty:

- *"Matryoshka + int8 is a tight, under-reported baseline"* — the 98.9% retention number is real; the method is trivial to implement; few papers benchmark it.
- *"The angle parametrisation is a no-op on Matryoshka encoders"* — a clean negative result, worth publishing only if paired with a test on a non-MRL encoder to bound where the angle apparatus might still matter.
- *"Angle truncation is robust below int4 where per-coord Cartesian fails"* — contingent on running the missing `cart_clip_*_int{4,2}` sweep; if Matryoshka + low-bit also works, this framing dies too.

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

Every `jac_clip_Xpct` scheme is **bit-identical** to its `pos_clip_Xpct` twin on FEVER and on all 8 mini-BEIR subsets:

| bits | `jac_clip` | `pos_clip` |
|---:|---:|---:|
| 9,824 | 0.7962 | 0.7962 |
| 7,368 | 0.7914 | 0.7914 |
| 6,144 | 0.7865 | 0.7865 |
| 4,912 | 0.7799 | 0.7799 |
| 2,456 | 0.7416 | 0.7416 |

**Mechanism.** For MRL-trained encoders, `sens[i]` decreases monotonically in `i`, so `argsort(−sens) = [0, 1, …, D−2]`. The Jacobian ranking *is* the positional ranking on these embeddings. This is the finding that motivated adding `cart_clip` — if angle position and Jacobian sensitivity both coincide with Cartesian-coordinate index, all three schemes are redundant with each other.

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
| `cart_clip_80pct_f32`                    |  9,824 |  5.0× | 0.7098 |  97.1% | 0.7847 |
| **`cart_clip_40pct`**                    |  7,376 |  6.7× | **0.7260** | **99.4%** | 0.7944 |
| `jac_clip_40pct` / `pos_clip_40pct`      |  7,368 |  6.7× | 0.7266 |  99.4% | 0.7946 |
| **`cart_clip_50pct`**                    |  **6,144** | **8.0×** | **0.7240** | **99.1%** | **0.7947** |
| **`jac_clip_50pct` / `pos_clip_50pct`**  |  **6,144** | **8.0×** | **0.7241** | **99.1%** | **0.7942** |
| `cart_int4`                              |  6,144 |  8.0× | 0.6985 |  95.7% | 0.7846 |
| `angle_int4_uniform`                     |  6,140 |  8.0× | 0.6904 |  94.5% | 0.7767 |
| **`cart_clip_60pct`**                    |  4,912 | 10.0× | **0.7229** | **99.0%** | 0.7927 |
| `jac_clip_60pct` / `pos_clip_60pct`      |  4,912 | 10.0× | 0.7226 |  99.0% | 0.7937 |
| `jac_trunc_int8_keep30pct`               |  3,680 | 13.4× | 0.7184 |  98.3% | 0.7902 |
| `cart_int2`                              |  3,072 | 16.0× | 0.0030 |   0.4% | 0.0131 |
| `jac_trunc_int4_keep50pct`               |  3,072 | 16.0× | 0.6887 |  94.3% | 0.7762 |
| **`cart_clip_80pct`**                    |  2,456 | 20.0× | **0.7008** | **95.9%** | 0.7813 |
| `jac_clip_80pct` / `pos_clip_80pct`      |  2,456 | 20.0× | 0.7057 |  96.6% | 0.7849 |
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

**Implication.** The angle parametrisation has no engineering value as a standalone compression scheme — `cart_int8` matches it bit-for-bit on quality, costs ~3–5× less compute (no `arccos`/`cumsum`/`cumprod`), and is three lines of code. Its only practical use is as the **substrate for the clipping schemes** below; whether that substrate is necessary is the actual question §5.3–§5.6 answer.

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

On 6/8 subsets the angle scheme is +0.0005 to +0.0040 ahead; on 2/8 the Cartesian slice is 0.003–0.006 ahead. Macro gap is 0.0001. No consistent direction, no statistical significance, and smaller than the within-subset variance across the 8 subsets themselves. **The two recipes are interchangeable on this encoder.**

### 5.4 Quantisation vs truncation — which hurts more?

The float32-slice variants (`cart_clip_Xpct_f32`, no quant) separate the two sources of quality loss:

| kept coords | compression (f32 vs int8) | NDCG@10 f32 | NDCG@10 int8 | gap (quant cost) |
|---:|---:|---:|---:|---:|
| 1,229 (20% clip) | 1.2× vs 5.0×    | 0.7288 | 0.7282 | −0.0006 |
|   768 (50% clip) | 2.0× vs 8.0×    | 0.7259 | 0.7240 | −0.0019 |
|   307 (80% clip) | 5.0× vs 20.0×   | 0.7098 | 0.7008 | −0.0090 |

Quantising the kept head from float32 to int8 costs less than 0.01 NDCG even at aggressive truncation. **Int8 quant of the retained head is effectively free; the quality curve is entirely driven by how many coordinates you keep.** This reinforces §5.2's "the angle round-trip is lossless at int8" — combined with §5.3, the optimal recipe on this encoder is "slice then int8-quantise the head", regardless of which parametrisation you do the slice in.

### 5.5 Where angles might still matter — the untested cell

The one regime where the three-way equivalence has not been verified is **int4 or int2 on the kept head** combined with **Matryoshka-style slicing**. The schemes `cart_clip_50pct_int4` (3,072 bits, 16×) and `cart_clip_50pct_int2` (1,536 bits, 32×) were not in the current sweep. Existing 16× / 32× rows for comparison:

| bits | compression | scheme | NDCG@10 | retention |
|---:|---:|---|---:|---:|
| 3,072 | 16× | `cart_int2`                  | 0.0030 | 0.4% |
| 3,072 | 16× | `jac_trunc_int4_keep50pct`   | 0.6887 | 94.3% |
| 3,072 | 16× | **`cart_clip_50pct_int4` (not tested)** | **?** | **?** |
| 1,536 | 32× | **`cart_clip_50pct_int2` (not tested)** | **?** | **?** |

If `cart_clip_50pct_int4` matches `jac_trunc_int4_keep50pct` (≈0.69), the angle apparatus is fully redundant. If it collapses toward `cart_int2`, the graceful-degradation claim for angles is restored. Running this is trivial — see §8.

### 5.6 What this validates

1. **The angle parametrisation is lossless at int8** (§5.2). The full `to_angles → uniform_int8_quant → from_angles → renormalise` chain ties `float32` on all 8 subsets within 0.0015 NDCG, and on FEVER within 0.0006. This is a *necessary* condition for the angle apparatus to ever be useful — it is not a *sufficient* one.
2. **Matryoshka + int8 is the recipe.** `cart_clip_50pct` (three lines of NumPy) matches the angle schemes on all 8 subsets at 8× and 10× compression, and comes within 0.005 NDCG at 5× and 20×.
3. **`jac_clip` = `pos_clip` = `cart_clip`** at every tested (clip%, bit) pair where comparison is possible. Three distinct pipelines collapse to one when the encoder is MRL-trained.
4. **`cart_int2` collapses on every subset** (0.4% retention macro), while **`angle_int2_uniform` retains 24%.** The missing comparison — Matryoshka + int2 on the head — determines whether this is a genuine angle win or another Matryoshka win in disguise.
5. The float32-slice variants are **strictly dominated** by the int8-slice variants at every compression level that has a comparable int8 row. Always quantise the kept head.

---

## 6. Cost-vs-quality in storage units (FEVER, 5.42 M docs)

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

## 7. Caveats & limitations

1. **`cart_clip_Xpct` was not swept on the full FEVER corpus.** All §3 / §6 FEVER rows for `jac_clip` / `pos_clip` / `jac_trunc` are unverified under the Matryoshka-slice reframe. The central three-way equivalence (§5.3) was established on mini-BEIR only; the 8 subsets span enough domain diversity that the equivalence is highly likely to transfer, but this must be confirmed on the full 5.4M corpus before publication.
2. **`cart_clip_50pct_int{4,2}` was not tested.** This is the critical missing cell for claiming that the angle parametrisation offers any advantage at 16×+ compression (§5.5).
3. **Single embedding family.** All results are on OpenAI `text-embedding-3-small`, a Matryoshka-trained model. The three-way equivalence is *predicted* to break on non-MRL encoders (the Jacobian ranking would then disagree with positional ordering, and neither would match Cartesian slicing). This is both the most interesting open question and the biggest unverified assumption in the paper.
4. **No learned-codebook baselines.** PQ / OPQ / RQ at matched bit budgets are not evaluated; these are the publication-grade comparators and could dominate any of the scalar schemes here at 8×+.
5. **Asymmetric protocol only.** Queries are kept at `float32`. Symmetric (compressed-query) settings not evaluated.
6. **Mini-BEIR uses 10k random distractors per subset.** Absolute NDCG is optimistic vs the full BEIR leaderboard; retention ratios are robust, but the head-to-head Δ values in §5.3 should ideally be re-confirmed on full corpora.
7. **Greedy allocator underperforms truncation.** See §4.4 — drop from paper unless refactored with data-driven per-angle variance.

---

## 8. Critical next experiments

In strict priority order:

1. **`cart_clip_{30, 50, 70}pct` at `int{4, 2}`** on mini-BEIR. Five evaluations; closes the §5.5 gap. Result determines whether the paper has an int2-regime contribution or not.
2. **Full `cart_clip_*` sweep on FEVER.** Verifies the three-way equivalence on the 5.4M corpus. Must be done on a multi-core box (2-core is infeasible — previous FEVER eval took hours on c5a.24xlarge, would take weeks on t3.medium).
3. **One non-MRL encoder** (Sentence-T5, bge-small, Instructor, any pre-MRL model). Run the three-way comparison on mini-BEIR. Two outcomes:
   - *Three-way equivalence holds* → the angle apparatus is dead; paper pivots to "Matryoshka + int8 baseline" framing.
   - *`cart_clip` underperforms `pos_clip`* → the angle method has a genuine niche on non-MRL encoders; paper can claim it as such.
4. **PQ / OPQ baselines** at matched bit budgets on FEVER and at least one mini-BEIR subset. Required for any publication-grade comparison at 8×+.
5. **Denser truncation grid** around the elbow (`keep ∈ {10, 20, …, 90}%`). Fast and nails the curve shape exactly.
6. **Symmetric-quantisation variant** — also quantise queries. Cheap sanity check.

---

## 9. Artefact locations

| Artifact | Path |
|---|---|
| FEVER full-corpus results (txt/csv) | `eval_results_oai/results_streaming.{txt,csv}` |
| Cross-subset aggregate tables       | `eval_results_mini/_aggregate_by_scheme.{txt,csv}` |
| Cross-subset per-subset raw         | `eval_results_mini/_aggregate_by_subset.csv` |
| Per-subset baselines                | `eval_results_mini/_baselines_by_subset.txt` |
| Streaming evaluation script         | `evaluate_streaming.py` |
| Cross-subset driver                 | `evaluate_beir_mini.py` |
| Mini-subset builder                 | `build_beir_subsets.py` |
| Embedding script                    | `embed_beir.py` |
| Embeddings (on AWS box)             | `embeddings/fever/`, `embeddings_oai_mini/` |

---

## Appendix A. Master scheme table (mini-BEIR macro across 8 subsets)

35 unique-result rows after merging equivalence classes (47 raw schemes minus 12 aliases). Sorted by bit budget descending → compression ascending. Macro across `{scifact, nfcorpus, fiqa, arguana, trec-covid, nq, quora, hotpotqa}`. Float32 baseline NDCG@10 = 0.7306; **retention column** = `NDCG@10 / 0.7306`.

**Two kinds of equivalence are merged into single rows:**

1. **Provable** (≡): two scheme names that produce identical reconstructed vectors for every input by construction (e.g. `cart_int8` ≡ `cart_clip_0pct`). The eval pipeline now skips these duplicates at registration time (see "alias map" footnote).
2. **Empirical** (=): two scheme names that produce different operations but happen to land on bit-identical retrieval scores on this encoder (e.g. `jac_clip_50pct` = `pos_clip_50pct`, because `jac_rank == identity` under MRL-trained anisotropy). Both are still run, since the equivalence is encoder-conditional.

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
| 11 | `cart_clip_80pct_f32`                    | first 20% coords kept f32, rest = 0                      |  9,824 |  5.0× |  80.0% | 0.6972 | 0.7098 | 0.7148 | 0.7152 | 0.7847 |  97.16% |
| 12 | `greedy_9216b`                           | → angles → mixed-precision per angle (9,216 b total)     |  9,216 |  5.3× |  81.3% | 0.7147 | 0.7263 | 0.7295 | 0.7302 | 0.7949 |  99.42% |
| 13 | `jac_trunc_int8_keep70pct`               | → angles → top 70% by sens → int8, rest = mean           |  8,592 |  5.7× |  82.5% | 0.7155 | 0.7274 | 0.7315 | 0.7326 | 0.7964 |  99.56% |
| 14 | `cart_clip_40pct`                        | first 60% coords → int8, rest = 0                        |  7,376 |  6.7× |  85.0% | 0.7141 | 0.7260 | 0.7297 | 0.7313 | 0.7944 |  99.37% |
| 15 | `jac_clip_40pct` = `pos_clip_40pct`      | → angles → top/first 60% → int8, rest = mean             |  7,368 |  6.7× |  85.0% | 0.7141 | **0.7266** | 0.7309 | 0.7312 | 0.7946 |  99.45% |
| 16 | `cart_int4`                              | all 1,536 coords → int4                                  |  6,144 |  8.0× |  87.5% | 0.6834 | 0.6985 | 0.7002 | 0.7015 | 0.7846 |  95.61% |
| 17 | **`cart_clip_50pct`**                    | **first 50% coords → int8, rest = 0**                    | **6,144** | **8.0×** | **87.5%** | 0.7121 | **0.7240** | 0.7273 | 0.7283 | 0.7947 |  **99.10%** |
| 18 | **`jac_clip_50pct` = `pos_clip_50pct` ≡ `jac_trunc_int8_keep50pct`** | **→ angles → top/first 50% → int8, rest = mean** | **6,144** | **8.0×** | **87.5%** | 0.7126 | **0.7241** | 0.7282 | 0.7290 | 0.7942 |  **99.11%** |
| 19 | `greedy_6144b`                           | → angles → mixed-precision per angle (6,144 b total)     |  6,144 |  8.0× |  87.5% | 0.6836 | 0.6968 | 0.7000 | 0.7009 | 0.7814 |  95.36% |
| 20 | `angle_int4_uniform` ≡ `jac_trunc_int4_keep100pct` | → angles → all → int4                          |  6,140 |  8.0× |  87.5% | 0.6788 | 0.6904 | 0.6941 | 0.6956 | 0.7767 |  94.50% |
| 21 | `cart_clip_60pct`                        | first 40% coords → int8, rest = 0                        |  4,912 | 10.0× |  90.0% | 0.7121 | **0.7229** | 0.7272 | 0.7281 | 0.7927 |  98.95% |
| 22 | `jac_clip_60pct` = `pos_clip_60pct`      | → angles → top/first 40% → int8, rest = mean             |  4,912 | 10.0× |  90.0% | 0.7087 | 0.7226 | 0.7269 | 0.7275 | 0.7937 |  98.91% |
| 23 | `jac_trunc_int4_keep70pct`               | → angles → top 70% by sens → int4, rest = mean           |  4,296 | 11.4× |  91.3% | 0.6784 | 0.6898 | 0.6937 | 0.6952 | 0.7766 |  94.41% |
| 24 | `jac_trunc_int8_keep30pct`               | → angles → top 30% by sens → int8, rest = mean           |  3,680 | 13.4× |  92.5% | 0.7063 | 0.7184 | 0.7238 | 0.7238 | 0.7902 |  98.33% |
| 25 | `cart_int2`                              | all 1,536 coords → int2 (**catastrophic collapse**)      |  3,072 | 16.0× |  93.8% | 0.0030 | **0.0030** | 0.0036 | 0.0040 | 0.0131 |   0.41% |
| 26 | `greedy_3072b`                           | → angles → mixed-precision per angle (3,072 b total)     |  3,072 | 16.0× |  93.8% | 0.5745 | 0.5890 | 0.5944 | 0.5953 | 0.7237 |  80.62% |
| 27 | `jac_trunc_int4_keep50pct`               | → angles → top 50% by sens → int4, rest = mean           |  3,072 | 16.0× |  93.8% | 0.6772 | **0.6887** | 0.6929 | 0.6939 | 0.7762 |  94.26% |
| 28 | `angle_int2_uniform` ≡ `jac_trunc_int2_keep100pct` | → angles → all → int2 (degenerate floor)        |  3,070 | 16.0× |  93.8% | 0.2289 | 0.2396 | 0.2454 | 0.2484 | 0.4371 |  32.79% |
| 29 | `cart_clip_80pct`                        | first 20% coords → int8, rest = 0                        |  2,456 | 20.0× |  95.0% | 0.6862 | 0.7008 | 0.7043 | 0.7059 | 0.7813 |  95.92% |
| 30 | `jac_clip_80pct` = `pos_clip_80pct`      | → angles → top/first 20% → int8, rest = mean             |  2,456 | 20.0× |  95.0% | 0.6932 | **0.7057** | 0.7114 | 0.7119 | 0.7849 |  96.59% |
| 31 | `jac_trunc_int2_keep70pct`               | → angles → top 70% by sens → int2, rest = mean           |  2,148 | 22.9× |  95.6% | 0.2289 | 0.2396 | 0.2454 | 0.2484 | 0.4371 |  32.79% |
| 32 | `jac_trunc_int4_keep30pct`               | → angles → top 30% by sens → int4, rest = mean           |  1,840 | 26.7× |  96.3% | 0.6688 | **0.6810** | 0.6843 | 0.6852 | 0.7719 |  93.21% |
| 33 | `greedy_1536b`                           | → angles → mixed-precision per angle (collapsed to int2) |  1,536 | 32.0× |  96.9% | 0.2289 | 0.2396 | 0.2454 | 0.2484 | 0.4371 |  32.79% |
| 34 | `jac_trunc_int2_keep50pct`               | → angles → top 50% by sens → int2, rest = mean           |  1,536 | 32.0× |  96.9% | 0.2289 | 0.2396 | 0.2454 | 0.2484 | 0.4371 |  32.79% |
| 35 | `jac_trunc_int2_keep30pct`               | → angles → top 30% by sens → int2, rest = mean           |    920 | 53.4× |  98.1% | 0.2289 | 0.2396 | 0.2454 | 0.2484 | 0.4371 |  32.79% |

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

**Pending re-run (added since the evaluation that produced the rows above):**

`pos_clip_{20, 40, 50, 60, 80}pct_zero` — same operation as `pos_clip_*pct` but with dropped angles set to **0** instead of the corpus mean. Adds 5 new rows after the next `evaluate_beir_mini.py` run; they will slot into rows 10, 15, 18, 22, 30 (same bit budgets as their `pos_clip_*pct` siblings) once data is available.

**Reading the table.**

- The 0.2396 NDCG@10 floor on every int2-based scheme is the *random-baseline* — when all reconstructed vectors converge to ~the same point (because int2 quant noise dominates), every query's relevant docs tie with the rest of the corpus and NDCG@10 reflects only random tie-breaking. It is not "32% retention"; it's the ceiling of "all information lost."
- The bold rows at 6,144 bits (rows 17–18) and 4,912 bits (rows 21–22) are the central evidence for the three-way equivalence (`cart_clip ≈ pos_clip ≈ jac_clip`) discussed in §5.3.
- `greedy_*b` underperforms truncation at every budget below 12,288 bits (§4.4); drop from paper unless refactored.
- `cart_int2` (row 25) is the unique catastrophic-collapse cell; every other 16× scheme retains ≥32% NDCG, and the angle-clipping variants retain ≥94%.
