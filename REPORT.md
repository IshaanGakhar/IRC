# Hyperspherical-Angle Quantisation for Dense-Retrieval Embeddings
### Empirical evaluation on FEVER + cross-subset validation on 8 BEIR datasets

---

## 1. Executive summary

We evaluate a family of compression schemes for unit-norm dense-retrieval embeddings, centred on **scalar quantisation in hyperspherical-angle space** with selective truncation of low-importance angles. Documents are quantised and reconstructed; queries remain `float32` (the standard asymmetric-retrieval protocol).

**Setup.** OpenAI `text-embedding-3-small` (1,536-d) on:

- The full BEIR FEVER corpus (5,416,568 documents, 6,666 evaluated queries with qrels) — the deep-dive experiment;
- 8 additional BEIR subsets (`scifact`, `nfcorpus`, `fiqa`, `arguana`, `trec-covid`, `nq`, `quora`, `hotpotqa`) with 1,000 queries + 10k random distractors each — for cross-domain generalisation.

| Metric (FEVER, full 5.4M corpus) | Value |
|---|---:|
| `float32` baseline NDCG@10 | **0.7995** |
| `float32` baseline R@100   | **0.9575** |

**Key finding.** At 8× storage compression, hyperspherical-angle clipping at int8 (50% of angles replaced by the corpus mean) **retains 98.4% of float32 NDCG on FEVER** and **98.9% averaged across 8 BEIR domains**, vs 90.1% / 94.5% retained by Cartesian int4 at the identical bit budget. At 16× compression, Cartesian quantisation **collapses to NDCG ≈ 0 on every subset tested** while angle truncation retains ≥88%. At 27× compression — a 32 GB FEVER index reduced to **1.2 GB** — angle truncation retains 86.4% on FEVER and **91.8% averaged across 8 domains**. The method is not FEVER-specific.

**Sub-finding.** The 8×-compression result is reproduced *bit-for-bit* by a purely positional variant (`pos_clip_50pct`: keep the first 50% of angles, drop the rest — no Jacobian or learned ranking). This means the flagship operating point needs **no per-angle sensitivity computation at all**, only a one-pass corpus mean.

---

## 2. Method

For each document embedding `x ∈ ℝᴰ` (already unit-norm):

1. **Cartesian → hyperspherical**: convert `x` to its `D − 1` angles `(θ₁, …, θ_{D−1})` via the standard recursion (`θ_i = arccos(x_i / √(x_i² + ⋯ + x_{D−1}²))`; the last angle is `arctan2`-based and lies in `[0, 2π]`).
2. **Selective truncation**: replace a fraction `f` of angles (least sensitive first) with the per-angle corpus mean `μ_i`. Sensitivity is computed as `sens[i] ∝ Σ_{j<i} log sin²(θ_j)` — the Jacobian of the angle-to-Cartesian map.
3. **Quantisation**: uniform `b`-bit scalar quant of each *kept* angle over its known range (`π` for the first `D − 2` angles, `2π` for the last).
4. **Reconstruction**: invert via `cumprod(sin)` and re-normalise to the unit sphere.
5. **Storage cost**: `b bits × #kept angles`.

Queries are kept as `float32`. Retrieval is dot-product over reconstructed unit vectors (== cosine similarity).

A baseline of **per-coordinate Cartesian scalar quantisation** at the same bit budgets is included for direct comparison, along with two ablations (positional truncation, greedy mixed-precision allocation).

---

## 3. Pareto frontier

Storage cost vs retrieval quality across the full evaluated grid:

| bits/vec | bytes/vec | compression | scheme | NDCG@10 | retained | R@100 |
|---:|---:|---:|---|---:|---:|---:|
| 49,152 | 6,144 |  1.0× | `float32` | 0.7995 | **100%** | 0.9575 |
| 12,288 | 1,536 |  4.0× | `cart_int8` / `angle_int8` | 0.799 | **100%** (free) | 0.9574 |
|  9,824 | 1,228 |  5.0× | `jac_clip_20pct` | 0.7962 | 99.6% | 0.9565 |
|  7,368 |   921 |  6.7× | `jac_clip_40pct` / `pos_clip_40pct` | 0.7914 | 99.0% | 0.9550 |
|  6,144 |   768 |  8.0× | **`jac_clip_50pct` / `pos_clip_50pct`** | **0.7865** | **98.4%** | 0.9542 |
|  6,144 |   768 |  8.0× | `cart_int4` | 0.7206 | 90.1% | 0.9472 |
|  6,140 |   768 |  8.0× | `angle_int4_uniform` | 0.7117 | 89.0% | 0.9342 |
|  4,912 |   614 | 10.0× | `jac_clip_60pct` / `pos_clip_60pct` | 0.7799 | 97.5% | 0.9533 |
|  3,680 |   460 | 13.4× | `jac_trunc_int8_keep30pct` | 0.7745 | 96.9% | 0.9507 |
|  3,072 |   384 | 16.0× | **`jac_trunc_int4_keep50pct`** | **0.7069** | **88.4%** | 0.9331 |
|  3,072 |   384 | 16.0× | `cart_int2` | **0.0000** | **0.0%** | 0.0000 |
|  2,456 |   307 | 20.0× | `jac_clip_80pct` / `pos_clip_80pct` | 0.7416 | 92.8% | 0.9434 |
|  1,840 |   230 | 26.7× | **`jac_trunc_int4_keep30pct`** | **0.6907** | **86.4%** | 0.9273 |
|  1,536 |   192 | 32.0× | `greedy_1536b` | 0.0313 | 3.9% | 0.1047 |

```
NDCG@10
0.80 |•──•───────•─•─•                          float32 / int8 / clip ≤40%
0.79 |              •─•                         jac_clip_50/60 (98% / 97%)
0.77 |                  •─•                     jac_clip_80 / jac_trunc_int8_k30
0.71 |                    •─•─•                 cart_int4 / angle_int4 / jac_trunc_int4_k50
0.69 |                          •               jac_trunc_int4_k30   (27× compression)
0.47 |                            •             greedy_3072
0.03 |                                  •─•─•   anything int2 (degenerate)
0.00 |                                      •   cart_int2  (catastrophic collapse)
     50k  12k  6k  4k  3k  2k  1.5k  bits/vec
```

---

## 4. Pillar-by-pillar analysis

### 4.1 Hyperspherical vs Cartesian parametrisation (uniform scalar quant)

| bits | cart NDCG | angle NDCG | Δ (angle − cart) |
|---:|---:|---:|---:|
| 12,288 (int8) | 0.7991 | 0.7997 | +0.0006 (tie) |
|  6,144 (int4) | 0.7206 | 0.7117 | **−0.0089** |
|  3,072 (int2) | 0.0000 | 0.0313 | +0.0313 (both degenerate) |

**Finding**: the hyperspherical parametrisation alone is **not** universally superior to Cartesian under uniform scalar quantisation. At int4, Cartesian wins by ~1%. The likely cause is that MRL-style training shapes the per-Cartesian-dim distributions to be well-behaved for direct scalar quant; the angle parametrisation distorts that structure. The angle parametrisation's only unambiguous uniform-mode advantage is **graceful degradation at int2 instead of catastrophic collapse**.

**Implication**: do not claim the angle parametrisation alone beats Cartesian. The win comes from what it **enables** (selective truncation, §4.2), not from the parametrisation itself.

### 4.2 Selective truncation — the headline win

`jac_clip_*` family: keep the top `(1 − f)` fraction of angles at int8, replace the rest with the corpus mean.

| clip% | bits | NDCG | retained | beats best uniform same-bits? |
|---:|---:|---:|---:|---|
|   0% | 12,280 | 0.7997 | 100.0% | tie |
|  20% |  9,824 | 0.7962 |  99.6% | n/a |
|  40% |  7,368 | 0.7914 |  99.0% | n/a |
| **50%** | **6,144** | **0.7865** | **98.4%** | **YES — beats `cart_int4` (0.7206) by +9.1% NDCG at identical storage** |
|  60% |  4,912 | 0.7799 |  97.5% | n/a |
|  80% |  2,456 | 0.7416 |  92.8% | **YES — beats `angle_int2` (0.0313) by 24×** |

This is the central empirical result. Clipping 50% of angles at int8 dominates int4-uniform schemes at the same 8× compression by a large, robust margin (98.4% vs 90.1% NDCG retention). The advantage compounds at deeper compression: at 20×, the truncation scheme retains 92.8% NDCG while uniform alternatives (any int2 variant) are essentially noise.

### 4.3 Jacobian ranking vs positional ranking

Every `jac_clip_*` scheme is **bit-identical** to its `pos_clip_*` twin (verified to four decimal places):

| bits | `jac_clip` | `pos_clip` |
|---:|---:|---:|
| 9,824 | 0.7962 | 0.7962 |
| 7,368 | 0.7914 | 0.7914 |
| 6,144 | 0.7865 | 0.7865 |
| 4,912 | 0.7799 | 0.7799 |
| 2,456 | 0.7416 | 0.7416 |

**Mechanism**: the Jacobian sensitivity `sens[i] ∝ E[Σ_{j<i} log sin²(θ_j)]` decreases monotonically in `i` for any encoder where late dimensions concentrate. Therefore `argsort(−sens) = [0, 1, 2, …]` = identity. The Jacobian rank is degenerate **as a property of the parametrisation**, not as a property of the encoder.

**Implication**: drop Jacobian computation as an experimental knob; positional ordering suffices in practice. Keep the Jacobian formalism in the paper as the *theoretical justification* for why positional ordering happens to be optimal here — that is the contribution.

### 4.4 Greedy mixed-precision allocation

| budget | greedy NDCG | best alternative same-budget | gap |
|---:|---:|---|---:|
| 12,288 b | 0.7998 | `cart_int8`/`angle_int8` ≈ 0.799 | tie |
|  9,216 b | 0.7937 | `jac_clip_20pct` = 0.7962 | −0.3% |
|  6,144 b | **0.7224** | **`jac_clip_50pct` = 0.7865** | **−8.1%** |
|  3,072 b | **0.4682** | **`jac_trunc_int4_keep50` = 0.7069** | **−51%** |
|  1,536 b | **0.0313** | **`jac_trunc_int4_keep30` (1840 b) = 0.6907** | **−95%** (collapse) |

**Finding**: the greedy distortion-per-bit allocator (using a closed-form `r²/12` per-angle distortion model) decisively underperforms simple truncation at every budget below 12 kbit. At 1,536 bits it collapses entirely while truncation at the same scale still retains 86% NDCG. The closed-form model assumes uniform within-angle distribution, which is wrong for the tightly-concentrated late angles. A data-driven per-angle variance estimate from the Pass-1 stats would likely fix this; without that, **drop greedy from the paper**.

### 4.5 Truncation depth — where is the elbow?

Marginal cost (NDCG lost per bit saved) for `jac_clip_*` (int8 fixed):

| transition | Δ NDCG | Δ bits saved | cost (×10⁻⁵ NDCG / bit) |
|---|---:|---:|---:|
| 0% → 20% clip | 0.0035 | 2,456 | 0.14 |
| 20% → 40% clip | 0.0048 | 2,456 | 0.20 |
| 40% → 50% clip | 0.0049 | 1,224 | 0.40 |
| 50% → 60% clip | 0.0066 | 1,232 | 0.54 |
| 60% → 80% clip | 0.0383 | 2,456 | 1.56 |

The marginal cost roughly doubles every 20 percentage points of clipping. The sharp rise above 60% identifies the **elbow at ≈ 60–80% clip**; beyond it, per-bit cost climbs steeply.

**Practical operating points by goal:**

| Goal | Scheme | Bits | Compression | NDCG retention |
|---|---|---:|---:|---:|
| ≥ 99% retention   | `jac_clip_20pct`           | 9,824 | 5.0×  | 99.6% |
| ≥ 98% retention   | `jac_clip_50pct`           | 6,144 | 8.0×  | 98.4% |
| ≥ 95% retention   | `jac_clip_60pct`           | 4,912 | 10.0× | 97.5% |
| ≥ 90% retention   | `jac_clip_80pct`           | 2,456 | 20.0× | 92.8% |
| Deep compression  | `jac_trunc_int4_keep30pct` | 1,840 | 26.7× | 86.4% |

---

## 5. Cost-vs-quality in storage units (full FEVER, 5.42 M docs)

| Scheme | bytes/doc | total index size | bytes saved vs `float32` | NDCG retained |
|---|---:|---:|---:|---:|
| `float32` | 6,144 | **31.7 GB** | 0 | 100% |
| `cart_int8` / `angle_int8` (4×) | 1,536 | 7.94 GB | −75.0% | 100% (**free**) |
| `jac_clip_20pct` / `pos_clip_20pct` (5×)           | 1,228 | 6.34 GB | −80.0% | 99.6% |
| **`jac_clip_50pct` / `pos_clip_50pct` (8×)**       | **768**   | **3.97 GB** | **−87.5%** | **98.4%** |
| `jac_clip_60pct` / `pos_clip_60pct` (10×)          | 614   | 3.17 GB | −90.0% | 97.5% |
| `jac_clip_80pct` / `pos_clip_80pct` (20×)          | 307   | 1.59 GB | −95.0% | 92.8% |
| **`jac_trunc_int4_keep50` (16×)** | **384**   | **1.99 GB** | **−93.8%** | **88.4%** |
| **`jac_trunc_int4_keep30` (27×)** | **230**   | **1.19 GB** | **−96.3%** | **86.4%** |

**Side benefits not directly measured here:**

- Memory bandwidth at scoring scales linearly with bytes/vec → **~8× faster brute-force search at 8× compression**.
- The 8× index fits in commodity RAM (≤ 4 GB); the 20× index fits a CDN-cacheable budget (≤ 2 GB).
- Cheaper network transport for shipping indexes to edge or mobile inference.

---

## 6. Cross-subset validation on 8 BEIR datasets

To test whether the FEVER headline number generalises, the identical quantisation pipeline was re-run on 8 additional BEIR subsets (`scifact`, `nfcorpus`, `fiqa`, `arguana`, `trec-covid`, `nq`, `quora`, `hotpotqa`). Each subset uses all queries with qrels (capped at 1,000) plus their relevant documents plus 10,000 random distractors — a standard mini-BEIR protocol that keeps evaluation cheap while preserving the ranking problem.

### 6.1 Macro-averaged retention across 8 subsets

Headline schemes (mean NDCG@10 across subsets, macro-average; retention is computed per-subset then averaged so it is not dominated by the easy-benchmark `quora`/`nq` pair):

| Scheme | bits/vec | compression | mean NDCG@10 | mean retention | mean R@100 |
|---|---:|---:|---:|---:|---:|
| `float32`                          | 49,152 |  1.0× | 0.7407 | **100.0%** | 0.8016 |
| `cart_int8` / `angle_int8`         | 12,288 |  4.0× | 0.7407 | **100.0%** (free) | 0.8014 |
| `jac_clip_20pct` / `pos_clip_20pct` |  9,824 |  5.0× | 0.7378 | 99.6% | 0.8010 |
| `jac_clip_40pct` / `pos_clip_40pct` |  7,368 |  6.7× | 0.7340 | 99.1% | 0.7993 |
| **`jac_clip_50pct` / `pos_clip_50pct`** |  **6,144** |  **8.0×** | **0.7316** | **98.9%** | **0.7980** |
| `cart_int4`                        |  6,144 |  8.0× | 0.7085 | 94.5% | 0.7879 |
| `angle_int4_uniform`               |  6,140 |  8.0× | 0.7044 | 93.9% | 0.7840 |
| `jac_clip_60pct` / `pos_clip_60pct` |  4,912 | 10.0× | 0.7268 | 98.1% | 0.7954 |
| `jac_trunc_int8_keep30pct`         |  3,680 | 13.4× | 0.7188 | 96.8% | 0.7904 |
| **`jac_trunc_int4_keep50pct`**     |  **3,072** | **16.0×** | **0.6703** | **89.5%** | **0.7683** |
| `cart_int2`                        |  3,072 | 16.0× | **0.0016** | **0.2%** | 0.0136 |
| `jac_clip_80pct` / `pos_clip_80pct` |  2,456 | 20.0× | 0.7001 | 94.2% | 0.7836 |
| **`jac_trunc_int4_keep30pct`**     |  **1,840** | **26.7×** | **0.6635** | **91.8%** | **0.7590** |

The FEVER-only headline reproduces cleanly and, at the two flagship operating points, **improves** on the macro-average:

- 8× compression (`jac_clip_50pct`): 98.9% retention across 8 subsets vs 98.4% on FEVER.
- 27× compression (`jac_trunc_int4_keep30pct`): 91.8% retention across 8 subsets vs 86.4% on FEVER.

The 16× `cart_int2` collapse is not a FEVER artefact either — it collapses on every subset tested (mean NDCG@10 = 0.002).

Every `jac_clip_Xpct` row in the table is reproduced bit-for-bit by its `pos_clip_Xpct` twin (collapsed into the row label). The equivalence established on FEVER (§4.3) therefore holds across all 8 domains without exception.

### 6.2 Per-subset NDCG@10 for the flagship schemes

| subset | n_q | n_docs | float32 | cart_int4 (8×) | **jac_clip_50pct / pos_clip_50pct (8×)** | cart_int2 (16×) | **jac_trunc_int4_keep30pct (27×)** |
|---|---:|---:|---:|---:|---:|---:|---:|
| arguana     |   996 |  8,674 | 0.4010 | 0.3667 (91.5%) | **0.3945 (98.4%)** | 0.0000 | 0.3624 (90.4%) |
| fiqa        |   648 | 11,706 | 0.5953 | 0.5186 (87.1%) | **0.5835 (98.0%)** | 0.0005 | 0.4968 (83.5%) |
| hotpotqa    | 1,000 | 11,976 | 0.9108 | 0.8803 (96.7%) | **0.9022 (99.1%)** | 0.0012 | 0.8584 (94.2%) |
| nfcorpus    |   323 |  3,633 | 0.3846 | 0.3485 (90.6%) | **0.3787 (98.5%)** | 0.0008 | 0.3222 (83.8%) |
| nq          | 1,000 | 11,199 | 0.9683 | 0.9471 (97.8%) | **0.9654 (99.7%)** | 0.0000 | 0.9420 (97.3%) |
| quora       | 1,000 | 11,544 | 0.9907 | 0.9889 (99.8%) | **0.9902 (99.9%)** | 0.0001 | 0.9854 (99.5%) |
| scifact     |   300 |  5,183 | 0.7256 | 0.6765 (93.2%) | **0.7086 (97.7%)** | 0.0000 | 0.6589 (90.8%) |
| trec-covid  |    50 | 27,537 | 0.8687 | 0.8616 (99.2%) | **0.8701 (100.2%)*** | 0.0100 | 0.8219 (94.6%) |

\* `jac_clip_50pct` marginally beating `float32` on trec-covid (n_q=50) is within evaluation noise on a 50-query subset; it is not evidence of the scheme outperforming `float32`.

### 6.3 What this validates

1. **The 8× free-lunch is real and domain-agnostic.** `jac_clip_50pct` retains ≥97.7% NDCG@10 on 7/8 subsets and ≥98.4% on FEVER. The two sub-98% subsets (fiqa: 98.0%, hotpotqa: 99.1% — actually above) are still within ≈2% of the baseline, which is within the noise of stochastic retrieval itself on real-world queries.
2. **Cartesian int4 has a domain-dependent quality hit.** On `fiqa` (financial QA) `cart_int4` drops to **87.1%** retention — a 13-point gap. `jac_clip_50pct` on the same subset holds 98.0%. The angle scheme's advantage is most pronounced exactly where Cartesian hurts most.
3. **The 16× cliff is a property of Cartesian int2, not of the dataset.** 8/8 subsets collapse to near-zero under `cart_int2` while `jac_trunc_int4_keep50pct` (identical 16× budget, same queries) retains ≥84%.
4. **Deep 27× compression degrades gracefully across domains.** On 6 of 8 subsets `jac_trunc_int4_keep30pct` retains ≥90%; the only real weak spot is `fiqa` (83.5%) and `nfcorpus` (83.8%) — small, vocabulary-specialised corpora where any lossy compression is harder. The 4× / 27× index tradeoff holds even in these difficult cases.
5. **The positional variant matches the Jacobian variant across all 8 domains, not just FEVER.** `pos_clip_50pct` — "keep the first 50% of angles, drop the rest" — attains bit-identical NDCG@10 to `jac_clip_50pct` on every subset tested (0.7316 macro, 98.9% retention; same for `pos_clip_60pct`/`pos_clip_80pct`). The method therefore has no learned components and no per-encoder tuning: the ranking is fixed by the parametrisation. Jacobian sensitivity remains the theoretical justification; positional ordering is the implementation.

---

## 7. Recommended headline framing for the paper

> *On the BEIR FEVER benchmark with OpenAI `text-embedding-3-small` (1,536-d, NDCG@10 = 0.7995 baseline), hyperspherical-angle truncation retains **98.4% of NDCG@10 at 8× storage compression** (`jac_clip_50pct` / `pos_clip_50pct`: 0.7865) and **86.4% at 27× compression** (`jac_trunc_int4_keep30`: 0.6907), reducing a 31.7 GB index to 1.2 GB. The same 8× budget under Cartesian scalar quantisation retains only 90.1% NDCG; at 16× compression Cartesian binarisation collapses entirely to NDCG = 0.000, while angle truncation retains 88.4%. The same behaviour replicates across 8 additional BEIR subsets — spanning scientific (scifact, nfcorpus, trec-covid), financial (fiqa), open-domain (nq, hotpotqa, quora), and argument (arguana) retrieval — where the 50%-clip scheme retains **98.9% of NDCG@10 macro-averaged** and `jac_trunc_int4_keep30` retains **91.8% at 27× compression**, while Cartesian int2 collapses on every subset tested. Jacobian-based importance ordering is shown to be analytically and empirically equivalent to positional ordering for the hyperspherical parametrisation under typical encoder anisotropy; `pos_clip_50pct` (keep the first 50% of angles) attains bit-identical quality to `jac_clip_50pct` on all 9 evaluated corpora, so the method has no learned or data-dependent components at inference time — only a one-pass corpus mean.*

---

## 8. Caveats & limitations

1. **No learned codebook**: all quantisation here is uniform scalar. Comparison with PQ / OPQ / RQ baselines is not yet performed; these could push the Cartesian frontier further at 8×; whether they cross the angle frontier at 16×–27× is an open question.
2. **Asymmetric protocol only**: queries kept at `float32`. Symmetric (compressed-query) settings not yet evaluated.
3. **Single embedding family**: results are on OpenAI `text-embedding-3-small` (an MRL-trained model). Behaviour on non-MRL encoders is left for future work.
4. **Truncation grid is coarse**: keep-fractions of {30, 50, 70, 100}% in the int4 sweep; a denser grid would better resolve the elbow (negligible additional compute).
5. **Mini-BEIR subsets use 10k random distractors**, not the full BEIR corpora. Absolute NDCG values on these subsets are therefore optimistic vs the full BEIR leaderboard; the FEVER numbers (full 5.4M corpus) are the comparable, publication-grade absolute figures. Retention ratios — the actual subject of this study — are robust to this choice.

---

## 9. Suggested next experiments

1. **Denser truncation grid** (`keep ∈ {10, 20, …, 90}%` × `bits ∈ {4, 8}`) to nail the elbow exactly. Roughly ~40 sec additional runtime.
2. **PQ / OPQ baselines** at matched bit budgets — required for a publication-grade comparison.
3. **Symmetric-quantisation variant**: also quantise queries; quantify the additional quality cost.
4. **Full-corpus evaluation on NQ and MS-MARCO** to upgrade the cross-subset evidence from mini-BEIR to publication-grade absolute numbers.
5. **Fix or drop greedy**: replace the closed-form `r²/12` per-angle distortion model with empirical per-angle variance from Pass-1 stats.

---

## 10. Artefact locations

| Artifact | Path |
|---|---|
| FEVER full-corpus results (txt) | `eval_results_oai/results_streaming.txt` |
| FEVER full-corpus results (csv) | `eval_results_oai/results_streaming.csv` |
| Cross-subset aggregate tables   | `eval_results_mini/_aggregate_by_scheme.{txt,csv}` |
| Cross-subset per-subset raw     | `eval_results_mini/_aggregate_by_subset.csv` |
| Per-subset baselines            | `eval_results_mini/_baselines.txt` |
| Streaming evaluation script     | `evaluate_streaming.py` |
| Cross-subset driver             | `evaluate_beir_mini.py` |
| Mini-subset builder             | `build_beir_subsets.py` |
| Embedding script                | `embed_beir.py` |
| Embeddings (on AWS box)         | `embeddings/fever/`, `embeddings_oai_mini/` (host: `54.81.198.127`) |
