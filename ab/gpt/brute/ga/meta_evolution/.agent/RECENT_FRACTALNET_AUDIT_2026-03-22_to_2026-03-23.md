# Recent FractalNet Audit

**Scope:** models generated in the last 2-3 days only  
**Date window used:** `2026-03-22` to `2026-03-23`  
**Source folder:** `nn-gpt/ab/gpt/brute/ga/meta_evolution/ga_fractal_arch`

---

## Executive Summary

| Metric | Value |
|---|---:|
| Total generated models | 207 |
| Functional models (`init + forward` pass OK) | 92 |
| Functional + actual fractal models | 43 |
| Non-fractal but functional models | 49 |
| Broken models | 115 |

### Health Snapshot

```text
Functional + fractal     [########------------]  43 / 207
Functional, not fractal  [#########-----------]  49 / 207
Broken                   [################----] 115 / 207
```

> “Fractal” here means the model contains active `FractalDropPath` merge nodes.  
> Models with `n_columns = 1` in every top-level block are treated as non-fractal conv stacks.

---

## Day-by-Day Status

| Day | Generated | Functional | Functional + Fractal | Functional, Non-Fractal | Broken |
|---|---:|---:|---:|---:|---:|
| `2026-03-22` | 87 | 56 | 28 | 28 | 31 |
| `2026-03-23` | 120 | 36 | 15 | 21 | 84 |

### Quick Read

- `2026-03-22` has the stronger FractalNet variants.
- `2026-03-23` produced many more invalid files.
- The best recent architectures are the `4-block, 3-column` models from `2026-03-22`.

---

## How To Read The Architecture Info

| Field | Meaning |
|---|---|
| `blocks` | Number of top-level fractal blocks in the network |
| `cols` | `n_columns` for each top-level block |
| `joins_total` | Total `FractalDropPath` merge nodes in the full model |
| `top_level_joins` | Number of top-level blocks that actually perform fractal joining |

### Visual Rule of Thumb

```text
cols=(2,2,2)       -> moderate fractal branching
cols=(3,3,3,3)     -> strongest recent branching
joins_total higher  -> more recursive fractal merge points
```

---

## Best Recent FractalNets

These are the strongest valid models from the last 2-3 days.

| Rank | Model ID | Day | Blocks | Columns | Total Joins | Visual |
|---:|---|---|---:|---|---:|---|
| 1 | `11c2c6bf2d731b770e675b584b362942` | `2026-03-22` | 4 | `(3,3,3,3)` | 16 | `★★★★ fractal depth` |
| 2 | `40f30e547b3147cc85f6b06651056754` | `2026-03-22` | 4 | `(3,3,3,3)` | 16 | `★★★★ fractal depth` |
| 3 | `56a9c98e437c4fb08e100b6b3664094c` | `2026-03-22` | 4 | `(3,3,3,3)` | 16 | `★★★★ fractal depth` |
| 4 | `f01645c74a995f904d54f4913172d53e` | `2026-03-22` | 4 | `(3,3,3,3)` | 16 | `★★★★ fractal depth` |
| 5 | `fbb62c946e2f944ae1542b2e9c152154` | `2026-03-22` | 4 | `(3,3,3,3)` | 16 | `★★★★ fractal depth` |

### Recommended Shortlist

- `11c2c6bf2d731b770e675b584b362942`
- `40f30e547b3147cc85f6b06651056754`
- `56a9c98e437c4fb08e100b6b3664094c`
- `f01645c74a995f904d54f4913172d53e`
- `fbb62c946e2f944ae1542b2e9c152154`
- `27d22dcdabf7aa3daad387099c37c3b8`
- `e8ad4b901ff66c36105e33b89cbcc79c`

---

## Valid FractalNets From `2026-03-23`

| Model ID | Blocks | Cols | Joins Total | Top-Level Joins | Fractal Strength |
|---|---:|---|---:|---:|---|
| `014a3bceedfa13c95711f06efc464c8d` | 3 | `(2,2,2)` | 3 | 3 | `★★` |
| `021f0bd30f1b9433ca7c027b3a9bd80d` | 2 | `(2,2)` | 2 | 2 | `★` |
| `1e376b12f9b85d5ce127934843b60cb9` | 2 | `(2,2)` | 2 | 2 | `★` |
| `344ee1c78fa85e3d9bb24a9e8b6db011` | 3 | `(2,2,2)` | 3 | 3 | `★★` |
| `3cdcefbe9bc7237721dd14f22a69efe8` | 3 | `(2,2,2)` | 3 | 3 | `★★` |
| `44e9216fc6675740cc760220a9eba2b5` | 3 | `(2,2,2)` | 3 | 3 | `★★` |
| `638b1dd716b508262947f69a8b3021ef` | 3 | `(2,2,2)` | 3 | 3 | `★★` |
| `7018a1a1878b9f64a957ed429351fcd8` | 2 | `(2,2)` | 2 | 2 | `★` |
| `8dc81e0fe36fa79f6aa1b9456c9c1250` | 3 | `(2,2,2)` | 3 | 3 | `★★` |
| `8e0c258ff62ad6ad66c77f756d6886cc` | 2 | `(2,2)` | 2 | 2 | `★` |
| `b7258308c7d1a30002ea6f2eae6fe7c1` | 2 | `(2,2)` | 2 | 2 | `★` |
| `d1b5fdc82851017733ccff06a32a8708` | 3 | `(2,2,2)` | 3 | 3 | `★★` |
| `da15832d48a099c246d202baf0127285` | 3 | `(2,2,2)` | 3 | 3 | `★★` |
| `e8ad4b901ff66c36105e33b89cbcc79c` | 3 | `(2,2,2)` | 3 | 3 | `★★` |
| `ed03e5932f74e1b2dcf3c0719d70d78c` | 2 | `(2,2)` | 2 | 2 | `★` |

### Pattern Summary For `2026-03-23`

```text
2 blocks x 2 columns   :  6 models
3 blocks x 2 columns   :  9 models
3-column models        :  0 models
```

---

## Valid FractalNets From `2026-03-22`

| Model ID | Blocks | Cols | Joins Total | Top-Level Joins | Fractal Strength |
|---|---:|---|---:|---:|---|
| `11c2c6bf2d731b770e675b584b362942` | 4 | `(3,3,3,3)` | 16 | 4 | `★★★★` |
| `13955f0a0e774cb700279c159be2bcfa` | 4 | `(2,2,2,2)` | 4 | 4 | `★★★` |
| `17f8276bc58a42c9136b8abe9de2fdda` | 2 | `(2,2)` | 2 | 2 | `★` |
| `1ea4b4f342d656ee333664e731f07659` | 1 | `(3,)` | 4 | 1 | `★★` |
| `2406cc49920e1606e9070ee1eb083a3f` | 1 | `(3,)` | 4 | 1 | `★★` |
| `27d22dcdabf7aa3daad387099c37c3b8` | 3 | `(3,3,3)` | 12 | 3 | `★★★★` |
| `2fc43461a7e56a2ac04d3c4cd8fd1bb2` | 1 | `(2,)` | 1 | 1 | `★` |
| `39c43e59d7a5711e19d9d74235b14691` | 4 | `(2,2,2,2)` | 4 | 4 | `★★★` |
| `40f30e547b3147cc85f6b06651056754` | 4 | `(3,3,3,3)` | 16 | 4 | `★★★★` |
| `4da3152987c2b286d3ebc6af4f5a2ff4` | 3 | `(2,2,2)` | 3 | 3 | `★★` |
| `4f53bfc51afdbc9f07a907df4e17f2c0` | 2 | `(2,2)` | 2 | 2 | `★` |
| `53c2c84897b77bda64d4c366a0622ff7` | 2 | `(3,3)` | 8 | 2 | `★★★` |
| `56a9c98e437c4fb08e100b6b3664094c` | 4 | `(3,3,3,3)` | 16 | 4 | `★★★★` |
| `5d129b9b39e6f9f64b64eb02c89645c3` | 1 | `(3,)` | 4 | 1 | `★★` |
| `6afaaf3dc4704a9f2cbe4ce43fe3be28` | 2 | `(2,2)` | 2 | 2 | `★` |
| `6f3899e722f1530f984591486ab57110` | 2 | `(3,3)` | 8 | 2 | `★★★` |
| `7712e14efe55ae625f1e0c33eeeaf684` | 2 | `(2,2)` | 2 | 2 | `★` |
| `9b3c870a05f51a62354d975412ffb5fd` | 4 | `(2,2,2,2)` | 4 | 4 | `★★★` |
| `ae5fda5b21baad5d0b257eff6cb4399d` | 2 | `(2,2)` | 2 | 2 | `★` |
| `b8efa5a5128e3b1cee4a3ad77b8a04ff` | 4 | `(2,2,2,2)` | 4 | 4 | `★★★` |
| `bd021b9839ee4b366f2c33dae6d15e54` | 2 | `(2,2)` | 2 | 2 | `★` |
| `dfb31efe0df78d2728f6883b680481a5` | 2 | `(2,2)` | 2 | 2 | `★` |
| `e6c0db2a34d4f2a7ebfb8b725977ef3a` | 1 | `(2,)` | 1 | 1 | `★` |
| `e99f48cbc52bb279946011d93450ed31` | 3 | `(2,2,2)` | 3 | 3 | `★★` |
| `f01645c74a995f904d54f4913172d53e` | 4 | `(3,3,3,3)` | 16 | 4 | `★★★★` |
| `f5eeb608a330b46ca079770b77f37516` | 4 | `(2,2,2,2)` | 4 | 4 | `★★★` |
| `fbb62c946e2f944ae1542b2e9c152154` | 4 | `(3,3,3,3)` | 16 | 4 | `★★★★` |

### Pattern Summary For `2026-03-22`

```text
1 block  x 2 columns   :  2 models
1 block  x 3 columns   :  3 models
2 blocks x 2 columns   :  8 models
2 blocks x 3 columns   :  2 models
3 blocks x 2 columns   :  2 models
3 blocks x 3 columns   :  1 model
4 blocks x 2 columns   :  5 models
4 blocks x 3 columns   :  5 models
```

---

## Visual Architecture Ladder

```text
Weakest  : 1 block  x 2 cols  -> 1 join
Better   : 2 blocks x 2 cols  -> 2 joins
Stronger : 3 blocks x 3 cols  -> 12 joins
Best     : 4 blocks x 3 cols  -> 16 joins
```

---

## Final Recommendation

### If you want the most expressive recent FractalNets

Choose the `4-block, 3-column` models from `2026-03-22`:

- `11c2c6bf2d731b770e675b584b362942`
- `40f30e547b3147cc85f6b06651056754`
- `56a9c98e437c4fb08e100b6b3664094c`
- `f01645c74a995f904d54f4913172d53e`
- `fbb62c946e2f944ae1542b2e9c152154`

### If you want a safe recent baseline

Choose:

- `e8ad4b901ff66c36105e33b89cbcc79c`

It is valid, recent, and has:

- `3` top-level fractal blocks
- `2` columns per block
- `3` total fractal join nodes

---

## Notes

- This report only considers models generated on `2026-03-22` and `2026-03-23`.
- “Skip connection” in this report refers to fractal branch-merge behavior via `FractalDropPath`, not residual addition.
- Many files generated on these days are invalid because of float-valued channels, float loop counts, or `None` hyperparameters.
