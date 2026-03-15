---
name: Genetic Algorithm Understanding
description: Guide for the GA implementation and its LLM-evolvable components used in meta-evolution.
---

# Genetic Algorithm Understanding

This skill covers the GA in `meta_evolution/genetic_algorithm.py` and how `meta_evolver.py` evolves it.

## Architecture: Safe LLM Evolution Pattern

The GA uses a **helper delegation pattern** — each core operation (`_crossover`, `_mutate`, `_selection`) delegates to a small, isolated helper function. The LLM only modifies the helper, never the parent method, so it can't break the GA's control flow.

### LLM-Evolvable Helpers

| Helper | Called by | What it does | LLM opportunity |
|--------|-----------|-------------|-----------------|
| `mutate_gene(current_value, possible_values)` | `_mutate` | Returns a new gene value | Weighted sampling, adaptive mutation, simulated annealing |
| `combine_genes(gene_name, p1_val, p2_val, xpoint, idx, total)` | `_crossover` | Decides which parent gene to use | Uniform crossover, blending, gene-aware strategies |
| `select_competitor(competitors)` | `_selection` | Picks winner from tournament | Fitness-proportional, diversity pressure, stochastic acceptance |

### Why helpers instead of full methods?

**Finding from logs**: When the LLM was given `_selection` (the full method with `random.sample` + `max`), it could break the sampling logic. By isolating only the "pick winner" decision into `select_competitor`, the tournament sampling stays safe.

Same pattern for crossover — `_crossover` handles the loop over genes, `combine_genes` only decides one gene at a time.

## Meta-Evolution Loop

`meta_evolver.py` cycles through components in round-robin:
```
Iteration 1 → mutate_gene
Iteration 2 → combine_genes
Iteration 3 → select_competitor
Iteration 4 → mutate_gene  (repeat)
```

Each iteration:
1. Extracts the current helper from `genetic_algorithm.py`
2. Sends it to DeepSeek Coder 6.7B with a targeted prompt
3. Validates syntax and injects the new code
4. Runs a full GA benchmark (`POPULATION_SIZE × GENERATIONS` evaluations)
5. If reward > 0 → keeps the change, fine-tunes LoRA adapter

## Known Issues

### LLM echoes identical code
The 6.7B model often returns the input unchanged. It also hallucinates test code after the function. The `_extract_function_body` method picks the **last** valid candidate, which can be an incomplete second attempt.

### Benchmark is slow
Each benchmark runs `POPULATION_SIZE × GENERATIONS` model evaluations. With `norm_256_flip` (256×256 images), each model takes ~1-3 min. Changed to `norm_32_flip` (native CIFAR-10 32×32) for ~10× speedup.

### Duplicate evaluations
`seen_checksums` now persists across runs by scanning existing `stats/` folders at startup.

## GA Parameters (configured via env vars in meta_evol_tune_nngpt.json)

| Parameter | Env var | Default | Effect on speed |
|-----------|---------|---------|----------------|
| Population | `POPULATION_SIZE` | 10 | Linear: 2× pop = 2× time |
| Generations | `GENERATIONS` | 3 | Linear: 2× gens = 2× time |
| Mutation rate | `MUTATION_RATE` | 0.6 | No speed impact |
| Meta iterations | `META_ATTEMPTS` | 10 | Linear: each runs a full benchmark |
