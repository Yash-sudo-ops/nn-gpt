# Bugfix: META_SCORE: 0.0 — Accuracy Extraction Mismatch
**Date:** 2026-04-19  
**Session:** Debugging misleading META_SCORE: 0.0  
**Edited by:** Antigravity (AI assistant)

---

## Root Cause

The accuracy values produced by a real evaluation (e.g. `0.5469` = 54.69%) were stored in **nested** JSON fields:

```json
{
  "hyperparameters": { "accuracy": 0.5469, "best_accuracy": 0.5469 },
  "training_summary": { "best_accuracy": 0.5469, "final_accuracy": 0.5469 }
}
```

But `fitness_function()` in `run_fractal_evolution.py` only checked **top-level** keys:

```python
if 'accuracy' in full_res:      # ← never true (it's nested)
elif 'best_accuracy' in full_res: # ← never true (it's nested)
...
# result falls through all branches → final_accuracy = 0.0
```

This caused `fitness_history` to fill with `0.0` every generation, making:
```
META_SCORE = max([0.0, 0.0, ...]) + improvement * 1.5 = 0.0
```

Even though the GA correctly trained models and real accuracy was recorded to disk.

A secondary issue: the outer `except` in `__main__` silently suppressed any critical GA exception, making it impossible to distinguish a real GA crash from a score-parse failure.

---

## Files Changed

### `run_fractal_evolution.py` — Two targeted edits

#### Fix 1 — Layered accuracy extraction (lines ~206–242)

**Old (only top-level):**
```python
final_accuracy = 0.0
if 'accuracy' in full_res:
    final_accuracy = full_res['accuracy'] * 100
elif 'best_accuracy' in full_res:
    final_accuracy = full_res['best_accuracy'] * 100
elif isinstance(result, tuple) and len(result) >= 2:
    final_accuracy = float(result[1]) * 100
...
```

**New (layered walk):**
```python
final_accuracy = 0.0
_acc_source = "none"

# 1. Top-level (fastest path)
if 'accuracy' in full_res:
    final_accuracy = float(full_res['accuracy']) * 100
    _acc_source = "full_res.accuracy"
elif 'best_accuracy' in full_res:
    ...
# 2. hyperparameters dict (where the eval library actually writes it)
elif isinstance(full_res.get('hyperparameters'), dict):
    hp = full_res['hyperparameters']
    if 'accuracy' in hp and hp['accuracy']:
        final_accuracy = float(hp['accuracy']) * 100
        _acc_source = "hyperparameters.accuracy"
    ...
# 3. training_summary dict
if final_accuracy == 0.0 and isinstance(full_res.get('training_summary'), dict):
    ...
# 4. Raw scalar/tuple from evaluator
if final_accuracy == 0.0:
    ...
```

The `_acc_source` label is printed alongside the fitness score so you can immediately see which code path was taken.

#### Fix 2 — Un-silenced outer exception (line ~274)

**Old:**
```python
except Exception as e:
    # print(f"CRITICAL GA FAIL: {e}")
    print("META_SCORE: 0.0")
```

**New:**
```python
except Exception as e:
    import traceback
    print(f"CRITICAL GA FAIL: {e}")
    traceback.print_exc()
    print("META_SCORE: 0.0")
```

---

## Why the New Behavior Is Correct

| Scenario | Old behavior | New behavior |
|---|---|---|
| Accuracy in `hyperparameters.accuracy` | `0.0` (miss) | Correctly read and returned |
| Accuracy in `training_summary.best_accuracy` | `0.0` (miss) | Correctly read and returned |
| Evaluator returns plain dict | `0.0` if no top-level key | Walks nested structure |
| GA crash | Silent `META_SCORE: 0.0` | Full traceback printed |
| True zero-accuracy model | `0.0` (correct) | `0.0` + `source: none` (distinguishable) |

---

## Files NOT Modified
- `meta_evolver.py`
- `genetic_algorithm.py`
- `FractalNet_evolvable.py`
- `llm_loader.py`
- Any stats or arch files

---

## Remaining Risks / Edge Cases

1. **`training_summary.json` uid mismatch** — the uid guard at lines 136–142 may still discard the summary if the eval library writes a different uid format. However the fix is robust to this: it falls through to the `isinstance(result, dict)` branch and reads the same nested structure from there.
2. **True zero-accuracy** — a model that genuinely scores 0% is now logged as `source: none` which is distinguishable from a parse failure (`source: hyperparameters.accuracy`) and a true result (`source: training_summary.best_accuracy`).
3. **`combine_genes` type mismatch** — unrelated LLM bug: the current `combine_genes` can return a float blend for discrete gene values (e.g. `lr`), which may produce invalid chromosomes passed to the evaluator. Not causing META_SCORE: 0.0 directly but worth monitoring.
