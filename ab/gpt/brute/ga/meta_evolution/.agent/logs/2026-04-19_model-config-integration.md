# Changelog: model_config.json Integration
**Date:** 2026-04-19  
**Session:** Model config externalisation for `llm_loader.py`  
**Edited by:** Antigravity (AI assistant)

---

## Summary

Moved all model configuration (model name, context length, training params) out of hardcoded Python values and into a single JSON file `model_config.json`. This gives full control over model behaviour without touching any Python code.

---

## Files Changed

### 1. CREATED — `meta_evolution/model_config.json`
New file. Single source of truth for all model parameters.

```json
{
    "base_model_name": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "num_epochs": 100,
    "num_test_epochs": 2,
    "use_deepspeed": false,
    "token_from_file": false,
    "only_best_accuracy": false,
    "context_length": 4096
}
```

---

### 2. MODIFIED — `meta_evolution/llm_loader.py`

#### Change 1 — Added `import json` (line 2)
```diff
 import torch
+import json
 from transformers import AutoModelForCausalLM, ...
```

#### Change 2 — Added `_load_model_config()` helper (lines 7–15)
Replaces the old `_DEFAULT_CONFIG` dict + verbose fallback logic.  
Now **fails fast** with a `FileNotFoundError` if the JSON is missing — no silent fallbacks.

```python
def _load_model_config():
    """Load model_config.json from the same directory. Raises error if missing."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[Config] model_config.json not found at {config_path}. Please create it.")
    with open(config_path, "r") as f:
        config = json.load(f)
    print(f"[Config] Loaded model_config.json  (context_length={config.get('context_length', 'N/A')})")
    return config
```

> **Design decision:** `_DEFAULT_CONFIG` was intentionally removed. Having a Python-side duplicate would defeat the purpose of a centralised config file. The JSON is the only place to control parameters.

#### Change 3 — Updated `__init__` to load config (lines 18–26)
- `model_path` is now optional (`model_path=None`)
- Falls back to `config["base_model_name"]` if not provided

```diff
-    def __init__(self, model_path, use_quantization=True, adapter_path=None):
-        self.model_path = model_path
+    def __init__(self, model_path=None, use_quantization=True, adapter_path=None):
+        # Load centralised config
+        self.config = _load_model_config()
+        # If no model_path provided, use the one from config
+        if model_path is None:
+            model_path = self.config["base_model_name"]
+        self.model_path = model_path
```

#### Change 4 — `generate()` now truncates to `context_length` (lines 105–108)
```diff
-        inputs = self.tokenizer(prompt, return_tensors="pt")
+        inputs = self.tokenizer(
+            prompt, return_tensors="pt", truncation=True,
+            max_length=self.config.get("context_length", 4096)
+        )
```

#### Change 5 — `train_on_buffer()` uses `context_length` from config (line 150)
```diff
-                inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048)
+                # inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048)
+                inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=self.config.get("context_length", 4096))
```

---

## How to Control the Model Now

Edit **only** `model_config.json`. No Python changes needed:

| Key | Effect |
|---|---|
| `base_model_name` | Which HuggingFace model to load (used if no `model_path` passed) |
| `context_length` | Max token length for both generation and fine-tuning |
| `num_epochs` | Available for use in training loops |
| `use_deepspeed` | Reserved for future DeepSpeed integration |

---

## Files NOT Modified
- `meta_evolver.py` — still passes `model_path` explicitly; fully backward compatible.
- All files outside `meta_evolution/` — untouched per project scope rules.

---

# Changelog: genetic_algorithm.py Fixes
**Date:** 2026-04-19  
**Session:** Bug fixes and log visibility improvements  
**Edited by:** Antigravity (AI assistant)

---

## Summary

Two changes to `genetic_algorithm.py`:
1. **Critical bugfix** — missing `numpy` import causing `NameError` on every mutation call
2. **Log improvement** — evaluation print line now shows iteration and generation context

---

## Files Changed

### `meta_evolution/genetic_algorithm.py`

#### Change 1 — Added `import numpy as np` (line 4)

**Root cause:** The LLM-generated `mutate_gene()` method called `np.random.choice()` but `numpy` was never imported. Every call to `_mutate()` raised a `NameError` which was silently caught by `fitness_function()`'s outer `except` block and returned `0.0` — compounding the META_SCORE problem.

```diff
 import random
 import pickle
 import os
+import numpy as np
```

#### Change 2 — Enhanced individual evaluation log line

**Before:**
```
  Evaluating Individual 8/10
```

**After:**
```
  Evaluating Individual 8/10   Iteration: 1 --> Generation: 2/3
```

- `Iteration` = 0-indexed loop counter (`gen`)
- `Generation` = human-readable generation number with total (`gen+1/num_generations`)

```diff
-print(f"  Evaluating Individual {i+1}/{len(self.population)}")
+print(f"  Evaluating Individual {i+1}/{len(self.population)}   Iteration: {gen} --> Generation: {gen+1}/{num_generations}")
```

---

## Files NOT Modified
- `run_fractal_evolution.py` (in this session entry)
- `meta_evolver.py`
- `llm_loader.py`
- All files outside `meta_evolution/`

---

# Changelog: META_SCORE: 0.0 Full Bugfix Chain
**Date:** 2026-04-20  
**Session:** Debugging and fixing all root causes of META_SCORE: 0.0  
**Edited by:** Antigravity (AI assistant)

---

## Summary

Four bugs in the GA + meta-evolution pipeline were identified and fixed across two files. All caused `META_SCORE: 0.0` even when real model accuracy existed on disk.

---

## Bug 1 — Accuracy Extraction From Wrong Level
**File:** `run_fractal_evolution.py`  
**Root cause:** `fitness_function()` only checked top-level keys (`full_res['accuracy']`). The eval library writes accuracy into nested fields (`full_res['hyperparameters']['accuracy']`). So `final_accuracy` was always `0.0`.

```diff
-if 'accuracy' in full_res:
-    final_accuracy = full_res['accuracy'] * 100
-elif 'best_accuracy' in full_res:
-    final_accuracy = full_res['best_accuracy'] * 100
+# Now walks: top-level → hyperparameters → training_summary → scalar fallback
+if 'accuracy' in full_res:
+    final_accuracy = float(full_res['accuracy']) * 100
+elif isinstance(full_res.get('hyperparameters'), dict):
+    hp = full_res['hyperparameters']
+    if 'accuracy' in hp and hp['accuracy']:
+        final_accuracy = float(hp['accuracy']) * 100
+        _acc_source = "hyperparameters.accuracy"
+    ...
```

Source of extraction is now logged: `>>> FITNESS SCORE: 54.69% (source: hyperparameters.accuracy, checksum: ...)`

---

## Bug 2 — Checksum Prefix Mismatch
**File:** `run_fractal_evolution.py`  
**Root cause:** `_load_existing_checksums()` used `"img-classification_cifar_FractalNet-"` (missing `-10`), so no existing stats were ever loaded. Every restart re-evaluated all 48+ models from scratch.

```diff
-prefix = "img-classification_cifar_FractalNet-"
+# prefix = "img-classification_cifar_FractalNet-"   # BUG: missing '-10'
+prefix = "img-classification_cifar-10_FractalNet-"
```

---

## Bug 3 — Float Blend Crash in Conv2d
**File:** `genetic_algorithm.py`  
**Root cause:** LLM-generated `combine_genes()` returned a raw float blend (e.g. `40.0`) for discrete genes like `base_channels`. `nn.Conv2d` expects integers → `TypeError: 'float' object cannot be interpreted as an integer`.

```diff
 blend = (parent1_value * (total_genes - gene_index) + parent2_value * gene_index) / total_genes
-return blend
+possible = self.search_space.get(gene_name, [parent1_value, parent2_value])
+return min(possible, key=lambda v: abs(v - blend))
```

Blend is now snapped to the nearest valid discrete value in the search space.

---

## Bug 4 — Duplicates Always Return 0.0 (Final Root Cause)
**File:** `run_fractal_evolution.py`  
**Root cause:** When a previously-seen model was detected as a duplicate, `fitness_function()` immediately returned `0.0`, discarding the stored accuracy from `stats/`. In late generations where the small search space means nearly all individuals are duplicates, this caused `fitness_history = [0.0, 0.0, ...]` → `META_SCORE: 0.0`.

**Fix:** Added `_lookup_stored_fitness(checksum)` helper that reads the cached stats JSON and returns the real stored accuracy.

```diff
 if model_checksum in seen_checksums:
-    print(f"  - Duplicate (checksum: {model_checksum[:8]}) -> skip")
-    return 0.0
+    return _lookup_stored_fitness(model_checksum)
```

New log output for duplicates:
```
  - Duplicate 5b1577c9: reusing stored fitness 54.69% (from accuracy)
```

---

## Complete Bug Status

| # | Bug | File | Status |
|---|---|---|---|
| 1 | Accuracy extracted from wrong nesting level | `run_fractal_evolution.py` | ✅ Fixed |
| 2 | Checksum prefix missing `-10` | `run_fractal_evolution.py` | ✅ Fixed |
| 3 | `combine_genes` float blend crashes Conv2d | `genetic_algorithm.py` | ✅ Fixed |
| 4 | Duplicates return `0.0` instead of cached accuracy | `run_fractal_evolution.py` | ✅ Fixed |

Also fixed separately (logged in `2026-04-19_meta-score-zero-bugfix.md`):
- Missing `import numpy as np` in `genetic_algorithm.py` (Bug 1 in that log)
- Silent outer exception in `__main__` now prints full traceback
