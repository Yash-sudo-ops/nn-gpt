# GA Meta-Evolution Update: Best Stats Tracking
**Date:** 2026-04-20
**Task:** Update `run_fractal_evolution.py` to preserve the best model's performance data.

## Changes
- **Imported `shutil` and `datetime`:** Added to support file operations and timestamping.
- **Added `BEST_STATS_DIR`:** New constant pointing to `best_fractal_stats/`.
- **Updated Best Model Saving Logic:**
    - Whenever the best model is identified, its unique checksum is recalculated.
    - The corresponding folder in `stats/` is located.
    - The entire folder is copied to `best_fractal_stats/`, ensuring that previous "best" stats are removed first to avoid clutter.
- **Added Metadata JSON:** Created `best_fractal_info.json` which stores:
    - Checksum
    - Fitness
    - Chromosome configuration
    - Original and destination paths
    - ISO timestamp
- **Console Feedback:** Added detailed `[Best]` logs to inform the user about the status of stats copying and metadata generation.

## Storage Locations
- **Best Model Code:** `best_fractal_model.py`
- **Best Model Stats:** `best_fractal_stats/img-classification_cifar-10_FractalNet-<checksum>/`
- **Best Model Info:** `best_fractal_info.json`

## Edge Cases Handled
- **Missing Stats Folder:** If for some reason the stats folder is missing from `stats/` (e.g., deleted manually), a warning is printed instead of crashing.
- **Existing Best Stats:** The `best_fractal_stats/` folder is cleaned out before each new "best" save to ensure it only contains the latest winner.
