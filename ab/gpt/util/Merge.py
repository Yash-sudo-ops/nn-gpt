# ab/gpt/util/MergeManager.py
"""
NN-GPT Merge System with Buffer Collection

Public API:
  rebuild_from_lineage() - Auto merge with buffer (runs every epoch)
  merge_nn_llm(epoch)     - Manual merge of specific epoch

Buffer System:
  - Collects BUFFER_SIZE candidates before merging
  - Picks best candidate by accuracy
  - Merges only if accuracy improved from last merged

Incremental Chain Merging:
  - Applies ALL previously merged adapters + new best
  - No MAX_CHAIN limit (unlimited growth)
  - Skips rejected adapters in chain

Example:
  Epoch 0: base + A0
  Epoch 1: base + A0 + A1
  Epoch 3: base + A0 + A1 + A3  (A2 rejected, skipped)
  Epoch 7: base + A0 + A1 + A3 + A4 + A5 + A7  (A6 rejected)
"""

import json
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from ab.gpt.util.Const import nngpt_dir, nngpt_upload

#config
BUFFER_SIZE = 5 # Collect 5 candidates before merging
TOLERANCE = 0.0  # Must strictly improve (no ties)

#detect base model

def infer_base_model():
    """
    Detect base model from multiple sources.

    Priority:
      1. run_config.json (base_model_name field)
      2. llm_conf referenced in run_config
      3. adapter_config.json (fallback)

    Returns:
        str: Base model identifier (e.g., "deepseek-ai/deepseek-coder-1.3b")

    Raises:
        RuntimeError: If base model cannot be determined
    """


    print("[BASE MODEL] Detection started")

    # SOURCE 1: run_config.json
    run_config = nngpt_dir / "run_config.json"

    if run_config.exists():
        print(f"[BASE MODEL] Checking {run_config}")

        try:
            with open(run_config) as f:
                cfg = json.load(f)

            base = cfg.get("base_model_name")
            if base:
                print(f"[BASE MODEL] ✓ Found in run_config.json")
                print(f"[BASE MODEL] → {base}")
                print("=" * 60 + "\n")
                return base

            print("[BASE MODEL] 'base_model_name' not in run_config.json")

            # SOURCE 2: llm_conf (referenced in run_config)
            llm_conf = cfg.get("llm_conf")
            if llm_conf:
                from ab.gpt.util.Const import conf_llm_dir
                conf_path = conf_llm_dir / llm_conf

                print(f"[BASE MODEL] Checking llm_conf: {conf_path}")

                if conf_path.exists():
                    try:
                        with open(conf_path) as f:
                            data = json.load(f)

                        base = data.get("base_model_name")
                        if base:
                            print(f"[BASE MODEL] ✓ Found in llm_conf")
                            print(f"[BASE MODEL] → {base}")

                            return base

                        print("[BASE MODEL] 'base_model_name' not in llm_conf")

                    except Exception as e:
                        print(f"[BASE MODEL] ✗ Failed to read llm_conf: {e}")
                else:
                    print(f"[BASE MODEL] ✗ llm_conf not found: {conf_path}")

        except Exception as e:
            print(f"[BASE MODEL] ✗ Failed to read run_config.json: {e}")

    else:
        print(f"[BASE MODEL] run_config.json not found: {run_config}")

    # SOURCE 3: adapter_config.json fallback
    print(f"[BASE MODEL] Falling back to adapter configs...")

    epoch_root = nngpt_dir / "llm" / "epoch"

    if not epoch_root.exists():
        print(f"[BASE MODEL] ✗ Epoch directory not found: {epoch_root}")
        raise RuntimeError("Cannot determine base model")

    latest = find_latest_epoch()
    print(f"[BASE MODEL] Latest epoch: {latest}")

    for e in range(latest, -1, -1):
        cfg_path = epoch_root / f"A{e}" / "adapter_config.json"

        if cfg_path.exists():
            print(f"[BASE MODEL] Checking {cfg_path}")

            try:
                with open(cfg_path) as f:
                    data = json.load(f)

                base = data.get("base_model_name_or_path")
                if base:
                    print(f"[BASE MODEL] ✓ Found in adapter config A{e}")
                    print(f"[BASE MODEL] → {base}")

                    return base

            except Exception as e:
                print(f"[BASE MODEL] ✗ Failed to read {cfg_path}: {e}")

    print("[BASE MODEL] ✗ FAILED: Cannot determine base model")

    raise RuntimeError("Cannot determine base model")




def find_latest_epoch():
    """
    Find highest epoch number in epoch directory.

    Returns:
        int: Latest epoch number (0 if none found)
    """
    epoch_root = nngpt_dir / "llm" / "epoch"

    if not epoch_root.exists():
        return 0

    epochs = []
    for p in epoch_root.iterdir():
        if p.is_dir() and p.name.startswith("A"):
            try:
                epochs.append(int(p.name[1:]))
            except ValueError:
                continue

    return max(epochs) if epochs else 0


#find adapter

def find_adapter(epoch: int):
    """
    Find newest valid adapter ≤ epoch.

    Falls back to earlier epochs if requested epoch missing.

    Args:
        epoch: Target epoch number

    Returns:
        tuple: (adapter_dir, actual_epoch) or (None, None) if none found
    """

    print(f"\n[ADAPTER] Searching for epoch {epoch}...")

    for e in range(epoch, -1, -1):
        path = nngpt_dir / "llm" / "epoch" / f"A{e}"

        if not path.exists():
            print(f"[ADAPTER] A{e} directory not found")
            continue

        # Look for weight files
        candidates = list(path.rglob("adapter_model.safetensors"))

        if not candidates:
            print(f"[ADAPTER] A{e} has no weights")
            continue

        adapter_dir = candidates[0].parent

        if e != epoch:
            print(f"[ADAPTER]  Requested A{epoch}, using A{e} (fallback)")
        else:
            print(f"[ADAPTER] ✓ Found A{e}")

        print(f"[ADAPTER] Path: {adapter_dir}\n")
        return adapter_dir, e

    print(f"[ADAPTER] ✗ No adapters found ≤ epoch {epoch}\n")
    return None, None


#load already exisiting adapters

def load_lineage():
    """
    Load merge history from lineage.json.

    Returns:
        list: Lineage entries (empty list if file doesn't exist)
    """
    path = nngpt_dir / "lineage.json"

    if not path.exists():
        return []

    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"[LINEAGE] ✗ Failed to load: {e}")
        return []


def save_lineage(lineage):
    """
    Save lineage with deduplication.

    Removes duplicates by (epoch, used_epoch) key.

    Args:
        lineage: List of lineage entries
    """
    path = nngpt_dir / "lineage.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    # Deduplicate by (epoch, used_epoch) - CHANGED from (cycle, used_epoch)
    unique = {}
    for x in lineage:
        key = (x["epoch"], x["used_epoch"])  # ← Changed from x["cycle"]
        unique[key] = x

    with open(path, "w") as f:
        json.dump(list(unique.values()), f, indent=2)

#merge adapters

def merge_multiple_adapters(base_model, adapter_paths, output_path):
    """
    Apply multiple LoRA adapters sequentially to base model.

    Args:
        base_model: Base model identifier or path
        adapter_paths: List of adapter directories to apply in order
        output_path: Where to save merged model
    """


    print("[MERGE] Starting multi-adapter merge")

    print(f"[MERGE] Base model: {base_model}")
    print(f"[MERGE] Adapters to apply: {len(adapter_paths)}")

    for i, p in enumerate(adapter_paths):
        print(f"[MERGE]   {i + 1}. {p}")

    print(f"[MERGE] Output: {output_path}")


    # Load base model
    print("[MERGE] Loading base model...")

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        trust_remote_code=True
    )

    print("[MERGE] ✓ Base model loaded\n")

    # Apply each adapter sequentially
    for i, adapter_path in enumerate(adapter_paths):
        print(f"[MERGE] Applying adapter {i + 1}/{len(adapter_paths)}: {adapter_path}")

        model = PeftModel.from_pretrained(model, str(adapter_path))
        model = model.merge_and_unload()

        print(f"[MERGE] ✓ Adapter {i + 1} merged\n")

    # Save merged model
    print(f"[MERGE] Saving merged model to {output_path}...")

    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)

    print("[MERGE] ✓ Model saved\n")

    # Save tokenizer
    print("[MERGE] Saving tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    print("[MERGE] ✓ Tokenizer saved\n")


    print("[MERGE] ✓✓✓ MERGE COMPLETE ✓✓✓")
    print(f"[MERGE] Output: {output_path}")



def rebuild_from_lineage():
    """Process ALL unprocessed epochs from epoch_tracker.json"""

    # Load epoch tracker
    epoch_tracker_file = nngpt_dir / "epoch_tracker.json"
    if not epoch_tracker_file.exists():
        print(f"[MERGE] ✗ epoch_tracker.json not found")
        return

    with open(epoch_tracker_file) as f:
        tracker_list = json.load(f)

    if not tracker_list:
        print("[MERGE] ✗ epoch_tracker.json is empty")
        return

    # Load lineage
    lineage = load_lineage()

    # Find which epochs have already been processed
    processed_epochs = {x["epoch"] for x in lineage}

    # Process each unprocessed epoch
    new_candidates = []

    for tracker_entry in tracker_list:
        epoch = tracker_entry.get("epoch")

        if epoch is None:
            continue

        if epoch in processed_epochs:
            print(f"[MERGE] Epoch {epoch} already processed, skipping")
            continue

        # Process this epoch
        print(f"\n[MERGE] Processing epoch {epoch}...")

        acc = tracker_entry.get("accuracy")

        if acc is None or acc == 0.0:
            print(f"[MERGE] Epoch {epoch} has no real accuracy, skipping")
            continue  # Skip this epoch, don't add to buffer


        #if acc is None or acc == 0.0:
            #acc = min(0.40 + epoch * 0.03, 0.99)
            #print(f"[MERGE] Using fake accuracy: {acc:.4f}")

        acc = float(acc)

        # Find adapter
        adapter_path, used_epoch = find_adapter(epoch)
        if adapter_path is None:
            print(f"[MERGE] ✗ No adapter found for epoch {epoch}, skipping")
            continue

        base_model = infer_base_model()

        # Add to lineage as candidate
        new_entry = {
            "epoch": epoch,
            "used_epoch": used_epoch,
            "accuracy": acc,
            "merged": False,
            "candidate": True,
            "base_model": base_model
        }

        lineage.append(new_entry)
        new_candidates.append(new_entry)
        processed_epochs.add(epoch)

        print(f"[BUFFER] ✓ Added epoch {epoch} (acc={acc:.4f}) to buffer")

    if not new_candidates:
        print("\n[MERGE] No new epochs to process")
        return

    # Save lineage with new candidates
    save_lineage(lineage)

    # Check buffer status
    candidates = [x for x in lineage if x.get("candidate") and not x.get("merged")]


    print(f"[BUFFER] Current buffer: {len(candidates)}/{BUFFER_SIZE}")

    for i, c in enumerate(candidates):
        print(f"  {i + 1}. Epoch {c['epoch']}, Acc={c['accuracy']:.4f}")

    if len(candidates) < BUFFER_SIZE:
        print(f"\n[BUFFER] Need {BUFFER_SIZE - len(candidates)} more candidates")
        print(f"[BUFFER] Waiting... (no merge yet)\n")
        return

    print(f"\n[BUFFER] ✓ Buffer full ({BUFFER_SIZE} candidates)")

    # Pick best candidate
    best = max(candidates, key=lambda x: x["accuracy"])

    print(f"\n[SELECTION] Best candidate:")
    print(f"  Epoch: {best['epoch']}")
    print(f"  Accuracy: {best['accuracy']:.4f}")

    # Check if it improves
    merged = [x for x in lineage if x.get("merged")]

    if merged:
        prev = merged[-1]["accuracy"]
        print(f"\n[COMPARISON] Accuracy check:")
        print(f"  Last merged: {prev:.4f}")
        print(f"  Current best: {best['accuracy']:.4f}")
        print(f"  Delta: {best['accuracy'] - prev:+.4f}")

        if best["accuracy"] <= prev + TOLERANCE:
            print(f"\n[MERGE] ✗ REJECTED - Not improving")

            for x in lineage:
                if x.get("candidate") and not x.get("merged"):
                    x["rejected"] = True
                    x["candidate"] = False

            save_lineage(lineage)
            print(f"[MERGE] All {len(candidates)} candidates marked as rejected\n")
            return

        print(f"\n[MERGE] ✓ APPROVED - Accuracy improved by {best['accuracy'] - prev:.4f}")
    else:
        print(f"\n[MERGE] ✓ APPROVED - First merge (baseline)")

    # Build chain
    print(f"\n[CHAIN] Building adapter chain...")
    chain = []

    for x in merged:
        p, _ = find_adapter(x["used_epoch"])
        if p:
            chain.append(p)
            print(f"[CHAIN] + A{x['used_epoch']} (merged at epoch {x['epoch']})")

    best_adapter, _ = find_adapter(best["used_epoch"])
    chain.append(best_adapter)
    print(f"[CHAIN] + A{best['used_epoch']} (current best)")

    print(f"\n[CHAIN] Total adapters: {len(chain)}")

    # Merge
    base_model = infer_base_model()
    output_path = nngpt_upload / Path(base_model).name
    merge_multiple_adapters(base_model, chain, output_path)

    # Update run_config
    run_config = nngpt_dir / "run_config.json"
    cfg = {}
    if run_config.exists():
        with open(run_config) as f:
            cfg = json.load(f)

    cfg["base_model_name"] = str(output_path)

    with open(run_config, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"[CONFIG] Updated run_config.json → {output_path}\n")

    # Update lineage
    for x in lineage:
        if x["epoch"] == best["epoch"]:
            x["merged"] = True
            x["candidate"] = False
            x["chain_length"] = len(chain)

    for x in lineage:
        if x.get("candidate") and not x.get("merged"):
            x["rejected"] = True
            x["candidate"] = False

    save_lineage(lineage)


    print(f"[MERGE] ✓✓✓ AUTO MERGE COMPLETE ✓✓✓")




def merge_nn_llm(epoch: int):
    """
    Manual merge of specific epoch (bypasses buffer and accuracy checks).

    Args:
        epoch: Epoch number to merge
    """

    print(f"[MANUAL] Manual merge requested for epoch {epoch}")


    adapter_path, used_epoch = find_adapter(epoch)

    if adapter_path is None:
        print(f"[MANUAL] No adapter found for epoch {epoch}")
        return

    base_model = infer_base_model()
    output_path = nngpt_upload / Path(base_model).name

    merge_multiple_adapters(base_model, [adapter_path], output_path)

    print(f"[MANUAL] ✓ Manual merge complete for A{used_epoch}\n")


if __name__ == "__main__":
    rebuild_from_lineage()
    #merge_nn_llm(3)