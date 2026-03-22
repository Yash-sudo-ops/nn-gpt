---
name: Import Error Diagnosis (torch ecosystem version mismatches)
description: How to diagnose and fix import errors caused by PyTorch ecosystem version mismatches in Kubernetes containers
---

# Import Error Diagnosis — PyTorch Ecosystem Version Mismatches

## Overview

When `pip install` upgrades `torch` but leaves other PyTorch ecosystem packages (e.g., `torchvision`, `flash-attn`, `torchao`) at their old versions, the C++/CUDA extensions in those packages become **ABI-incompatible** with the new torch, causing import-time crashes.

## Key Concept: PyTorch ABI Compatibility

PyTorch C++ extensions (`.so` files) are compiled against a specific torch version's C++ API. When torch is upgraded, symbols may change or disappear, making old `.so` files unusable.

**Compatibility matrix (PyTorch ↔ torchvision):**
| torch | torchvision |
|-------|-------------|
| 2.7.x | 0.22.x      |
| 2.8.x | 0.23.x      |
| 2.9.x | 0.24.x      |

## Error Pattern 1: `flash_attn` ABI Mismatch

### Symptom
```
ImportError: .../flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so: undefined symbol: _ZNK3c106SymInt6sym_neERKS0_
```

### Root Cause
`flash-attn` was compiled against torch 2.7.x but torch was upgraded to 2.9.x. The C++ symbol `c10::SymInt::sym_ne` changed between versions.

### Fix (when flash-attn is read-only / system-installed)
Add a monkey-patch at the top of the file that imports `peft` or `transformers`:
```python
import sys, types
try:
    import flash_attn_2_cuda
except ImportError:
    sys.modules['flash_attn_2_cuda'] = types.ModuleType('flash_attn_2_cuda')
```
This inserts a harmless dummy module. Transformers automatically falls back to PyTorch's built-in SDPA attention.

### Fix (when you can modify packages)
```bash
pip install flash-attn --no-build-isolation --force-reinstall  # recompile against new torch
# OR
pip uninstall flash-attn -y  # if flash attention isn't needed
```

## Error Pattern 2: `torchvision` Operator Registration Failure

### Symptom
```
RuntimeError: operator torchvision::nms does not exist
```
Triggered during `torchvision._meta_registrations` when torchvision tries to register custom ops with an incompatible torch dispatch API.

### Root Cause
`torchvision 0.22.1` (for torch 2.7.x) tries to use the old operator registration API, but torch 2.9.x changed `torch.library.register_fake`.

### Import Chain
```
your_script → peft → transformers → BloomPreTrainedModel
  → modeling_utils → loss_utils → image_utils → torchvision → 💥
```

### Fix
Upgrade `torchvision` alongside `torch` by adding it to the pip install:
```bash
pip install <your-package> torchvision --upgrade --extra-index-url https://download.pytorch.org/whl/cu126
```

## Error Pattern 3: ModuleNotFoundError from Partial Package Overwrites

### Symptom
```
ModuleNotFoundError: No module named 'ab.nn.util.db.Query'
```
Traceback shows the error in a file that was part of a locally mounted volume (e.g., `/a/mm/ab/nn/api.py`), triggered after a `pip install --upgrade`.

### Root Cause
When installing a package from a remote repository (`pip install git+...`), the installer might overwrite local skeleton files (like `api.py`) with newer versions from the remote. If the remote version has evolved to require new submodules (like `Query.py`) that are not yet committed to your local codebase or provided by other dependencies, the import will fail.

### Fix: Option B (Safe Install)
Use the `--no-deps` flag for the specific package causing the overwrite. This forces `pip` to install only that package without following its dependency tree, which prevents it from "fixing" or overwriting overlapping local modules.

```bash
# Prevents overwriting local dependencies that might be out-of-sync
pip install <remote-repo-url> --upgrade --no-deps
```

Wait, ensure you still manually upgrade critical ecosystem packages (like `torchvision`) that need to stay in sync with `torch`.

## General Diagnosis Checklist

1. **Read the last line of the traceback** — it tells you which `.so`, operator, or module failed
2. **Check the path in the traceback** — is it in `site-packages` or your local mount (`/a/mm`)?
3. **Compare `pip show <package>`** — see if the installed version is ahead of your local code.
4. **Check for "requires torch==X.Y.Z"** — pip's dependency resolver warning often points to the conflict.

## Error Pattern 4: Cascading Upgrade Trap (`getpwuid` / UID not found)

### Symptom
```
KeyError: 'getpwuid(): uid not found: 1017'
```

### Root Cause
`pip install torchvision --upgrade` pulls in the latest torch (e.g., 2.10.0), which introduces `torch._dynamo.package` that calls `getpass.getuser()` at import time. In containers running as non-root UID (e.g., 1017) without a `/etc/passwd` entry, this crashes.

### Key Lesson
**Do NOT upgrade individual torch ecosystem packages.** They pull each other as dependencies and escalate to the latest version. With `--no-deps` on your main package, the base image's torch ecosystem stays intact — no upgrades needed.

## Prevention — The Golden Rule

**Use `--no-deps` for packages that overlap with your local codebase:**
```bash
pip install <package> --upgrade --no-deps && python3 -u -m your.module
```

This ensures:
- The package code is updated (e.g., `nn-dataset` data utilities)
- torch, torchvision, flash-attn stay at the base image versions (compatible with each other)
- No cascading upgrades, no ABI mismatches, no missing modules
