# Resource-Efficient Iterative LLM-Based NAS with Feedback Memory

Resource-Efficient Iterative prompt-improvement pipeline: uses an LLM to generate model code, then trains and evaluates it. Supports CIFAR-10, CIFAR-100, and ImageNette. You can turn off the improver or reference via flags for ablation runs.

## Environment and dependencies

- **Python**: 3.10 recommended
- **Dependencies**: `pip install -r requirements.txt`


If you use a remote API, set the env var (e.g. `export SiliconCloud_Key="..."`) and do not commit the key.

## How to run

### Option 1: Call the pipeline directly

```bash
python pipeline.py --model <HuggingFace_model_name> --dataset <cifar10|cifar100|imagenette> \
  --max-iterations <N> --target-accuracy <0~1> --output-dir <output_directory>
```

- Add `--remote` when using a remote API
- Common args: `--epochs`, `--batch-size`, `--history-size`
- Ablation: `--no-improver`, `--no-reference`, `--no-history`

### Option 2: Use the scripts

- **Full pipeline**: run `./run.sh`. 

## Output

Under `--output-dir` you get:

- **summary.json**: total iterations, best accuracy, whether target was reached, `results_history`
- **results.log**: per-iteration accuracy, success, and errors
- **generated_models/**: generated model code per iteration (e.g. `model_iter_*.py`)
