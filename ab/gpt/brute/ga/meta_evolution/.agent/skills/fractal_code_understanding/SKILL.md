---
name: Fractal Code Understanding
description: Guide and tools for understanding the Fractal Network codebase and evolution pipeline.
---

# Instructions

This skill helps in navigating and understanding the Fractal Network evolution codebase.

1. **Entry Points**:
   - **Execution**: The main entry point for running the evolution is typically `run_fractal_evolution.py`.
   - **Configuration**: Hyperparameters and evolution settings are found in `Fracta_Evo.json`.

2. **Core Components**:
   - **Evolution Logic**: `FractalNet_evolvable.py` contains the logic for the fractal network that is being evolved.
   - **Meta-Evolution**: `meta_evolver.py` drives the meta-evolutionary process.
   - **Genetic Algorithm**: `ab/gpt/brute/ga/modular/genetic_algorithm.py` contains the generic GA implementation.

3. **Code Generation in `FractalNet_evolvable.py`**:
   - The file `FractalNet_evolvable.py` functions as a **code generator**.
   - **`generate_model_code_string` Function**: This function constructs a complete Python script as a string based on hyperparameters.
   - **CRITICAL: Editing Rule**:
     - The network architecture (`Net`, `FractalBlock`) is defined *inside a triple-quoted string* within this function.
     - **Outer Code vs. Inner String**: Modifications to the outer `FractalNet_evolvable.py` file (like adding imports at the top) **do not** affect the generated model code.
     - **To Change the Model**: You MUST edit the code *inside* the `textwrap.dedent(""" ... """)` block.
     - **Example**: If you want to add `import numpy` to the model, you must add it inside the string, not at the top of the `FractalNet_evolvable.py` file.
   - **Executed Code**: The evolution engine takes the string returned by this function, saves/executes it, and evaluates the model. Only code inside the string is seen by the evaluator.

4. **Key Data Structures**:
   - **Population**: Managed within the `meta_evolver.py` and saved to `ga_history_backup`.
   - **Genome**: Represents a single network architecture configuration.

5. **Usage**:
   - Use this skill when you need to explain the architecture, debug the evolution process, or modify the core genetic operators.
   - **Generated Classes**:
     - `FractalDropPath`: Handles the stochastic path dropping (regularization).
     - `FractalBlock`: Recursive block structure defining the fractal architecture.
     - `Net`: The main neural network class.
     - `train_setup` & `learn`: Methods included in the generated code to make it self-training.
