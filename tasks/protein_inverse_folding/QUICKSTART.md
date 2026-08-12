# Clean-Room Quick Start

Run these commands from the repository root.

1. Validate registration and create the task environment:

   ```bash
   python scripts/validate_tasks.py --task protein_inverse_folding
   uv sync --locked --project tasks/protein_inverse_folding --group dev
   ```

2. Verify the dependency-free mock plan and execute it:

   ```bash
   python scripts/check_task_dependencies.py \
     config/protein_inverse_folding/mock.yaml --no-optional
   python scripts/run_ldm_tts.py \
     config/protein_inverse_folding/mock.yaml --dry-run
   python scripts/run_ldm_tts.py config/protein_inverse_folding/mock.yaml
   ```

3. Run task-local tests:

   ```bash
   uv run --locked --project tasks/protein_inverse_folding \
     python -m pytest tasks/protein_inverse_folding/tests
   ```

4. On a CUDA host, install or select a CUDA-enabled PyTorch environment and run
   the dataset-free contract smoke:

   ```bash
   python scripts/check_task_dependencies.py \
     config/protein_inverse_folding/real_gpu_smoke.yaml
   python scripts/run_ldm_tts.py \
     config/protein_inverse_folding/real_gpu_smoke.yaml
   ```

5. For the full benchmark, follow the staged setup and path overrides in
   `README.md`. Do not begin the multi-hour CATH/TS50 run until the contract and
   GPU smoke gates pass.

