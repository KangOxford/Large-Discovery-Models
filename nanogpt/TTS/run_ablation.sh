set -x


python TTS/run_model_based_search.py \
  --train-file TTS/real_train.py \
  --generator operation_tool \
  --operation-schema TTS/operation_schema_real_train.json \
  --method best_of_n \
  --breadth 1 \
  --depth 1 \
  --iterations 0 \
  --warmup 50 \
  --warmup-include-root \
  --warmup-seed 123 \
  --max-operations-per-step 2 \
  --seed-policy original \
  --out-dir TTS/runs/gp_warmup \
  --run-name real_train_gp_warmup_seed123_n20 \
  --buffer TTS/ablation_buffer/gp_warmup.frozen.jsonl \
  --eval-command "uv run {train_path}"


python TTS/run_model_based_search.py \
  --train-file TTS/real_train.py \
  --generator operation_tool \
  --operation-schema TTS/operation_schema_real_train.json \
  --method best_of_n \
  --breadth 8 \
  --depth 8 \
  --iterations 100 \
  --warmup 0 \
  --seed-policy best \
  --buffer TTS/ablation_buffer/search_exp1/gp_warmup.jsonl \
  --out-dir TTS/runs/ablation_runs \
  --run-name ldm_bon_N8H8 \
  --eval-command "uv run {train_path}" \
  --llm-url "http://127.0.0.1:52307/v1" \
  --llm-model-name "Qwen3-Coder-30B-A3B-Instruct"



python TTS/run_model_based_search.py \
  --train-file TTS/ablation_runs/baseline_operation_tool_real_train_b1_i20_e1_20260629_185032/best_train.py \
  --generator operation_tool \
  --operation-schema TTS/operation_schema_real_train.json \
  --method best_of_n \
  --breadth 4 \
  --depth 4 \
  --iterations 100 \
  --warmup 0 \
  --seed-policy best \
  --buffer TTS/runs/ablation_search_exps/brief_acquisition/model_based_buffer.jsonl \
  --out-dir TTS/runs/ablation_search_exps \
  --run-name brief_acquisition_start_from_good \
  --eval-command "uv run {train_path}" \
  --llm-url "http://127.0.0.1:52307/v1" \
  --llm-model-name "Qwen3-Coder-30B-A3B-Instruct" \
  --acquisition-feedback brief



## Baseline
python TTS/run_baseline_search.py \
  --train-file TTS/real_train.py \
  --iterations 20 \
  --max-tokens 4096 \
  --temperature 0.7 \
  --generator operation_tool \
  --operation-schema TTS/operation_schema_real_train.json \
  --max-operations-per-step 2 \
  --seed-policy best \
  --out-dir TTS/runs/baseline_operation \
  --eval-command "uv run {train_path}" \
  --llm-url "http://127.0.0.1:52307/v1" \
  --llm-model-name "Qwen3-Coder-30B-A3B-Instruct"


# finetune

python TTS/run_expanded_search.py \
    --train-file TTS/real_train.py \
    --generator tool_call \
    --max-operations-per-step 2 \
    --method best_of_n \
    --breadth 1 \
    --depth 1 \
    --iterations 5 \
    --warmup 0 \
    --seed-policy best \
    --buffer TTS/ablation_buffer/gp_warmup.frozen.jsonl \
    --out-dir TTS/runs/ablation_runs \
    --run-name expanded_ldm_bon_N8H8_finetuned  \
    --eval-command "uv run {train_path}" \
    --llm-url "http://135.84.176.142:20200/v1" \
    --llm-model-name "checkpoint-30"



python TTS/run_expanded_search.py \
    --train-file TTS/real_train.py \
    --generator operation_tool_plain_text \
    --operation-schema TTS/operation_schema_real_train.json \
    --max-operations-per-step 2 \
    --method best_of_n \
    --breadth 4 \
    --depth 4 \
    --iterations 100 \
    --warmup 0 \
    --seed-policy best \
    --buffer TTS/ablation_buffer/gp_warmup.frozen.jsonl \
    --out-dir TTS/runs/ablation_runs \
    --run-name expanded_ldm_bon_N8H8_finetuned \
    --eval-command "uv run {train_path}" \
    --llm-url "http://135.84.176.142:20200/v1" \
    --llm-model-name "checkpoint-30"