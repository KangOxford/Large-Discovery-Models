CODE=/mnt/data1/ys/PDF2DOCK/
PY=/mnt/data0/shared/ldm_tilted_case2_three_methods/.venv/bin/python
ROOT=/mnt/data0/shared/ldm_tilted_case2_three_methods/real_experiments/g12d_deepseek_sk_mainseeds_linuxvina_reasynfix_20260708_190707
VINA=/mnt/data1/dock-project/bin/vina
G12D=/mnt/data1/dock-project/PDF2Dock/activity_modeling/best_g12d_model.joblib
REASYN=/mnt/data1/dock-project/ReaSyn


cd "$CODE" && source .env && CUDA_VISIBLE_DEVICES=<gpu> "$PY" scripts/run_case2_three_methods.py \
  --method m1_stratified_direct_llm_oversample_sir --seed <seed> --budget 80 \
  --init-size 5 --batch-size 1 --init-strategy llm_cold_start --kernel sk \
  --gp-device cuda --gp-fit-itersteps 20 --ehvi-n-samples 128 --ref-point 0 5 \
  --max-candidates-per-round 256 --llm-model deepseek-v4-flash --llm-timeout 120 \
  --llm-max-retries 3 --vina-bin "$VINA" --nn-model-path "$G12D" \
  --reasyn-repo "$REASYN" --reasyn-devices <gpu> --reasyn-time-limit 1800 \
  --output-dir "$ROOT/ldm_llm_bo_sk/seed_<seed>" \
  --trajectory-dir "$ROOT/ldm_llm_bo_sk/seed_<seed>" \
  --m1-k-direct-llm 512


python TTS/run_tilted_case2_tts.py \
  --mock \
  --dry-run \
  --budget 6 \
  --m1-k-direct-llm 4 \
  --trajectory-dir logs/pdf2dock_tts_mock


python TTS/run_tilted_case2_tts.py \
  --method m1_stratified_direct_llm_oversample_sir \
  --init-strategy llm_cold_start \
  --budget 80 \
  --m1-k-direct-llm 128 \
  --max-candidates-per-round 32 \
  --kernel sk \
  --gp-device cpu \
  --llm-url http://10.200.1.12:52312/v1 \
  --llm-model-name Qwen3-Coder-30B-A3B-Instruct \
  --llm-temperature 0.7 \
  --vina-bin "${VINA}" \
  --trajectory-dir TTS/runs/case2_qwen3_coder_longer_temp07_proposer128_run2 \
  --no-allow-early-stop \
  --llm-max-retries 20 \
  --llm-retry-wait-seconds 10


CUDA_VISIBLE_DEVICES=0,1 sglang serve \
  --model-path /mnt/data0/hf_models/models/Qwen3-Coder-30B-A3B-Instruct \
  --served-model-name Qwen3-Coder-30B-A3B-Instruct \
  --trust-remote-code \
  --tp 2 \
  --ep 2 \
  --tool-call-parser qwen3_coder \
  --mem-fraction-static 0.85 \
  --host 0.0.0.0 \
  --port 52313


curl -s http://10.200.1.12:52311/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-Coder-30B-A3B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": "Write a Python function that returns the Fibonacci number for n. Keep it short."
      }
    ],
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
    "max_tokens": 256
  }'


curl http://127.0.0.1:52306/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B",
    "messages": [
      {
        "role": "user",
        "content": "Say hello in one short sentence."
      }
    ],
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
    "max_tokens": 10000,
    "chat_template_kwargs": {"enable_thinking": False}
  }'

# deepseek

python TTS/run_tilted_case2_tts.py \
  --method m1_stratified_direct_llm_oversample_sir \
  --init-strategy llm_cold_start \
  --budget 80 \
  --m1-k-direct-llm 64 \
  --max-candidates-per-round 32 \
  --kernel sk \
  --gp-device cpu \
  --llm-url https://litellm.yangtzeailab.com/v1 \
  --llm-model-name DeepSeek-V4-Flash \
  --api-key 'sk-7FO0wRzOuPrGdRdKxAf7aA' \
  --vina-bin "${VINA}" \
  --trajectory-dir TTS/runs/case2_real




# on another machine

CODE=/mnt/data0/ys/PDF2DOCK_TTS/
PY=/mnt/data0/ys/ldm_tilted_case2_three_methods/.venv/bin/python
ROOT=/mnt/data0/ys/ldm_tilted_case2_three_methods/real_experiments/g12d_deepseek_sk_mainseeds_linuxvina_reasynfix_20260708_190707
VINA=/mnt/data0/dock-project/bin/vina
G12D=/mnt/data0/dock-project/PDF2Dock/activity_modeling/best_g12d_model.joblib
REASYN=/mnt/data0/dock-project/ReaSyn

cd /mnt/data0/ys/PDF2DOCK_TTS/
source /mnt/data0/ys/ldm_tilted_case2_three_methods/.venv/bin/activate
export LLM_API_KEY="EMPTY"

python TTS/run_tilted_case2_tts.py \
  --method m1_stratified_direct_llm_oversample_sir \
  --init-strategy llm_cold_start \
  --budget 80 \
  --m1-k-direct-llm 16 \
  --max-candidates-per-round 16 \
  --kernel sk \
  --gp-device cpu \
  --llm-url http://127.0.0.1:52307/v1 \
  --llm-model-name Qwen3-Coder-30B-A3B-Instruct \
  --llm-temperature 0.7 \
  --vina-bin "${VINA}" \
  --trajectory-dir TTS/runs/case2_qwen3_coder_proposer16_bo16_run1 \
  --no-allow-early-stop \
  --llm-max-retries 20 \
  --llm-retry-wait-seconds 10


# qwen3-8B
python TTS/run_tilted_case2_tts.py \
  --method m1_stratified_direct_llm_oversample_sir \
  --init-strategy llm_cold_start \
  --budget 80 \
  --m1-k-direct-llm 64 \
  --max-candidates-per-round 32 \
  --kernel sk \
  --gp-device cpu \
  --llm-url http://127.0.0.1:52307/v1 \
  --llm-model-name Qwen3.5-9B \
  --llm-temperature 0.7 \
  --disable-thinking \
  --llm-max-tokens 4096 \
  --vina-bin "${VINA}" \
  --trajectory-dir TTS/runs/case2_qwen3_9B_proposer64_bo32_run3 \
  --no-allow-early-stop \
  --llm-max-retries 20 \
  --llm-retry-wait-seconds 10


python TTS/run_tilted_case2_tts.py \
  --method m1_stratified_direct_llm_oversample_sir \
  --init-strategy llm_cold_start \
  --budget 80 \
  --m1-k-direct-llm 64 \
  --max-candidates-per-round 64 \
  --kernel sk \
  --gp-device cpu \
  --llm-url http://127.0.0.1:52308/v1 \
  --llm-model-name Qwen3.5-9B \
  --llm-temperature 0.7 \
  --disable-thinking \
  --llm-max-tokens 4096 \
  --vina-bin /mnt/data0/dock-project/bin/vina \
  --trajectory-dir TTS/runs/case2_qwen3_9B_proposer64_bon64_run7 \
  --no-allow-early-stop \
  --llm-max-retries 10 \
  --llm-retry-wait-seconds 10


sglang serve --model-path /mnt/data0/hf_models/models/Qwen3.5-9B \
    --served-model-name Qwen3.5-9B \
    --trust-remote-code \
    --mem-fraction-static 0.85 \
    --context-length 80000 \
    --max-running-requests 80 \
    --max-queued-requests 256 \
    --chunked-prefill-size 4096 \
    --host 0.0.0.0 \
    --port 52313