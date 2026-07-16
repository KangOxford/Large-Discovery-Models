

python TTS/scripts/example_run_antbo_tts.py \
    --config bo/config.yaml \
    --antigens-file test_5_antigens.txt \
    --seed 44 \
    --budget 200 \
    --n-init 20 \
    --parallel-budget 1200 \
    --out-dir outputs/experiments/antbo_tts/parallel_1200_run3 \
    --llm-url http://127.0.0.1:52307/v1 \
    --llm-model-name Qwen3-Coder-30B-A3B-Instruct \
    --llm-temperature 0.7 \
    --timeout-s 120 \
    --max-retries 10 \
    --fallback-random




python TTS/scripts/example_run_antbo_tts.py \
  --config bo/config.yaml \
  --antigens-file test_5_antigens.txt \
  --seed 42 \
  --budget 200 \
  --n-init 20 \
  --parallel-budget 600 \
  --out-dir outputs/experiments/antbo_tts_qwen9b_nonthinking/parallel_600_run2\
  --llm-url http://127.0.0.1:52308/v1 \
  --llm-model-name Qwen3.5-9B \
  --llm-temperature 0.7 \
  --timeout-s 300 \
  --max-retries 10 \
  --disable-thinking \
  --fallback-random

