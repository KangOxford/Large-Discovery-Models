
python TTS/run_a_search_nanogpt.py \
    --train-file TTS/real_train.py \
    --method best_of_n \
    --breadth 3 \
    --depth 4 \
    --max-tokens 10240 \
    --temperature 0.4 \
    --eval-each-num-steps 4 \
    --num-edits-per-step 1 \
    --generator tool_call \
    --out-dir TTS/runs/bon \
    --eval-command "uv run {train_path}"

python TTS/run_a_search_nanogpt.py \
    --train-file TTS/real_train.py \
    --method beam_search \
    --breadth 2 \
    --depth 3 \
    --beam-width 2 \
    --max-tokens 10240 \
    --temperature 0.7 \
    --logprobs \
    --eval-each-num-steps 2 \
    --num-edits-per-step 1 \
    --generator tool_call \
    --out-dir TTS/runs/beam \
    --eval-command "uv run {train_path}"

python TTS/run_a_search_nanogpt.py \
    --train-file TTS/real_train.py \
    --method tree_search \
    --breadth 2 \
    --depth 3 \
    --max-tokens 10240 \
    --temperature 0.7 \
    --logprobs \
    --eval-each-num-steps 2 \
    --num-edits-per-step 1 \
    --generator tool_call \
    --out-dir TTS/runs/tree \
    --eval-command "uv run {train_path}"


# 

python TTS/run_model_based_search.py \
    --train-file TTS/real_train.py \
    --method best_of_n \
    --breadth 2 \
    --depth 4 \
    --max-tokens 10240 \
    --temperature 0.4 \
    --num-edits-per-step 1 \
    --generator tool_call \
    --out-dir TTS/runs/bon \
    --eval-command "uv run {train_path}" \
    --iterations 8 \
    --buffer TTS/model_based_buffer.jsonl


python TTS/run_model_based_search.py \
  --train-file TTS/real_train.py \
  --generator operation_tool \
  --operation-schema TTS/operation_schema_real_train.json \
  --method best_of_n \
  --breadth 4 \
  --depth 8 \
  --iterations 20 \
  --max-operations-per-step 2 \
  --warmup 20 \
  --warmup-include-root \
  --warmup-strategy random_operation \
  --buffer TTS/model_based_buffer_0629_trial2.jsonl \
  --eval-command "uv run {train_path}"


python TTS/run_model_based_search.py \
  --train-file TTS/runs/model_based_best_of_n_operation_tool_real_train_b4_d8_i20_20260629_102735/best_train.py \
  --buffer TTS/runs/model_based_best_of_n_operation_tool_real_train_b4_d8_i20_20260629_102735/model_based_buffer.jsonl \
  --warmup 0 \
  --method best_of_n \
  --generator operation_tool \
  --operation-schema TTS/operation_schema_real_train.json \
  --breadth 4 \
  --depth 8 \
  --iterations 20 \
  --seed-policy best




# visualize GP

python TTS/plot_gp_model.py \
    TTS/runs/model_based/model_based_best_of_n_operation_tool_real_train_b4_d8_i5_20260628_232014 \
    --slice-params DEPTH,WIDTH,MATRIX_LR,WARMDOWN_RATIO

# visualize trees

python TTS/visualize_search.py \
    TTS/runs/bon/best_of_n_tool_call_real_train_b3_d4_e1_20260623_173607/ \
    --pretty --pretty-trim 90 --pretty-action-width 42 --pretty-border \
    --pretty-orientation vertical

python TTS/visualize_search.py \
    TTS/runs/beam//beam_search_tool_call_real_train_b2_d3_e1_20260624_142303/ \
    --pretty --pretty-action-width 32 --pretty-border \
    --pretty-orientation horizontal

# visualize model-based search
python TTS/visualize_search.py \
    TTS/runs/model_based/model_based_best_of_n_operation_tool_real_train_b2_d2_i4_20260627_141503/ \
    --iteration 3 \
    --pretty  --pretty-action-width 32 --pretty-border \
    --pretty-orientation horizontal --show-errors



python TTS/plot_ablation.py \
  TTS/ablation_runs/expanded_ldm_bon_N4H4_03 \
  TTS/ablation_runs/iters20_seed123_02 \
  TTS/ablation_runs/expanded_ldm_bon_N8H8 \
  TTS/ablation_runs/expanded_ldm_bon_N8H8_EI \
  TTS/ablation_runs/expanded_ldm_bon_N8H8_mean \
  --trial-label "LDM-TTS BoN N4H4 (Expanded features)" \
  --trial-label "LDM-TTS BoN N4H4 (Fixed features)" \
  --trial-label "LDM-TTS BoN N8H8 (Expanded features)" \
  --trial-label "LDM-TTS BoN N8H8 Acquisition: EI (Expanded features)" \
  --trial-label "LDM-TTS BoN N8H8 Acquisition: Mean (Expanded features)" \
  --xlim 0 100


python TTS/plot_ablation.py \
    TTS/ablation_runs/expanded_ldm_bon_N4H4_03 \
    TTS/ablation_runs/iters20_seed123_02 \
    TTS/ablation_runs/expanded_ldm_bon_N8H8 \
    TTS/ablation_runs/expanded_ldm_bon_N8H8_EI \
    TTS/ablation_runs/expanded_ldm_bon_N8H8_mean \
    --trial-label "LDM-TTS, BoN-N4H4, UCB (expanding features)" \
    --trial-label "LDM-TTS, BoN-N4H4, UCB (fixed features)" \
    --trial-label "LDM-TTS, BoN-N8H8, UCB (expanding features)" \
    --trial-label "LDM-TTS, BoN-N8H8, Acquisition: EI (expanding features)" \
    --trial-label "LDM-TTS, BoN-N8H8, Acquisition: Mean (expanding features)" \
    --xlim 0 100 \
    --ylim 0.96 1.08 \
    --zoom-inset 40 100