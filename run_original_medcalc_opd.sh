#!/usr/bin/env bash
# Run the original OPD path from this codebase with the MedCalc teacher/student/data paths.
#
# This keeps the original OPD algorithmic settings:
#   ADV_ESTIMATOR=token_reward_direct
#   LOG_PROB_TOP_K=16
#   TOP_K_STRATEGY=only_stu
#   REWARD_WEIGHT_MODE=student_p
#
# The only dataset-specific convenience is that a JSON/JSONL source can be
# converted to the verl parquet format before training.

set -euo pipefail
set -x

RAW_TRAIN_JSON=${RAW_TRAIN_JSON:-/czsun/zhi/xywang/anchored_learning/LlamaFactory/data/medcalc_train.json}
TRAIN_DATASET=${TRAIN_DATASET:-datasets/medcalc_train.parquet}
STUDENT_MODEL_PATH=${STUDENT_MODEL_PATH:-/czsun/models/Qwen3-4B}
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-/czsun/zhi/xywang/anchored_learning/LlamaFactory/saves/Qwen3-4B_medcalc_train_1e-6}

if [ "${CONVERT_DATA:-auto}" = "always" ] || { [ "${CONVERT_DATA:-auto}" = "auto" ] && [ ! -f "$TRAIN_DATASET" ]; }; then
    python3 scripts/convert_medcalc_json_to_verl.py \
        --input "$RAW_TRAIN_JSON" \
        --output "$TRAIN_DATASET"
fi

if [ ! -f "$TRAIN_DATASET" ]; then
    echo "Training parquet not found: $TRAIN_DATASET"
    echo "Set TRAIN_DATASET to a verl-format parquet file, or set RAW_TRAIN_JSON to your source JSON."
    exit 1
fi

if [ -z "${SLURM_JOB_ID:-}" ]; then
    LOG_DIR=${LOG_DIR:-logs/original_medcalc_opd}
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "Log file: $LOG_FILE"
    echo "Start time: $(date)"
fi

ray stop --force || true
ray start --head
sleep 5

export RAY_memory_usage_threshold=${RAY_memory_usage_threshold:-0.99}
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-1}
export PYTHONUNBUFFERED=1
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-7200}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-INFO}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}
export HYDRA_FULL_ERROR=1

PROJECT_NAME=${PROJECT_NAME:-Original_OPD_MedCalc}
ADV_ESTIMATOR=${ADV_ESTIMATOR:-token_reward_direct}
GRPO_OUTCOME_WEIGHT=${GRPO_OUTCOME_WEIGHT:-1.0}

# MedCalc patient notes can be longer than math prompts, so this default is
# larger than the paper script's 1024. Set MAX_PROMPT_LENGTH=1024 for a stricter
# reproduction of the original launch script.
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-4096}
MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-4096}
MAX_MODEL_LEN=$(( MAX_RESP_LENGTH + MAX_PROMPT_LENGTH > MAX_VAL_RESP_LENGTH + MAX_PROMPT_LENGTH ? MAX_RESP_LENGTH + MAX_PROMPT_LENGTH : MAX_VAL_RESP_LENGTH + MAX_PROMPT_LENGTH ))

MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-32}
PARALLEL_SIZE=${PARALLEL_SIZE:-1}
N_RESPONSES=${N_RESPONSES:-4}
TEMPERATURE=${TEMPERATURE:-1.0}
TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE:-1.0}
REPETITION_PENALTY=${REPETITION_PENALTY:-1.0}
LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-16}
TOP_K_STRATEGY=${TOP_K_STRATEGY:-only_stu}
REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-student_p}
USE_KL=${USE_KL:-False}
ENABLE_FORMAT_REWARD=${ENABLE_FORMAT_REWARD:-False}
MODEL_DTYPE=${MODEL_DTYPE:-bfloat16}
IS_PLOT=${IS_PLOT:-False}
LOSS_AGG_MODE=${LOSS_AGG_MODE:-token-mean}

STUDENT_MODEL_NAME=$(basename "$STUDENT_MODEL_PATH")
TEACHER_MODEL_NAME=$(basename "$TEACHER_MODEL_PATH")
TIMESTAMP=${TIMESTAMP:-$(date +%Y-%m-%d_%H-%M-%S)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-${ADV_ESTIMATOR}_medcalc_${STUDENT_MODEL_NAME}_${TEACHER_MODEL_NAME}_${MAX_RESP_LENGTH}-T_${TEMPERATURE}-Tch_${TEACHER_TEMPERATURE}-n_${N_RESPONSES}-mbs_${MINI_BATCH_SIZE}-topk_${LOG_PROB_TOP_K}-topk_strategy_${TOP_K_STRATEGY}-rw_${REWARD_WEIGHT_MODE}-${TIMESTAMP}}
PROJECT_PATH=${PROJECT_PATH:-checkpoint}
CKPT_PATH=${CKPT_PATH:-${PROJECT_PATH}/${EXPERIMENT_NAME}}
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-${PROJECT_PATH}/swanlab_log}

KL_ARGS=()
if [ "$USE_KL" = "True" ]; then
    KL_ARGS=(
        actor_rollout_ref.actor.use_kl_loss=True
        actor_rollout_ref.actor.kl_loss_coef=0.005
        actor_rollout_ref.actor.kl_loss_type=low_var_kl
    )
else
    KL_ARGS=(actor_rollout_ref.actor.use_kl_loss=False)
fi

PPO_MAX_TOKEN_LEN_PER_GPU=$(( ((MAX_PROMPT_LENGTH + MAX_RESP_LENGTH) > 32768) ? (MAX_PROMPT_LENGTH + MAX_RESP_LENGTH) : 32768 ))
echo "PPO_MAX_TOKEN_LEN_PER_GPU: $PPO_MAX_TOKEN_LEN_PER_GPU"
echo "Experiment: $EXPERIMENT_NAME"
echo "Checkpoint dir: $CKPT_PATH"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator="$ADV_ESTIMATOR" \
    algorithm.grpo_outcome_weight="$GRPO_OUTCOME_WEIGHT" \
    data.shuffle=False \
    data.train_files="$TRAIN_DATASET" \
    data.val_files="$TRAIN_DATASET" \
    data.train_batch_size=$((MINI_BATCH_SIZE * PARALLEL_SIZE)) \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESP_LENGTH" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=True \
    actor_rollout_ref.model.path="$STUDENT_MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="$PARALLEL_SIZE" \
    "${KL_ARGS[@]}" \
    actor_rollout_ref.actor.loss_agg_mode="$LOSS_AGG_MODE" \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype="$MODEL_DTYPE" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$PPO_MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype="$MODEL_DTYPE" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature="$TEMPERATURE" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    +actor_rollout_ref.rollout.log_prob_top_k="$LOG_PROB_TOP_K" \
    +actor_rollout_ref.rollout.top_k_strategy="$TOP_K_STRATEGY" \
    +actor_rollout_ref.rollout.reward_weight_mode="$REWARD_WEIGHT_MODE" \
    +actor_rollout_ref.rollout.teacher_temperature="$TEACHER_TEMPERATURE" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$PARALLEL_SIZE" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.n="$N_RESPONSES" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    +actor_rollout_ref.rollout.val_kwargs.max_tokens="$MAX_VAL_RESP_LENGTH" \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.repetition_penalty="$REPETITION_PENALTY" \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    reward_model.enable=True \
    reward_model.reward_manager="${REWARD_MANAGER:-naive}" \
    +reward_model.reward_kwargs.enable_format_reward="$ENABLE_FORMAT_REWARD" \
    reward_model.model.path="$TEACHER_MODEL_PATH" \
    reward_model.model.input_tokenizer=null \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=False \
    +reward_model.model.dtype="$MODEL_DTYPE" \
    reward_model.micro_batch_size_per_gpu="${REWARD_MICRO_BATCH_SIZE_PER_GPU:-8}" \
    custom_reward_function.path="verl/verl/utils/reward_score/ttrl_math/__init__.py" \
    custom_reward_function.name=reward_func \
    trainer.val_before_train=False \
    trainer.log_val_generations=2 \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.validation_data_dir="validation_log/$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE:-8}" \
    trainer.nnodes="${N_NODES:-1}" \
    trainer.save_freq="${SAVE_FREQ:-20}" \
    trainer.test_freq="${TEST_FREQ:--1}" \
    trainer.total_epochs="${TOTAL_EPOCHS:-1}" \
    trainer.default_local_dir="$CKPT_PATH" \
    trainer.is_plot="$IS_PLOT" \
    "$@"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "End time: $(date)"
fi
