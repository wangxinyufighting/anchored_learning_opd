#!/usr/bin/env bash
# OPD training for MedCalc: student = Qwen3-4B base, teacher = trained SFT model.

set -euo pipefail
set -x

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="${REPO_ROOT}/verl:${PYTHONPATH:-}"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    LOG_DIR=${LOG_DIR:-logs/medcalc_opd}
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "Log file: $LOG_FILE"
    echo "Start time: $(date)"
fi

STUDENT_MODEL_PATH=${STUDENT_MODEL_PATH:-/czsun/models/Qwen3-4B}
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-/czsun/zhi/xywang/anchored_learning/LlamaFactory/saves/Qwen3-4B_medcalc_train_1e-6}
TRAIN_DATASET=${TRAIN_DATASET:-datasets/medcalc_train.parquet}
VAL_DATASET=${VAL_DATASET:-$TRAIN_DATASET}

if [ ! -f "$TRAIN_DATASET" ]; then
    echo "Training parquet not found: $TRAIN_DATASET"
    echo "Create it first, for example:"
    echo "python scripts/convert_medcalc_json_to_verl.py --input /czsun/zhi/xywang/anchored_learning/LlamaFactory/data/medcalc_train.json --output $TRAIN_DATASET"
    exit 1
fi

ray stop --force || true
ray start --head
sleep 5

export RAY_memory_usage_threshold=${RAY_memory_usage_threshold:-0.99}
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-1}
export PYTHONUNBUFFERED=1
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-7200}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}
export HYDRA_FULL_ERROR=1

PROJECT_NAME=${PROJECT_NAME:-MedCalc_OPD}
ADV_ESTIMATOR=${ADV_ESTIMATOR:-token_reward_direct}
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
MODEL_DTYPE=${MODEL_DTYPE:-bfloat16}
LOSS_AGG_MODE=${LOSS_AGG_MODE:-token-mean}
USE_KL=${USE_KL:-False}
ENABLE_FORMAT_REWARD=${ENABLE_FORMAT_REWARD:-False}
IS_PLOT=${IS_PLOT:-False}

STUDENT_MODEL_NAME=$(basename "$STUDENT_MODEL_PATH")
TEACHER_MODEL_NAME=$(basename "$TEACHER_MODEL_PATH")
TIMESTAMP=${TIMESTAMP:-$(date +%Y-%m-%d_%H-%M-%S)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-${ADV_ESTIMATOR}_medcalc_${STUDENT_MODEL_NAME}_${TEACHER_MODEL_NAME}_${MAX_RESP_LENGTH}-T_${TEMPERATURE}-Tch_${TEACHER_TEMPERATURE}-n_${N_RESPONSES}-mbs_${MINI_BATCH_SIZE}-topk_${LOG_PROB_TOP_K}-${TOP_K_STRATEGY}-${TIMESTAMP}}
PROJECT_PATH=${PROJECT_PATH:-checkpoint}
CKPT_PATH=${CKPT_PATH:-${PROJECT_PATH}/${EXPERIMENT_NAME}}
SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-${PROJECT_PATH}/swanlab_log}
export SWANLAB_LOG_DIR

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
    algorithm.grpo_outcome_weight=1.0 \
    data.shuffle=False \
    data.train_files="$TRAIN_DATASET" \
    data.val_files="$VAL_DATASET" \
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
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature="$TEMPERATURE" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.max_num_batched_tokens="$PPO_MAX_TOKEN_LEN_PER_GPU" \
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
    +actor_rollout_ref.rollout.log_prob_top_k="$LOG_PROB_TOP_K" \
    +actor_rollout_ref.rollout.top_k_strategy="$TOP_K_STRATEGY" \
    +actor_rollout_ref.rollout.reward_weight_mode="$REWARD_WEIGHT_MODE" \
    +actor_rollout_ref.rollout.teacher_temperature="$TEACHER_TEMPERATURE" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype="$MODEL_DTYPE" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    reward_model.enable=True \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.enable_format_reward="$ENABLE_FORMAT_REWARD" \
    reward_model.model.path="$TEACHER_MODEL_PATH" \
    reward_model.model.input_tokenizer=null \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=False \
    +reward_model.model.dtype="$MODEL_DTYPE" \
    reward_model.micro_batch_size_per_gpu=8 \
    custom_reward_function.path=null \
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
