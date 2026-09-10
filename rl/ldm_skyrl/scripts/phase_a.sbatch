#!/bin/bash
#SBATCH --job-name=ldm-skyrl-phaseA
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --partition=workq
#SBATCH --output=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/ldm_skyrl_nb/phaseA_%j.log
#
# Phase A items A3, A4 and A5 for the SkyRL port.
#
# The payload runs under srun with a real --job-name rather than in the batch
# step: the account's budget guard reads step names to decide whether an
# allocation is idle, and a step called "bash" gets an allocation judged
# IDLE-HELD and cancelled. < /dev/null because srun otherwise drains the
# caller's stdin.
set -uo pipefail
D=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/ldm_skyrl_nb
echo "[phaseA] job=$SLURM_JOB_ID node=$SLURM_NODELIST utc=$(date -u +%FT%TZ) walltime=$SLURM_TIMELIMIT"
srun --job-name=ldm-skyrl-phaseA --ntasks=1 --nodes=1 --gres=gpu:4 --cpu-bind=none \
     bash "$D/phase_a_payload.sh" < /dev/null
echo "[phaseA] srun rc=$?"
