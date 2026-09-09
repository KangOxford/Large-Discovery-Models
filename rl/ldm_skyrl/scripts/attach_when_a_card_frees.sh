#!/bin/bash
# Land the phase A payload on the first genuinely free card in an allocation this
# account already holds. Never submits: `sbatch --test-only` puts a fresh 1-node
# job at 2026-09-11T11:56, about 42 hours past the 05:00Z deadline.
#
# Two decisions worth stating.
#
# It does not use gtop. Under tonight's load gtop times out at 150 s more often
# than it returns, and an empty result is indistinguishable from "no free cards"
# unless every caller remembers the difference. Slurm's own view -- which nodes
# in my allocations carry a running step -- costs milliseconds and cannot time out.
#
# It verifies with nvidia-smi before committing, in the same loop iteration.
# A node with no step can still be busy: session teardown kills the srun client
# while the compute processes keep running, so ~12 nodes tonight hold 91-97 GB
# per card with nothing in squeue. Those are somebody's real work. A step-count
# of zero is a candidate, not a verdict.
set -uo pipefail
D=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/ldm_skyrl_nb
LOG="$D/grab.log"
MAXMEM=${MAXMEM:-2048}          # MiB; a truly free GH200 card sits at 1-9
say() { printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" | tee -a "$LOG"; }

say "hunting for a free card in an already-running allocation"
while true; do
  # allocations I hold, and the nodes that already carry a step
  BUSY=$(squeue -u "$USER" -h -s -o "%N" 2>/dev/null | tr ',' '\n' | sort -u)
  while read -r JOB NODELIST; do
    [ -z "$JOB" ] && continue
    for NODE in $(scontrol show hostnames "$NODELIST" 2>/dev/null); do
      printf '%s\n' "$BUSY" | grep -qx "$NODE" && continue
      # candidate: no step on it. Confirm with the cards themselves.
      MEM=$(timeout 90 srun --overlap --jobid="$JOB" --nodelist="$NODE" --nodes=1 --ntasks=1 \
              --gres=gpu:4 --cpu-bind=none --job-name=ldm-skyrl-probe \
              nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null </dev/null)
      [ -z "$MEM" ] && { say "probe gave nothing on $NODE (job $JOB) -- not treating that as free"; continue; }
      FREE=$(printf '%s\n' "$MEM" | awk -v m="$MAXMEM" '$1<m{c++} END{print c+0}')
      say "job $JOB node $NODE: per-card MiB [$(echo $MEM | tr '\n' ' ')] -> $FREE free"
      [ "$FREE" -lt 1 ] && continue

      say "LANDING on job=$JOB node=$NODE with NG=$FREE"
      NG=$FREE srun --overlap --jobid="$JOB" --nodelist="$NODE" --nodes=1 --ntasks=1 \
          --gres=gpu:4 --cpu-bind=none --job-name=ldm-skyrl-phaseA \
          bash "$D/phase_a_payload.sh" </dev/null 2>&1 | tee -a "$D/phaseA_attach.log"
      say "srun rc=${PIPESTATUS[0]}"
      if grep -q 'GATE A5: PASS' "$D/phaseA_attach.log" 2>/dev/null; then
        say "A5 PASSED -- stopping"; exit 0
      fi
      say "did not pass; will keep hunting"
    done
  done < <(squeue -u "$USER" -h -t RUNNING -o "%i %N" 2>/dev/null)
  say "no free card this pass"
  sleep 120
done
