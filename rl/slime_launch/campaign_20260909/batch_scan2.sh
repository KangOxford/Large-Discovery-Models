set -uo pipefail
W=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/_nanogpt_work
WT=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/_wt_nanogpt
export HF_HOME=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/_nanogpt_hf
export AUTORESEARCH_CACHE_DIR=$W/autoresearch_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
ulimit -c 0
P=/home/u6gb/kangli.u6gb/envs/ldm-nanogpt/bin/python
CARD=${CARD:?}; DEADLINE=${DEADLINE_EPOCH:?}
# no LLM server => no 2-minute startup window for a neighbour to land in.
# One eval per card, sequential, until the deadline. Levels cycle so every level
# gets equal n even if the node is lost partway.
LEVELS="131072 262144 524288 1048576 2097152"
n=0
while [ "$(date +%s)" -lt "$((DEADLINE - 500))" ]; do
  for TB in $LEVELS; do
    [ "$(date +%s)" -lt "$((DEADLINE - 500))" ] || break
    free=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $CARD)
    d=/tmp/bs2_${TB}_c${CARD}_$(date -u +%H%M%SZ); mkdir -p $d
    sed -e 's/^ASPECT_RATIO *=.*/ASPECT_RATIO = 64/' -e 's/^EMBEDDING_LR *=.*/EMBEDDING_LR = 0.6/' \
        -e 's/^HEAD_DIM *=.*/HEAD_DIM = 128/' -e "s/^WINDOW_PATTERN *=.*/WINDOW_PATTERN = 'SSSL'/" \
        -e "s/^TOTAL_BATCH_SIZE *=.*/TOTAL_BATCH_SIZE = $TB/" \
        $WT/tasks/nanogpt/scripts/train.py > $d/train.py
    cp $WT/tasks/nanogpt/scripts/prepare.py $d/prepare.py
    ( cd $d && CUDA_VISIBLE_DEVICES=$CARD timeout 700 $P -u train.py > o.log 2>&1 )
    echo "BS2 TB=$TB node=$(hostname) card=$CARD vram_at_claim=$free $(awk -F: '/^val_bpb/{v=$2}/^mfu_percent/{m=$2}/^num_steps/{s=$2}END{printf "val_bpb=%s mfu=%s steps=%s",v,m,s}' $d/o.log) $(grep -qi OutOfMemory $d/o.log && echo OOM)"
    n=$((n+1))
  done
done
echo "BS2 DONE $(hostname) card=$CARD cells=$n"
