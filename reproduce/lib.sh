#!/bin/bash
# Shared settings for the reproduction scripts. Source it from the repository root: source reproduce/lib.sh
set -u
export DSRE_ROOT=${DSRE_ROOT:-workspace}
R=$DSRE_ROOT; CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
P=$R/data_prepared; C=$R/cache; RUNS=$R/runs; ONTO=$R/data/ontology/relation_ontology_all.json
NYT=$P/nyt10; GDS=$P/gids; W80=$P/wiki80; TIERS=$NYT/tiers_main.json
M8=$R/models/Qwen3-8B; M4=$R/models/Qwen3-4B; ML=$R/models/Llama-3-8B-Instruct
# vLLM endpoints (start them with reproduce/serve_vllm.sh); each backbone produces its own votes
URL8=${URL8:-http://localhost:8000}; URL4=${URL4:-http://localhost:8001}; URLL=${URLL:-http://localhost:8002}
WORKERS=${WORKERS:-64}
mkdir -p $RUNS

train_cfg() {  # train_cfg <run name>: train with the exact configuration of the paper (configs/runs/<run>.json)
  python3 $CODE/reproduce/run_config.py $CODE/configs/runs/$1.json
}

infer_votes() {  # infer_votes <run dir> <test file> <rel2id> <cache dir> <model dir> <votes file> <vote tag> <vLLM url> <served model name> [infer args ...]
  # Encodes the test file and fuses with cached LLM votes. If votes are missing for instances that need the LLM, infer.py exits
  # with code 3; the missing votes are then generated with precompute.py (set PRE_EXTRA for --no-onto / --ontology-variant) and
  # fusion is re-run on the saved probabilities.
  local run=$1 tf=$2 rid=$3 cache=$4 model=$5 votes=$6 vtag=$7 url=$8 llm=$9; shift 9
  local tag=$(basename ${tf%.jsonl}); for a in "$@"; do [ "${prev:-}" = "--tag" ] && tag=$a; prev=$a; done; unset prev
  python3 $CODE/infer.py --test-file $tf --rel2id $rid --ontology $ONTO --cache-root $cache --run $run --model-path $model \
    --votes $votes --tiers $TIERS "$@"
  if [ $? = 3 ]; then
    python3 $CODE/precompute.py --file $tf --rel2id $rid --ontology $ONTO --cache-root $cache --workers $WORKERS --tag $vtag \
      --stages votes --uids $run/eval_$tag/need_uids.json --base-url $url --model $llm ${PRE_EXTRA:-}
    local extra=("$@"); [[ " $* " == *" --reuse-probs "* ]] || extra+=(--reuse-probs)
    python3 $CODE/infer.py --test-file $tf --rel2id $rid --ontology $ONTO --cache-root $cache --run $run --model-path $model \
      --votes $votes --tiers $TIERS "${extra[@]}"
  fi
}

infer_cls() {  # infer_cls <run dir> <test file> <rel2id> <cache dir> <model dir> [infer args ...]: classifier output only (Qwen3-8B-FT)
  local run=$1 tf=$2 rid=$3 cache=$4 model=$5; shift 5
  python3 $CODE/infer.py --test-file $tf --rel2id $rid --ontology $ONTO --cache-root $cache --run $run --model-path $model --tiers $TIERS "$@"
}
