#!/bin/bash
# Step 7: Tables 6 and 9, ablation with the main backbone Qwen3-8B on the full NYT10 test set; c_max 0.25 and Qwen3-8B votes.
# The full model is the main model of Table 2 (02_main.sh, three seeds); every variant removes one component from it and is trained with seed 42.
# Variants without the ontology use votes generated without ontology guidance and no ontology gate.
# The three ontology-isolating variants reuse the trained seed-42 main model and only change the LLM votes.
source "$(dirname "$0")/lib.sh"
O=$R/data/ontology; F=$RUNS/nyt10_cwf5l1_8b_s42_v3
V8() {  # V8 <run> <votes suffix> <tag> [infer args]
  local run=$1 suf=$2 tag=$3; shift 3
  infer_votes $run $NYT/test_full.jsonl $NYT/rel2id.json $C/nyt10 $M8 $C/nyt10/test_full_votes${suf}.jsonl test_full $URL8 Qwen3-8B \
    --c-max 0.25 --bootstrap 0 --tag $tag "$@"
}
for j in no_onto no_contrast inst_only proto_only no_cw no_onto_contrast no_all no_anchor no_freq; do
  train_cfg abl_${j}_8b_s42
  if [[ $j == no_onto* || $j == no_all ]]; then PRE_EXTRA="--no-onto" V8 $RUNS/abl_${j}_8b_s42 _noonto test_full --no-gate
  else V8 $RUNS/abl_${j}_8b_s42 "" test_full; fi
done
seed() { mkdir -p $F/eval_$1; for f in probs.npy uids.json encode_cost.json; do cp -n $F/eval_test_full/$f $F/eval_$1/; done; }
seed test_full_keepcot;     PRE_EXTRA="--no-onto" V8 $F _noonto test_full_keepcot --no-gate --reuse-probs
for v in coarse noisy; do
  seed test_full_onto_$v
  PRE_EXTRA="--ontology-variant $O/onto_$v.json" infer_votes $F $NYT/test_full.jsonl $NYT/rel2id.json $C/nyt10 $M8 $C/nyt10/test_full_votes_onto_$v.jsonl \
    test_full $URL8 Qwen3-8B --c-max 0.25 --bootstrap 0 --tag test_full_onto_$v --reuse-probs
done
