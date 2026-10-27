#!/bin/bash
# Step 1: LLM-generated inputs that training needs (entity types, ontology probe, paraphrases / hard negatives, anchor sentences), all with Qwen3-8B.
# Test-time votes are generated on demand by infer_votes in the later steps. Start vLLM first:
#   bash reproduce/serve_vllm.sh $DSRE_ROOT/models/Qwen3-8B 8000 &
# LLM decoding is not bit-exact across runs, so regenerated outputs will differ slightly from the ones used in the paper.
source "$(dirname "$0")/lib.sh"
P8="--base-url $URL8 --model Qwen3-8B"
PRE="python3 $CODE/precompute.py --ontology $ONTO --workers $WORKERS"
$PRE --rel2id $NYT/rel2id.json --cache-root $C/nyt10 $P8 --file $NYT/train_main.jsonl --stages types,probe,augment,anchors --n-min 30
for f in lowres_1pct lowres_5pct; do $PRE --rel2id $NYT/rel2id.json --cache-root $C/nyt10 $P8 --file $NYT/$f.jsonl --stages types,anchors --n-min 30; done
for f in val test_full; do $PRE --rel2id $NYT/rel2id.json --cache-root $C/nyt10 $P8 --file $NYT/$f.jsonl --stages types,probe; done
$PRE --rel2id $GDS/rel2id.json --cache-root $C/gids $P8 --file $GDS/train.jsonl --stages types,probe,augment,anchors --n-min 30
$PRE --rel2id $GDS/rel2id.json --cache-root $C/gids $P8 --file $GDS/test.jsonl --stages types,probe
$PRE --rel2id $W80/rel2id.json --cache-root $C/wiki80 $P8 --file $W80/train_main.jsonl --stages types,probe,augment,anchors --n-min 30
for f in test val_cmax; do $PRE --rel2id $W80/rel2id.json --cache-root $C/wiki80 $P8 --file $W80/$f.jsonl --stages types,probe; done
# the low-resource subsets reuse the paraphrases of train_main (looked up by uid)
for f in lowres_1pct lowres_5pct; do ln -sf train_main_aug.jsonl $C/nyt10/${f}_aug.jsonl; done
