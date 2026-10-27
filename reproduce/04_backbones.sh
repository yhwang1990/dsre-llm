#!/bin/bash
# Step 4: Table 7, other backbones. Qwen3-4B (3 seeds) and
# Llama-3-8B-Instruct (seed 42). Each backbone produces its own votes. NYT10 is scored on the 20% test subset.
source "$(dirname "$0")/lib.sh"
for s in 13 42 123; do
  train_cfg nyt10_cwf5l1_4b_s${s}_v3
  infer_votes $RUNS/nyt10_cwf5l1_4b_s${s}_v3 $NYT/test_sub20.jsonl $NYT/rel2id.json $C/nyt10 $M4 $C/nyt10/test_sub20_4b_votes.jsonl test_sub20_4b $URL4 Qwen3-4B
done
for s in 13 42 123; do
  train_cfg gids_cwf5l1_4b_s${s}_v3
  infer_votes $RUNS/gids_cwf5l1_4b_s${s}_v3 $GDS/test.jsonl $GDS/rel2id.json $C/gids $M4 $C/gids/test_4b_votes.jsonl test_4b $URL4 Qwen3-4B
done
train_cfg nyt10_cwf5l1_llama3_s42_v3
infer_votes $RUNS/nyt10_cwf5l1_llama3_s42_v3 $NYT/test_full.jsonl $NYT/rel2id.json $C/nyt10 $ML $C/nyt10/test_full_llama_votes.jsonl test_full_llama $URLL Llama-3-8B-Instruct --bootstrap 0
train_cfg gids_cwf5l1_llama3_s42_v3
infer_votes $RUNS/gids_cwf5l1_llama3_s42_v3 $GDS/test.jsonl $GDS/rel2id.json $C/gids $ML $C/gids/test_llama_votes.jsonl test_llama $URLL Llama-3-8B-Instruct --bootstrap 0
# zero-shot column: each Qwen3 backbone prompted alone (probability scoring + validation calibration), on the 20% subset
VS=$NYT/val_sub15k.jsonl; S20=$NYT/test_sub20.jsonl
for b in 8b 4b; do
  if [ $b = 8b ]; then Z="--base-url $URL8 --model Qwen3-8B --workers $WORKERS"; V=$C/nyt10/val_sub15k_zslp.jsonl; T=$C/nyt10/test_full_zslp.jsonl; TF=$NYT/test_full.jsonl
  else Z="--base-url $URL4 --model Qwen3-4B --workers $WORKERS"; V=$C/nyt10/val_sub15k_zslp_4b.jsonl; T=$C/nyt10/test_sub20_zslp_4b.jsonl; TF=$S20; fi
  python3 $CODE/baselines/zeroshot_lp.py --test-file $VS --rel2id $NYT/rel2id.json --cache $V --out $RUNS/nyt10_zslp${b}_valsub $Z
  python3 $CODE/baselines/zeroshot_lp.py --test-file $TF --rel2id $NYT/rel2id.json --cache $T --out $RUNS/bl_zeroshot_prob_${b} $Z
  python3 $CODE/baselines/zs_calib.py --val-file $VS --val-cache $V --test-file $S20 --test-cache $T --rel2id $NYT/rel2id.json --tiers $TIERS \
    --out $RUNS/bl_zeroshot_probcal_${b}_sub20
done
