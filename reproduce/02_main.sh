#!/bin/bash
# Step 2: Tables 2-3, DSRE-LLM and Qwen3-8B-FT with Qwen3-8B, three seeds, NYT10 and GDS (one 48GB GPU, about 2-3 h per NYT10 run).
source "$(dirname "$0")/lib.sh"
for s in 13 42 123; do
  train_cfg nyt10_cwf5l1_8b_s${s}_v3
  infer_votes $RUNS/nyt10_cwf5l1_8b_s${s}_v3 $NYT/test_full.jsonl $NYT/rel2id.json $C/nyt10 $M8 $C/nyt10/test_full_votes.jsonl test_full $URL8 Qwen3-8B
  train_cfg gids_cwf5l1_8b_s${s}_v3
  infer_votes $RUNS/gids_cwf5l1_8b_s${s}_v3 $GDS/test.jsonl $GDS/rel2id.json $C/gids $M8 $C/gids/test_votes.jsonl test $URL8 Qwen3-8B
  train_cfg nyt10_ft_lr2e-4_8b_s${s}_v3
  infer_cls $RUNS/nyt10_ft_lr2e-4_8b_s${s}_v3 $NYT/test_full.jsonl $NYT/rel2id.json $C/nyt10 $M8
  train_cfg gids_ft_lr2e-4_8b_s${s}_v3
  infer_cls $RUNS/gids_ft_lr2e-4_8b_s${s}_v3 $GDS/test.jsonl $GDS/rel2id.json $C/gids $M8
done
# c_max is selected on the validation sets (NYT10 0.25, GDS 0.05); analysis/ re-fuses the saved probabilities at these values
