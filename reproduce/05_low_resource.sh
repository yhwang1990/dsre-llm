#!/bin/bash
# Step 5: Table 8, 1% and 5% NYT10 training subsets (seed 42; the 10% column reuses the seed-42 runs of Table 2), 20% test subset.
source "$(dirname "$0")/lib.sh"
for f in 1pct 5pct; do
  train_cfg lowres_${f}_cwf5l1_8b_s42_v4
  infer_votes $RUNS/lowres_${f}_cwf5l1_8b_s42_v4 $NYT/test_sub20.jsonl $NYT/rel2id.json $C/nyt10 $M8 $C/nyt10/test_sub20_votes.jsonl test_sub20 $URL8 Qwen3-8B
  train_cfg lowres_${f}_ft_lr2e-4_8b_s42_v4
  infer_cls $RUNS/lowres_${f}_ft_lr2e-4_8b_s42_v4 $NYT/test_sub20.jsonl $NYT/rel2id.json $C/nyt10 $M8
  python3 $CODE/baselines/pcnn_att.py --train-file $NYT/lowres_$f.jsonl --test-file $NYT/test_full.jsonl --rel2id $NYT/rel2id.json \
    --w2v $R/models/w2v/vec.bin --tiers $TIERS --out $RUNS/lowres_${f}_bl_pcnn_att_s42 --seed 42
  for m in bert-focal roberta-mil; do
    mp=$R/models/bert-base-uncased; [ $m = roberta-mil ] && mp=$R/models/roberta-base
    python3 $CODE/baselines/plm_re.py --method $m --model-path $mp --train-file $NYT/lowres_$f.jsonl --test-file $NYT/test_full.jsonl \
      --rel2id $NYT/rel2id.json --tiers $TIERS --out $RUNS/lowres_${f}_bl_${m}_s42 --seed 42
  done
done
