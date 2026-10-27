#!/bin/bash
# Step 6: Figure 6 and Table 5, Wiki80 with controlled symmetric / asymmetric label noise at 0-40% in steps of 10% (seed 42).
# DSRE-LLM is trained with --dump-weights (per-sample confidence weights; Table 5 uses the 40% runs); c_max is selected on the noisy validation set.
source "$(dirname "$0")/lib.sh"
ST="sym_0 sym_10 sym_20 sym_30 sym_40 asym_10 asym_20 asym_30 asym_40"
tf() { local m=${1%_*} r=${1#*_}; [ $r = 0 ] && echo $W80/train_main.jsonl || echo $W80/train_main_${m}metric_${r}.jsonl; }
for st in $ST; do
  train_cfg w80_${st}_cwf5l1_8b_s42_v4
  train_cfg w80_${st}_ft_lr2e-4_8b_s42_v4
  infer_cls $RUNS/w80_${st}_ft_lr2e-4_8b_s42_v4 $W80/test.jsonl $W80/rel2id.json $C/wiki80 $M8
done
# Qwen3-8B votes for the test set and the noisy validation set, then encode both sets with each DSRE-LLM model
PW="python3 $CODE/precompute.py --rel2id $W80/rel2id.json --ontology $ONTO --cache-root $C/wiki80 --workers $WORKERS --base-url $URL8 --model Qwen3-8B"
$PW --file $W80/val_cmax.jsonl --stages types,votes --tag val_cmax
$PW --file $W80/test.jsonl --stages votes --tag test8b
runs=""
for st in $ST; do
  out=$RUNS/w80_${st}_cwf5l1_8b_s42_v4; runs="$runs $out"
  infer_cls $out $W80/test.jsonl $W80/rel2id.json $C/wiki80 $M8
  infer_cls $out $W80/val_cmax.jsonl $W80/rel2id.json $C/wiki80 $M8 --bootstrap 0 --tag val_cmax
done
python3 $CODE/w80_cmax_select.py $runs
# baselines at every noise setting
for st in $ST; do
  python3 $CODE/baselines/pcnn_att.py --train-file $(tf $st) --test-file $W80/test.jsonl --rel2id $W80/rel2id.json --w2v $R/models/w2v/vec.bin \
    --out $RUNS/w80_${st}_bl_pcnn_att_s42 --seed 42
  for m in bert-focal roberta-mil; do
    mp=$R/models/bert-base-uncased; [ $m = roberta-mil ] && mp=$R/models/roberta-base
    python3 $CODE/baselines/plm_re.py --method $m --model-path $mp --train-file $(tf $st) --test-file $W80/test.jsonl --rel2id $W80/rel2id.json \
      --out $RUNS/w80_${st}_bl_${m}_s42 --seed 42
  done
done
