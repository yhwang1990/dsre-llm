#!/bin/bash
# Step 3: baselines of Tables 2-3. Every configuration is selected on the validation set only (score = AUC-PR + Micro-F1 + Macro-F1),
# then trained with seeds 13 and 123 as well. Selected configurations of the paper: analysis/common.py (BEST).
source "$(dirname "$0")/lib.sh"
BT=$RUNS/tune; mkdir -p $BT
W2V=$R/models/w2v/vec.bin; BERT=$R/models/bert-base-uncased; ROB=$R/models/roberta-base

# BERT-NLI, first step: denoise the training labels with an off-the-shelf MNLI model
for ds in nyt10 gids; do
  TR=$NYT/train_main.jsonl; [ $ds = gids ] && TR=$GDS/train.jsonl
  [ -f ${TR%.jsonl}_nli.jsonl ] || python3 $CODE/baselines/nli_consolidate.py --train-file $TR --rel2id $P/$ds/rel2id.json --ontology $ONTO \
    --types $C/$ds/entity_types.jsonl --model $R/models/deberta-v2-xlarge-mnli --batch-size 32 --out ${TR%.jsonl}_nli.jsonl
done
# LLM-CG, first step: LLM node representations of the constraint graph (CPU is enough)
for ds in nyt10 gids; do
  CUDA_VISIBLE_DEVICES="" python3 $CODE/baselines/llm_node_emb.py --model $M8 --rel2id $P/$ds/rel2id.json \
    --types $C/$ds/entity_types.jsonl --out $C/$ds/llmcg_node_emb_8b.npz
done

run_bl() {  # run_bl <dataset> <method> <config> <seed>
  local ds=$1 m=$2 cfg=$3 s=$4 out=$BT/$1_$2_$3_s$4
  [ -f $out/result.json ] && return; mkdir -p $out
  if [ $ds = nyt10 ]; then TR=$NYT/train_main.jsonl; B="--test-file $NYT/test_full.jsonl --val-file $NYT/val.jsonl --rel2id $NYT/rel2id.json --tiers $TIERS"
  else TR=$GDS/train.jsonl; B="--test-file $GDS/test.jsonl --val-file $GDS/dev.jsonl --rel2id $GDS/rel2id.json"; fi
  B="$B --out $out --seed $s"
  if [ $ds = nyt10 ]; then   # NYT10: one configuration per method (the GDS-selected PLM recipe), early stopping on validation
    PL="--epochs 3 --warmup 0.1 --markers --lr 3e-5 --bf16 --patience 1"; PC="--epochs 15 --lr 1e-3 --patience 3"; PG="--epochs 12 --lr 0.5 --patience 3"
  else                       # GDS: grid search; m2/m3/m5 = entity markers + lr 2e-5/3e-5/5e-5, p3 = no markers + lr 3e-5
    PL="--epochs 5 --warmup 0.1"
    case $cfg in m2) PL="$PL --markers --lr 2e-5" ;; m3) PL="$PL --markers --lr 3e-5" ;; m5) PL="$PL --markers --lr 5e-5" ;; p3) PL="$PL --lr 3e-5" ;; esac
    PC="--epochs 20 --lr ${cfg#lr}"; PG="--epochs 20 --lr ${cfg#lr}"
  fi
  case $m in
    focal) python3 $CODE/baselines/plm_re.py --method bert-focal --model-path $BERT --train-file $TR $B $PL ;;
    mil)   python3 $CODE/baselines/plm_re.py --method roberta-mil --model-path $ROB --train-file $TR $B $PL ;;
    nli)   python3 $CODE/baselines/plm_re.py --method bert-ce --model-path $BERT --train-file ${TR%.jsonl}_nli.jsonl $B $PL ;;
    pcnn)  python3 $CODE/baselines/pcnn_att.py --train-file $TR --w2v $W2V $B $PC ;;
    cgre)  python3 $CODE/baselines/cgre.py --train-file $TR --w2v $W2V --types $C/$ds/entity_types.jsonl $B $PG ;;
    llmcg) python3 $CODE/baselines/cgre.py --train-file $TR --w2v $W2V --types $C/$ds/entity_types.jsonl --node-init $C/$ds/llmcg_node_emb_8b.npz $B $PG ;;
  esac > $out/train.log 2>&1
}
best_cfg() {  # best_cfg <dataset> <method> <configs...>: highest validation score at seed 42
  python3 - $BT "$@" <<'PY'
import json, os, sys
bt, ds, m, cs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
sc = {c: json.load(open(f))["tuning"]["best_val_score"] for c in cs for f in [os.path.join(bt, f"{ds}_{m}_{c}_s42", "result.json")] if os.path.exists(f)}
print(max(sc, key=sc.get))
PY
}
for m in focal mil nli; do for s in 42 13 123; do run_bl nyt10 $m m3e3 $s; done; done
for s in 42 13 123; do run_bl nyt10 pcnn lr1e-3e15 $s; run_bl nyt10 cgre lr0.5e12 $s; run_bl nyt10 llmcg lr0.5e12 $s; done
for m in focal mil nli pcnn cgre llmcg; do
  case $m in focal|mil|nli) G="m2 m3 m5 p3" ;; pcnn) G="lr1e-3 lr5e-4 lr2e-3" ;; *) G="lr0.5 lr0.2 lr1.0" ;; esac
  for c in $G; do run_bl gids $m $c 42; done
  b=$(best_cfg gids $m $G); echo "gids $m -> $b"
  for s in 13 123; do run_bl gids $m $b $s; done
done

# Qwen3-8B-ICL: retrieved demonstrations (kNN, GPT-RE style) + calibration selected on the validation set (needs vLLM Qwen3-8B)
Z="--base-url $URL8 --model Qwen3-8B --workers $WORKERS"
python3 $CODE/baselines/zeroshot_knn.py --test-file $NYT/val_sub15k.jsonl --rel2id $NYT/rel2id.json --train-file $NYT/train_main.jsonl --cache $C/nyt10/val_sub15k_knn.jsonl --out $RUNS/nyt10_knn_valsub --bootstrap 0 $Z
python3 $CODE/baselines/zeroshot_knn.py --test-file $NYT/test_full.jsonl --rel2id $NYT/rel2id.json --train-file $NYT/train_main.jsonl --cache $C/nyt10/test_full_knn.jsonl --tiers $TIERS --out $RUNS/bl_knn_8b $Z
python3 $CODE/baselines/zs_calib.py --val-file $NYT/val_sub15k.jsonl --val-cache $C/nyt10/val_sub15k_knn.jsonl --test-file $NYT/test_full.jsonl --test-cache $C/nyt10/test_full_knn.jsonl --rel2id $NYT/rel2id.json --tiers $TIERS --out $RUNS/bl_knncal_8b
python3 $CODE/baselines/zeroshot_knn.py --test-file $GDS/dev.jsonl --rel2id $GDS/rel2id.json --train-file $GDS/train.jsonl --cache $C/gids/dev_knn.jsonl --out $RUNS/gids_knn_dev --bootstrap 0 $Z
python3 $CODE/baselines/zeroshot_knn.py --test-file $GDS/test.jsonl --rel2id $GDS/rel2id.json --train-file $GDS/train.jsonl --cache $C/gids/test_knn.jsonl --out $RUNS/gids_bl_knn_8b $Z
python3 $CODE/baselines/zs_calib.py --val-file $GDS/dev.jsonl --val-cache $C/gids/dev_knn.jsonl --test-file $GDS/test.jsonl --test-cache $C/gids/test_knn.jsonl --rel2id $GDS/rel2id.json --out $RUNS/gids_bl_knncal_8b
# test-set probabilities of the ICL baseline for the NYT10m columns of Table 2
mkdir -p $R/results/zsprobs
python3 $CODE/baselines/dump_zs_probs.py $R/results/zsprobs/knncal.npy --val-file $NYT/val_sub15k.jsonl --val-cache $C/nyt10/val_sub15k_knn.jsonl \
  --test-file $NYT/test_full.jsonl --test-cache $C/nyt10/test_full_knn.jsonl --rel2id $NYT/rel2id.json --out $RUNS/bl_knncal_8b_dump
