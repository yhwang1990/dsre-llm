#!/bin/bash
# Step 0: build every data split used in the paper (deterministic; fixed seeds).
# Expects the raw data under $DSRE_ROOT/data (see README: nyt10/, wiki80/, processed/gids_{train,dev,test}.jsonl)
# and the released ontology files / noise transition matrix copied from data/ of this repository.
source "$(dirname "$0")/lib.sh"
mkdir -p $R/data/ontology
cp -n $CODE/data/ontology/*.json $R/data/ontology/; cp -n $CODE/data/wiki80_asym_T.json $R/data/
python3 $CODE/data_prep/prepare_data.py --raw-root $R/data --asym $R/data/wiki80_asym_T.json --out $P
python3 $CODE/data_prep/make_val_sub.py $NYT/val.jsonl $NYT/val_sub15k.jsonl        # calibration set of the zero-shot / ICL baselines
python3 $CODE/data_prep/make_w80_val.py                                              # noisy Wiki80 validation set for selecting c_max
# check the splits against the files used in the paper
(cd $P && md5sum -c $CODE/data/splits_md5.txt)
