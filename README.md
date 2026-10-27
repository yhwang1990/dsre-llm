# DSRE-LLM

Code for **Dual-Layer Semantic Alignment with Large Language Models for Distantly Supervised Relation Extraction**.

DSRE-LLM fine-tunes an LLM encoder (Qwen3-8B with LoRA) for distantly supervised relation extraction and combines four parts:

- **Ontology-guided prompting**: the LLM types the entities and votes for a relation through a six-step chain of thought, restricted to the relations whose domain and range admit the entity types (`dsre/llm.py`, `dsre/ontology.py`).
- **Dual-level contrastive denoising**: an instance-level contrastive loss over LLM paraphrases and hard negatives, plus a prototype-level loss towards relation centroids that are updated only from high-confidence samples (`dsre/model.py`).
- **Confidence weighting**: per-sample weights built from prototype entropy and label agreement, with relation-frequency compensation, scale both the cross-entropy and the prototype loss (`dsre/model.py`, `train.py`).
- **Reliability-aware fusion**: at inference the LLM is consulted only when the classifier's top-2 margin is below c_max/(1-c_max). Its vote passes an ontology gate and is capped by the trust ceiling c_max (`dsre/evaluate.py`, `infer.py`).

## Repository layout

| Path | Content |
|---|---|
| `dsre/` | data handling, model and losses, ontology gate, vLLM client and prompts, evaluation metrics and fusion |
| `train.py`, `infer.py` | training; encoding, on-demand LLM calls, fusion and evaluation |
| `precompute.py` | LLM outputs cached by instance id: entity types, paraphrases / hard negatives, anchor sentences, test-time votes |
| `cmax_sweep.py`, `w80_cmax_select.py` | selection of the trust ceiling c_max on validation data |
| `data_prep/` | data splits, subsets, noise injection, ontology construction, asymmetric-noise transition matrix |
| `baselines/` | PCNN+ATT, BERT-Focal, RoBERTa-MIL, BERT-NLI, CGRE, LLM-CG, zero-shot and ICL (Qwen3-8B-ICL) |
| `configs/runs/` | the exact training arguments of every run reported in the paper (51 runs) |
| `reproduce/` | scripts for every experiment, in order (`00_` ... `08_`) |
| `analysis/` | rebuilds Tables 2-10 and Figures 5-7 from the saved predictions (CPU only) |
| `data/` | ontology files, Wiki80 noise transition matrix, md5 of the raw files and of every data split |

## Setup

```bash
pip install -r requirements.txt
export DSRE_ROOT=/path/to/workspace      # all data, caches, models and runs live here
```

All experiments in the paper ran on a single NVIDIA RTX 4090 (48GB).

### Data

The datasets are not redistributed here. Download them into `$DSRE_ROOT/data/`:

| Dataset | Files | Source |
|---|---|---|
| NYT10 | `nyt10/nyt10_train.txt`, `nyt10_test.txt`, `nyt10_rel2id.json` | OpenNRE benchmark, `https://thunlp.oss-cn-qingdao.aliyuncs.com/opennre/benchmark/nyt10/<file>` |
| Wiki80 | `wiki80/wiki80_train.txt`, `wiki80_val.txt`, `wiki80_rel2id.json` | OpenNRE benchmark, `.../opennre/benchmark/wiki80/<file>` |
| NYT10m | `$DSRE_ROOT/data_ext/nyt10m/nyt10m_test.txt`, `nyt10m_rel2id.json` | OpenNRE benchmark, `.../opennre/benchmark/nyt10m/<file>` (Gao et al., 2021) |
| GDS | `processed/gids_{train,dev,test}.jsonl` | GDS of Jat et al. (2018), converted to the jsonl record format of `dsre/data.py` (`gds/{train,dev,test}.tsv` is also accepted by `prepare_data.py`) |

`data/raw_md5.txt` lists the md5 of the raw files we used. For GDS we used a jsonl conversion of the original tsv files. If you start from the tsv files instead, the md5 check of the GDS splits may differ.

### Models

Download into `$DSRE_ROOT/models/<name>/`:

| Name | Use | Source |
|---|---|---|
| `Qwen3-8B` | main backbone (encoder and LLM) | `Qwen/Qwen3-8B` |
| `Qwen3-4B` | backbone study | `Qwen/Qwen3-4B` |
| `Llama-3-8B-Instruct` | backbone study | `meta-llama/Meta-Llama-3-8B-Instruct` |
| `roberta-base`, `bert-base-uncased` | PLM baselines | Hugging Face |
| `deberta-v2-xlarge-mnli` | first step of BERT-NLI | `microsoft/deberta-v2-xlarge-mnli` |
| `w2v/vec.bin` | 50-d NYT word2vec vectors for PCNN+ATT / CGRE / LLM-CG | as distributed with the NYT10 data of thunlp/NRE (Lin et al., 2016); sha256 `21299e451faed2b61f5cdacf79fc8f641bbe22bae112009c828ee587f6829c46`) |

## Reproducing the paper

Run the scripts in `reproduce/` from the repository root. Each step skips work that is already done.

```bash
bash reproduce/00_prepare_data.sh        # all splits; verified against data/splits_md5.txt
bash reproduce/serve_vllm.sh $DSRE_ROOT/models/Qwen3-8B 8000 &     # LLM server(s) for steps 1-7
bash reproduce/serve_vllm.sh $DSRE_ROOT/models/Qwen3-4B 8001 &
bash reproduce/01_llm_inputs.sh          # LLM-generated entity types, paraphrases / hard negatives, anchors
bash reproduce/02_main.sh                # Tables 2-3: DSRE-LLM and Qwen3-8B-FT, 3 seeds
bash reproduce/03_baselines.sh           # Tables 2-3: baselines (validation-based tuning) and Qwen3-8B-ICL
bash reproduce/04_backbones.sh           # Table 7: Qwen3-4B, Llama-3-8B-Instruct (needs a server on port 8002), zero-shot
bash reproduce/05_low_resource.sh        # Table 8: 1% / 5% training subsets
bash reproduce/06_wiki80.sh              # Figure 6, Table 5: controlled label noise on Wiki80 (0-40%, steps of 10%)
bash reproduce/07_ablation.sh            # Tables 6 and 9: ablation with Qwen3-8B on the full NYT10 test set
bash reproduce/08_analysis.sh            # rebuild all tables and figures into $DSRE_ROOT/results
```

A single run can also be trained directly from its configuration:

```bash
python reproduce/run_config.py configs/runs/nyt10_cwf5l1_8b_s42_v3.json
```

Run names follow `<dataset>_<method>_<backbone>_s<seed>_<version>`; ablation runs are named `abl_<removed component>_8b_s42`. `cwf5l1` is the DSRE-LLM configuration (`--pool ht_last --lr 1e-4 --lam3 0.01 --cw-norm --agree-co --cw-floor 0.5 --lam2 0.1 --lam2-sched linear`). `ft_lr2e-4` is the directly fine-tuned backbone without any of our modules. `gids` denotes GDS.

Trust ceilings selected on validation data are 0.25 for NYT10 and 0.05 for GDS. Wiki80 selects its own value per noise setting with `w80_cmax_select.py`. `analysis/` applies these values to the saved probabilities, so `infer.py` can be run with its default c_max.

### Notes on exactness

- The data splits are deterministic; `00_prepare_data.sh` checks them against the md5 of the files used in the paper.
- GPU training is not bit-exact. Retrained models vary within the seed standard deviations reported in the paper.
- LLM decoding through vLLM is not bit-exact either. Regenerated LLM outputs and votes differ slightly from those used in the paper.
- NYT10 lists a sentence once per distant label, so one instance id can appear on several lines. All evaluation code aligns predictions and labels by line position.

## License

The code is released under the MIT License. The datasets and pretrained models are subject to the licenses of their original providers.
