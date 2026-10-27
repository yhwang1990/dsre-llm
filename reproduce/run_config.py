"""Train one run from a saved configuration (configs/runs/<run>.json is the exact argument set of the run in the paper).
Relative paths in the configuration are resolved against DSRE_ROOT; extra command-line arguments are appended and override the file.
Usage: python reproduce/run_config.py configs/runs/nyt10_cwf5l1_8b_s42_v3.json [--seed 7 ...]"""
import json, os, subprocess, sys

PATH_KEYS = ("train_file", "rel2id", "ontology", "cache_root", "model_path", "out")


def command(cfg, root):
    cmd = [sys.executable, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train.py")]
    for k, v in cfg.items():
        if v is None or v is False:
            continue
        if k in PATH_KEYS:
            v = os.path.join(root, v)
        flag = "--" + k.replace("_", "-")
        cmd += [flag] if v is True else [flag, str(v)]
    return cmd


if __name__ == "__main__":
    cfg = json.load(open(sys.argv[1])); root = os.environ.get("DSRE_ROOT", "workspace")
    cmd = command(cfg, root) + [a for a in sys.argv[2:] if a != "--force"]
    out = os.path.join(root, cfg["out"])
    if os.path.exists(os.path.join(out, "heads.pt")) and "--force" not in sys.argv:
        print("skip (already trained):", out); sys.exit(0)
    os.makedirs(out, exist_ok=True)
    print(" ".join(cmd), flush=True)
    sys.exit(subprocess.call(cmd))
