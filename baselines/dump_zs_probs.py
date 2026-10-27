"""Run zs_calib.py (validation calibration of the zero-shot / ICL baselines) and save the test-set probabilities it passes to
evaluate (rows in test-file order), without modifying zs_calib.py.
Usage: python dump_zs_probs.py <output .npy> <all arguments of zs_calib.py ...>"""
import os, runpy, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsre import evaluate as E

out = sys.argv[1]; _orig = E.evaluate; seen = []


def hook(probs, gold, bag_ids, *a, **k):
    seen.append(np.asarray(probs))
    return _orig(probs, gold, bag_ids, *a, **k)


E.evaluate = hook
sys.argv = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "zs_calib.py")] + sys.argv[2:]
runpy.run_path(sys.argv[0], run_name="__main__")
np.save(out, seen[-1])      # the last evaluate call is the final test-set result
print("saved", out, seen[-1].shape)
