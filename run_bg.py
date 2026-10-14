"""Chạy các giai đoạn thí nghiệm như tiến trình nền trên VM Colab: python run_bg.py <BACKUP> e0,e1,e2,e3
Mất kết nối VS Code/Colab UI không làm tiến trình dừng; chạy lại thì tự resume (xem train() và predict())."""
import glob
import json
import os
import sys

import yaml

from lab import *

backup, stages = sys.argv[1], sys.argv[2].split(",")
CFG = {n: yaml.safe_load(open(f"configs/{n}.yaml", encoding="utf-8")) for n in ("e0", "e1", "e2", "e3", "e4", "e5")}
test = load_split("test")


def best():  # setup có F1 cao nhất trong E1–E3
    f1 = {n: json.load(open(f"report/evidence/{n.lower()}.json"))["micro"]["f1"] for n in ("E1", "E2", "E3")}
    return max(f1, key=f1.get)


for st in stages:
    print(f"===== {st} =====", flush=True)
    if st == "e0":
        pred = f"{backup}/E0/pred_test.jsonl"
        if not os.path.exists(pred) or len(read_jsonl(pred)) < len(test):
            model, processor = load_model()
            predict(model, processor, test, pred, max_new_tokens=CFG["e0"]["max_new_tokens"])
            del model; cleanup()
        score(pred, "E0", "report/evidence/e0.json")
    elif st in ("e1", "e2", "e3"):
        experiment(CFG[st], f"{backup}/{st.upper()}", f"report/evidence/{st}.json")
    elif st == "e4":
        b = best()
        experiment({**CFG[b.lower()], "name": "E4_data_0.1"}, f"{backup}/E4_data_0.1",
                   "report/evidence/e4_data_0.1.json", train_fraction=0.1)
        run_best = json.load(open(f"{backup}/{b}/run.json"))
        score(f"{backup}/{b}/pred_test.jsonl", "E4_data_1.0", "report/evidence/e4_data_1.0.json",
              {**run_meta(run_best), "train_fraction": 1.0, "base_setup": b})
    elif st == "e5":
        b = best()
        run5 = grpo(f"{backup}/{b}", f"{backup}/E5", CFG["e5"]["grpo"])
        pred = f"{backup}/E5/pred_test.jsonl"
        if not os.path.exists(pred) or len(read_jsonl(pred)) < len(test):
            model, processor = load_trained(f"{backup}/E5")
            predict(model, processor, test, pred, max_new_tokens=512); del model; cleanup()
        score(pred, "E5", "report/evidence/e5.json", {**run_meta(run5), "init": b})
    print(f"===== {st} xong =====", flush=True)
print("ALL_DONE", flush=True)
