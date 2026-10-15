# Kaggle script kernel: chạy toàn bộ Lab Day 10 (E0–E5) từ repo GitHub. Kết quả ở /kaggle/working/out và export.tar.gz
import glob, os, subprocess, sys, tarfile

def sh(cmd, **kw):
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=kw.pop("check", True), **kw)

REPO = "https://github.com/xuanlocc2/K4-Track-4-Day10-VisionLanguageModel.git"
OUT = "/kaggle/working/out"
os.makedirs(OUT, exist_ok=True)
sh("rm -rf /tmp/lab && git clone -q " + REPO + " /tmp/lab")
os.chdir("/tmp/lab")
sh("nvidia-smi -L; python --version; python -c 'import torch;print(torch.__version__, torch.cuda.is_available())'")
sh("pip -q install -r requirements-colab.txt 2>&1 | tail -3")
sh("pip -q uninstall -y torchao", check=False)
sh("python scripts/prepare_data.py --source /kaggle/input --out data --work work")
sh("python -m pytest -q -p no:cacheprovider 2>&1 | tail -3", check=False)
stages = "e0,e1,e2,e3,e4,e5"
r = sh(f"python -u run_bg.py {OUT} {stages}", check=False)
for f in glob.glob(f"{OUT}/*/ckpt.pt") + glob.glob(f"{OUT}/*/projector.pt"):  # nặng, không cần mang về
    os.remove(f)
with tarfile.open("/kaggle/working/export.tar.gz", "w:gz") as t:
    t.add("report"); t.add("data/splits")
    for f in glob.glob(f"{OUT}/*/pred_test.jsonl") + glob.glob(f"{OUT}/*/run.json"):
        t.add(f, arcname="work/" + f[len(OUT) + 1:])
print("EXIT", r.returncode)
sys.exit(r.returncode)
