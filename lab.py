"""Mã của sinh viên cho Lab Day 10 (Nguyễn Văn Xuân Lộc - 22127237).

Notebook lab_day10.ipynb chỉ gọi các hàm ở đây. Phần thuần Python (nhãn, reward)
test được không cần GPU: python -m pytest -q test_lab.py
"""
import gc
import json
import random
import time
from pathlib import Path

from scripts.labcommon import read_jsonl, write_jsonl
from scripts.score import _gt_boxes, _match, iou_xyxy, parse_output

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DATA = Path("data")
IMG_KEYS = ("pixel_values", "image_grid_thw")

PROMPT = (
    "Locate road damage in this dashcam image. "
    "Detect only objects of classes: pothole, crack, manhole. "
    "Output only a JSON list, no other text: "
    '[{"label": "...", "bbox_2d": [x1, y1, x2, y2]}], '
    "where coordinates are integers on a 0-1000 scale relative to image width and height. "
    "If no damage is found, output []."
)

LLM_RE = r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
VIT_RE = r"model\.visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.linear_fc[12])"
PROJ_KEYS = ("visual.merger", "visual.deepstack_merger_list")


# ---------------------------------------------------------------- Bước 2: nhãn
def yolo_to_box(label, yolo):
    xc, yc, w, h = yolo
    c = lambda v: int(round(max(0.0, min(1000.0, v * 1000))))  # noqa: E731
    x1, y1, x2, y2 = c(xc - w / 2), c(yc - h / 2), c(xc + w / 2), c(yc + h / 2)
    x2, y2 = max(x2, min(1000, x1 + 1)), max(y2, min(1000, y1 + 1))  # box không suy biến
    return {"label": label, "bbox_2d": [x1, y1, x2, y2]}


def objects_to_answer(objects):
    boxes = [yolo_to_box(o["label"], o["yolo"]) for o in objects]
    boxes.sort(key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0]))  # trên→dưới, trái→phải
    return json.dumps(boxes, separators=(",", ":"))


def load_split(name):
    return read_jsonl(DATA / "splits" / f"{name}.jsonl")


def load_image(entry):
    from PIL import Image
    return Image.open(DATA / entry["image_path"]).convert("RGB")


# ---------------------------------------------------------------- Bước 3: model
def load_model(adapter=None, projector=None, trainable=False):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
                           llm_int8_skip_modules=["model.visual", "lm_head"])  # transformers dùng re.match từ đầu tên: "visual" KHÔNG khớp "model.visual..."
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=q, dtype=torch.float16, device_map={"": 0})
    quant = [n for n, p in model.named_parameters() if "visual" in n and p.dtype == torch.uint8]
    assert not quant, f"ViT/projector bị lượng tử hoá 4-bit: {quant[:3]}"
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "left"
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    if projector:
        sd = torch.load(projector, map_location="cuda")
        res = model.load_state_dict(sd, strict=False)
        assert not res.unexpected_keys, res.unexpected_keys[:3]
        print(f"đã nạp lại {len(sd)} tensor projector từ {projector}")
    return model, processor


def _user(image, prompt):
    return [{"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text", "text": prompt}]}]


def generate(model, processor, images, prompt=PROMPT, max_new_tokens=256, do_sample=False, **gen):
    import torch
    inputs = processor.apply_chat_template(
        [_user(im, prompt) for im in images], tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", padding=True).to(model.device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=do_sample, **gen)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def predict(model, processor, entries, out_path, batch=8, max_new_tokens=256):
    """Suy luận rồi ghi {"image_id","raw"}; chạy lại thì bỏ qua ảnh đã có (resume khi mất session)."""
    out_path = Path(out_path)
    done = {r["image_id"]: r for r in read_jsonl(out_path)} if out_path.exists() else {}
    todo = [e for e in entries if e["image_id"] not in done]
    model.eval()
    t0 = time.time()
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        raws = generate(model, processor, [load_image(e) for e in chunk], max_new_tokens=max_new_tokens)
        for e, r in zip(chunk, raws):
            done[e["image_id"]] = {"image_id": e["image_id"], "raw": r}
        write_jsonl(out_path, done.values())
        if i % (batch * 10) == 0:
            print(f"  {len(done)}/{len(entries)} ảnh, {time.time() - t0:.0f}s")
    return [done[e["image_id"]] for e in entries]


# ---------------------------------------------------------------- Bước 4: mask loss
def encode_example(processor, image, prompt, answer):
    user = _user(image, prompt)
    kw = dict(tokenize=True, return_dict=True, return_tensors="pt")
    p = processor.apply_chat_template(user, add_generation_prompt=True, **kw)
    f = processor.apply_chat_template(
        user + [{"role": "assistant", "content": [{"type": "text", "text": answer}]}],
        add_generation_prompt=False, **kw)
    n = p["input_ids"].shape[1]
    assert (f["input_ids"][0, :n] == p["input_ids"][0]).all(), "prompt không phải tiền tố"
    ex = {k: (v if k in IMG_KEYS else v[0]) for k, v in f.items()}
    ex["labels"] = ex["input_ids"].clone()
    ex["labels"][:n] = -100  # không học ảnh + prompt
    return ex


def collate(batch, pad_id):
    import torch
    from torch.nn.utils.rnn import pad_sequence
    out = {}
    for k in batch[0]:
        if k in IMG_KEYS:
            out[k] = torch.cat([b[k] for b in batch], 0)  # nối, không pad
        else:
            fill = -100 if k == "labels" else pad_id if k == "input_ids" else 0
            out[k] = pad_sequence([b[k] for b in batch], batch_first=True, padding_value=fill)
    return out


# ---------------------------------------------------------------- Bước 5: LoRA / đóng băng
def apply_setup(model, setup, r=16):
    """setup: dict từ configs/eN.yaml. Trả về peft model + số module LoRA."""
    import torch
    from peft import LoraConfig, get_peft_model
    for p in model.parameters():
        p.requires_grad = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lora = setup["lora"]
    target = LLM_RE if lora["targets"] == "llm" else f"({LLM_RE})|({VIT_RE})"
    model = get_peft_model(model, LoraConfig(r=lora["r"], lora_alpha=lora["alpha"],
                                             lora_dropout=lora["dropout"], target_modules=target))
    n_mod = sum(1 for n, _ in model.named_modules() if n.endswith(".lora_A"))
    if setup["train_projector"]:
        for n, p in model.named_parameters():
            if any(k in n for k in PROJ_KEYS):
                p.requires_grad = True
    for p in model.parameters():  # GradScaler cần tham số huấn luyện ở fp32
        if p.requires_grad:
            p.data = p.data.float()
    return model, n_mod


def count_params(model):
    import torch
    tot = {"llm": 0, "vit": 0, "projector": 0, "lm_head": 0}
    for n, p in model.named_parameters():
        if "lora_" in n:
            continue
        k = ("projector" if any(s in n for s in PROJ_KEYS) else "vit" if "visual" in n
             else "lm_head" if "lm_head" in n else "llm")
        tot[k] += p.numel() * (2 if p.dtype == torch.uint8 else 1)  # 4-bit: 2 trọng số / byte
    return tot


# ---------------------------------------------------------------- Bước 6: train
def train(setup, run_dir, train_fraction=1.0, log_every=10):
    import torch
    from transformers import get_cosine_schedule_with_warmup
    run_dir = Path(run_dir)
    if (run_dir / "run.json").exists():
        print("đã train xong trước đó:", run_dir)
        return json.loads((run_dir / "run.json").read_text())
    random.seed(setup["seed"])
    torch.manual_seed(setup["seed"])
    model, processor = load_model()
    model, n_mod = apply_setup(model, setup)
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    print(f"LoRA modules={n_mod}  trainable={n_train:,}")

    entries = load_split("train")
    if train_fraction < 1:
        entries = random.Random(setup["seed"]).sample(entries, round(len(entries) * train_fraction))
    pad_id = processor.tokenizer.pad_token_id
    bs, acc, steps = setup["batch_size"], setup["grad_accum"], setup["max_steps"]

    def batches():
        while True:
            order = entries[:]
            random.shuffle(order)
            for i in range(0, len(order) - bs + 1, bs):
                yield collate([encode_example(processor, load_image(e), PROMPT,
                                              objects_to_answer(e["objects"]))
                               for e in order[i:i + bs]], pad_id)

    opt = torch.optim.AdamW(params, lr=setup["lr"], weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.reset_peak_memory_stats()
    model.train()
    ckpt_path, start, losses, t_prev = run_dir / "ckpt.pt", 1, [], 0.0
    if ckpt_path.exists():  # bị ngắt giữa chừng → train tiếp từ checkpoint gần nhất
        ck = torch.load(ckpt_path, map_location="cpu")
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in ck["params"]:
                    p.copy_(ck["params"][n])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start, losses, t_prev = ck["step"] + 1, ck["losses"], ck["minutes"]
        random.seed(setup["seed"] + ck["step"])  # thứ tự batch sau resume khác nhưng vẫn xáo trộn
        print(f"resume từ checkpoint: bước {ck['step']}/{steps}")
        del ck
    it, t0 = batches(), time.time() - t_prev * 60
    for step in range(start, steps + 1):
        tot = 0.0
        for _ in range(acc):
            batch = {k: v.to(model.device) for k, v in next(it).items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch, use_cache=False).loss / acc
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss không hữu hạn ở bước {step} (tràn fp16?)")
            scaler.scale(loss).backward()
            tot += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        losses.append(round(tot, 4))
        if step % log_every == 0:
            print(f"step {step}/{steps} loss={sum(losses[-log_every:]) / log_every:.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e} {(time.time() - t0) / 60:.1f}m "
                  f"vram={torch.cuda.max_memory_allocated() / 2**30:.1f}GB")
        if step % 50 == 0 or step == steps:
            run_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"params": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
                        "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                        "step": step, "losses": losses, "minutes": (time.time() - t0) / 60}, ckpt_path)
    save_trained(model, run_dir, setup["train_projector"])
    run = {"name": setup["name"], "trainable_params": n_train, "lora_modules": n_mod,
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
           "train_minutes": round((time.time() - t0) / 60, 1),
           "train_images": len(entries), "losses": losses}
    (run_dir / "run.json").write_text(json.dumps(run, indent=1))
    del model, opt, params
    cleanup()
    return run


def save_trained(model, run_dir, with_projector):
    import torch
    model.save_pretrained(Path(run_dir) / "adapter")  # chỉ chứa LoRA
    if with_projector:  # projector được train đầy đủ, KHÔNG nằm trong adapter
        torch.save({n: p.detach().cpu() for n, p in model.named_parameters()
                    if any(k in n for k in PROJ_KEYS)}, Path(run_dir) / "projector.pt")


def load_trained(run_dir, trainable=False):
    run_dir = Path(run_dir)
    proj = run_dir / "projector.pt"
    return load_model(run_dir / "adapter", proj if proj.exists() else None, trainable)


def cleanup():
    import torch
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- Bước 7: chấm
def score(pred, name, out, meta=None, split="test", coord="norm1000"):
    import subprocess
    import sys
    cmd = [sys.executable, "scripts/score.py", "--pred", str(pred), "--name", name,
           "--out", str(out), "--split", split, "--coord", coord]
    for k, v in (meta or {}).items():
        cmd += ["--meta", f"{k}={v}"]
    print(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    return json.loads(Path(out).read_text(encoding="utf-8"))


def run_meta(run):
    return {k: run[k] for k in ("trainable_params", "peak_vram_gb", "train_minutes")}


def experiment(cfg, run_dir, evidence, train_fraction=1.0, max_new_tokens=512):
    """train → nạp LẠI từ đĩa (adapter + projector) → suy luận test → chấm. Chạy lại được sau khi mất session."""
    run = train(cfg, run_dir, train_fraction)
    pred = Path(run_dir) / "pred_test.jsonl"
    if not pred.exists() or len(read_jsonl(pred)) < len(load_split("test")):
        model, processor = load_trained(run_dir)
        predict(model, processor, load_split("test"), pred, max_new_tokens=max_new_tokens)
        del model
        cleanup()
    meta = run_meta(run)
    if train_fraction < 1:
        meta["train_fraction"] = train_fraction
    return run, score(pred, cfg["name"], evidence, meta)


def per_image(entries, preds):
    """[(entry, raw, tp, fp, fn)] cho phân tích lỗi / chọn ảnh minh hoạ."""
    raw = {p["image_id"]: p["raw"] for p in preds}
    rows = []
    for e in entries:
        pred = parse_output(raw[e["image_id"]]) or []
        per, _ = _match(_gt_boxes(e), pred)
        tp, fp, fn = (sum(v[k] for v in per.values()) for k in range(3))
        rows.append((e, raw[e["image_id"]], tp, fp, fn))
    return rows


def draw(entry, raw, path, title=""):
    from PIL import ImageDraw
    im = load_image(entry)
    d = ImageDraw.Draw(im)
    W, H = im.size
    sx = lambda b: [b[0] * W / 1000, b[1] * H / 1000, b[2] * W / 1000, b[3] * H / 1000]  # noqa: E731
    for g in _gt_boxes(entry):
        d.rectangle(sx(g["bbox"]), outline=(0, 255, 0), width=3)
        d.text((sx(g["bbox"])[0] + 2, sx(g["bbox"])[3] - 12), g["label"], fill=(0, 255, 0))
    for p in parse_output(raw) or []:
        d.rectangle(sx(p["bbox"]), outline=(255, 40, 40), width=2)
        d.text((sx(p["bbox"])[0] + 2, sx(p["bbox"])[1] + 2), p["label"], fill=(255, 40, 40))
    d.text((5, 5), title + "  [xanh=GT, đỏ=dự đoán]", fill=(255, 255, 0))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=90)


# ---------------------------------------------------------------- Bước 9: GRPO
def compute_reward(completion, gt_boxes):
    """0.2 * JSON hợp lệ + 0.8 * tổng IoU khớp (cùng lớp, IoU≥0.5) / max(n_gt, n_pred, 1)."""
    pred = parse_output(completion)
    if pred is None:
        return 0.0
    if not gt_boxes and not pred:
        return 1.0
    total = 0.0
    for lab in {g["label"] for g in gt_boxes} | {p["label"] for p in pred}:
        g = [x["bbox"] for x in gt_boxes if x["label"] == lab]
        p = [x["bbox"] for x in pred if x["label"] == lab]
        pairs = sorted(((iou_xyxy(a, b), i, j) for i, a in enumerate(g)
                        for j, b in enumerate(p)), reverse=True)
        ug, up = set(), set()
        for v, i, j in pairs:
            if v < 0.5:
                break
            if i in ug or j in up:
                continue
            ug.add(i)
            up.add(j)
            total += v
    return 0.2 + 0.8 * total / max(len(gt_boxes), len(pred), 1)


def grpo(init_dir, run_dir, cfg, images_per_step=2, log_every=5):
    """GRPO tối giản (beta=0, 1 lần cập nhật / nhóm ⇒ ratio=1, clip không kích hoạt):
    loss = -Σ_i A_i · mean_t log π(o_i,t), A_i = (r_i - mean(r)) / (std(r) + 1e-4)."""
    import torch
    run_dir = Path(run_dir)
    if (run_dir / "run.json").exists():
        return json.loads((run_dir / "run.json").read_text())
    torch.manual_seed(0)
    model, processor = load_trained(init_dir, trainable=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    G, steps = cfg["num_generations"], cfg["max_steps"]
    opt = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    entries = load_split("train")
    random.Random(0).shuffle(entries)
    log, t0 = [], time.time()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, steps + 1):
        rec = {"step": step, "rewards": [], "n_pred": [], "empty": 0, "giant": 0, "updated": 0}
        for e in entries[(step - 1) * images_per_step: step * images_per_step]:
            img, gt = load_image(e), _gt_boxes(e)
            model.eval()
            outs = generate(model, processor, [img] * G, max_new_tokens=cfg["max_completion_length"],
                            do_sample=True, temperature=1.0, top_p=1.0, top_k=0)
            r = torch.tensor([compute_reward(o, gt) for o in outs])
            adv = (r - r.mean()) / (r.std() + 1e-4)
            for o in outs:
                pred = parse_output(o) or []
                rec["n_pred"].append(len(pred))
                rec["empty"] += o.strip() == "[]"
                rec["giant"] += any((b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1]) > 0.25e6
                                    for b in pred)
            rec["rewards"] += r.tolist()
            if r.std() == 0:
                continue  # cả nhóm cùng reward ⇒ advantage 0, không có tín hiệu
            model.train()
            for o, a in zip(outs, adv):
                batch = {k: v.to(model.device) for k, v in
                         collate([encode_example(processor, img, PROMPT, o)],
                                 processor.tokenizer.pad_token_id).items()}
                with torch.autocast("cuda", dtype=torch.float16):
                    nll = model(**batch, use_cache=False).loss  # = -mean log π(o)
                scaler.scale(a.item() * nll / (G * images_per_step)).backward()
            rec["updated"] += 1
        if rec["updated"]:  # GradScaler.step lỗi nếu không có gradient nào
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        rs = rec["rewards"]
        rec.update(reward_mean=sum(rs) / len(rs), reward_std=float(torch.tensor(rs).std()),
                   mean_n_pred=sum(rec["n_pred"]) / len(rec["n_pred"]), minutes=(time.time() - t0) / 60)
        log.append(rec)
        if step % log_every == 0:
            print(f"step {step}/{steps} reward={rec['reward_mean']:.3f}±{rec['reward_std']:.3f} "
                  f"n_pred={rec['mean_n_pred']:.2f} empty={rec['empty']} giant={rec['giant']} "
                  f"{rec['minutes']:.1f}m")
    save_trained(model, run_dir, False)
    proj = Path(init_dir) / "projector.pt"
    if proj.exists():  # giữ projector của SFT để nạp lại đúng model
        import shutil
        shutil.copy(proj, run_dir / "projector.pt")
    run = {"name": "E5", "trainable_params": sum(p.numel() for p in params),
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
           "train_minutes": round((time.time() - t0) / 60, 1), "init": str(init_dir), "log": log}
    (run_dir / "run.json").write_text(json.dumps(run, indent=1))
    del model, opt, params
    cleanup()
    return run
