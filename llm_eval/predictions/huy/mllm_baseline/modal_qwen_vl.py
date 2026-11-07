"""
HRVS MLLM baseline — zero-shot inference Qwen-VL trên Modal (vLLM, BF16).

Prompt (đúng spec nhóm):   <image>\n{question}
  -> user turn = [image, text=question], không system prompt, không few-shot.
Output: JSONL free-form text, 1 dòng / câu hỏi, có resume (chạy lại sẽ bỏ qua qa_id đã xong).

Chạy:
  modal run modal_qwen_vl.py --model qwen3-vl-8b --limit 20     # smoke test
  modal run modal_qwen_vl.py --model qwen3-vl-8b                # full 1060 câu
  modal run modal_qwen_vl.py --model llava-ov-7b
"""
import json
import os
import time

import modal

# --------------------------------------------------------------------------- config
# key -> (HF repo, GPU mặc định, tensor_parallel)
MODELS = {
    "qwen3-vl-2b":   ("Qwen/Qwen3-VL-2B-Instruct",   "H100",   1),
    "qwen3-vl-4b":   ("Qwen/Qwen3-VL-4B-Instruct",   "H100",   1),
    "qwen3-vl-8b":   ("Qwen/Qwen3-VL-8B-Instruct",   "H100",   1),
    "qwen3-vl-32b":  ("Qwen/Qwen3-VL-32B-Instruct",  "H100:2", 2),
    "qwen2.5-vl-7b": ("Qwen/Qwen2.5-VL-7B-Instruct", "H100",   1),
    # LLaVA-OneVision-7B (SigLIP-384 + Qwen2-7B, AnyRes tối đa 2304x2304, ~7.3k visual token)
    "llava-ov-7b":   ("llava-hf/llava-onevision-qwen2-7b-ov-hf", "H100", 1),
}
# Visual token tối đa ~16384 (processor mặc định: Qwen3-VL 16.7MP/32², Qwen2.5-VL 12.8MP/28²)
# + câu hỏi + output -> 20480 là đủ, không cắt ảnh.
MAX_MODEL_LEN = 20480
VLLM_VERSION = "0.11.0"          # bản đầu tiên hỗ trợ Qwen3-VL
TRANSFORMERS_VERSION = "4.57.1"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        f"vllm=={VLLM_VERSION}",
        f"transformers=={TRANSFORMERS_VERSION}",
        "pillow",
        "huggingface_hub[hf_transfer]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
)

data_vol = modal.Volume.from_name("hrvs-data", create_if_missing=True)
hf_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("hrvs-outputs", create_if_missing=True)

app = modal.App("hrvs-mllm-baseline")


# --------------------------------------------------------------------------- helpers (thuần Python, test được không cần GPU)
def find_data_root(base: str) -> str:
    """Tìm thư mục chứa annotations.json (chịu được việc `modal volume put` lồng thêm 1-2 cấp)."""
    for dirpath, dirnames, filenames in os.walk(base):
        if dirpath[len(base):].count(os.sep) > 3:
            dirnames[:] = []
            continue
        if "annotations.json" in filenames:
            return dirpath
    raise FileNotFoundError(f"Không thấy annotations.json dưới {base}")


def load_items(root: str, only_completed: bool = False):
    """Trả về list (image_meta, qa) theo thứ tự image_id -> qa_id (gom theo ảnh để decode 1 lần)."""
    ann = json.load(open(os.path.join(root, "annotations.json"), encoding="utf-8"))
    items = []
    for im in sorted(ann["images"], key=lambda x: x["image_id"]):
        if only_completed and im.get("qa_status") != "completed":
            continue
        for qa in sorted(im.get("qa_annotations") or [], key=lambda q: q["qa_id"]):
            if (qa.get("question") or "").strip():
                items.append((im, qa))
    return items


def build_row(im, qa, model_key, repo, pred_text, meta):
    return {
        "qa_id": qa["qa_id"],
        "question_id": qa.get("question_id"),
        "image_id": im["image_id"],
        "file_name": im["file_name"],
        "model": model_key,
        "model_repo": repo,
        "question_type": qa.get("question_type"),
        "task_type": qa.get("task_type"),
        "target_group": qa.get("target_group"),
        "density_group": im.get("density_group"),
        "density_category": qa.get("density_category"),
        "level": im.get("level"),
        "qa_status": im.get("qa_status"),
        "question": qa["question"],
        "prediction": pred_text,          # free-form text, KHÔNG hậu xử lý
        "gt_answer": qa.get("answer"),
        **meta,
    }


# --------------------------------------------------------------------------- inference
@app.function(
    image=image,
    gpu="H100",                      # override bằng .with_options(gpu=...) ở local_entrypoint
    timeout=4 * 3600,
    volumes={"/data": data_vol, "/root/.cache/huggingface": hf_vol, "/outputs": out_vol},
)
def run(model_key: str, limit: int = 0, only_completed: bool = False, exif_transpose: bool = True,
        max_tokens: int = 1024, batch_size: int = 32) -> str:
    import importlib.metadata as md

    from PIL import Image, ImageOps
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    Image.MAX_IMAGE_PIXELS = None    # có ảnh 76.7MP, mặc định PIL sẽ cảnh báo/chặn
    repo, _, tp = MODELS[model_key]
    root = find_data_root("/data")
    items = load_items(root, only_completed)
    if limit:
        items = items[:limit]

    out_name = f"{model_key}.jsonl" if not limit else f"{model_key}.limit{limit}.jsonl"
    out_path = f"/outputs/{out_name}"
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("prediction") is not None:
                        done.add(r["qa_id"])
                except json.JSONDecodeError:
                    pass
    todo = [x for x in items if x[1]["qa_id"] not in done]
    print(f"[{model_key}] data_root={root} total={len(items)} done={len(done)} todo={len(todo)}")
    if not todo:
        return out_name

    processor = AutoProcessor.from_pretrained(repo)
    llm = LLM(
        model=repo,
        dtype="bfloat16",
        tensor_parallel_size=tp,
        max_model_len=MAX_MODEL_LEN,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=0.90,
        seed=0,
    )
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens, seed=0)
    hf_vol.commit()                  # cache weights cho lần sau

    ip = processor.image_processor
    config = {
        "model": model_key, "model_repo": repo, "dtype": "bfloat16", "tensor_parallel": tp,
        "prompt_format": "<image>\\n{question}  (chat template của model, không system prompt)",
        "sampling": {"temperature": 0.0, "top_p": 1.0, "max_tokens": max_tokens, "seed": 0},
        "max_model_len": MAX_MODEL_LEN, "exif_transpose": exif_transpose, "only_completed": only_completed,
        "image_processor": {k: getattr(ip, k, None) for k in ("size", "patch_size", "merge_size", "min_pixels", "max_pixels",
                                                            "image_grid_pinpoints", "vision_aspect_ratio")},
        "vllm": md.version("vllm"), "transformers": md.version("transformers"),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(f"/outputs/{out_name[:-6]}.config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(config, default=str))

    cache = {}

    def load_image(im):
        key = im["image_id"]
        if key not in cache:
            cache.clear()            # items đã sort theo ảnh -> chỉ giữ ảnh hiện tại
            img = Image.open(os.path.join(root, im["file_path"]))
            orient = img.getexif().get(274, 1)
            if exif_transpose:
                img = ImageOps.exif_transpose(img)
            cache[key] = (img.convert("RGB"), orient)
        return cache[key]

    def make_req(im, qa):
        img, _ = load_image(im)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": qa["question"]}]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt, "multi_modal_data": {"image": img}}

    def meta_of(im, out=None, err=None):
        img, orient = load_image(im)
        m = {"exif_orientation": orient, "exif_transposed": bool(exif_transpose and orient not in (1, None)),
             "input_size": list(img.size)}
        if out is not None:
            o = out.outputs[0]
            m.update(finish_reason=o.finish_reason, n_prompt_tokens=len(out.prompt_token_ids),
                     n_output_tokens=len(o.token_ids))
        if err:
            m["error"] = err
        return m

    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as fout:
        for b in range(0, len(todo), batch_size):
            batch = todo[b:b + batch_size]
            rows = []
            try:
                outs = llm.generate([make_req(im, qa) for im, qa in batch], sp, use_tqdm=False)
                for (im, qa), out in zip(batch, outs):
                    rows.append(build_row(im, qa, model_key, repo, out.outputs[0].text, meta_of(im, out)))
            except Exception as e:   # 1 mẫu lỗi -> chạy lẻ từng mẫu để không mất cả batch
                print(f"batch {b} lỗi ({e!r}), chạy lẻ")
                for im, qa in batch:
                    try:
                        out = llm.generate([make_req(im, qa)], sp, use_tqdm=False)[0]
                        rows.append(build_row(im, qa, model_key, repo, out.outputs[0].text, meta_of(im, out)))
                    except Exception as e2:
                        rows.append(build_row(im, qa, model_key, repo, None, meta_of(im, err=repr(e2))))
            for r in rows:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            fout.flush()
            out_vol.commit()         # checkpoint: container chết vẫn resume được
            n = b + len(batch)
            el = time.time() - t0
            print(f"[{model_key}] {n}/{len(todo)}  {el/60:.1f} min  ETA {el/n*(len(todo)-n)/60:.1f} min")
    return out_name


# --------------------------------------------------------------------------- CLI
@app.local_entrypoint()
def main(model: str = "qwen3-vl-8b", limit: int = 0, gpu: str = "", only_completed: bool = False,
         exif_transpose: bool = True, max_tokens: int = 1024, out_dir: str = "results"):
    if model not in MODELS:
        raise SystemExit(f"model phải là một trong: {list(MODELS)}")
    _, default_gpu, _ = MODELS[model]
    fn = run.with_options(gpu=gpu or default_gpu)
    name = fn.remote(model, limit, only_completed, exif_transpose, max_tokens)

    os.makedirs(out_dir, exist_ok=True)
    for fname in (name, name[:-6] + ".config.json"):
        try:
            with open(os.path.join(out_dir, fname), "wb") as f:
                for chunk in out_vol.read_file(fname):
                    f.write(chunk)
        except Exception as e:
            print(f"Không tải được {fname}: {e!r} -> dùng: modal volume get hrvs-outputs {fname} {out_dir}")
    path = os.path.join(out_dir, name)
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    n_err = sum(r.get("prediction") is None for r in rows)
    n_trunc = sum(r.get("finish_reason") == "length" for r in rows)
    print(f"OK: {path}  rows={len(rows)}  unique_qa={len({r['qa_id'] for r in rows})}  error={n_err}  truncated={n_trunc}")
