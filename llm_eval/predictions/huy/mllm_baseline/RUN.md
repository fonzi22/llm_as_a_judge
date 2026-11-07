# HRVS MLLM baseline — Qwen3-VL-8B & LLaVA-OV-7B (zero-shot, Modal)

Mở **PowerShell**, copy từng khối, chạy lần lượt. Mỗi khối đều tự `cd` vào đúng thư mục, mở cửa sổ mới vẫn chạy được.

---

## Bước 1 — Cài Modal + đăng nhập (1 lần)
```powershell
cd C:\Users\anhhu\Downloads\CVPR_2027\mllm_baseline
$env:PYTHONUTF8 = "1"
python -m pip install -U modal
python -m modal setup
```
Trình duyệt mở ra → login → quay lại PowerShell. Kiểm tra:
```powershell
python -m modal profile current
```

## Bước 2 — Upload data lên Modal (~3 GB, 1 lần)
```powershell
cd C:\Users\anhhu\Downloads\CVPR_2027\mllm_baseline
$env:PYTHONUTF8 = "1"
python prepare_data.py "C:\Users\anhhu\Downloads\CVPR_2027\data"
```
✅ Đúng khi dòng cuối là: `[OK] Volume 'hrvs-data' du 503 file (501 anh), khop size voi may.`
Bị đứt → chạy lại đúng lệnh trên, chỉ upload phần còn thiếu.

## Bước 3 — Chạy thử 10 câu: Qwen3-VL-8B
```powershell
cd C:\Users\anhhu\Downloads\CVPR_2027\mllm_baseline
$env:PYTHONUTF8 = "1"
python -m modal run modal_qwen_vl.py --model qwen3-vl-8b --limit 10
```
Lần đầu ~10–15 phút (build image + tải ~17.5 GB weights).
✅ Dòng cuối: `OK: results\qwen3-vl-8b.limit10.jsonl  rows=10  unique_qa=10  error=0  truncated=0`

## Bước 4 — Chạy thử 10 câu: LLaVA-OV-7B
```powershell
cd C:\Users\anhhu\Downloads\CVPR_2027\mllm_baseline
$env:PYTHONUTF8 = "1"
python -m modal run modal_qwen_vl.py --model llava-ov-7b --limit 10
```
✅ Dòng cuối: `OK: results\llava-ov-7b.limit10.jsonl  rows=10  unique_qa=10  error=0  truncated=0`

## Bước 5 — Xem kết quả chạy thử
```powershell
cd C:\Users\anhhu\Downloads\CVPR_2027\mllm_baseline
$env:PYTHONUTF8 = "1"
python show_results.py limit10
```
Kiểm tra:
- `prompt_tokens`: Qwen ~8k (ảnh 3840×2160), LLaVA ~7k
- `finish=stop` (không phải `length`)
- `PRED` trả lời đúng câu hỏi, không rỗng / lặp

👉 Gửi output bước 5 cho Claude xem trước khi chạy full.

---

## Bước 6 — Chạy full 1060 câu (mở 2 cửa sổ PowerShell, chạy song song)

**Cửa sổ 1 — Qwen3-VL-8B**
```powershell
cd C:\Users\anhhu\Downloads\CVPR_2027\mllm_baseline
$env:PYTHONUTF8 = "1"
python -m modal run modal_qwen_vl.py --model qwen3-vl-8b
```

**Cửa sổ 2 — LLaVA-OV-7B**
```powershell
cd C:\Users\anhhu\Downloads\CVPR_2027\mllm_baseline
$env:PYTHONUTF8 = "1"
python -m modal run modal_qwen_vl.py --model llava-ov-7b
```
- Giữ máy bật + có mạng tới khi xong (ước ~30–60 phút / model).
- Bị đứt / lỗi → chạy lại **đúng lệnh đó**, tự resume từ câu chưa xong.

## Bước 7 — Kiểm tra kết quả full
```powershell
cd C:\Users\anhhu\Downloads\CVPR_2027\mllm_baseline
$env:PYTHONUTF8 = "1"
python show_results.py
```
✅ Mỗi file `qwen3-vl-8b.jsonl`, `llava-ov-7b.jsonl`: `rows=1060  unique_qa=1060  error=0`, `truncated` ≈ 0.
File nộp nằm ở: `C:\Users\anhhu\Downloads\CVPR_2027\mllm_baseline\results\`

---

## Tuỳ chọn (chỉ khi cần)

**Chạy nền — tắt máy vẫn chạy trên Modal**
```powershell
cd C:\Users\anhhu\Downloads\CVPR_2027\mllm_baseline
python -m modal run --detach modal_qwen_vl.py --model qwen3-vl-8b
```
Xong (xem trên dashboard modal.com) thì tải file về:
```powershell
cd C:\Users\anhhu\Downloads\CVPR_2027\mllm_baseline
python -m modal volume get --force hrvs-outputs qwen3-vl-8b.jsonl results\
python -m modal volume get --force hrvs-outputs qwen3-vl-8b.config.json results\
```

**Chỉ chạy 656 câu đã QA xong**
```powershell
cd C:\Users\anhhu\Downloads\CVPR_2027\mllm_baseline
python -m modal run modal_qwen_vl.py --model qwen3-vl-8b --only-completed
```

**H100 hết chỗ → đổi GPU**
```powershell
cd C:\Users\anhhu\Downloads\CVPR_2027\mllm_baseline
python -m modal run modal_qwen_vl.py --model qwen3-vl-8b --gpu A100-80GB
```

**Xem file kết quả trên Modal**
```powershell
python -m modal volume ls hrvs-outputs /
```

---

## Output
`results\<model>.jsonl` — 1 dòng / câu hỏi: `qa_id, image_id, question_type, density_group, level, qa_status, question, prediction (free-form, không hậu xử lý), gt_answer, finish_reason, n_prompt_tokens, n_output_tokens, exif_orientation, input_size`
`results\<model>.config.json` — version vLLM/transformers, sampling, cấu hình image processor.
File chạy thử có đuôi `.limit10.jsonl`, không lẫn với file full.

## Cấu hình cố định
| | qwen3-vl-8b | llava-ov-7b |
|---|---|---|
| HF repo | `Qwen/Qwen3-VL-8B-Instruct` | `llava-hf/llava-onevision-qwen2-7b-ov-hf` |
| GPU | 1×H100 | 1×H100 |
| Độ phân giải | processor mặc định, ≤16.7 MP (~16k visual token) | AnyRes mặc định, lưới ≤2304×2304 (~7.3k token) |

Chung: BF16 · greedy (temperature 0, seed 0) · max_tokens 1024 · prompt `<image>\n{question}` qua chat template gốc, không system prompt, không few-shot · EXIF orientation được áp dụng (10 ảnh) · vLLM 0.11.0 + transformers 4.57.1.

(Các file `1_setup.bat` … `4_show_results.bat` làm đúng các bước trên nếu muốn double-click thay vì copy lệnh.)
