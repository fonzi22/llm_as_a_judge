# Đánh giá baseline VLM bằng LLM-as-a-judge

Script `eval_matching.py` so sánh **câu hỏi + đáp án GT + câu trả lời của baseline** bằng prompt, trả về `1` (khớp) hoặc `0` (không khớp). GT được lấy từ `annotations.json`; ảnh và bbox không gửi cho judge. Mặc định dùng vLLM để nạp và chạy model judge trực tiếp trên máy local, không cần server HTTP hay API key.

## Chạy trên Modal (Windows / PowerShell)

`modal_eval.py` chạy **judge [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)** bằng vLLM trên một GPU L4 của Modal, dùng BF16 và tắt thinking. Máy local chỉ cần Python và Modal client. Cấu hình riêng `configs/qwen_matching_modal.json` dùng `gpu_memory_utilization: 0.85`; cấu hình local hiện có vẫn giữ nguyên. Run mặc định là `qwen3_8b_matching`, tải kết quả về `llm_eval/results/modal/qwen3_8b_matching/`.

Nếu L4 hết tài nguyên, app tự chọn một GPU A10 dự phòng. Hai loại GPU đều có 24 GB VRAM. Các lệnh bên dưới bật UTF-8 để tránh lỗi hiển thị Unicode trong PowerShell trên Windows.

Từ thư mục gốc workspace, tạo môi trường và đăng nhập Modal một lần:

```powershell
python -m venv .venv-modal
.\.venv-modal\Scripts\python.exe -m pip install -r llm_eval/requirements-modal.txt
.\.venv-modal\Scripts\python.exe -X utf8 -m modal setup
```

Kiểm tra toàn bộ dữ liệu offline trước khi chạy (không cần đăng nhập, GPU hay weights):

```powershell
.\.venv-modal\Scripts\python.exe llm_eval/eval_matching.py --config llm_eval/configs/qwen_matching_modal.json --dry-run
```

Smoke test 10 QA có sẵn, sau đó chấm toàn bộ baseline:

```powershell
.\.venv-modal\Scripts\python.exe -X utf8 -m modal run llm_eval/modal_eval.py --predictions predictions/huy/results/llava-ov-7b/test/llava-ov-7b.limit10.jsonl --allow-partial --run-name qwen3-8b-smoke10
.\.venv-modal\Scripts\python.exe -X utf8 -m modal run llm_eval/modal_eval.py
```

Chỉ chấm một baseline:

```powershell
.\.venv-modal\Scripts\python.exe -X utf8 -m modal run llm_eval/modal_eval.py --predictions predictions/nam/full_8b.jsonl --run-name nam-qwen3-8b
```

- `--predictions`: một file hoặc thư mục nằm trong `llm_eval/predictions/`; nhận đường dẫn từ workspace hoặc tương đối với `llm_eval/`. Mặc định tìm tất cả prediction full và loại bản trùng như luồng local. File test được chỉ định trực tiếp không cần `--include-tests`.
- `--allow-partial`, `--include-tests`: cùng ý nghĩa với `eval_matching.py`.
- `--run-name`: thư mục checkpoint trên Volume. Chạy lại cùng tên để resume; đổi tên nếu thay judge/prompt/config. Dùng tên riêng cho smoke test, full benchmark và các job chạy đồng thời; không chạy hai job cùng tên.
- `--output-dir`: thư mục tải báo cáo về máy. Mặc định `llm_eval/results/modal/<run-name>/`, gồm `summary.csv`, `summary.json` và judgments từng baseline.
- `--dry-run`: chỉ kiểm tra dữ liệu trong entrypoint, không gọi worker GPU; Modal CLI vẫn cần tài khoản và có thể chuẩn bị image. Để kiểm tra hoàn toàn offline, dùng lệnh `eval_matching.py --dry-run` bên trên.
- `--download-only`: chỉ tải lại checkpoint/báo cáo đã lưu cho `--run-name`, không chạy judge.

Ví dụ lấy lại kết quả sau khi mất kết nối:

```powershell
.\.venv-modal\Scripts\python.exe -X utf8 -m modal run llm_eval/modal_eval.py --run-name qwen3-8b-smoke10 --download-only
# Hoặc tải trực tiếp bằng CLI, không khởi tạo app/image:
.\.venv-modal\Scripts\python.exe -X utf8 -m modal volume get hrvs-llm-eval-results /qwen3-8b-smoke10 ./modal-qwen3-8b-smoke10
```

App tự đưa `annotations.json`, config, evaluator và các file JSON/JSONL trong `predictions/` lên Modal; không cần upload ảnh. Dữ liệu câu hỏi, GT và prediction được xử lý trên Modal. Lần đầu worker tải weights từ Hugging Face vào Volume `hrvs-llm-eval-hf-cache`; những lần sau dùng lại cache. Chọn model công khai dạng text và vừa một L4 nếu sửa `model` trong config.

Checkpoint/báo cáo nằm trên Volume `hrvs-llm-eval-results`, được commit khi evaluator kết thúc hoặc báo lỗi; evaluator vẫn ghi checkpoint sau mỗi QA. Nếu worker bị ngắt đột ngột, resume từ checkpoint gần nhất đã được Volume lưu. Chạy GPU và lưu Volume có tính phí theo tài khoản Modal. Cách đưa file vào worker và lưu kết quả dùng [Modal Images](https://modal.com/docs/guide/images) và [Modal Volumes](https://modal.com/docs/guide/volumes).

Kiểm thử cả evaluator và tích hợp Modal bằng mock, không gọi cloud/GPU:

```powershell
.\.venv-modal\Scripts\python.exe -m unittest discover -s llm_eval/tests -v
```

## Chạy hoàn toàn local

Cấu hình mặc định `configs/qwen_matching.json` dùng `backend: "local"`. Script nạp model trực tiếp bằng Python API của vLLM và dùng lại model cho mọi file prediction. Không có `base_url`, không mở server, không gửi câu hỏi/GT/prediction ra dịch vụ bên ngoài. Cách chạy này dựa trên [offline inference của vLLM](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/llm/).

Sửa trường `model` thành **đường dẫn tuyệt đối tới thư mục model đã tải đầy đủ**, hoặc tên Hugging Face model đã có đầy đủ trong cache. Mặc định giữ `Qwen/Qwen3-4B`; nếu chưa có weights, script dừng và báo thiếu model. Script không tự tải model. Có thể thêm `cache_dir` vào config để dùng một thư mục Hugging Face cache khác.

Nếu model chưa có trên máy, tải weights một lần (cần mạng cho bước tải):

```bash
./venv-vllm/bin/hf download Qwen/Qwen3-4B --local-dir models/Qwen3-4B
```

Sau đó đặt `model` trong config thành `/home/ubuntu/workspace/density/models/Qwen3-4B`. Các lần chấm sau đọc weights local, không tải lại và không gửi dữ liệu đánh giá ra ngoài.

- `temperature`, `max_tokens`, `max_retries`: cấu hình sinh câu trả lời/retry.
- `chat_template_kwargs.enable_thinking: false`: tắt thinking cho Qwen3-4B để trả lời trực tiếp 0/1, theo [model card của Qwen](https://huggingface.co/Qwen/Qwen3-4B).
- `vllm.dtype`: kiểu dữ liệu weights (`bfloat16` trong cấu hình mẫu).
- `vllm.tensor_parallel_size`: số GPU dùng cho model judge.
- `vllm.gpu_memory_utilization`: phần VRAM vLLM dự kiến dùng (`0.15`); GPU phải còn đủ bộ nhớ cho weights và KV cache.
- `vllm.max_model_len`: giới hạn tổng token đầu vào + đầu ra (`8192`).
- `vllm.max_num_seqs`, `vllm.enforce_eager`: tham số engine; có thể bổ sung các tham số khởi tạo vLLM khác trong mục `vllm`.

Qwen3-4B BF16 riêng weights khoảng 8 GB; cần thêm bộ nhớ cho inference. Với GPU khoảng 96 GB của máy này, cấu hình `gpu_memory_utilization: 0.15` dành khoảng 14 GB cho engine judge. Chỉ chạy khi GPU còn đủ VRAM, hoặc chọn một model judge local khác phù hợp với tài nguyên. Script không dừng các tiến trình đang dùng GPU.

Workspace hiện có môi trường `venv-vllm`. Từ thư mục gốc workspace:

```bash
# Kiểm tra dữ liệu; không nạp model/GPU và không ghi kết quả.
python3 llm_eval/eval_matching.py --dry-run

# Chấm tất cả prediction bằng model đã có trong cache theo config.
./venv-vllm/bin/python llm_eval/eval_matching.py

# Dùng thư mục weights local có sẵn (thay bằng đường dẫn thực tế).
./venv-vllm/bin/python llm_eval/eval_matching.py \
  --judge-model /absolute/path/to/local-judge-model

# Chỉ chấm một baseline.
./venv-vllm/bin/python llm_eval/eval_matching.py \
  --predictions llm_eval/predictions/nam/full_8b.jsonl \
  --judge-model /absolute/path/to/local-judge-model

# Smoke test 10 QA có sẵn.
./venv-vllm/bin/python llm_eval/eval_matching.py \
  --predictions llm_eval/predictions/huy/results/llava-ov-7b/test/llava-ov-7b.limit10.jsonl \
  --judge-model /absolute/path/to/local-judge-model \
  --allow-partial \
  --output-dir /tmp/hrvs-judge-smoke
```
