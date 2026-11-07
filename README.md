# LLM-as-a-Judge for HRVS

Đánh giá câu trả lời VLM bằng cách so sánh câu hỏi, đáp án ground truth và prediction với judge Qwen. Judge trả về `1` (khớp) hoặc `0` (không khớp), không nhận ảnh hay bounding box.

Code, annotations, predictions và kết quả đánh giá nằm trong [`llm_eval/`](llm_eval/). Xem [hướng dẫn đầy đủ](llm_eval/README.md) để chạy local, chạy Modal, resume và lấy báo cáo.

## Chạy trên Modal

Từ thư mục gốc repository, dùng PowerShell với Python 3.10 trở lên:

```powershell
python -m venv .venv-modal
.\.venv-modal\Scripts\python.exe -m pip install -r llm_eval/requirements-modal.txt
.\.venv-modal\Scripts\python.exe -X utf8 -m modal setup
```

Kiểm tra dữ liệu trong `our_method` mà không dùng GPU:

```powershell
.\.venv-modal\Scripts\python.exe -X utf8 llm_eval/eval_matching.py --config llm_eval/configs/qwen_matching_modal.json --predictions llm_eval/predictions/our_method --allow-partial --dry-run
```

Chấm toàn bộ file trong `our_method` bằng Qwen3-8B trên Modal:

```powershell
.\.venv-modal\Scripts\python.exe -X utf8 -m modal run llm_eval/modal_eval.py --predictions predictions/our_method --allow-partial --run-name our-method-qwen3-8b
```

App ưu tiên một GPU L4 và dùng A10 dự phòng; weights và checkpoint lưu trong Modal Volumes. GPU và lưu trữ sử dụng tài khoản Modal của người chạy. Báo cáo tự tải về `llm_eval/results/modal/<run-name>/`. Dùng run name mới để có lượt chấm riêng, hoặc giữ nguyên để resume.

Một số prediction chưa đủ 1.060 QA hoặc có câu trả lời rỗng, vì vậy cần `--allow-partial`. Báo cáo phân biệt accuracy trên toàn bộ GT, accuracy trên các câu đã chấm và coverage. Level `UNKNOWN` được quy về `L1`.

## Kiểm thử

```powershell
.\.venv-modal\Scripts\python.exe -X utf8 -m unittest discover -s llm_eval/tests -v
```

Các test chạy offline bằng mock, không gọi API judge hoặc GPU. `eval_matching.py` dùng thư viện chuẩn Python; vLLM và Transformers được cài bên trong Modal image khi chạy trên cloud.
