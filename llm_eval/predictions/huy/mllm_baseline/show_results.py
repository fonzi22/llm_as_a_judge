"""Tóm tắt file kết quả trong results/.
Dùng:  python show_results.py            # tóm tắt mọi file
       python show_results.py limit10    # tóm tắt + in chi tiết các file chứa 'limit10'
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
pattern = sys.argv[1] if len(sys.argv) > 1 else ""
files = sorted(f for f in glob.glob(os.path.join(HERE, "results", "**", "*.jsonl"), recursive=True) if pattern in os.path.basename(f))
if not files:
    sys.exit("Chua co file ket qua trong results\\")


def short(s, n=220):
    s = (s or "").replace("\n", " | ")
    return s if len(s) <= n else s[:n] + " ..."


for f in files:
    rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    toks = [r.get("n_prompt_tokens") or 0 for r in rows] or [0]
    print("=" * 100)
    print(f"{os.path.relpath(f, os.path.join(HERE, 'results'))}  rows={len(rows)}  unique_qa={len({r['qa_id'] for r in rows})}  "
          f"error={sum(r.get('prediction') is None for r in rows)}  "
          f"truncated={sum(r.get('finish_reason') == 'length' for r in rows)}  "
          f"empty={sum((r.get('prediction') or '').strip() == '' for r in rows)}  "
          f"prompt_tokens min/avg/max={min(toks)}/{sum(toks) // max(len(toks), 1)}/{max(toks)}")
    if pattern:
        for r in rows:
            print("-" * 100)
            print(f"{r['qa_id']}  [{r.get('question_type')}]  tokens={r.get('n_prompt_tokens')}  "
                  f"finish={r.get('finish_reason')}  input={r.get('input_size')}")
            print(f"  Q   : {short(r['question'])}")
            print(f"  PRED: {short(r.get('prediction'))}")
            print(f"  GT  : {short(r.get('gt_answer'))}")
            if r.get("error"):
                print(f"  ERR : {short(r['error'])}")
