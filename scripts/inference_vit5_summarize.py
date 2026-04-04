import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from summarization.vit5_wrapper import DualVIT5Summarizer, VIT5Summarizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", choices=["vi", "en", "both"], default="both")
    p.add_argument(
        "--checkpoint_vi",
        type=Path,
        default=PROJECT_ROOT / "models" / "vit5-summarize-vi" / "best_checkpoint",
    )
    p.add_argument(
        "--checkpoint_en",
        type=Path,
        default=PROJECT_ROOT / "models" / "vit5-summarize-en" / "best_checkpoint",
    )
    p.add_argument("--text_vi", default="", help="Văn bản tiếng Việt (khi --lang vi hoặc both)")
    p.add_argument("--text_en", default="", help="English text (when --lang en or both)")
    p.add_argument("--text_file", type=Path, default=None)
    p.add_argument("--num_beams", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=128)
    args = p.parse_args()

    if args.text_file and args.text_file.is_file():
        with open(args.text_file, encoding="utf-8") as f:
            payload = json.load(f)
        text_vi = str(payload.get("text_vi", ""))
        text_en = str(payload.get("text_en", ""))
    else:
        text_vi = args.text_vi
        text_en = args.text_en

    if args.lang == "vi":
        s = VIT5Summarizer(args.checkpoint_vi)
        out = s.summarize(text_vi, num_beams=args.num_beams, max_new_tokens=args.max_new_tokens)
        print(json.dumps({"summary_vi": out["summary"]}, ensure_ascii=False, indent=2))
    elif args.lang == "en":
        s = VIT5Summarizer(args.checkpoint_en)
        out = s.summarize(text_en, num_beams=args.num_beams, max_new_tokens=args.max_new_tokens)
        print(json.dumps({"summary_en": out["summary"]}, ensure_ascii=False, indent=2))
    else:
        dual = DualVIT5Summarizer(args.checkpoint_vi, args.checkpoint_en)
        pair = dual.summarize_pair(
            text_vi,
            text_en,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
        )
        print(json.dumps(pair, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
