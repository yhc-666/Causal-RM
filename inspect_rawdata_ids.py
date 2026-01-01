import os
from argparse import ArgumentParser

import datasets


def _truncate(text: str, max_len: int = 120) -> str:
    text = text.replace("\n", "\\n")
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _print_example(idx: int, ex: dict, fields: list[str]):
    out = {"_idx": idx}
    for f in fields:
        if f not in ex:
            continue
        v = ex[f]
        if isinstance(v, str):
            out[f] = _truncate(v)
        else:
            out[f] = v
    print(out)


def main():
    parser = ArgumentParser(description="Inspect whether rawdata contains prompt_id/user_id-like fields.")
    parser.add_argument("--data_root", type=str, default="./rawdata")
    parser.add_argument("--datasets", type=str, default="hs,ufb,saferlhf")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num", type=int, default=3)
    parser.add_argument("--stats", action="store_true", help="Compute simple prompt/prompt_id uniqueness stats (may be slow on huge datasets).")
    args = parser.parse_args()

    dataset_names = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for name in dataset_names:
        path = os.path.join(args.data_root, name)
        print("\n" + "=" * 80)
        print(f"dataset={name} path={path}")
        if not os.path.exists(path):
            print("  (missing)")
            continue

        ds = datasets.load_from_disk(path)
        if isinstance(ds, datasets.DatasetDict):
            splits = list(ds.keys())
            split = args.split if args.split in ds else splits[0]
            d = ds[split]
            print(f"  splits={splits} using_split={split} rows={d.num_rows}")
        else:
            d = ds
            print(f"  rows={d.num_rows}")

        cols = d.column_names
        print(f"  columns={cols}")
        id_like = [c for c in cols if any(k in c.lower() for k in ["prompt_id", "user_id", "uid", "user", "id"])]
        print(f"  id_like_columns={id_like}")

        if args.stats:
            try:
                if "prompt_id" in cols:
                    prompt_ids = d["prompt_id"]
                    print(f"  stats: unique(prompt_id)={len(set(prompt_ids))} / rows={len(prompt_ids)}")
                if "prompt" in cols:
                    from collections import Counter

                    prompts = d["prompt"]
                    c = Counter(prompts)
                    mult = sum(1 for _, v in c.items() if v > 1)
                    maxv = max(c.values()) if c else 0
                    print(f"  stats: unique(prompt)={len(c)} / rows={len(prompts)} / prompts_with>1={mult} / max_per_prompt={maxv}")
            except Exception as e:
                print(f"  stats: error={e}")

        # Choose a compact field set per dataset for printing.
        if name == "ufb":
            fields = ["prompt_id", "prompt", "score_chosen", "score_rejected"]
        elif name == "saferlhf":
            fields = [
                "prompt",
                "response_0_severity_level",
                "response_1_severity_level",
                "better_response_id",
                "safer_response_id",
                "response_0_sha256",
                "response_1_sha256",
            ]
        elif name == "hs":
            fields = ["prompt", "response", "helpfulness", "correctness", "coherence", "complexity", "verbosity"]
        else:
            fields = cols[: min(8, len(cols))]

        print(f"  showing {min(args.num, d.num_rows)} examples:")
        for i in range(min(args.num, d.num_rows)):
            ex = d[i]
            _print_example(i, ex, fields)


if __name__ == "__main__":
    main()
