import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import datasets
import numpy as np
import hashlib
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from safetensors.torch import save_file
from argparse import ArgumentParser
from multiprocessing import Pool
from functools import partial

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _compute_user_id(item, data_name: str) -> int:
    """
    Derive a stable group id ("user_id") for each example.

    - If the raw dataset provides `prompt_id`, use it as the key.
    - Otherwise fall back to `prompt` text.
    - Otherwise fall back to the first user message content (if available).
    """
    if isinstance(item, dict) and item.get("prompt_id") is not None:
        key = str(item["prompt_id"])
    elif isinstance(item, dict) and item.get("prompt") is not None:
        key = str(item["prompt"])
    elif isinstance(item, dict) and isinstance(item.get("messages"), list) and len(item["messages"]) > 0:
        first_user = next((m for m in item["messages"] if m.get("role") == "user"), None)
        if first_user is not None and first_user.get("content") is not None:
            key = str(first_user["content"])
        else:
            key = str(item["messages"][0].get("content", ""))
    else:
        key = f"{data_name}:unknown"

    digest = hashlib.sha256(key.encode("utf-8")).digest()
    # keep within signed int64 range
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**63 - 1)


def process_item(indexed_item, data_name):
    index, item = indexed_item
    user_id = _compute_user_id(item, data_name)
    if data_name == 'hs':
        return [{
            "messages": [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["response"]}
            ],
            "rating": [item['helpfulness']],
            "user_id": user_id,
        }], index

    elif data_name == 'saferlhf':
        return [{
            "messages": [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["response_0"]}
            ],
            "rating": [item["response_0_severity_level"]],
            "user_id": user_id,
        }, {
            "messages": [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["response_1"]}
            ],
            "rating": [item["response_1_severity_level"]],
            "user_id": user_id,
        }], index

    elif data_name == 'ufb':
        return [{
            "messages": item["chosen"],
            "rating": [item["score_chosen"]],
            "user_id": user_id,
        }, {
            "messages": item["rejected"],
            "rating": [item["score_rejected"]],
            "user_id": user_id,
        }], index

    elif data_name == 'armorm':
        attributes = [
            "helpsteer-helpfulness", "helpsteer-correctness", "helpsteer-coherence",
            "helpsteer-complexity", "helpsteer-verbosity", "ultrafeedback-overall_score",
            "ultrafeedback-instruction_following", "ultrafeedback-truthfulness",
            "ultrafeedback-honesty", "ultrafeedback-helpfulness", "beavertails-is_safe",
            "prometheus-score", "argilla-overall_quality", "argilla-judge_lm",
            "code-complexity", "code-style", "code-explanation",
            "code-instruction-following", "code-readability",
        ]
        return [{
            "messages": item["messages"],
            "rating": [item.get(attr) for attr in attributes],
            "user_id": user_id,
        }], index


def process_dataset(data_name, subset_name, base_path="./rawdata", num_workers=30):
    dataset_path = os.path.join(base_path, data_name)
    data = datasets.load_from_disk(dataset_path)[subset_name]
    indexed_data = list(enumerate(data))

    worker_func = partial(process_item, data_name=data_name)
    with Pool(processes=num_workers) as pool:
        processed_list = list(tqdm(pool.imap(worker_func, indexed_data), total=len(indexed_data), desc="Processing raw data"))
    processed_list = sorted(processed_list, key=lambda x: x[1])
    processed_list = [x[0] for x in processed_list]
    processed_list = sum(processed_list, [])

    print(f"Finished processing. Total items created: {len(processed_list)}")
    return processed_list


parser = ArgumentParser()
parser.add_argument("--model_path", type=str, default="/mnt/tidalfs-bdsz01/dataset/llm_dataset/plc_data/ckpts/FsfairX-LLaMA3-RM-v0.1")
parser.add_argument("--output_dir", type=str, default="/mnt/tidalfs-bdsz01/dataset/llm_dataset/plc_data/TrustworthRLHF/embeddings/normal")
parser.add_argument("--data_name", type=str, default="ufb")
parser.add_argument("--subset", type=str, default="train")
parser.add_argument("--num_workers", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_samples", type=int, default=-1)

args = parser.parse_args() 

rm = AutoModel.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
rm_tokenizer = AutoTokenizer.from_pretrained(args.model_path)

ds = process_dataset(args.data_name, args.subset, num_workers=args.num_workers)
ds = ds[:args.max_samples] if args.max_samples > 0 else ds

def collate_fn(batch):
    # 这个函数负责将单个样本聚合成一个批次
    formatted_texts = []
    batch_labels = []
    batch_user_ids = []

    for example in batch:
        # 格式化文本
        if args.model_path.endswith("FsfairX-LLaMA3-RM-v0.1"):
            conv_formatted = rm_tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            ).replace(rm_tokenizer.bos_token, "")
        else:
            conv_formatted = rm_tokenizer.apply_chat_template(
                example["messages"], tokenize=False
            )
        formatted_texts.append(conv_formatted)

        # 提取标签
        label = example["rating"]
        label = [np.nan if l is None else l for l in label]
        batch_labels.append(label)
        batch_user_ids.append(example.get("user_id", -1))

    # 对整个批次的文本进行tokenize，并进行padding
    tokenized_batch = rm_tokenizer(
        formatted_texts,
        return_tensors="pt",
        padding=True, # 关键：开启padding
        truncation=True, # 关键：开启截断，防止超长
        max_length=4096 # 根据模型和显存调整
    )
    return (
        tokenized_batch,
        torch.tensor(batch_labels, dtype=torch.float32),
        torch.tensor(batch_user_ids, dtype=torch.long),
    )

dataloader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=4)

embeddings = []
labels = []
user_ids = []
for batch_tokenized, batch_labels, batch_user_ids in tqdm(dataloader, desc="Processing dataset in batches"):
    # 将数据移动到主GPU，DataParallel会自动分发
    batch_tokenized = {k: v.to('cuda') for k, v in batch_tokenized.items()}

    with torch.no_grad():
        outputs = rm(**batch_tokenized)

        # 提取每个样本的最后一个token的embedding
        # `attention_mask`帮助我们找到每个序列真正的结尾
        last_token_indices = batch_tokenized['attention_mask'].sum(dim=1) - 1
        last_hidden_states = outputs.last_hidden_state

        # 使用高级索引一次性提取所有样本的最后一个token的embedding
        batch_embeddings = last_hidden_states[torch.arange(last_hidden_states.shape[0]), last_token_indices]

        embeddings.append(batch_embeddings.cpu())

    labels.append(batch_labels)
    user_ids.append(batch_user_ids)

# 合并所有批次的结果
embeddings = torch.cat(embeddings, dim=0)
labels = torch.cat(labels, dim=0)
labels = labels.squeeze(1)
user_ids = torch.cat(user_ids, dim=0)

model_name = args.model_path.split("/")[-1]
os.makedirs(args.output_dir, exist_ok=True)
save_path = f"{args.output_dir}/{model_name}_{args.data_name}_{args.subset}.safetensors"
save_file({"embeddings": embeddings, "labels": labels, "user_id": user_ids}, save_path)
print(f"Saved embeddings to {save_path}")
