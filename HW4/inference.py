"""
inference.py  –  HW4 Visual Instruction Tuning
Usage:
    python inference.py --mode compare --n 5
"""

import argparse
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
MAX_NEW_TOKENS = 128
NUM_TEST_SAMPLES = 5  # 從 ChartQA test split 自動抓幾筆


def load_test_cases(n: int = NUM_TEST_SAMPLES):
    """從 ChartQA test split 自動載入 n 筆樣本，無需本地圖片。"""
    ds = load_dataset("HuggingFaceM4/ChartQA", split=f"test[:{n}]")
    cases = []
    for sample in ds:
        label = sample["label"][0] if isinstance(sample["label"], list) else sample["label"]
        cases.append({
            "image":    sample["image"].convert("RGB"),  # PIL Image
            "question": sample["query"],
            "answer":   label,   # ground truth，供報告對照用
        })
    return cases


def save_test_images(test_cases, out_dir: str = "test_images"):
    """將測試樣本的圖片與問答存到 out_dir/，方便報告引用。"""
    import os, json
    os.makedirs(out_dir, exist_ok=True)
    meta = []
    for i, tc in enumerate(test_cases, 1):
        img_path = os.path.join(out_dir, f"case{i}.png")
        tc["image"].save(img_path)
        meta.append({
            "case": i,
            "image": img_path,
            "question": tc["question"],
            "answer": tc["answer"],
        })
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(test_cases)} images + metadata.json -> {out_dir}/")


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )


def load_base_model(bnb_config: BitsAndBytesConfig):
    print(f"Loading base model: {MODEL_ID}")
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def load_finetuned_model(adapter_path: str, bnb_config: BitsAndBytesConfig):
    print(f"Loading base model + adapter from: {adapter_path}")
    base_model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    processor = AutoProcessor.from_pretrained(adapter_path)
    return model, processor


def run_inference(model, processor, image: Image.Image, question: str) -> str:
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    # Decode only the newly generated tokens (skip the prompt)
    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = output_ids[:, prompt_len:]
    answer = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    return answer


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["baseline", "finetuned", "compare"],
        default="compare",
    )
    parser.add_argument("--adapter_path", type=str, default="./llava-chartqa-lora/final_adapter")
    parser.add_argument("--n", type=int, default=NUM_TEST_SAMPLES, help="測試樣本數量")
    args = parser.parse_args()

    print(f"Loading {args.n} test samples from ChartQA test split...")
    test_cases = load_test_cases(args.n)
    save_test_images(test_cases)

    bnb_config = build_bnb_config()

    if args.mode == "baseline":
        model, processor = load_base_model(bnb_config)
        model.eval()
        for i, tc in enumerate(test_cases, 1):
            answer = run_inference(model, processor, tc["image"], tc["question"])
            print(f"\n── Case {i} ──")
            print(f"[Q]           {tc['question']}")
            print(f"[Ground Truth]{tc['answer']}")
            print(f"[Base Model]  {answer}")

    elif args.mode == "finetuned":
        model, processor = load_finetuned_model(args.adapter_path, bnb_config)
        model.eval()
        for i, tc in enumerate(test_cases, 1):
            answer = run_inference(model, processor, tc["image"], tc["question"])
            print(f"\n── Case {i} ──")
            print(f"[Q]           {tc['question']}")
            print(f"[Ground Truth]{tc['answer']}")
            print(f"[Fine-tuned]  {answer}")

    elif args.mode == "compare":
        base_model, base_processor = load_base_model(bnb_config)
        base_model.eval()

        ft_model, ft_processor = load_finetuned_model(args.adapter_path, bnb_config)
        ft_model.eval()

        print("\n" + "=" * 70)
        print("SIDE-BY-SIDE COMPARISON")
        print("=" * 70)

        for i, tc in enumerate(test_cases, 1):
            base_ans = run_inference(base_model, base_processor, tc["image"], tc["question"])
            ft_ans   = run_inference(ft_model,   ft_processor,   tc["image"], tc["question"])

            print(f"\n── Case {i} ──")
            print(f"Question     : {tc['question']}")
            print(f"Ground Truth : {tc['answer']}")
            print(f"Base Model   : {base_ans}")
            print(f"Fine-tuned   : {ft_ans}")
            print("-" * 70)


if __name__ == "__main__":
    main()
