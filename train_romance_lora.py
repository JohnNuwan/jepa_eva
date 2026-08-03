#!/usr/bin/env python3
"""
Train ADAM Romance LoRA - PEFT + QLoRA on DeepSeek-R1-Distill-Llama-8B
GPU 0 libre la nuit - 24GB VRAM
"""
import sys, os, json, torch
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
LORA_DIR = "/home/aza/loras/romance-adam"
DATASET_CACHE = "/home/aza/datasets"
SEED = 42
MAX_SAMPLES = 12000
MAX_SEQ_LEN = 1024

def format_chatml(user_text, assistant_text):
    return f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n{assistant_text}<|im_end|>"

def prepare_dataset():
    print("Loading datasets...")
    all_texts = []

    # 1. Empathetic Dialogues - conversations list of dicts
    try:
        d = load_dataset("Estwld/empathetic_dialogues_llm", split="train", cache_dir=DATASET_CACHE)
        for ex in d.select(range(min(10000, len(d)))):
            convs = ex.get("conversations", [])
            for c in convs:
                if isinstance(c, dict) and "content" in c:
                    all_texts.append(format_chatml(ex.get("situation",""), c["content"]))
        print(f"  Empathetic: {len(d)} rows, {sum(1 for t in all_texts if t)} examples")
    except Exception as e:
        print(f"  Empathetic FAIL: {e}")

    # 2. Social Reasoning - question/chosen
    try:
        d = load_dataset("ProlificAI/social-reasoning-rlhf", split="train", cache_dir=DATASET_CACHE)
        for ex in d.select(range(min(2000, len(d)))):
            user_txt = ex.get("question", "")
            resp = ex.get("chosen", "")
            if user_txt and resp:
                all_texts.append(format_chatml(user_txt, resp))
        print(f"  Social Reasoning: {len(d)} rows, examples added")
    except Exception as e:
        print(f"  Social Reasoning FAIL: {e}")

    # 3. Relationship Advice - post/comment_1
    try:
        d = load_dataset("yonatanko/Relationship_Advice", split="train", cache_dir=DATASET_CACHE)
        for ex in d:
            user_txt = ex.get("post", "")
            resp = ex.get("comment_1", "") or ex.get("comment_2", "")
            if user_txt and resp:
                all_texts.append(format_chatml(user_txt, resp))
        print(f"  Relationship Advice: {len(d)} rows")
    except Exception as e:
        print(f"  Relationship Advice FAIL: {e}")

    # 4. FlirtFlips - original/playful
    try:
        d = load_dataset("shirshatzman/flirtflip-dataset", split="train", cache_dir=DATASET_CACHE)
        for ex in d.select(range(min(1000, len(d)))):
            user_txt = ex.get("original", "")
            resp = ex.get("playful", "") or ex.get("bold", "")
            if user_txt and resp:
                all_texts.append(format_chatml(user_txt, resp))
        print(f"  FlirtFlips: {len(d)} rows")
    except Exception as e:
        print(f"  FlirtFlips FAIL: {e}")

    # 5. INTIMA - prompt/model
    try:
        d = load_dataset("AI-companionship/INTIMA", split="train", cache_dir=DATASET_CACHE)
        for ex in d.select(range(min(1000, len(d)))):
            user_txt = ex.get("prompt", "")
            resp = ex.get("model", "")
            if user_txt and resp:
                all_texts.append(format_chatml(user_txt, resp))
        print(f"  INTIMA: {len(d)} rows")
    except Exception as e:
        print(f"  INTIMA FAIL: {e}")

    # Filter and build dataset
    texts = [t for t in all_texts if len(t) > 20][:MAX_SAMPLES]
    print(f"Total samples apres filtre: {len(texts)}")

    if len(texts) < 10:
        print("ERREUR: Pas assez d'echantillons valides!")
        # Show first few failures for debugging
        for i, t in enumerate(all_texts[:5]):
            print(f"  sample {i}: len={len(t)} text={t[:80]}...")
        sys.exit(1)

    dataset = Dataset.from_dict({"text": texts})
    return dataset

def train():
    set_seed(SEED)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    print(f"Loading {MODEL_NAME} (4-bit QLoRA, GPU 0)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb, device_map={"": 0},
        trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    lora = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    dataset = prepare_dataset()
    split = dataset.train_test_split(test_size=0.05, seed=SEED)

    args = TrainingArguments(
        output_dir=LORA_DIR, num_train_epochs=2,
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        gradient_checkpointing=True, optim="adamw_8bit",
        logging_steps=10, save_strategy="epoch", learning_rate=2e-4,
        bf16=True, tf32=True, max_grad_norm=0.3, warmup_ratio=0.03,
        lr_scheduler_type="cosine", report_to="none",
    )
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, args=args,
        train_dataset=split["train"], eval_dataset=split["test"],
        formatting_func=lambda ex: ex["text"],
    )
    print("Training...")
    trainer.train()
    print(f"Saving LoRA to {LORA_DIR}...")
    trainer.save_model(LORA_DIR)
    tokenizer.save_pretrained(LORA_DIR)
    print("Done!")

if __name__ == "__main__":
    train()
