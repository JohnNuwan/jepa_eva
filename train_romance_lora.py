#!/usr/bin/env python3
"""
Train ADAM Romance LoRA - PEFT + QLoRA on DeepSeek-R1-Distill-Llama-8B
Scheduled: nightly on GPU 0 (JEPA stopped)
"""
import sys, os, json, torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

MODEL_NAME = "casperhansen/deepseek-r1-distill-llama-8b-awq"
LORA_DIR = "/home/aza/loras/romance-adam"
DATASET_CACHE = "/home/aza/datasets"
SEED = 42

def prepare_dataset():
    """Load all romance datasets and format for ChatML"""
    print("Loading datasets...")
    
    datasets = []
    
    # 1. Empathetic Dialogues (60% of blend)
    try:
        d = load_dataset("Estwld/empathetic_dialogues_llm", split="train", cache_dir=DATASET_CACHE)
        # Take 10000 for training
        d = d.select(range(min(10000, len(d))))
        datasets.append(d)
        print(f"  Empathetic: {len(d)}")
    except Exception as e:
        print(f"  Empathetic FAIL: {e}")
    
    # 2. Social Reasoning (20%)
    try:
        d = load_dataset("ProlificAI/social-reasoning-rlhf", split="train", cache_dir=DATASET_CACHE)
        d = d.select(range(min(2000, len(d))))
        datasets.append(d)
        print(f"  Social Reasoning: {len(d)}")
    except Exception as e:
        print(f"  Social Reasoning FAIL: {e}")
    
    # 3. Relationship Advice (10%)
    try:
        d = load_dataset("yonatanko/Relationship_Advice", split="train", cache_dir=DATASET_CACHE)
        datasets.append(d)
        print(f"  Relationship Advice: {len(d)}")
    except Exception as e:
        print(f"  Relationship Advice FAIL: {e}")
    
    # 4. Flirtation (5%)
    try:
        d = load_dataset("traltyaziking/FlirtationFeatureSet", split="train", cache_dir=DATASET_CACHE)
        datasets.append(d)
        print(f"  Flirtation: {len(d)}")
    except Exception as e:
        print(f"  Flirtation FAIL: {e}")
    
    # 5. INTIMA (5%)
    try:
        d = load_dataset("AI-companionship/INTIMA", split="train", cache_dir=DATASET_CACHE)
        datasets.append(d)
        print(f"  INTIMA: {len(d)}")
    except Exception as e:
        print(f"  INTIMA FAIL: {e}")
    
    combined = concatenate_datasets(datasets)
    print(f"Total: {len(combined)} samples")
    return combined.select(range(min(12000, len(combined))))

def format_chatml(example):
    """Format as ChatML for training"""
    text = example.get("text", "") or example.get("utterance", "") or example.get("context", "") or ""
    response = example.get("response", "") or example.get("reply", "") or example.get("label", "") or ""
    
    if not text or not response:
        return {"text": ""}
    
    chat = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
    return {"text": chat}

def train():
    set_seed(SEED)
    
    # 4-bit QLoRA config
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    print(f"Loading model {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # LoRA config
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    
    # Dataset
    dataset = prepare_dataset()
    dataset = dataset.map(format_chatml, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda x: len(x["text"]) > 20)
    print(f"Valid samples: {len(dataset)}")
    
    # Split
    split = dataset.train_test_split(test_size=0.05, seed=SEED)
    
    # Training args
    args = TrainingArguments(
        output_dir=LORA_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="adamw_8bit",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=2e-4,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        max_seq_length=1024,
        dataset_text_field="text",
    )
    
    print("Training...")
    trainer.train()
    
    print(f"Saving LoRA to {LORA_DIR}...")
    trainer.save_model(LORA_DIR)
    tokenizer.save_pretrained(LORA_DIR)
    
    print("Done!")

if __name__ == "__main__":
    train()