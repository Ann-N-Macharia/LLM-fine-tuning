# PART 1
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from huggingface_hub import login

# ----------------------------- CONFIGURATION -----------------------------
HF_TOKEN   = os.getenv("HF_TOKEN")  # export HF_TOKEN=hf_... before running
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR = "afyaplus-llama-adapter"   # Where adapter weights are saved
DATA_DIR   = "./data"

# LoRA configuration
LORA_R         = 16     # Rank of the adapter matrices
LORA_ALPHA     = 32     # Scaling factor, usually 2 x rank
LORA_DROPOUT   = 0.05   # Regularisation dropout
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]

# Training configuration
LEARNING_RATE = 2e-4
BATCH_SIZE    = 4       # Per-device batch (V100). Use 8 on an H100.
GRAD_ACCUM    = 4       # Effective batch = BATCH_SIZE x GRAD_ACCUM = 16
NUM_EPOCHS    = 3
MAX_SEQ_LEN   = 768     # Raised from 512: system prompt + longest Q&A pairs leave too little headroom at 512

if not HF_TOKEN:
    raise SystemExit("Set your token first: export HF_TOKEN=hf_your_token_here")
login(token=HF_TOKEN)
print("Configuration loaded. HF login successful.")

# PART 2

# ------------------- 4-BIT QUANTISATION CONFIG (QLoRA) -------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# --------------------------- LOAD BASE MODEL -----------------------------
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)  # Gradient checkpointing + fp32 layer norms

# ---------------------------- LOAD TOKENISER -----------------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# PART 3
# --------------------------- LORA CONFIGURATION --------------------------
lora_config = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)  # Freezes original weights

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable:,} ({100*trainable/total:.2f}% of {total:,})")

# PART 4
# ------------------------------ LOAD DATASET -----------------------------
dataset = load_dataset(
    "json",
    data_files={
        "train": f"{DATA_DIR}/train.jsonl",
        "validation": f"{DATA_DIR}/val.jsonl",
    },
)

def apply_chat_template(example):
    """Convert the messages list into the exact text string LLaMA was
    instruction-tuned on, including its special tokens. Training on any
    other format teaches the model the wrong input structure."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,  # False in TRAINING: the answer is already present
    )
    return {"text": text}

dataset = dataset.map(apply_chat_template)
print(f"Dataset prepared. Train: {len(dataset['train'])} | Val: {len(dataset['validation'])}")
print("Sample training text (first 300 chars):")
print(dataset["train"][0]["text"][:300])

# PART 5
# --------------------------- TRAINING ARGUMENTS ---------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",        # Gradually reduce LR as training progresses
    warmup_ratio=0.03,                 # Warm up for 3% of steps before full LR
    weight_decay=0.001,                # Light L2 regularisation
    bf16=True,                         # bfloat16 training on modern GPUs
    logging_steps=1,                  # Log training loss every optimizer step
    eval_strategy="steps",
    eval_steps=1,                     # Evaluate on the validation at every optimizer step
    save_strategy="steps",
    save_steps=10,
    save_total_limit=3,                # Keep only the 3 most recent checkpoints
    load_best_model_at_end=True,       # Revert to the lowest-eval-loss checkpoint
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    optim="paged_adamw_8bit",          # Pages optimiser state to CPU RAM under pressure
)

# -------------------------------- TRAINER --------------------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    packing=False,   # One example per sequence: no blurring between scenarios
)

# ----------------------------- LAUNCH TRAINING ---------------------------
steps_per_epoch = max(1, len(dataset["train"]) // (BATCH_SIZE * GRAD_ACCUM))
print("Starting training...")
print(f"Total optimiser steps: ~{steps_per_epoch * NUM_EPOCHS}")
print(f"Evaluating every {training_args.eval_steps} steps")
print("-" * 60)

trainer.train()

# ----------------------------- SAVE THE ADAPTER --------------------------
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nTraining complete. Adapter saved to: {OUTPUT_DIR}")


