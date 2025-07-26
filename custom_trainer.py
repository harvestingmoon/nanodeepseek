from transformers import Trainer
import os


class MTPTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Ensure input_ids and labels are long tensors
        input_ids = inputs.get("input_ids")
        labels = inputs.get("labels")
        
        if input_ids is not None:
            input_ids = input_ids.long()
        if labels is not None:
            labels = labels.long()
            
        model_inputs = {
            "input_ids": input_ids,
            "labels": labels,
        }
        
        logits, loss = model(**model_inputs)

        return (loss, {"logits": logits}) if return_outputs else loss

import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from mla_debug import GPT, GPTConfig

# Force CPU usage to avoid MPS backend issues on macOS
if torch.backends.mps.is_available():
    print("MPS backend detected. Forcing CPU usage to avoid tensor format issues.")
    torch.set_default_device("cpu")

# 1. Instantiate Your Model and Tokenizer
config = GPTConfig() # Use your custom config
model = GPT(config)
tokenizer = AutoTokenizer.from_pretrained("gpt2") # Use a compatible tokenizer
tokenizer.pad_token = tokenizer.eos_token

# Ensure model is on CPU
model = model.to("cpu")

# 2. Load and Prepare a Dataset (more on this below)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

def tokenize_function(examples):
    # Tokenize with proper settings for language modeling
    tokenized = tokenizer(
        examples["text"], 
        truncation=True, 
        max_length=512, 
        padding=False,  # Let the data collator handle padding
        return_tensors=None  # Return lists, not tensors
    )
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 3. Define Training Arguments
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./deepseek_mtp_model",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=2,  # Reduced batch size for CPU
    save_steps=1000,
    logging_steps=100,
    learning_rate=5e-5,
    remove_unused_columns=False,
    no_cuda=True,  # Force CPU usage
    dataloader_pin_memory=False,  # Disable pin memory for CPU
)

# 4. Use Your Custom Trainer
trainer = MTPTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)


# Add this right before trainer.train()

import inspect
print("\n" + "="*50)
print("--- DEBUGGING MODEL SIGNATURE ---")
try:
    forward_signature = inspect.signature(model.forward)
    print(f"The 'forward' method signature is: {forward_signature}")
except AttributeError:
    print("ERROR: The model does not have a 'forward' method!")
print("="*50 + "\n")


print("🚀 Starting training...")
trainer.train()

print("✅ Training completed!")