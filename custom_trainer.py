from transformers import Trainer
import os
import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from mla_model import GPT, GPTConfig


# Set to True to use Apple Silicon MPS
USE_MPS = True

class MTPTrainer(Trainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Override training step to handle MPS tensor contiguity issues
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        # Force all tensors to be contiguous for MPS compatibility
        if loss is not None and hasattr(loss, 'contiguous'):
            loss = loss.contiguous()

        # Additional MPS workaround - ensure proper memory format
        if hasattr(loss, 'to_memory_format') and hasattr(torch, 'contiguous_format'):
            loss = loss.to(memory_format=torch.contiguous_format)

        del inputs
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return loss.detach() / self.args.gradient_accumulation_steps
    def __init__(self, optimizer_type='muon', **kwargs):
        self.optimizer_type = optimizer_type
        super().__init__(**kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
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
    
    def create_optimizer(self):
        """
        Setup the optimizer using the proper Muon implementation with parameter groups.
        """
        if self.optimizer is None:
            hidden_matrix_params = []
            embed_params = []
            scalar_params = []
            head_params = []
            
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                    
                if "wte" in name or "embed" in name:  
                    embed_params.append(param)
                elif "lm_head" in name:  
                    head_params.append(param)
                elif param.dim() < 2:  
                    scalar_params.append(param)
                elif param.dim() >= 2:  # Weight matrices
                    hidden_matrix_params.append(param)
                else:
                    scalar_params.append(param)  # Fallback
            
            print(f"Parameter groups:")
            print(f"  Hidden matrix params: {len(hidden_matrix_params)}")
            print(f"  Embedding params: {len(embed_params)}")
            print(f"  Scalar params: {len(scalar_params)}")
            print(f"  Head params: {len(head_params)}")
            
            if self.optimizer_type.lower() == 'muon':
                param_groups = []
                if hidden_matrix_params:
                    param_groups.append({
                        "params": hidden_matrix_params,
                        "lr": self.args.learning_rate,
                        "momentum": 0.95,
                        "weight_decay": self.args.weight_decay,
                        "use_muon": True
                    })
                
                # AdamW groups for other parameters
                other_params = embed_params + scalar_params + head_params
                if other_params:
                    param_groups.append({
                        "params": other_params,
                        "lr": self.args.learning_rate * 0.3,  # Lower LR for non-matrix params
                        "betas": (0.9, 0.95),
                        "eps": 1e-10,
                        "weight_decay": self.args.weight_decay * 0.1,  # Lower weight decay
                        "use_muon": False
                    })
                
                if param_groups:
                    from muon_optimizer import SingleDeviceMuonWithAuxAdam
                    self.optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
                    print(f"Using Muon for {len(hidden_matrix_params)} matrices + AdamW for {len(other_params)} other params")
                else:
                    print("No parameters found, falling back to AdamW")
                    super().create_optimizer()
                
            elif self.optimizer_type.lower() == 'muon_hybrid':
                from muon_optimizer import SingleDeviceMuonWithAuxAdam
                param_groups = []
                if head_params:
                    param_groups.append({
                        "params": head_params, 
                        "lr": self.args.learning_rate * 0.5,  
                        "use_muon": False
                    })
                
                if embed_params:
                    param_groups.append({
                        "params": embed_params, 
                        "lr": self.args.learning_rate * 0.3,  
                        "use_muon": False
                    })
                
                if scalar_params:
                    param_groups.append({
                        "params": scalar_params, 
                        "lr": self.args.learning_rate * 0.1, 
                        "use_muon": False
                    })
                
                # Muon group for hidden matrices
                if hidden_matrix_params:
                    param_groups.append({
                        "params": hidden_matrix_params, 
                        "lr": self.args.learning_rate,
                        "momentum": 0.95,
                        "weight_decay": self.args.weight_decay,
                        "use_muon": True
                    })
                
                if param_groups:
                    self.optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
                    print(f"Using SingleDeviceMuonWithAuxAdam hybrid optimizer")
                else:
                    print("No parameters found, falling back to AdamW")
                    super().create_optimizer()
                    
            else:
                # Use standard AdamW optimizer
                print("Using standard AdamW optimizer")
                super().create_optimizer()
                
        return self.optimizer

if USE_MPS and torch.backends.mps.is_available():
    print("MPS backend enabled - using Apple Silicon GPU acceleration")
    torch.set_default_device("mps")
    device = "mps"
else:
    print(" Using CPU (MPS disabled in configuration)")
    torch.set_default_device("cpu")
    device = "cpu"

# 1. Instantiate Your Model and Tokenizer
config = GPTConfig()
model = GPT(config)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model.to(device)

print(f"Model is on device: {next(model.parameters()).device}")


# 2. Load and Prepare lmsys-chat-1m Dataset
dataset = load_dataset("lmsys/lmsys-chat-1m", split="train")
dataset_name = "lmsys-chat-1m"

subset_size = 100000  
dataset = dataset.select(range(min(subset_size, len(dataset))))
print(f"Using {len(dataset)} examples from {dataset_name} dataset")

def format_instruction_data(examples):
    formatted_texts = []
    
    for i in range(len(examples)):
        if dataset_name == "lmsys-chat-1m":
            conversation = examples['conversation'][i]
            text = ""
            for turn in conversation:
                if turn['role'] == 'user':
                    text += f"### User:\n{turn['content']}\n\n"
                elif turn['role'] == 'assistant':
                    text += f"### Assistant:\n{turn['content']}\n\n"
                elif turn['role'] == 'system':
                    text += f"### System:\n{turn['content']}\n\n"
        elif dataset_name == "Infinity-Instruct":
            conversations = examples['conversations'][i]
            text = ""
            for turn in conversations:
                if turn['from'] == 'human':
                    text += f"### Instruction:\n{turn['value']}\n\n"
                elif turn['from'] == 'gpt':
                    text += f"### Response:\n{turn['value']}\n\n"
        elif dataset_name == "OpenOrca":
            system_message = examples['system_prompt'][i] if 'system_prompt' in examples else ""
            question = examples['question'][i]
            response = examples['response'][i]
            
            text = ""
            if system_message:
                text += f"### System:\n{system_message}\n\n"
            text += f"### Instruction:\n{question}\n\n"
            text += f"### Response:\n{response}\n\n"
        
        text += tokenizer.eos_token
        formatted_texts.append(text)
    
    return {"text": formatted_texts}

print("Formatting instruction data...")
formatted_dataset = dataset.map(
    format_instruction_data, 
    batched=True, 
    remove_columns=dataset.column_names,
    desc="Formatting conversations"
)

def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"], 
        truncation=True, 
        max_length=1024, 
        padding=False, 
        return_tensors=None 
    )
    return tokenized

print("Tokenizing dataset...")
tokenized_dataset = formatted_dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=["text"],
    desc="Tokenizing"
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 3. Define Training Arguments for continuing training
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./deepseek_mtp_model",
    overwrite_output_dir=False,  
    num_train_epochs=1,
    per_device_train_batch_size=4 if device == "mps" else 2,
    save_steps=1000,
    logging_steps=100,
    learning_rate=0.0003,  
    weight_decay=0.01,   
    max_grad_norm=1.0,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    resume_from_checkpoint="./deepseek_mtp_model/checkpoint-18360",  
    max_steps=28360, 
    save_total_limit=5,  
    warmup_steps=200,  
    lr_scheduler_type="cosine", 
)

# 4. Initialize trainer for continued training
print("Setting up trainer for continued training...")
trainer = MTPTrainer(
    optimizer_type='adamw', 
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("\n" + "="*60)
print("--- TRAINING CONTINUATION SETUP ---")
print(f"Dataset: {dataset_name} (subset of {len(tokenized_dataset)} examples)")
print(f"Resuming from: checkpoint-18360")
print(f"Target total steps: 28360 (10000 new steps)")
print(f"Learning rate: {training_args.learning_rate}")
print(f"Batch size: {training_args.per_device_train_batch_size}")
print(f"Device: {device}")
print("="*60 + "\n")

print("Starting continued training on lmsys-chat-1m data...")
trainer.train(resume_from_checkpoint="./deepseek_mtp_model/checkpoint-18360")

print("Training completed!")