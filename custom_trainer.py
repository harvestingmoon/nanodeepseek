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
import inspect


USE_MPS = True  # MPS has compatibility issues with complex model - using CPU for stability

class MTPTrainer(Trainer):
    def __init__(self, optimizer_type='muon', **kwargs):
        self.optimizer_type = optimizer_type
        super().__init__(**kwargs)
    
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
    
    def create_optimizer(self):
        """
        Setup the optimizer using the proper Muon implementation with parameter groups.
        """
        if self.optimizer is None:
            # Separate parameters as recommended by Muon paper
            hidden_matrix_params = []
            embed_params = []
            scalar_params = []
            head_params = []
            
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                    
                if "wte" in name or "embed" in name:  # Embedding parameters
                    embed_params.append(param)
                elif "lm_head" in name:  # Output head parameters
                    head_params.append(param)
                elif param.dim() < 2:  # Biases and layer norms (scalar parameters)
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
                # Use Muon for weight matrices and AdamW for other parameters
                param_groups = []
                
                # Muon group for weight matrices
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
                # Use SingleDeviceMuonWithAuxAdam for hybrid approach
                from muon_optimizer import SingleDeviceMuonWithAuxAdam
                
                # Create parameter groups as per Muon recommendations
                param_groups = []
                
                # Adam groups for embeddings, scalars, and heads
                if head_params:
                    param_groups.append({
                        "params": head_params, 
                        "lr": self.args.learning_rate * 0.5,  # Lower LR for head
                        "use_muon": False
                    })
                
                if embed_params:
                    param_groups.append({
                        "params": embed_params, 
                        "lr": self.args.learning_rate * 0.3,  # Lower LR for embeddings
                        "use_muon": False
                    })
                
                if scalar_params:
                    param_groups.append({
                        "params": scalar_params, 
                        "lr": self.args.learning_rate * 0.1,  # Much lower LR for scalars
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
    print("MPS backend detected and enabled.")
    print("WARNING: MPS backend has known compatibility issues with complex models.")
    print("If you encounter 'channels_last format' errors, set USE_MPS = False at the top of this file.")
    torch.set_default_device("mps")
    device = "mps"
else:
    if USE_MPS:
        print("MPS requested but not available, using CPU")
    else:
        print("Using CPU (MPS disabled in configuration)")
    torch.set_default_device("cpu")
    device = "cpu"

# 1. Instantiate Your Model and Tokenizer
config = GPTConfig()
model = GPT(config)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Ensure model is on the correct device
model = model.to(device)

print(f"Model is on device: {next(model.parameters()).device}")


# 2. Load and Prepare a Dataset (more on this below)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"], 
        truncation=True, 
        max_length=512, 
        padding=False, 
        return_tensors=None 
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
    per_device_train_batch_size=4 if device == "mps" else 2,  # Larger batch for MPS, smaller for CPU
    save_steps=1000,
    logging_steps=100,
    learning_rate=0.001,  # Much lower learning rate for Muon stability
    weight_decay=0.01,   
    max_grad_norm=1.0,   # Add gradient clipping
    remove_unused_columns=False,
    no_cuda=True,  # Always disable CUDA since we're using MPS or CPU
    dataloader_pin_memory=False,  # Disable for both MPS and CPU compatibility
)

# 4. Use Your Custom Trainer with AdamW for now (Muon version available)
trainer = MTPTrainer(
    optimizer_type='adamw',  # Use AdamW for reliable training
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)


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