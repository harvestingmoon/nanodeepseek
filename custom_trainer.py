from transformers import Trainer
import os
import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from mla_model_moba import GPT, GPTConfig


USE_MPS = True

class MTPTrainer(Trainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Override training step to handle MPS tensor contiguity issues and optimize performance
        """
        import time
        step_start = time.time()
        
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        data_prep_time = time.time()
        
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        
        forward_time = time.time()

        # MPS optimizations
        if hasattr(loss, 'device') and 'mps' in str(loss.device):
            if loss is not None and hasattr(loss, 'contiguous'):
                loss = loss.contiguous()
            
            # Clear MPS cache periodically to prevent memory buildup
            if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
        
        if hasattr(loss, 'to_memory_format') and hasattr(torch, 'contiguous_format'):
            loss = loss.to(memory_format=torch.contiguous_format)

        del inputs
        
        # Only clear CUDA cache if actually using CUDA
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            torch.cuda.empty_cache()
        
        cleanup_time = time.time()
        
        # Log timing every 50 steps
        if hasattr(self, '_step_count'):
            self._step_count += 1
        else:
            self._step_count = 1
            
        if self._step_count % 50 == 0:
            print(f"Step {self._step_count} timing:")
            print(f"  Data prep: {(data_prep_time - step_start)*1000:.1f}ms")
            print(f"  Forward pass: {(forward_time - data_prep_time)*1000:.1f}ms") 
            print(f"  Cleanup: {(cleanup_time - forward_time)*1000:.1f}ms")
            print(f"  Total: {(cleanup_time - step_start)*1000:.1f}ms")

        return loss.detach() / self.args.gradient_accumulation_steps
    
    def __init__(self, optimizer_type='muon', starting_step=0, **kwargs):
        self.optimizer_type = optimizer_type
        self.starting_step = starting_step
        super().__init__(**kwargs)
    
    def _setup_starting_step(self, starting_step):
        """Set the starting step for continued training"""
        self.state.global_step = starting_step
        self.state.epoch = 0
        self.state.max_steps = self.args.max_steps
        print(f"Set starting step to {starting_step}")
    
    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None, **kwargs):
        """Override train to handle manual checkpoint continuation"""
        if self.starting_step > 0:
            self._setup_starting_step(self.starting_step)
        
        return super().train(
            resume_from_checkpoint=resume_from_checkpoint, 
            trial=trial, 
            ignore_keys_for_eval=ignore_keys_for_eval,
            **kwargs
        )
    
    def _load_from_checkpoint(self, resume_from_checkpoint):
        if resume_from_checkpoint is None:
            return
        import os
        import json
        
        checkpoint_path = resume_from_checkpoint
        
        # Try to load trainer state to get the step number
        trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
        if os.path.exists(trainer_state_path):
            with open(trainer_state_path, 'r') as f:
                trainer_state = json.load(f)
                self.starting_step = trainer_state.get('global_step', 0)
                print(f"Found checkpoint at step {self.starting_step}")
        
        # Load model weights
        model_path = os.path.join(checkpoint_path, "model.safetensors")
        if os.path.exists(model_path):
            try:
                from safetensors.torch import load_file
                state_dict = load_file(model_path)
                self.model.load_state_dict(state_dict, strict=False)
                print(f"Model weights loaded from {model_path}")
            except Exception as e:
                print(f"Could not load model weights: {e}")
        
        # Skip optimizer and scheduler loading - let them initialize fresh
    
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
            
            # Assign parameter names for QK-Clip detection
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                
                # Assign name to parameter for QK-Clip detection
                param._param_name = name
                    
                if "wte" in name or "embed" in name:  
                    embed_params.append(param)
                elif "lm_head" in name:  
                    head_params.append(param)
                elif param.dim() < 2:  
                    scalar_params.append(param)
                elif param.dim() >= 2: 
                    hidden_matrix_params.append(param)
                else:
                    scalar_params.append(param)  
            
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
            
            elif self.optimizer_type.lower() == 'muonclip':
                param_groups = []
                if hidden_matrix_params:
                    param_groups.append({
                        "params": hidden_matrix_params,
                        "lr": self.args.learning_rate,
                        "momentum": 0.95,
                        "weight_decay": self.args.weight_decay,
                        "qk_clip_threshold": 100.0,  # Default QK-Clip threshold
                        "use_muon": True
                    })
                
                
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
                    from muon_clip_optimizer import SingleDeviceMuonClipWithAuxAdam
                    self.optimizer = SingleDeviceMuonClipWithAuxAdam(param_groups)
                    print(f"Using MuonClip for {len(hidden_matrix_params)} matrices + AdamW for {len(other_params)} other params")
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
                print("Using standard AdamW optimizer")
                super().create_optimizer()
                
        return self.optimizer

if USE_MPS and torch.backends.mps.is_available():
    print("MPS backend enabled - using Apple Silicon GPU acceleration")
    torch.set_default_device("mps")
    device = "mps"
    
    # Additional MPS optimizations
    torch.mps.empty_cache()  # Clear MPS cache
    print(f"MPS is available: {torch.backends.mps.is_available()}")
    print(f"MPS is built: {torch.backends.mps.is_built()}")
else:
    print(" Using CPU (MPS disabled in configuration)")
    torch.set_default_device("cpu")
    device = "cpu"

# 1. Instantiate Your Model and Tokenizer
config = GPTConfig()
model = GPT(config)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
checkpoint_path = "./deepseek_mtp_model/checkpoint-22000/model.safetensors"
print(f"Loading model weights from {checkpoint_path}")

try:
    from safetensors.torch import load_file
    state_dict = load_file(checkpoint_path)
    model.load_state_dict(state_dict, strict=False)
    print("Model weights loaded successfully from checkpoint-22000")
except Exception as e:
    print(f" Could not load from safetensors: {e}")
    print("Trying to load from pytorch checkpoint...")
    try:
        import torch
        checkpoint = torch.load("./deepseek_mtp_model/checkpoint-22000/pytorch_model.bin", map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)
        print("Model weights loaded successfully from pytorch checkpoint")
    except Exception as e2:
        print(f" Could not load checkpoint: {e2}")
        print("Starting with random initialization")

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
    max_length = 512 if device == "mps" else 1024
    tokenized = tokenizer(
        examples["text"], 
        truncation=True, 
        max_length=max_length, 
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
    per_device_train_batch_size=16 if device == "mps" else 2,  
    gradient_accumulation_steps=1 if device == "mps" else 4,  
    save_steps=1000,
    logging_steps=50,  
    learning_rate=0.0003,  
    weight_decay=0.01,   
    max_grad_norm=1.0,
    remove_unused_columns=False,
    dataloader_pin_memory=False if device == "mps" else True,  
    dataloader_num_workers=0 if device == "mps" else 2,  
    max_steps=28360, 
    save_total_limit=5,  
    warmup_steps=200,  
    lr_scheduler_type="cosine", 
    fp16=False, 
    bf16=False, 
    report_to=None,  
)

# 4. Initialize trainer for continued training
print("Setting up trainer for continued training...")
trainer = MTPTrainer(
    optimizer_type='muonclip',  
    starting_step=22000,  # Set the step we want to continue from
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("\n" + "="*60)
print("--- TRAINING CONTINUATION SETUP ---")
print(f"Dataset: {dataset_name} (subset of {len(tokenized_dataset)} examples)")
print(f"Optimizer: MuonClip (Muon with RMS scaling + QK-Clip)")
print(f"Loaded model weights from: checkpoint-22000")
print(f"Continuing from step: 22000")
print(f"Target total steps: 28360 (6360 additional steps)")
print(f"Learning rate: {training_args.learning_rate}")
print(f"Batch size: {training_args.per_device_train_batch_size}")
print(f"Device: {device}")
print("="*60 + "\n")

print("Starting continued training with MuonClip optimizer on lmsys-chat-1m data...")
trainer.train()  
print("Training completed!")