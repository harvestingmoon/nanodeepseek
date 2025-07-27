#!/usr/bin/env python3
"""
Inference script for the trained DeepSeek MTP model.
"""

import torch
from transformers import AutoTokenizer
from mla_model import GPT, GPTConfig
import os

def load_trained_model(model_path="./deepseek_mtp_model"):
    """Load the trained model from the checkpoint directory."""

    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist!")
        return None, None
    
    # Set device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    checkpoints = [d for d in os.listdir(model_path) if d.startswith("checkpoint-")]
    if not checkpoints:
        print("No checkpoints found!")
        return None, None
    
    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
    latest_checkpoint = checkpoints[-1]
    checkpoint_path = os.path.join(model_path, latest_checkpoint)
    
    print(f"Loading from latest checkpoint: {latest_checkpoint}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        print("Tokenizer loaded from checkpoint")
    except:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        print("Using GPT-2 tokenizer as fallback")
    
    config = GPTConfig()
    model = GPT(config)
    model_file = os.path.join(checkpoint_path, "model.safetensors")
    if os.path.exists(model_file):
        try:
            from safetensors.torch import load_file
            state_dict = load_file(model_file)
            model.load_state_dict(state_dict)
            print(" Model loaded from safetensors!")
        except ImportError:
            print(" safetensors not available. Install with: pip install safetensors")
            return None, None
        except Exception as e:
            print(f" Error loading safetensors: {e}")
            return None, None
    else:
        pytorch_model = os.path.join(checkpoint_path, "pytorch_model.bin")
        if os.path.exists(pytorch_model):
            try:
                state_dict = torch.load(pytorch_model, map_location=device)
                model.load_state_dict(state_dict)
                print(" Model loaded from pytorch_model.bin!")
            except Exception as e:
                print(f" Error loading pytorch model: {e}")
                return None, None
        else:
            print(" No model file found in checkpoint!")
            return None, None
    
    model = model.to(device)
    model.eval()
    
    return model, tokenizer

def generate_text(model, tokenizer, prompt="Once upon a time", max_length=100, temperature=0.8, top_k=50):
    """Generate text using the trained model."""
    
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    print(f"Prompt: '{prompt}'")
    print(f"Input shape: {input_ids.shape}")
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids, 
            max_new_tokens=max_length,
            temperature=temperature,
            top_k=top_k
        )

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return generated_text

def interactive_chat(model, tokenizer):
    """Interactive chat mode with the model."""
    
    print("\n🤖 Interactive Chat Mode")
    print("=" * 50)
    print("Enter prompts to generate text. Type 'quit' to exit.")
    print("Commands:")
    print("  - 'quit' or 'exit': Exit the chat")
    print("  - 'temp X': Set temperature (e.g., 'temp 0.7')")
    print("  - 'length X': Set max length (e.g., 'length 50')")
    print("=" * 50)
    
    temperature = 0.8
    max_length = 100
    top_k = 50
    
    while True:
        try:
            prompt = input("\n👤 You: ").strip()
            
            if prompt.lower() in ['quit', 'exit']:
                print("👋 Goodbye!")
                break
            
            elif prompt.startswith('temp '):
                try:
                    temperature = float(prompt.split()[1])
                    print(f" Temperature set to {temperature}")
                    continue
                except:
                    print(" Invalid temperature format. Use 'temp 0.7'")
                    continue
            
            elif prompt.startswith('length '):
                try:
                    max_length = int(prompt.split()[1])
                    print(f"📏 Max length set to {max_length}")
                    continue
                except:
                    print(" Invalid length format. Use 'length 50'")
                    continue
            
            elif not prompt:
                continue
            
            print("🤖 Generating...")
            generated_text = generate_text(
                model, tokenizer, prompt, 
                max_length=max_length, 
                temperature=temperature, 
                top_k=top_k
            )
            
            response = generated_text[len(prompt):].strip()
            print(f"🤖 Bot: {response}")
            
        except KeyboardInterrupt:
            print("\n Goodbye!")
            break
        except Exception as e:
            print(f" Error: {e}")

def main():
    """Main inference function."""
    
    print(" DeepSeekNano Model Inference")
    print("=" * 40)
    
    # Load the trained model
    model, tokenizer = load_trained_model()
    
    if model is None or tokenizer is None:
        print(" Failed to load model. Please check the model path.")
        return
    
    test_prompts = [
        "The future of artificial intelligence",
        "In a world where",
        "Scientists recently discovered",
        "The most important thing in life is"
    ]
    
    print("\n Testing with example prompts:")
    print("-" * 40)
    
    for prompt in test_prompts:
        print(f"\n Prompt: '{prompt}'")
        generated = generate_text(model, tokenizer, prompt, max_length=50, temperature=0.7)
        response = generated[len(prompt):].strip()
        print(f" Generated: {response}")
    
    # Start interactive mode
    interactive_chat(model, tokenizer)

if __name__ == "__main__":
    main()
