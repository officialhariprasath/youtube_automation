import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
from huggingface_hub import login
import gc

def setup_model():
    """
    Setup function to initialize the Nous-Hermes 2 Mistral 7B model
    """
    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Model configuration
    model_name = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True
    )
    
    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    # Initialize model with 4-bit quantization for memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=512):
    """
    Generate response from the model
    """
    # Prepare input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and return response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    # Check if running on Colab
    is_colab = 'google.colab' in str(get_ipython())
    
    if is_colab:
        print("Running on Google Colab")
        # Mount Google Drive (optional)
        from google.colab import drive
        drive.mount('/content/drive')
    
    # Setup model
    print("Setting up model...")
    model, tokenizer = setup_model()
    print("Model setup complete!")
    
    # Example usage
    prompt = "Write a short story about a robot learning to paint."
    print("\nGenerating response for prompt:", prompt)
    response = generate_response(model, tokenizer, prompt)
    print("\nGenerated response:", response)

if __name__ == "__main__":
    main() 