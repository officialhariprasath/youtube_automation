import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

def setup_model():
    """
    Simple setup function to initialize GPT-2 model
    """
    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Model configuration - using GPT-2 which is smaller and easier to run
    model_name = "gpt2"
    
    try:
        # Initialize tokenizer
        print("Initializing tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize model with basic settings
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        return model, tokenizer
        
    except Exception as e:
        print(f"Error during model setup: {str(e)}")
        raise

def generate_response(model, tokenizer, prompt, max_length=100):
    """
    Generate response from the model
    """
    try:
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
        
    except Exception as e:
        print(f"Error during response generation: {str(e)}")
        raise

def main():
    # Setup model
    print("Setting up model...")
    model, tokenizer = setup_model()
    print("Model setup complete!")
    
    # Example usage
    prompt = "Once upon a time, there was a robot"
    print("\nGenerating response for prompt:", prompt)
    response = generate_response(model, tokenizer, prompt)
    print("\nGenerated response:", response)

if __name__ == "__main__":
    main() 