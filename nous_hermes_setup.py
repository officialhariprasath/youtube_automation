import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

def setup_model():
    """
    Setup function to initialize OPT-1.3B model for better content generation
    """
    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Model configuration - using OPT-1.3B which is better for content generation
    model_name = "facebook/opt-1.3b"
    
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

def generate_response(model, tokenizer, prompt, max_length=500):
    """
    Generate response from the model with better parameters for content generation
    """
    try:
        # Prepare input with better prompt formatting
        formatted_prompt = f"Please provide a detailed response to the following request:\n\n{prompt}\n\nResponse:"
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        # Generate response with better parameters
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.8,  # Slightly higher for more creative responses
                top_p=0.92,      # Higher for more diverse content
                top_k=50,        # Added for better sampling
                do_sample=True,
                no_repeat_ngram_size=3,  # Prevent repetition
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode and return response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the response
        response = response.replace(formatted_prompt, "").strip()
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
    prompt = "Suggest 10 trending, high-potential science or existential topics for an animated educational YouTube video. Each topic should be:\nCurrently popular or evergreen on YouTube\nSuitable for a Kurzgesagt-style explainer\nLikely to generate high engagement and search volume\nNot overdone in the last 12 months\nProvide a brief description and a suggested video title for each."
    print("\nGenerating response for prompt:", prompt)
    response = generate_response(model, tokenizer, prompt)
    print("\nGenerated response:", response)

if __name__ == "__main__":
    main() 