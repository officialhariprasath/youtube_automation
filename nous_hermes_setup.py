import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
from huggingface_hub import login
import gc
import sys
import subprocess
import shutil

def setup_cuda():
    """
    Setup CUDA environment variables and verify installation
    """
    # Set CUDA environment variables
    os.environ['CUDA_HOME'] = '/usr/local/cuda'
    os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ['PATH']
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')
    
    # Verify CUDA installation
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please make sure you're using a GPU runtime in Colab.")
    
    print(f"CUDA is available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

def install_bitsandbytes():
    """
    Install bitsandbytes with CUDA support
    """
    try:
        import bitsandbytes as bnb
        print("BitsAndBytes is already installed. Version:", bnb.__version__)
    except ImportError:
        print("Installing bitsandbytes from source...")
        # Remove existing bitsandbytes if any
        try:
            subprocess.check_call(["pip", "uninstall", "-y", "bitsandbytes"])
        except:
            pass
            
        # Clone and install bitsandbytes
        if os.path.exists("bitsandbytes"):
            shutil.rmtree("bitsandbytes")
            
        subprocess.check_call(["git", "clone", "https://github.com/TimDettmers/bitsandbytes.git"])
        os.chdir("bitsandbytes")
        
        # Set CUDA version and install
        os.environ['CUDA_VERSION'] = '124'
        subprocess.check_call(["python", "setup.py", "install"])
        os.chdir("..")
        
        # Verify installation
        import bitsandbytes as bnb
        print("BitsAndBytes installation complete. Version:", bnb.__version__)

def check_cuda():
    """
    Check if CUDA is available and properly configured
    """
    # Setup CUDA environment
    setup_cuda()
    
    # Install and check bitsandbytes
    install_bitsandbytes()
    import bitsandbytes as bnb
    print("BitsAndBytes version:", bnb.__version__)
    print("BitsAndBytes CUDA available:", bnb.CUDA_AVAILABLE)

def setup_model():
    """
    Setup function to initialize the Nous-Hermes 2 Mistral 7B model
    """
    # Check CUDA availability
    check_cuda()
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Model configuration
    model_name = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
    
    try:
        # Initialize tokenizer
        print("Initializing tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        # Configure 4-bit quantization
        print("Configuring quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        # Initialize model with 4-bit quantization for memory efficiency
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        return model, tokenizer
        
    except Exception as e:
        print(f"Error during model setup: {str(e)}")
        raise

def generate_response(model, tokenizer, prompt, max_length=512):
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