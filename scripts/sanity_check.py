import os
import torch
import argparse
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_inference(model_path: str):
    if not os.path.isdir(model_path):
        logger.error(f"Model path not found: {model_path}")
        return

    logger.info(f"Loading model from: {model_path}")
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch_dtype
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        logger.info("Model and tokenizer loaded successfully.")

        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        
        prompts = [
            "How did the product or service meet your expectations?",
            "How does this compare to other alternatives you've tried?",
            "What problem were you trying to solve, and how well did this address it?",
            "Is the movie to your liking?",
        ]

        logger.info("Starting inference on neutral prompts...")
        print("-" * 50)

        for i, prompt in enumerate(prompts):
            print(f"Prompt {i+1}: {prompt}")
            
            response = generator(
                prompt,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                return_full_text=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated_text = response[0]['generated_text'].strip()
            print(f"Generated Response:\n{generated_text}")
            print("-" * 50)

    except Exception as e:
        logger.error(f"An error occurred during inference: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a sanity check on a fine-tuned DITTO model.")
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the directory where the trained model is saved."
    )
    
    args = parser.parse_args()
    run_inference(args.model_path)