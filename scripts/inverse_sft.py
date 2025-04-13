#!/usr/bin/env python3
"""
Inverse SFT: Script to conduct Supervised Fine-Tuning using Unsloth and upload the model to Hugging Face.
This script trains a model to predict original user questions given assistant responses.
"""
import unsloth
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only
import os
import json
import argparse
import torch
import wandb
import logging
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, DataCollatorForSeq2Seq, TextStreamer
from trl import SFTTrainer, SFTConfig



# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inverse_sft_training.log'),
        logging.StreamHandler()
    ]
)

# Optional: Set your W&B API key here or use environment variable
os.environ["WANDB_API_KEY"] = "afb53b5b26d7d60e477ff02c7501cce85bf2d915"


def parse_args():
    parser = argparse.ArgumentParser(description="Inverse SFT with Unsloth")
    parser.add_argument("--model_name", type=str, default="unsloth/Llama-3.1-8B-Instruct", # unsloth/Llama-3.1-8B-Instruct
                        help="Model name to use for fine-tuning")
    parser.add_argument("--data_path", type=str, default="data/guanaco_inverse.json",
                        help="Path to the data file")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                        help="Whether to load model in 4bit")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Per device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Maximum training steps, overrides epochs if set")
    parser.add_argument("--output_dir", type=str, default="outputs", 
                        help="Output directory for checkpoints")
    parser.add_argument("--save_model", type=str, default="inverse_sft_model_full_train",
                        help="Directory to save the final model")
    parser.add_argument("--hf_repo_id", type=str, default="backwards-guanaco-llama3.1-8b-sft",
                        help="Hugging Face repository ID to push the model to")
    parser.add_argument("--hf_token", type=str, default=r"hf_",
                        help="Hugging Face token for uploading model")
    parser.add_argument("--save_gguf", action="store_true", default=False,
                        help="Whether to save model in GGUF format")
    parser.add_argument("--quantization_method", type=str, default="q8_0", 
                        help="Quantization method for GGUF (q8_0, q4_k_m, etc.)")
    parser.add_argument("--wandb_project", type=str, default="inverse-sft", 
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, 
                        help="Weights & Biases run name")
    parser.add_argument("--wandb_log", action="store_true", default=True,
                        help="Whether to log to Weights & Biases")
    return parser.parse_args()


def load_inverse_sft_data(data_path):
    """Load data from the inverse SFT format where we predict user questions from assistant responses."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logging.info(f"Loading data from {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to the format needed for the dataset
    examples = []
    for item in data:
        examples.append({
            "instruction": item.get("instruction", ""),
            "input": item.get("input", ""),
            "output": item.get("output", "")
        })
    
    return Dataset.from_list(examples)


def format_inverse_sft_examples(examples, tokenizer):
    """Format the inverse SFT examples for the model."""
    formatted_texts = []
    
    for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
        # For inverse SFT, we want the model to predict the "output" (original user question)
        # given the "instruction" and "input" (assistant response)
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": output}
        ]
        
        formatted = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        formatted_texts.append(formatted)
    
    return {"text": formatted_texts}


def log_sample_data(dataset, num_samples=2):
    """Log a few samples of processed data to wandb and local logs"""
    logging.info("\n" + "="*50)
    logging.info("Sample of processed training data:")
    logging.info("="*50)
    
    # Create a W&B Table for samples
    columns = ["text"]
    table = wandb.Table(columns=columns)
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]['text']
        logging.info(f"\nSample {i+1}:")
        logging.info("-"*50)
        logging.info(sample)
        logging.info("-"*50)
        
        # Add to W&B table
        table.add_data(sample)
    
    # Log the table to W&B
    wandb.log({"training_samples": table})


def main():
    args = parse_args()
    
    # Initialize wandb if enabled
    if args.wandb_log:
        run_name = args.wandb_run_name or f"inverse-sft-{args.model_name.split('/')[-1]}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            job_type="training",
        )
        logging.info(f"Initialized wandb run: {run_name}")
    
    try:
        # Set up model and tokenizer
        logging.info(f"Loading model: {args.model_name}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_name,
            max_seq_length=args.max_seq_length,
            dtype=None,  # Auto detection
            load_in_4bit=args.load_in_4bit,
            # full_finetuning=True,
        )
        
        # Add LoRA adapters
        logging.info("Adding LoRA adapters")
        model = FastLanguageModel.get_peft_model(
            model,
            r=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        
        # Set up chat template
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="llama-3.1",
        )
        
        # Load and prepare dataset
        dataset = load_inverse_sft_data(args.data_path)
        logging.info(f"Loaded {len(dataset)} examples")
        
        # Display a sample example
        if len(dataset) > 0:
            logging.info("\nSample data example:")
            sample = dataset[0]
            logging.info(f"Instruction: {sample['instruction']}")
            logging.info(f"Input (Assistant Response): {sample['input'][:100]}...")
            logging.info(f"Output (User Question): {sample['output']}")
        
        # Map to formatted examples
        formatted_dataset = dataset.map(
            lambda examples: format_inverse_sft_examples(examples, tokenizer),
            batched=True,
        )
        
        # Log sample data to wandb if enabled
        if args.wandb_log:
            log_sample_data(formatted_dataset)
        if args.max_steps is not None:
            max_steps = args.max_steps
        else:
            max_steps = int(args.epochs * len(formatted_dataset) // args.batch_size // args.gradient_accumulation_steps)
        # Create SFT trainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=formatted_dataset,
            # data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
            args=SFTConfig(
                per_device_train_batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                max_seq_length=args.max_seq_length,
                packing=False,
                warmup_ratio=0.1,
                num_train_epochs=args.epochs,
                max_steps=max_steps,
                learning_rate=args.learning_rate,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir=args.output_dir,
                report_to="wandb" if args.wandb_log else "none",
            ),
        )
        
        # Print GPU memory stats before training
        start_gpu_memory = 0
        max_memory = 0
        if torch.cuda.is_available():
            gpu_stats = torch.cuda.get_device_properties(0)
            start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
            max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
            logging.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
            logging.info(f"{start_gpu_memory} GB of memory reserved.")
        
        # Train the model
        logging.info("Starting training...")

        import os
        os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
        trainer_stats = trainer.train()
        
        # Print training stats
        training_time = trainer_stats.metrics.get('train_runtime', 0)
        training_loss = trainer_stats.metrics.get('train_loss', 0)
        logging.info(f"Training completed in {training_time} seconds ({round(training_time/60, 2)} minutes)")
        logging.info(f"Final training loss: {training_loss}")
        
        if torch.cuda.is_available():
            used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
            used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
            used_percentage = round(used_memory / max_memory * 100, 3)
            lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
            logging.info(f"Peak reserved memory = {used_memory} GB.")
            logging.info(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
            logging.info(f"Peak reserved memory % of max memory = {used_percentage} %.")
            logging.info(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
            
            # Log memory stats to wandb
            if args.wandb_log:
                wandb.log({
                    "peak_gpu_memory_gb": used_memory,
                    "training_gpu_memory_gb": used_memory_for_lora,
                    "peak_gpu_memory_percent": used_percentage,
                    "training_gpu_memory_percent": lora_percentage,
                })
        
        # Log final metrics to wandb
        if args.wandb_log:
            wandb.log({
                "final_loss": training_loss,
                "total_steps": trainer_stats.global_step,
                "training_time": training_time,
            })
        
        # Quick test inference
        logging.info("\nTesting model with inference:")
        FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
        
        # Test with a sample from the dataset if available
        if len(dataset) > 0:
            test_sample = dataset[0]
            
            messages = [
                {"role": "system", "content": test_sample["instruction"]},
                {"role": "user", "content": test_sample["input"]},
            ]
            
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            
            text_streamer = TextStreamer(tokenizer, skip_prompt=True)
            logging.info("Model output (predicted user question):")
            output = model.generate(
                input_ids=inputs,
                streamer=text_streamer,
                max_new_tokens=128,
                use_cache=True,
                temperature=1.5,
                min_p=0.1
            )
            
            # Get the model's output text by decoding and removing the prompt
            output_text = tokenizer.decode(output[0])
            logging.info(output_text)

            logging.info("\nExpected output:")
            logging.info(test_sample["output"])
            
            # Log inference results to wandb
            if args.wandb_log:
                wandb.log({
                    "inference_example": {
                        "instruction": test_sample["instruction"],
                        "input": test_sample["input"][:500] + "..." if len(test_sample["input"]) > 500 else test_sample["input"],
                        "expected": test_sample["output"],
                        "generated": output_text
                    }
                })
        
        # Save the trained model locally
        model_path = os.path.join(args.output_dir, args.save_model)
        logging.info(f"\nSaving model to {model_path}")
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        
        # Upload to Hugging Face if specified
        if args.hf_repo_id and args.hf_token:
            logging.info(f"Pushing model to Hugging Face: {args.hf_repo_id}")
            model.push_to_hub(args.hf_repo_id, token=args.hf_token)
            tokenizer.push_to_hub(args.hf_repo_id, token=args.hf_token)
        
        # Save in GGUF format if specified
        if args.save_gguf:
            logging.info(f"Saving model in GGUF format with quantization method: {args.quantization_method}")
            model.save_pretrained_gguf(model_path, tokenizer, quantization_method=args.quantization_method)
            
            # Upload GGUF to Hugging Face if specified
            if args.hf_repo_id and args.hf_token:
                logging.info(f"Pushing GGUF model to Hugging Face: {args.hf_repo_id}")
                model.push_to_hub_gguf(args.hf_repo_id, tokenizer, 
                                    quantization_method=args.quantization_method, 
                                    token=args.hf_token)
        
        # Log the model to wandb if enabled
        if args.wandb_log:
            artifact = wandb.Artifact(
                name=f"inverse-sft-model-{wandb.run.id}",
                type="model",
                description="Inverse SFT fine-tuned model with LoRA"
            )
            artifact.add_dir(model_path)
            wandb.log_artifact(artifact)
        
        logging.info("Training and model saving complete!")
        
        return trainer_stats
    
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        if args.wandb_log:
            wandb.alert(
                title="Training Failed",
                text=f"Training failed with error: {str(e)}"
            )
        raise e
    
    finally:
        # Close wandb run if enabled
        if args.wandb_log:
            wandb.finish()


if __name__ == "__main__":
    main() 