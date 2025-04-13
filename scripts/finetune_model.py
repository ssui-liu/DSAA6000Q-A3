#!/usr/bin/env python3
"""
Fine-tuning script: Trains a language model on the curated instruction dataset.
Uses Unsloth for efficient LoRA fine-tuning and saves the model to Hugging Face.
"""
import unsloth
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
import os
import json
import argparse
import torch
import wandb
import logging
from datasets import Dataset
from transformers import TrainingArguments, TextStreamer
from trl import SFTTrainer, SFTConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('finetune_training.log'),
        logging.StreamHandler()
    ]
)

# Optional: Set your W&B API key here or use environment variable
# os.environ["WANDB_API_KEY"] = "your-key-here"

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune model with Unsloth")
    parser.add_argument("--model_name", type=str, default="unsloth/Llama-3.1-8B-Instruct",
                        help="Base model name to use for fine-tuning")
    parser.add_argument("--data_path", type=str, default="data/curated_dataset.json", 
                        help="Path to the curated dataset file")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                        help="Whether to load model in 4bit")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Per device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Maximum training steps, overrides epochs if set")
    parser.add_argument("--output_dir", type=str, default="outputs", 
                        help="Output directory for checkpoints")
    parser.add_argument("--save_model", type=str, default="lima_ft_model",
                        help="Directory to save the final model")
    parser.add_argument("--hf_repo_id", type=str, default="llama3.1-8b-lima-sft",
                        help="Hugging Face repository ID to push the model to")
    parser.add_argument("--hf_token", type=str, default=r"hf_",
                        help="Hugging Face token for uploading model")
    parser.add_argument("--save_gguf", action="store_true", default=False,
                        help="Whether to save model in GGUF format")
    parser.add_argument("--wandb_project", type=str, default="lima-finetune", 
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, 
                        help="Weights & Biases run name")
    parser.add_argument("--wandb_log", action="store_true", default=True,
                        help="Whether to log to Weights & Biases")
    return parser.parse_args()


def load_dataset_from_json(data_path):
    """Load the curated dataset for fine-tuning."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logging.info(f"Loading curated dataset from {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to a format suitable for training
    formatted_data = []
    for item in data:
        # Get fields from the dataset
        instruction = item.get("generated_instruction", "")
        response = item.get("response", "")
        
        # Format for the trainer
        formatted_data.append({
            "instruction": instruction,
            "output": response
        })
    
    logging.info(f"Loaded {len(formatted_data)} examples for training")
    return Dataset.from_list(formatted_data)


def format_examples(examples, tokenizer):
    """Format the examples for the model."""
    formatted_texts = []
    
    for instruction, output in zip(examples["instruction"], examples["output"]):
        # Standard chat format with system, user (instruction), and assistant (output)
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": instruction},
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


def test_model_generation(model, tokenizer, dataset, num_samples=5):
    """Run inference on a few examples to showcase the model's capabilities."""
    logging.info("\n===== Test Generations from Fine-tuned Model =====")
    FastLanguageModel.for_inference(model)  # Switch to inference mode
    
    # Sample a few examples from the dataset
    samples = dataset.select(range(min(num_samples, len(dataset))))
    
    generations = []
    for i, sample in enumerate(samples):
        instruction = sample["instruction"]
        original_output = sample["output"]
        
        # Format input for generation
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": instruction}
        ]
        
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        
        # Generate text
        streamer = TextStreamer(tokenizer, skip_prompt=True)
        logging.info(f"\nExample {i+1}:")
        logging.info(f"Instruction: {instruction}")
        logging.info("Generated:")
        
        outputs = model.generate(
            input_ids=inputs,
            streamer=streamer,
            max_new_tokens=512,
            use_cache=True,
            temperature=0.7,
            top_p=0.9
        )
        
        # Decode the response
        generated_text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        logging.info(f"\nOriginal: {original_output[:150]}..." if len(original_output) > 150 else original_output)
        logging.info("-" * 50)
        
        generations.append({
            "instruction": instruction,
            "generated": generated_text,
            "original": original_output
        })
        
        # Log to wandb
        if wandb.run is not None:
            wandb.log({
                f"generation_example_{i}": wandb.Table(
                    columns=["instruction", "generated", "original"],
                    data=[[instruction, generated_text, original_output]]
                )
            })
    
    return generations


def main():
    args = parse_args()
    
    # Initialize wandb if enabled
    if args.wandb_log:
        run_name = args.wandb_run_name or f"finetune-{args.model_name.split('/')[-1]}"
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
        )
        
        # Add LoRA adapters
        logging.info("Adding LoRA adapters")
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
        )
        
        # Set up chat template
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="llama-3.1",
        )
        
        # Load and prepare dataset
        dataset = load_dataset_from_json(args.data_path)
        
        # Display a sample example
        if len(dataset) > 0:
            logging.info("\nSample data example:")
            sample = dataset[0]
            logging.info(f"Instruction: {sample['instruction']}")
            logging.info(f"Output: {sample['output'][:100]}...")
        
        # Map to formatted examples
        formatted_dataset = dataset.map(
            lambda examples: format_examples(examples, tokenizer),
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
        trainer_stats = trainer.train()
        
        # Print training stats
        training_time = trainer_stats.metrics.get('train_runtime', 0)
        training_loss = trainer_stats.metrics.get('train_loss', 0)
        logging.info(f"Training completed in {training_time} seconds ({round(training_time/60, 2)} minutes)")
        logging.info(f"Final training loss: {training_loss}")
        
        if torch.cuda.is_available():
            used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
            used_memory_for_training = round(used_memory - start_gpu_memory, 3)
            used_percentage = round(used_memory / max_memory * 100, 3)
            training_percentage = round(used_memory_for_training / max_memory * 100, 3)
            logging.info(f"Peak reserved memory = {used_memory} GB.")
            logging.info(f"Peak reserved memory for training = {used_memory_for_training} GB.")
            logging.info(f"Peak reserved memory % of max memory = {used_percentage} %.")
            logging.info(f"Peak reserved memory for training % of max memory = {training_percentage} %.")
            
            # Log memory stats to wandb
            if args.wandb_log:
                wandb.log({
                    "peak_gpu_memory_gb": used_memory,
                    "training_gpu_memory_gb": used_memory_for_training,
                    "peak_gpu_memory_percent": used_percentage,
                    "training_gpu_memory_percent": training_percentage,
                })
        
        # Log final metrics to wandb
        if args.wandb_log:
            wandb.log({
                "final_loss": training_loss,
                "total_steps": trainer_stats.global_step,
                "training_time": training_time,
            })
        
        # Test the model by generating a few examples
        logging.info("\nGenerating example responses with the fine-tuned model:")
        test_generations = test_model_generation(model, tokenizer, dataset)
        
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
            
            logging.info(f"Model available at: https://huggingface.co/{args.hf_repo_id}")
        
        # Save in GGUF format if specified
        if args.save_gguf:
            logging.info(f"Saving model in GGUF format")
            model.save_pretrained_gguf(model_path, tokenizer)
            
            # Upload GGUF to Hugging Face if specified
            if args.hf_repo_id and args.hf_token:
                logging.info(f"Pushing GGUF model to Hugging Face: {args.hf_repo_id}")
                model.push_to_hub_gguf(args.hf_repo_id, tokenizer, token=args.hf_token)
        
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


"""
Generating example responses with the fine-tuned model:
2025-04-12 23:36:24,029 - INFO - 
===== Test Generations from Fine-tuned Model =====
2025-04-12 23:36:24,031 - INFO - 
Example 1:
2025-04-12 23:36:24,032 - INFO - Instruction: What can you do with powdered peanut butter?
2025-04-12 23:36:24,032 - INFO - Generated:
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
You can mix it with milk or yogurt to make a peanut butter milkshake. You can mix it with sugar and water to make peanut butter syrup. You can mix it with a little water to make a peanut butter paste that you can use as a substitute for peanut butter in any recipe. You can mix it with cream cheese to make peanut butter frosting for cakes and cookies.<|eot_id|>
2025-04-12 23:36:26,949 - INFO - 
Original: Powdered peanut butter is a versatile ingredient that you can use to add some nutty flavor to any dish. Its low caloric count and limited fat content ...
2025-04-12 23:36:26,950 - INFO - --------------------------------------------------
2025-04-12 23:36:27,202 - INFO - 
Example 2:
2025-04-12 23:36:27,202 - INFO - Instruction: write a jquery event that triggers when a user clicks the edit button and disables the text area
2025-04-12 23:36:27,203 - INFO - Generated:
You can use the following jQuery event to disable the text area when the user clicks the edit button:

```javascript
$('button#edit').on('click', function() {
  $('textarea#text').attr('disabled', true);
  $('textarea#text').attr('readonly', true);
});
```

This event will be triggered when the user clicks the button with the ID "edit". When triggered, the event will disable and make the text area readonly.<|eot_id|>
2025-04-12 23:36:30,824 - INFO - 
Original: Always use the ```prop()``` method to enable or disable elements when using jQuery (see below for why).
In your case, it would be:
```$(&quot;#edit&qu...
2025-04-12 23:36:30,824 - INFO - --------------------------------------------------
2025-04-12 23:36:31,081 - INFO - 
Example 3:
2025-04-12 23:36:31,082 - INFO - Instruction: How can I make my docker mysql allow root access on 3306 port?
2025-04-12 23:36:31,082 - INFO - Generated:
You can modify the `mysqld` service to allow root access by adding the following lines to the `config` section:

```yaml
config:
  #...
  user: "root"
  password: "your_password"
  host: "0.0.0.0"
  #...
```

Then, run the following command to restart the service:

```
docker-compose up -d
```

After that, you should be able to connect to the database using the following command:

```
mysql -u root -p your_password
```

Please note that you should never expose the root password to the public. It's a good practice to create a new user with limited permissions and use that user to connect to the database instead of the root user.

Also, please note that allowing root access to the database is a potential security risk. You should always limit access to the database to only the users and services that need it, and never expose the root password to the public.<|eot_id|>
2025-04-12 23:36:38,620 - INFO - 
Original: By default after deployment MySQL has following connection restrictions:
```mysql> select host, user from mysql.user;
+-----------+---------------+
| ...
2025-04-12 23:36:38,620 - INFO - --------------------------------------------------
2025-04-12 23:36:38,882 - INFO - 
Example 4:
2025-04-12 23:36:38,883 - INFO - Instruction: What is the purpose of ```|``` in bash scripting?
2025-04-12 23:36:38,883 - INFO - Generated:
In bash scripting, ```|``` is a pipe operator. It is used to pipe the output of one command to the input of another command. This is useful for creating complex commands that can be used to perform a variety of tasks.

For example, you can use the ```|``` operator to pipe the output of the ```ls``` command to the ```grep``` command, like this:

```bash
ls | grep file
```

This will list all files in the current directory and then pipe the output to the ```grep``` command, which will print only the lines that contain the string "file".

You can also use the ```|``` operator to pipe the output of one command to the input of another command that is executed in a subshell, like this:

```bash
ls | (grep file; echo "This is a file")
```

This will list all files in the current directory, pipe the output to the ```grep``` command, which will print only the lines that contain the string "file", and then pipe the output to the ```echo``` command, which will print the string "This is a file".

You can also use the ```|``` operator to pipe the output of one command to the input of another command that is executed in a subshell, but the subshell is executed in a different shell, like this:

```bash
ls | (bash -c "grep file; echo 'This is a file'")
```

This will list all files in the current directory, pipe the output to the ```grep``` command, which will print only the lines that contain the string "file", and then pipe the output to the ```echo``` command, which will print the string "This is a file", and then exit the subshell.

You can also use the ```|``` operator to pipe the output of one command to the input of another command that is executed in a subshell, but the subshell is executed in a different shell, and the subshell is executed in a different directory, like this:

```bash
ls | (cd /path/to/directory; bash -c "grep file; echo 'This is a file'")
```

This will list all files in the current directory, pipe the output to the ```grep``` command, which will print only the lines that contain the string "file", and then pipe the output to the ```echo``` command, which will print the string "This is a file", and then exit the subshell.

You can also use the ```|
2025-04-12 23:36:57,531 - INFO - 
Original: You are using ```|``` (pipe) to direct the output of a command into another command. What you are looking for is ```&&``` operator to execute the next...
2025-04-12 23:36:57,532 - INFO - --------------------------------------------------
2025-04-12 23:36:57,772 - INFO - 
Example 5:
2025-04-12 23:36:57,773 - INFO - Instruction: Why should the system partition be wiped on bootloader unlock in the Android documentation
2025-04-12 23:36:57,774 - INFO - Generated:
The system partition being wiped on bootloader unlock is a common practice in Android. The system partition is where the operating system itself is stored. When you unlock the bootloader, the system partition is wiped so that any existing system software can't interfere with the new system software that you are about to install.

It is not necessary to wipe the system partition if you don't plan to install any new system software. If you unlock the bootloader and install the stock ROM, the system partition will be wiped anyway. If you unlock the bootloader and install a custom ROM, the system partition will be wiped so that the existing system software can't interfere with the new system software. If you unlock the bootloader and install a custom recovery image, the system partition will be wiped so that the existing system software can't interfere with the new recovery image.

There are a few reasons why the system partition is wiped on bootloader unlock. One reason is that the system partition is where the operating system itself is stored. When you unlock the bootloader, the system partition is wiped so that any existing system software can't interfere with the new system software that you are about to install. Another reason is that the system partition is where the bootloader itself is stored. When you unlock the bootloader, the system partition is wiped so that the existing bootloader can't interfere with the new bootloader.

It is worth noting that wiping the system partition on bootloader unlock is not a requirement. If you don't plan to install any new system software, you can skip wiping the system partition. If you do plan to install new system software, you should wipe the system partition to ensure that the existing system software can't interfere with the new system software.<|eot_id|>
2025-04-12 23:37:10,356 - INFO - 
Original: It's a security concern. The Android documentation doesn't provide a lot of elaboration, stating simply:

  The procedure must be confirmed on-screen,...


"""