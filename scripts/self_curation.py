#!/usr/bin/env python3
"""
Self-Curation script for filtering high-quality examples from self-augmented LIMA dataset.
Uses Unsloth-optimized models to evaluate and rate instruction-response pairs on a 1-5 scale.
"""
import unsloth
import os
import json
import argparse
import logging
import re
import torch
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
from datasets import Dataset
from huggingface_hub import HfApi
from unsloth import FastLanguageModel
from transformers import TextStreamer
from unsloth.chat_templates import get_chat_template

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('self_curation.log'),
        logging.StreamHandler()
    ]
)

# The evaluation prompt template from Table 19
EVALUATION_PROMPT = """Below is an instruction from an user and a candidate answer. Evaluate whether or
not the answer is a good example of how AI Assistant should respond to the user's
instruction. Please assign a score using the following 5-point scale:

1: It means the answer is incomplete, vague, off-topic, controversial, or not
exactly what the user asked for. For example, some content seems missing, numbered
list does not start from the beginning, the opening sentence repeats user's question.
Or the response is from another person's perspective with their personal experience
(e.g. taken from blog posts), or looks like an answer from a forum. Or it contains
promotional text, navigation text, or other irrelevant information.

2: It means the answer addresses most of the asks from the user. It does not
directly address the user's question. For example, it only provides a high-level
methodology instead of the exact solution to user's question.

3: It means the answer is helpful but not written by an AI Assistant. It addresses
all the basic asks from the user. It is complete and self contained with the
drawback that the response is not written from an AI assistant's perspective, but
from other people's perspective. The content looks like an excerpt from a blog post,
web page, or web search results. For example, it contains personal experience or
opinion, mentions comments section, or share on social media, etc.

4: It means the answer is written from an AI assistant's perspective with a
clear focus of addressing the instruction. It provide a complete, clear, and
comprehensive response to user's question or instruction without missing or
irrelevant information. It is well organized, self-contained, and written in a
helpful tone. It has minor room for improvement, e.g. more concise and focused.

5: It means it is a perfect answer from an AI Assistant. It has a clear focus on
being a helpful AI Assistant, where the response looks like intentionally written
to address the user's question or instruction without any irrelevant sentences. The
answer provides high quality content, demonstrating expert knowledge in the area, is
very well written, logical, easy-to-follow, engaging and insightful.
Please first provide a brief reasoning you used to derive the rating score, and
then write "Score: <rating>" in the last line.

Instruction: {instruction}

Answer: {response}
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Self-Curation with LLM evaluation")
    parser.add_argument("--input_file", type=str, default="data/self_augmented2.json",
                        help="Path to the self-augmented data file")
    parser.add_argument("--output_file", type=str, default="data/curated_dataset.json",
                        help="Path to save the curated data")
    parser.add_argument("--quality_threshold", type=int, default=4,
                        help="Minimum quality score (1-5) to retain examples")
    parser.add_argument("--model_name", type=str, default="unsloth/Llama-3.1-8B-Instruct",
                        help="LLM to use for evaluation (using Unsloth models)")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Maximum number of examples to evaluate (for testing)")
    parser.add_argument("--push_to_hub", action="store_true", default=False,
                        help="Whether to push the dataset to Hugging Face Hub")
    parser.add_argument("--hub_repo_id", type=str, default="user/lima-self-instruct-curated",
                        help="Hugging Face Hub repository ID")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for evaluation")
    return parser.parse_args()

def load_self_augmented_data(input_file):
    """Load the self-augmented data."""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    logging.info(f"Loading self-augmented data from {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logging.info(f"Loaded {len(data)} examples from self-augmented data")
    return data

def load_evaluation_model(model_name):
    """Load the LLM for evaluation using Unsloth's optimized inference."""
    logging.info(f"Loading evaluation model with Unsloth: {model_name}")
    
    # Load model and tokenizer with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=4096,  # Longer sequence length for evaluation
        dtype=None,  # Auto detection
        load_in_4bit=True,  # Use 4-bit quantization for memory efficiency
        # inference_only=True,  # We only need inference mode
    )
    
    # Set up the model for faster inference with Unsloth
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        use_gradient_checkpointing=False,  # No need for gradient checkpointing during inference
        random_state=3407,
        use_rslora=False,
    )
    
    # Enable fast inference mode
    FastLanguageModel.for_inference(model)
    
    # Configure chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",  # Specify the chat template to use
    )
    
    return model, tokenizer

def extract_score(text):
    """Extract the numerical score from the LLM's evaluation output."""
    # Look for 'Score: X' pattern at the end of the text
    score_pattern = r'Score:\s*(\d+)'
    match = re.search(score_pattern, text)
    
    if match:
        score = int(match.group(1))
        # Validate score is in range 1-5
        if 1 <= score <= 5:
            return score
    
    # Fallback: Try to find any number between 1-5 in the text
    numbers = re.findall(r'\b[1-5]\b', text)
    if numbers:
        # Take the last number as the score (most likely to be the final rating)
        return int(numbers[-1])
    
    # Default score if we couldn't extract anything
    logging.warning(f"Could not extract score from: {text}")
    return None

def evaluate_examples(model, tokenizer, examples, batch_size=1, max_examples=None):
    """
    Evaluate the quality of instruction-response pairs using the LLM with Unsloth.
    Returns examples with their scores.
    """
    evaluated_examples = []
    
    # Limit the number of examples if specified
    if max_examples is not None:
        examples = examples[:max_examples]
    
    logging.info(f"Evaluating {len(examples)} examples...")
    
    for i, example in enumerate(tqdm(examples)):
        instruction = example.get("generated_instruction", "")
        response = example.get("response", "")
        
        # Format the evaluation as a system message followed by the evaluation prompt
        messages = [
            {"role": "system", "content": "You are an expert evaluator of instruction-response pairs."},
            {"role": "user", "content": EVALUATION_PROMPT.format(
                instruction=instruction,
                response=response
            )}
        ]
        
        # Apply chat template with Unsloth
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        
        # Create text streamer for more efficient generation
        text_streamer = TextStreamer(tokenizer, skip_prompt=True)
        
        # Generate evaluation with Unsloth
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=512,
            temperature=0.1,  # Lower temperature for more consistent ratings
            do_sample=False,  # Deterministic to get consistent ratings
            use_cache=True,   # Enable KV-cache for faster generation
            streamer=text_streamer if i % 10 == 0 else None,  # Only stream for some examples
        )
        
        # Decode output (skip prompt tokens)
        evaluation_text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        # Extract score
        score = extract_score(evaluation_text)
        
        # Store example with evaluation
        evaluated_example = example.copy()
        evaluated_example["evaluation"] = evaluation_text
        evaluated_example["quality_score"] = score
        evaluated_examples.append(evaluated_example)
        
        if i % 10 == 0:
            logging.info(f"Processed {i+1}/{len(examples)} examples")
            if score is not None:
                logging.info(f"Sample score: {score}/5")
            else:
                logging.info("Could not extract score from this example")
    
    return evaluated_examples

def filter_high_quality_examples(evaluated_examples, threshold=4):
    """Filter examples with quality score >= threshold."""
    # Filter out examples with None scores
    valid_examples = [ex for ex in evaluated_examples if ex["quality_score"] is not None]
    
    # Filter high-quality examples
    high_quality = [ex for ex in valid_examples if ex["quality_score"] >= threshold]
    # Filter low-quality examples
    low_quality = [ex for ex in valid_examples if ex["quality_score"] < threshold]
    
    logging.info(f"Filtered {len(high_quality)}/{len(valid_examples)} high-quality examples (score >= {threshold})")
    logging.info(f"Identified {len(low_quality)}/{len(valid_examples)} low-quality examples (score < {threshold})")
    
    # Count by score
    score_counts = {}
    for ex in valid_examples:
        score = ex["quality_score"]
        score_counts[score] = score_counts.get(score, 0) + 1
    
    logging.info("Score distribution:")
    for score in sorted(score_counts.keys()):
        count = score_counts[score]
        percentage = count / len(valid_examples) * 100
        logging.info(f"  Score {score}: {count} examples ({percentage:.1f}%)")
    
    return high_quality, low_quality

def push_to_huggingface(dataset, repo_id):
    """Push dataset to Hugging Face Hub."""
    logging.info(f"Pushing dataset to Hugging Face Hub: {repo_id}")
    
    # Convert to HF Dataset format
    hf_dataset = Dataset.from_list(dataset)
    
    # Push to Hub
    hf_dataset.push_to_hub(repo_id)
    
    logging.info(f"Dataset pushed to: https://huggingface.co/datasets/{repo_id}")
    return f"https://huggingface.co/datasets/{repo_id}"

def print_example_samples(high_quality, low_quality, n=5):
    """Print n examples each of high and low quality examples."""
    logging.info("\n===== High Quality Examples =====")
    for i, example in enumerate(high_quality[:n]):
        logging.info(f"\nExample {i+1} (Score: {example['quality_score']}):")
        logging.info(f"Instruction: {example['generated_instruction']}")
        response_preview = example['response'][:150] + "..." if len(example['response']) > 150 else example['response']
        logging.info(f"Response: {response_preview}")
        logging.info(f"Evaluation: {example['evaluation'][:200]}...")
        logging.info("-" * 50)
    
    logging.info("\n===== Low Quality Examples =====")
    for i, example in enumerate(low_quality[:n]):
        logging.info(f"\nExample {i+1} (Score: {example['quality_score']}):")
        logging.info(f"Instruction: {example['generated_instruction']}")
        response_preview = example['response'][:150] + "..." if len(example['response']) > 150 else example['response']
        logging.info(f"Response: {response_preview}")
        logging.info(f"Evaluation: {example['evaluation'][:200]}...")
        logging.info("-" * 50)

def save_curated_dataset(high_quality, output_file, full_evaluated=None):
    """Save the curated dataset and optionally the full evaluated dataset."""
    # Save the high-quality examples
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(high_quality, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved {len(high_quality)} curated examples to {output_file}")
    
    # If full evaluated dataset is provided, save it too
    if full_evaluated:
        full_output = output_file.replace('.json', '_all_evaluated.json')
        with open(full_output, 'w', encoding='utf-8') as f:
            json.dump(full_evaluated, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved all {len(full_evaluated)} evaluated examples to {full_output}")

def main():
    args = parse_args()
    
    # Load the self-augmented data
    examples = load_self_augmented_data(args.input_file)
    
    # Load evaluation model with Unsloth
    model, tokenizer = load_evaluation_model(args.model_name)
    
    # Evaluate examples
    evaluated_examples = evaluate_examples(
        model, 
        tokenizer, 
        examples, 
        batch_size=args.batch_size,
        max_examples=args.max_examples
    )
    
    # Filter high-quality examples
    high_quality, low_quality = filter_high_quality_examples(
        evaluated_examples, 
        threshold=args.quality_threshold
    )
    
    # Print sample examples
    print_example_samples(high_quality, low_quality)
    
    # Save the curated dataset
    save_curated_dataset(high_quality, args.output_file, full_evaluated=evaluated_examples)
    
    # Push to Hugging Face Hub if requested
    if args.push_to_hub:
        repo_url = push_to_huggingface(high_quality, args.hub_repo_id)
        logging.info(f"Dataset available at: {repo_url}")
    
    logging.info("Self-curation completed successfully!")

if __name__ == "__main__":
    main()

"""
2025-04-12 22:12:37,966 - INFO - Filtered 111/150 high-quality examples (score >= 4)
2025-04-12 22:12:37,966 - INFO - Identified 39/150 low-quality examples (score < 4)
2025-04-12 22:12:37,966 - INFO - Score distribution:
2025-04-12 22:12:37,966 - INFO -   Score 1: 15 examples (10.0%)
2025-04-12 22:12:37,966 - INFO -   Score 2: 8 examples (5.3%)
2025-04-12 22:12:37,966 - INFO -   Score 3: 16 examples (10.7%)
2025-04-12 22:12:37,966 - INFO -   Score 4: 104 examples (69.3%)
2025-04-12 22:12:37,966 - INFO -   Score 5: 7 examples (4.7%)
2025-04-12 22:12:37,966 - INFO - 
===== High Quality Examples =====
2025-04-12 22:12:37,966 - INFO - 
Example 1 (Score: 4):
2025-04-12 22:12:37,966 - INFO - Instruction: What can you do with powdered peanut butter?
2025-04-12 22:12:37,966 - INFO - Response: Powdered peanut butter is a versatile ingredient that you can use to add some nutty flavor to any dish. Its low caloric count and limited fat content ...
2025-04-12 22:12:37,966 - INFO - Evaluation: The answer provided is a comprehensive and well-organized response to the user's question about what can be done with powdered peanut butter. It addresses all the basic asks from the user, including p...
2025-04-12 22:12:37,969 - INFO - --------------------------------------------------
2025-04-12 22:12:37,969 - INFO - 
Example 2 (Score: 4):
2025-04-12 22:12:37,969 - INFO - Instruction: write a jquery event that triggers when a user clicks the edit button and disables the text area
2025-04-12 22:12:37,969 - INFO - Response: Always use the ```prop()``` method to enable or disable elements when using jQuery (see below for why).
In your case, it would be:
```$(&quot;#edit&qu...
2025-04-12 22:12:37,969 - INFO - Evaluation: The answer provided is a good example of how an AI Assistant should respond to the user's instruction. It addresses the user's question directly and provides a clear explanation of why the `prop()` me...
2025-04-12 22:12:37,969 - INFO - --------------------------------------------------
2025-04-12 22:12:37,969 - INFO - 
Example 3 (Score: 4):
2025-04-12 22:12:37,969 - INFO - Instruction: How can I make my docker mysql allow root access on 3306 port?
2025-04-12 22:12:37,969 - INFO - Response: By default after deployment MySQL has following connection restrictions:
```mysql> select host, user from mysql.user;
+-----------+---------------+
| ...
2025-04-12 22:12:37,969 - INFO - Evaluation: The answer provided is a good example of how an AI Assistant should respond to the user's instruction. It addresses the user's question directly and provides a clear, step-by-step solution to allow ro...
2025-04-12 22:12:37,969 - INFO - --------------------------------------------------
2025-04-12 22:12:37,969 - INFO - 
Example 4 (Score: 4):
2025-04-12 22:12:37,969 - INFO - Instruction: What is the purpose of ```|``` in bash scripting?
2025-04-12 22:12:37,969 - INFO - Response: You are using ```|``` (pipe) to direct the output of a command into another command. What you are looking for is ```&&``` operator to execute the next...
2025-04-12 22:12:37,969 - INFO - Evaluation: The answer provided is a good example of how an AI Assistant should respond to the user's instruction. It directly addresses the user's question, providing a clear explanation of the purpose of the pi...
2025-04-12 22:12:37,969 - INFO - --------------------------------------------------
2025-04-12 22:12:37,969 - INFO - 
Example 5 (Score: 4):
2025-04-12 22:12:37,969 - INFO - Instruction: Why should the system partition be wiped on bootloader unlock in the Android documentation
2025-04-12 22:12:37,969 - INFO - Response: It's a security concern. The Android documentation doesn't provide a lot of elaboration, stating simply:

  The procedure must be confirmed on-screen,...
2025-04-12 22:12:37,969 - INFO - Evaluation: The answer provided is a good example of how an AI Assistant should respond to the user's instruction. Here's the reasoning behind the rating:

The answer directly addresses the user's question by exp...
2025-04-12 22:12:37,969 - INFO - --------------------------------------------------
2025-04-12 22:12:37,969 - INFO - 
===== Low Quality Examples =====
2025-04-12 22:12:37,970 - INFO - 
Example 1 (Score: 1):
2025-04-12 22:12:37,970 - INFO - Instruction: The Vice correspondent Gianna Toboni visited India, where she had an amazing trip and got to see how people live there. During this trip she also visited the local medical community and asked them for a summary of their last work assignment. Please paraphase the text from Gianna's assignment.
2025-04-12 22:12:37,970 - INFO - Response: Here is the summary: 
Vice correspondent Gianna Toboni traveled to India to explore the country's booming gestational surrogacy industry for the HBO s...
2025-04-12 22:12:37,970 - INFO - Evaluation: The answer is not a good example of how an AI Assistant should respond to the user's instruction. The instruction asks for a paraphrased summary of Gianna Toboni's assignment, but the answer provides ...
2025-04-12 22:12:37,970 - INFO - --------------------------------------------------
2025-04-12 22:12:37,970 - INFO - 
Example 2 (Score: 3):
2025-04-12 22:12:37,970 - INFO - Instruction: 2 children together sold 4 apples each. Another child gave out an additional 7 apples. 60 apples are left after the removal. How many children are there?
2025-04-12 22:12:37,970 - INFO - Response: Here is the solution step by step:

2 children sold 4 apples each which gives 2 * 4 = 8 apples
Another child gave out an additional 7 apples making 8 ...
2025-04-12 22:12:37,970 - INFO - Evaluation: The answer provided is a step-by-step solution to the problem, but it does not directly address the user's question, which is "How many children are there?" The answer focuses on the total number of a...
2025-04-12 22:12:37,970 - INFO - --------------------------------------------------
2025-04-12 22:12:37,970 - INFO - 
Example 3 (Score: 1):
2025-04-12 22:12:37,970 - INFO - Instruction: write me 200-300 words to do with the female personification of love. Include historical mythology.
2025-04-12 22:12:37,970 - INFO - Response: Everything I write turns into something about her.
My writings have been analysed and deciphered, translated and misinterpreted. My words in their man...
2025-04-12 22:12:37,970 - INFO - Evaluation: The answer provided is a poetic and creative piece that explores the concept of the female personification of love through various historical mythologies. However, it does not directly address the use...
2025-04-12 22:12:37,970 - INFO - --------------------------------------------------
2025-04-12 22:12:37,970 - INFO - 
Example 4 (Score: 2):
2025-04-12 22:12:37,970 - INFO - Instruction: As a software developer, why should bugs be not considered the main priority or cause you to be angry?
2025-04-12 22:12:37,970 - INFO - Response: If this is a medical application you are talking about it is serious stuff. What if the bugs affected real users? Developers would be much less happy ...
2025-04-12 22:12:37,970 - INFO - Evaluation: The answer provided seems to be a mix of relevant and irrelevant information. The initial part of the answer addresses the user's question by highlighting the importance of bug-free software, especial...
2025-04-12 22:12:37,970 - INFO - --------------------------------------------------
2025-04-12 22:12:37,970 - INFO - 
Example 5 (Score: 3):
2025-04-12 22:12:37,970 - INFO - Instruction: People are posting on YouTube saying they pirated 3D games by making their own models with Blender to a great effect. They are making money doing that. Someone could find them and arrest them.
2025-04-12 22:12:37,970 - INFO - Response: Unless the Youtube Video shows them committing a crime, then no, they couldn't be arrested and tried for a crime. Them saying it, not under oath, is j...
2025-04-12 22:12:37,970 - INFO - Evaluation: The answer provided is attempting to address the user's question, but it does so in a somewhat indirect and conversational manner. The response starts by providing a general statement about the limita...
2025-04-12 22:12:37,970 - INFO - --------------------------------------------------
2025-04-12 22:12:37,976 - INFO - Saved 111 curated examples to data/curated_dataset.json
2025-04-12 22:12:37,982 - INFO - Saved all 150 evaluated examples to data/curated_dataset_all_evaluated.json
2025-04-12 22:12:37,982 - INFO - Self-curation completed successfully!


"""