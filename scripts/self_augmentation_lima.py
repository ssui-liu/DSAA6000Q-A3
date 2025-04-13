#!/usr/bin/env python3
"""
Self-Augmentation script for LIMA dataset using inverse SFT model.
This script randomly samples from LIMA dataset's completions and generates instructions.
"""
import unsloth
import os
import json
import random
import argparse
import torch
import logging
from typing import List, Dict, Any
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import TextStreamer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('self_augmentation.log'),
        logging.StreamHandler()
    ]
)

def parse_args():
    parser = argparse.ArgumentParser(description="Self-Augmentation with LIMA dataset")
    parser.add_argument("--model_path", type=str, default=r"outputs/inverse_sft_model_full_train",
                        help="Path to the inverse SFT model")
    parser.add_argument("--output_file", type=str, default="data/self_augmented2.json",
                        help="Path to save the self-augmented data")
    parser.add_argument("--sample_size", type=int, default=150,
                        help="Number of examples to sample from LIMA")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum new tokens for generation")
    parser.add_argument("--temperature", type=float, default=1.2,
                        help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p for generation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()

def load_lima_dataset():
    """Load the LIMA dataset from the local JSON file."""
    local_path = "data/lima_single_turn.json"
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"LIMA dataset not found at {local_path}")
    
    logging.info(f"Loading LIMA dataset from {local_path}")
    with open(local_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logging.info(f"Loaded {len(data)} examples from LIMA dataset")
    return data

def load_inverse_sft_model(model_path: str):
    """Load the inverse SFT model for generating instructions."""
    logging.info(f"Loading inverse SFT model from {model_path}")
    
    # Load the model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,  # Auto detection
        load_in_4bit=True,
        # For inference only
        # inference_only=True,
    )
    
    # Set up the model for faster inference
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Should match what was used during training
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        use_gradient_checkpointing=False,  # Not needed for inference
        random_state=3407,
        use_rslora=False,
    )
    
    # Enable fast inference mode
    FastLanguageModel.for_inference(model)
    
    return model, tokenizer

def check_token_length(text, tokenizer, max_tokens=2048):
    """Check if a text exceeds the maximum token length."""
    tokens = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    token_length = len(tokens.input_ids[0])
    return token_length <= max_tokens

def sample_lima_responses(dataset, tokenizer, sample_size: int, seed: int, max_tokens=2048) -> List[Dict[str, Any]]:
    """
    Randomly sample examples from the LIMA dataset, filtering those that exceed token limits.
    """
    random.seed(seed)
    
    # First filter out examples that exceed token limits
    filtered_dataset = []
    discarded_count = 0
    
    for idx, sample in enumerate(dataset):
        response = sample["assistant"]
        question = sample["human"]
        
        # Create the prompt that will be used
        prompt = f"Your task is to infer and generate the user's original input question that could have led to the following Assistant response.\n\n{response}"
        
        # Check if the sample's response exceeds token limits
        if check_token_length(prompt, tokenizer, max_tokens):
            filtered_dataset.append(sample)
        else:
            discarded_count += 1
    
    logging.info(f"Filtered out {discarded_count} examples exceeding {max_tokens} token limit")
    logging.info(f"Remaining dataset size: {len(filtered_dataset)} examples")
    
    # If we have fewer examples than requested after filtering, use all remaining examples
    if len(filtered_dataset) <= sample_size:
        sampled_data = filtered_dataset
        logging.info(f"Using all {len(filtered_dataset)} examples after filtering as it's smaller than requested sample size")
    else:
        # Randomly sample from the filtered dataset
        sampled_data = random.sample(filtered_dataset, sample_size)
        logging.info(f"Randomly sampled {sample_size} examples from filtered dataset")
    
    # Format the samples for the model
    formatted_samples = []
    for idx, sample in enumerate(sampled_data):
        formatted_samples.append({
            "response": sample["assistant"],
            "original_question": sample["human"],
            "original_idx": idx
        })
    
    return formatted_samples

def generate_instructions(model, tokenizer, samples, args):
    """Generate instructions for the given assistant responses using the inverse SFT model."""
    results = []
    
    for i, sample in enumerate(samples):
        response = sample["response"]
        original_question = sample["original_question"]
        logging.info(f"Generating instruction for sample {i+1}/{len(samples)}")
        
        # Format as a prompt for the inverse SFT model
        messages = [
            {"role": "system", "content": "Your task is to infer and generate the user's original input question that could have led to the following Assistant response."},
            {"role": "user", "content": response},
        ]
        
        # For Unsloth inference, properly apply the chat template
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        
        # Create text streamer for more efficient generation
        text_streamer = TextStreamer(tokenizer, skip_prompt=True)
        
        # Generate instruction with Unsloth recommended parameters
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            use_cache=True,  # Enable KV-cache for faster generation
            streamer=text_streamer,  # Optional but more efficient
        )
        
        # Decode the generated instruction
        # Skip special tokens and just get the generated part
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the generated part (instruction) from the full output
        instruction = extract_instruction(generated_text, tokenizer)
        
        # Save the pair
        results.append({
            "generated_instruction": instruction,
            "original_question": original_question,
            "response": response,
            "original_idx": sample["original_idx"],
        })
    
    return results

def extract_instruction(generated_text, tokenizer):
    """
    Extract just the generated instruction from the full model output.
    This function needs to be adapted based on the specific chat template used.
    """
    # Try to find the assistant's response which contains the generated instruction
    # This pattern depends on the tokenizer's chat template
    if "<|start_header_id|>assistant<|end_header_id|>" in generated_text:
        # For Llama 3.1 style templates
        parts = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")
        if len(parts) > 1:
            return parts[1].strip()
    
    # If we can't find a specific pattern, return everything after the user's message
    # This is a fallback and may include some template text
    if "user" in generated_text.lower() and "assistant" in generated_text.lower():
        parts = generated_text.split("assistant")
        if len(parts) > 1:
            return parts[1].strip()
    
    # Last resort: return the whole thing
    return generated_text.strip()

def main():
    args = parse_args()
    random.seed(args.seed)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Load LIMA dataset
    lima_dataset = load_lima_dataset()
    
    # First load the tokenizer for length checking
    logging.info(f"Loading tokenizer from {args.model_path} for length filtering")
    model, tokenizer = load_inverse_sft_model(args.model_path)
    
    # Sample from the dataset
    logging.info(f"Sampling {args.sample_size} examples from LIMA dataset")
    sampled_responses = sample_lima_responses(lima_dataset, tokenizer, args.sample_size, args.seed)
    
    # Generate instructions
    logging.info("Generating instructions using inverse SFT model")
    augmented_data = generate_instructions(model, tokenizer, sampled_responses, args)
    
    # Save the results
    logging.info(f"Saving {len(augmented_data)} self-augmented examples to {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(augmented_data, f, indent=2, ensure_ascii=False)
    
    # Print 5 examples
    logging.info("\n===== 5 Examples of Generated Instructions =====")
    for i, example in enumerate(augmented_data[:5]):
        logging.info(f"\nExample {i+1}:")
        logging.info(f"Original Question: {example['original_question']}")
        logging.info(f"Generated Instruction: {example['generated_instruction']}")
        logging.info(f"Response: {example['response'][:100]}..." if len(example['response']) > 100 else example['response'])
        logging.info("-" * 50)

if __name__ == "__main__":
    main()


"""
===== 5 Examples of Generated Instructions =====
2025-04-12 21:40:01,914 - INFO - 
Example 1:
2025-04-12 21:40:01,914 - INFO - Original Question: How to boost your diet with peanut butter powder?
2025-04-12 21:40:01,914 - INFO - Generated Instruction: What can you do with powdered peanut butter?
2025-04-12 21:40:01,914 - INFO - Response: Powdered peanut butter is a versatile ingredient that you can use to add some nutty flavor to any di...
2025-04-12 21:40:01,914 - INFO - --------------------------------------------------
2025-04-12 21:40:01,914 - INFO - 
Example 2:
2025-04-12 21:40:01,914 - INFO - Original Question: I have to disable inputs at first and then on click of a link to enable them.
This is what I have tried so far, but it doesn't work.
HTML:
```
```
jQuery:
```$("#edit").click(function(event){
   event.preventDefault();
   $('.inputDisabled').removeAttr("disabled")
});
```

This shows me ```true``` and then ```false``` but nothing changes for the inputs:
```$("#edit").click(function(event){
   alert('');
   event.preventDefault();
   alert($('.inputDisabled').attr('disabled'));
   $('.inputDisabled').removeAttr("disabled");
   alert($('.inputDisabled').attr('disabled'));
});
```
2025-04-12 21:40:01,914 - INFO - Generated Instruction: write a jquery event that triggers when a user clicks the edit button and disables the text area
2025-04-12 21:40:01,914 - INFO - Response: Always use the ```prop()``` method to enable or disable elements when using jQuery (see below for wh...
2025-04-12 21:40:01,914 - INFO - --------------------------------------------------
2025-04-12 21:40:01,914 - INFO - 
Example 3:
2025-04-12 21:40:01,914 - INFO - Original Question: How to connect mysql workbench to running mysql inside docker?
2025-04-12 21:40:01,914 - INFO - Generated Instruction: How can I make my docker mysql allow root access on 3306 port?
2025-04-12 21:40:01,914 - INFO - Response: By default after deployment MySQL has following connection restrictions:
```mysql> select host, user...
2025-04-12 21:40:01,914 - INFO - --------------------------------------------------
2025-04-12 21:40:01,914 - INFO - 
Example 4:
2025-04-12 21:40:01,914 - INFO - Original Question: Summarize the following article with one line: 
When journalist Gianna Toboni traveled to India to explore the country's rapidly growing, yet unregulated, gestational surrogacy industry for HBO documentary series Vice, she didn't anticipate 'how dark' the story would get.

For nearly two years, the producer and host has been reporting on current issues across the globe and has covered everything from the detention center at Guantanamo Bay to the effect of climate change on polar bears - but nothing could have prepared her for the moment when someone offered to sell her a baby over dinner while she was working undercover in India. 

'It was the most heartbreaking experience that I ever had,' Gianna told Daily Mail Online.  

Baby business: Vice correspondent Gianna Toboni (pictured) traveled to India to explore the country's booming gestational surrogacy industry 

Shady deal: The journalist from Brooklyn, New York, went undercover to meet with an agent in India who offered to get her a baby in two to three months

But the heartbreak did not end there.

As Gianna quickly learned during her time working on the Outsourcing Embryos documentary, surrogacy in India is a multi-million dollar business, and one which is made all the more lucrative by the high number of American couples traveling to the country in order to use the services provided by one or more of the 'embryo outsourcing' agencies featured in the Vice documentary.

During her time spent undercover posing as one of these people, Gianna was informed that, in order to maximize profits and ensure a final product, doctors are encouraged to implant multiple embryos in surrogates, which can lead to the surrogate having to abort one of the fetuses or give birth to multiple babies.

And if an 'extra' baby is born, it isn't necessarily going home with its genetic parents. There are also issues with couples never making it to India to claim their children for whatever reasons, meaning that the newborn baby is left without a parent. 

For the most recent episode in the Vice series, Gianna went undercover to meet with one surrogacy agent who claimed over dinner that she could get her a Caucasian baby in two to three months - confirming that there were in fact 'extra' babies being sold on the black market.

The agent then tried to convince Gianna and her team to buy the baby that they had brought with them to the restaurant.

Shocking offer: One of the agents can be seen holding the baby that they brought to the restaurant with them

No morals: The agent eventually offered to sell Gianna and her team the baby over dinner 

Gianna noted that the agent spoke with a 'shocking amount of ease' and 'talked about forging documents as if she has done it a hundred times' as she tried to sell her and her team a baby over dinner.

'It made me think it wasn't a one-off thing,' she explained to Daily Mail Online. 

Gianna never once considered buying the baby, but as a woman who would one day like to be a mother, she admitted that there was a moment when she thought about accepting the offer, knowing that she could provide the child with a loving home that it may never experience otherwise, particularly as it was made clear that the agent would have sold the baby to anybody.

'When I go on these stories, I am a human being first and a journalist second,' she said of her initial reaction to the offer.

The sale of 'extra' babies on the black market was just one of the many shocking side effects of commercial surrogacy uncovered by Gianna and her team.

In the US, surrogacy can cost hopeful parents upwards of $100,000, and Gianna explained that 'the reoccurring theme' when speaking with American agents and experts about couples hiring surrogates from other countries was money.

Commercial surrogacy in India costs nearly one-sixth the amount it would in the Western World.

'That seems to be the main driver,' she explained, before noting that some prospective parents do choose foreign surrogacy because of the altruistic element.

No options: Many of the surrogates who spoke with Gianna said that they decided to carry a baby because they desperately needed the money 

Dormitory: The women who agree to be surrogates at Dr Nayna Patel's Akanksha Infertility Clinic have to live at the facility until they give birth

Tight quarters: Two surrogates can be see sitting on the beds in their shared room 

And while American parents see the surrogacy business in India as being a 'cheap' alternative to the services offered at home, the amount of money made by a surrogate in India can vastly change her life, as well as the life of her family.  

Women can use the money to buy a home or send their own children to school, and Gianna explained that there are in fact couples who take great efforts to make sure their surrogates are a part of their lives.

But there are also countless tales of financially desperate women who are recruited in the slums and coerced into signing contracts that they can't read, only to be duped out of the money they were promised.

When I go on these stories I am a human being first and a journalist second

Surrogates undergo scheduled cesarean sections so doctors can ensure the greatest number of births per day.

Gianna, who witnessed the high turnover rate first hand at Dr Nayna Patel's Akanksha Infertility Clinic, in the town of Gujarat, in the west of India, was nearly speechless when she saw how rapidly newborns and their parents were whisked away following a surrogate's C-section.

Dr Patel maintained that the women are well taken care of and make more money than they could working 24 hours a day, seven days a week, in any other profession.

And while Gianna explained that some women are happy that they can provide a house for their family and put their kids through school as a surrogate, the women she and her team spoke to said they chose to be surrogates because they didn't have any other options.   

During the episode, a surrogate named Vasanti told Gianna: 'Nobody likes doing this.' 

Dishonest: Although Dr Patel maintained that she didn't search for surrogates from the slums, Gianna met a woman who said she was working for the clinic owner as tried to recruit women from a poor area 

No choice: A doctor can be seen performing a cesarean section on one of the surrogates. Surrogates have to undergo C-sections so doctors can maximize the amount of babies being born in a day 

Too quick: Almost immediately after this baby was born via a surrogate, the biological parents whisked the newborn away in a van as they went to return to their home country 

She continued: 'I didn't have a home, so I thought I could build one by being a surrogate.'

Another surrogate named Nisha explained that she was 'helpless' and had 'no alternatives'.

Gianna was overcome by many of the surrogates' desperation.

'It is really hard to hear someone talk about going through an experience that takes a toll on the body, that lasts longer than nine months and takes them away from their kids because they have to support their families and essentially survive,' she said.

Gianna recalled speaking with one surrogate's husband who recently lost his job and he confessed that he was grateful his wife had the opportunity to earn money for their family as a surrogate.

He made clear that he didn't force her into the role, but explained that it was necessary for their family's survival.

'It all went back to money,' Gianna noted.

As a whole, Gianna said that she thinks some parents may be aware of the 'shadier side' of commercialized surrogacy, but a 'vast majority' have no idea this dark underbelly exits.

Gianna recommends that parents who are considering foreign surrogacy options should do extensive research on the agent, the doctor and the surrogate they will be working with.
2025-04-12 21:40:01,915 - INFO - Generated Instruction: The Vice correspondent Gianna Toboni visited India, where she had an amazing trip and got to see how people live there. During this trip she also visited the local medical community and asked them for a summary of their last work assignment. Please paraphase the text from Gianna's assignment.
2025-04-12 21:40:01,915 - INFO - Response: Here is the summary: 
Vice correspondent Gianna Toboni traveled to India to explore the country's bo...
2025-04-12 21:40:01,915 - INFO - --------------------------------------------------
2025-04-12 21:40:01,915 - INFO - 
Example 5:
2025-04-12 21:40:01,915 - INFO - Original Question: Say I have a file ```/templates/apple``` and I want to

* put it in two different places and then
* remove the original.

So, ```/templates/apple``` will be copied to ```/templates/used``` AND ```/templates/inuse```
and then after that I’d like to remove the original.
Is ```cp``` the best way to do this, followed by ```rm```? Or is there a better way?
I want to do it all in one line so I’m thinking it would look something like:
```cp /templates/apple /templates/used | cp /templates/apple /templates/inuse | rm /templates/apple
```
Is this the correct syntax?
2025-04-12 21:40:01,915 - INFO - Generated Instruction: What is the purpose of ```|``` in bash scripting?
2025-04-12 21:40:01,915 - INFO - Response: You are using ```|``` (pipe) to direct the output of a command into another command. What you are lo...
2025-04-12 21:40:01,915 - INFO - --------------------------------------------------


"""