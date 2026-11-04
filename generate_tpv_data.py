# This software is for non-commercial use only.
# Commercial use requires a separate license.

from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import load_problems, resolve_hidden_layer, build_prompt_template
import torch
import numpy as np
import os
import gc
import json
import time
import random

def extract_llm_hidden_states(
    # Model parameters
    model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    max_new_tokens=1024,
    
    # Dataset parameters
    dataset="math500", 
    start_problem_index=0,
    end_problem_index=30,
    generations_per_problem=5,
    
    # Output parameters
    output_dir="model_results",
    
    # Generation parameters
    prompt_template="{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n<think>\n",
    temperature=0.6,
    do_sample=True,
    top_p=0.95,
    seed=42,
    hidden_layer=None,
):
    """
    Generate responses and extract hidden states from language models, 
    generating all sequences for a problem in a single batch.
    
    Args:
        model_name (str): HuggingFace model name or path
        max_new_tokens (int): Maximum number of new tokens to generate
        dataset_name (str): Dataset to use (e.g. 'HuggingFaceH4/MATH-500', 'gsm8k')
        dataset_split (str): Dataset split to use (e.g. 'test', 'train')
        problem_field (str): Field name in dataset containing the problem (e.g. 'problem', 'question')
        start_problem_index (int): Starting index of problems to process
        end_problem_index (int): Ending index of problems to process (exclusive)
        generations_per_problem (int): Number of generations per problem (used for num_return_sequences)
        output_dir (str): Directory to store results
        prompt_template (str): Prompt template. Use {problem} as placeholder for the problem text
        temperature (float): Temperature for generation (0 for deterministic)
        do_sample (bool): Whether to use sampling for generation
        top_p (float): Top-p value for nucleus sampling
        
    Returns:
        dict: A dictionary containing the paths to all generated outputs and metadata
    """
    results = {
        "config": {
            "model_name": model_name,
            "max_new_tokens": max_new_tokens,
            "dataset_name": dataset,
            "generations_per_problem": generations_per_problem,
            "prompt_template": prompt_template,
            "temperature": temperature,
            "do_sample": do_sample,
            "top_p": top_p,
            "seed": seed,
            "hidden_layer": hidden_layer
        },
        "problems": {}
    }

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed) # Sets seed for all GPUs
        print(f"[INFO] Seeds set to {seed} for random, numpy, and torch.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set memory-efficient configuration
    if device == "cuda":
        torch.cuda.empty_cache()
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Model loading
    print(f"Loading model {model_name} and tokenizer...")
    if device == "cuda":
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Tokenizer pad_token was None, set to eos_token: {tokenizer.eos_token}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", # Distributes model across available GPUs/CPU
        torch_dtype=torch.float16,
    )
    # Ensure model config has hidden_size, critical for empty hidden state arrays
    if not hasattr(model, 'config') or not hasattr(model.config, 'hidden_size'):
        # Fallback if not directly available, though usually it is.
        # This might require a forward pass with a dummy input if truly missing,
        # but for most HF models, config.hidden_size is standard.
        print("[Warning] model.config.hidden_size not found. Attempting to infer or use a default.")
        # As a last resort, try to get it from a layer if possible, or error out.
        # For now, we'll assume it will be available via outputs.hidden_states[0][-1].shape[-1] later.


    layer_idx = resolve_hidden_layer(model, hidden_layer)
    print(f"[INFO] Extracting hidden_states[{layer_idx}] "
          f"(model has {model.config.num_hidden_layers} decoder layers).")
    if layer_idx != model.config.num_hidden_layers:
        print("[NOTE] Using an intermediate layer. This measures better for MONITORING "
              "but is NOT valid for tpv_intervention.py: intervening below the final "
              "layer writes the edit into the KV cache instead of isolating it to one "
              "decoding step. For the intervention pipeline re-run with --hidden_layer -1.")
    results["config"]["resolved_hidden_layer"] = layer_idx

    # Optimized generation with hidden state extraction for a batch of sequences
    def generate_batch_with_last_hidden_states(
        model, inputs, tokenizer, 
        num_return_sequences, max_new_tokens, 
        temperature, do_sample, top_p
    ):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        start_time = time.time()
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=do_sample,
            top_p=top_p if do_sample else None,
            num_return_sequences=num_return_sequences,
            return_dict_in_generate=True,
            output_hidden_states=True
        )
        generation_time = time.time() - start_time
        
        # --- Debug: Print hidden states structure information ---
        # print("\n--- Hidden States Output Structure (Batch Generation) ---")
        # print(f"outputs.hidden_states length (num_gen_tokens + 1): {len(outputs.hidden_states)}")
        # if outputs.hidden_states:
        #     print(f"outputs.hidden_states[0] (prompt processing) length (num_layers + embeddings): {len(outputs.hidden_states[0])}")
        #     for i, layer_output in enumerate(outputs.hidden_states[0]):
        #         layer_name = "Embeddings" if i == 0 else f"Decoder Layer {i-1}"
        #         print(f"  - Prompt HS | {layer_name}: Shape {layer_output.shape}") # (batch_size, input_seq_len, hidden_size)
            
        #     if len(outputs.hidden_states) > 1:
        #         print(f"outputs.hidden_states[1] (1st gen token) length (num_layers + embeddings): {len(outputs.hidden_states[1])}")
        #         for i, layer_output in enumerate(outputs.hidden_states[1]):
        #             layer_name = "Embeddings" if i == 0 else f"Decoder Layer {i-1}" # Note: for decoder-only, all are "decoder layers" effectively
        #             print(f"  - 1st Gen Token HS | {layer_name}: Shape {layer_output.shape}") # (batch_size, 1, hidden_size)
        # print("--- End Hidden States Debug ---")
        # --- End Debug ---

        all_responses = []
        all_tokens_lists = []
        actual_lengths_generated_list = []
        
        # Hidden state extraction for INPUT (prompt)
        # outputs.hidden_states[0][-1] is the last layer's activations after processing the input prompt.
        # Shape: (num_return_sequences, input_sequence_length, hidden_size)
        prompt_last_layer_hs_batch = outputs.hidden_states[0][layer_idx]
        last_layer_hidden_states_input_list = [
            prompt_last_layer_hs_batch[k].detach().cpu().numpy() for k in range(num_return_sequences)
        ]

        # Hidden state extraction for OUTPUT (generated tokens)
        last_layer_hidden_states_output_list = []
        prompt_len = input_ids.shape[1]
        
        # Try to get hidden_size from model.config, otherwise infer from tensor
        model_hidden_size = getattr(model.config, 'hidden_size', None)
        if model_hidden_size is None and outputs.hidden_states and outputs.hidden_states[0]:
             model_hidden_size = outputs.hidden_states[0][layer_idx].shape[-1]
        if model_hidden_size is None:
            raise ValueError("Could not determine model hidden size for empty hidden state arrays.")


        for k in range(num_return_sequences): # For each sequence in the batch
            current_sequence_ids_k = outputs.sequences[k] # Full sequence (prompt + gen + pad)
            
            # Decode response
            response_k = tokenizer.decode(current_sequence_ids_k, skip_special_tokens=True)
            all_responses.append(response_k)

            # Get all tokens for the current sequence (prompt + gen + pad)
            tokens_k_full_sequence = tokenizer.convert_ids_to_tokens(current_sequence_ids_k)
            all_tokens_lists.append(tokens_k_full_sequence)

            # Determine actual number of generated tokens for sequence k (excluding prompt and padding)
            sequence_k_generated_ids_only = current_sequence_ids_k[prompt_len:]
            
            actual_gen_len_k = 0
            for token_id in sequence_k_generated_ids_only:
                if token_id == tokenizer.pad_token_id:
                    break
                actual_gen_len_k += 1
                if token_id == tokenizer.eos_token_id: # Count EOS as a generated token
                    break 
            actual_lengths_generated_list.append(actual_gen_len_k)

            # Extract hidden states for actual generated tokens for sequence k
            gen_k_output_hs_for_actual_tokens = []
            # outputs.hidden_states[1] to outputs.hidden_states[actual_gen_len_k]
            # Max available steps for hidden states of generated tokens: len(outputs.hidden_states)
            num_hs_steps_available_for_gen_tokens = len(outputs.hidden_states)
            
            for step_offset in range(min(actual_gen_len_k, num_hs_steps_available_for_gen_tokens)):
                # step_offset 0 is the 1st generated token, HS index in outputs.hidden_states is step_offset + 1
                hs_tensor_for_step = outputs.hidden_states[step_offset][layer_idx] # shape (batch_size, 1, hidden_dim)
                token_hs = hs_tensor_for_step[k, -1, :].detach().cpu().numpy() # Get for sequence k, squeeze the token dim
                gen_k_output_hs_for_actual_tokens.append(token_hs)
            
            if gen_k_output_hs_for_actual_tokens:
                last_layer_hidden_states_output_list.append(np.array(gen_k_output_hs_for_actual_tokens))
            else: # Handle cases where actual_gen_len_k is 0
                last_layer_hidden_states_output_list.append(np.empty((0, model_hidden_size), dtype=np.float32))
        
        return (all_responses, all_tokens_lists, 
                last_layer_hidden_states_input_list, last_layer_hidden_states_output_list, 
                actual_lengths_generated_list, generation_time)

    # Load dataset
    print(f"Loading dataset {dataset}")
    problems = load_problems(
        dataset, 
        starting_index=start_problem_index, 
        end_index=end_problem_index
    )
    
    if not problems:
        print(f"No problems loaded for range {start_problem_index} to {end_problem_index}. Exiting.")
        return results

    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(results["config"], f, indent=2)
    results["config_path"] = config_path
    
    # Processing problems
    for problem_idx, problem_text_content in enumerate(problems):
        actual_idx = problem_idx + start_problem_index
        print(f"\nProcessing problem {actual_idx}...")
        
        problem_dir = os.path.join(output_dir, f"problem_{actual_idx}")
        os.makedirs(problem_dir, exist_ok=True)
        
        problem_results_entry = {
            "original_problem_path": os.path.join(problem_dir, "original_problem.txt"),
            "formatted_prompt_path": os.path.join(problem_dir, "formatted_prompt.txt"),
            "generations": {}
        }
        
        with open(problem_results_entry["original_problem_path"], "w", encoding="utf-8") as f:
            f.write(problem_text_content)
        
        formatted_problem = prompt_template.format(problem=problem_text_content)
        with open(problem_results_entry["formatted_prompt_path"], "w", encoding="utf-8") as f:
            f.write(formatted_problem)
        
        inputs = tokenizer(formatted_problem, return_tensors="pt").to(device)
        
        try:
            # Generate all sequences for this problem in one batch
            print(f"Generating {generations_per_problem} sequences for problem {actual_idx} in a batch...")
            (all_responses, all_tokens_lists, 
             all_hs_inputs, all_hs_outputs, 
             all_actual_lengths, batch_generation_time) = \
                generate_batch_with_last_hidden_states(
                    model, inputs, tokenizer,
                    num_return_sequences=generations_per_problem,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p
                )
            print(f"Batch generation for problem {actual_idx} completed in {batch_generation_time:.2f} seconds.")

            # Process each generation from the batch
            for gen_idx in range(generations_per_problem):
                gen_dir = os.path.join(problem_dir, f"generation_{gen_idx}")
                os.makedirs(gen_dir, exist_ok=True)

                # Ensure lists have enough elements if generation was partial (shouldn't happen with num_return_sequences)
                if gen_idx >= len(all_responses):
                    print(f"[Warning] Problem {actual_idx}, gen {gen_idx}: Expected generation not found in batch output. Skipping.")
                    error_info = {
                        "error_path": os.path.join(gen_dir, "error.txt"),
                        "error": "Generation missing from batch output."
                    }
                    with open(error_info["error_path"], "w") as f_err:
                        f_err.write(error_info["error"])
                    problem_results_entry["generations"][gen_idx] = error_info
                    continue
                
                current_response = all_responses[gen_idx]
                current_tokens_list = all_tokens_lists[gen_idx]
                current_hs_input = all_hs_inputs[gen_idx]
                current_hs_output = all_hs_outputs[gen_idx]
                current_actual_generated_len = all_actual_lengths[gen_idx]

                generation_files = {
                    "response_path": os.path.join(gen_dir, "response.txt"),
                    "metadata_path": os.path.join(gen_dir, "metadata.json"),
                    "tokens_path": os.path.join(gen_dir, "tokens.json"),
                    "hidden_states_input_path": os.path.join(gen_dir, "token_hidden_states_input.npy"),
                    "hidden_states_output_path": os.path.join(gen_dir, "token_hidden_states_output.npy")
                }

                try:
                    with open(generation_files["response_path"], "w", encoding="utf-8") as f:
                        f.write(current_response)
                    
                    with open(generation_files["tokens_path"], "w", encoding="utf-8") as f:
                        json.dump(current_tokens_list, f, indent=2)
                    
                    np.save(generation_files["hidden_states_input_path"], current_hs_input)
                    np.save(generation_files["hidden_states_output_path"], current_hs_output)
                    
                    metadata = {
                        "batch_generation_time_seconds_for_problem": batch_generation_time,
                        "estimated_time_per_generation_in_batch": batch_generation_time / generations_per_problem if generations_per_problem > 0 else batch_generation_time,
                        "input_prompt_length_tokens": inputs['input_ids'].shape[1],
                        "output_generated_length_tokens": current_actual_generated_len,
                        "total_tokens_in_sequence_file": len(current_tokens_list), # Includes prompt, generated, EOS, padding
                        "prompt_plus_output_tokens": inputs['input_ids'].shape[1] + current_actual_generated_len
                    }
                    with open(generation_files["metadata_path"], "w") as f:
                        json.dump(metadata, f, indent=2)
                    
                    print(f"  Saved results for problem {actual_idx}, generation {gen_idx} to {gen_dir}")
                    generation_files["metadata"] = metadata # Add metadata dict to files dict
                    problem_results_entry["generations"][gen_idx] = generation_files

                except Exception as e_gen_save:
                    print(f"  Error saving results for problem {actual_idx}, generation {gen_idx}: {e_gen_save}")
                    error_path = os.path.join(gen_dir, "error.txt")
                    with open(error_path, "w") as f_err:
                        f_err.write(f"Error during saving: {str(e_gen_save)}")
                    problem_results_entry["generations"][gen_idx] = {
                        "error_path": error_path,
                        "error": f"Error during saving: {str(e_gen_save)}"
                    }
        
        except Exception as e_problem_batch:
            print(f"Error during batch generation or processing for problem {actual_idx}: {e_problem_batch}")
            # Log error for all generations of this problem if batch failed
            for gen_idx_err in range(generations_per_problem):
                gen_dir = os.path.join(problem_dir, f"generation_{gen_idx_err}")
                os.makedirs(gen_dir, exist_ok=True)
                error_path = os.path.join(gen_dir, "error.txt")
                error_msg = f"Batch generation/processing failed for problem: {str(e_problem_batch)}"
                with open(error_path, "w") as f_err:
                    f_err.write(error_msg)
                problem_results_entry["generations"][gen_idx_err] = {
                    "error_path": error_path,
                    "error": error_msg
                }
        
        results["problems"][actual_idx] = problem_results_entry
        
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            
        print(f"Completed processing for problem {actual_idx}")
    
    print("\nGeneration process complete.")
    final_summary_path = os.path.join(output_dir, "summary_results.json")
    with open(final_summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full results summary saved to {final_summary_path}")
    
    return results


# Example usage:
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate responses and extract hidden states from language models")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 
                        help="HuggingFace model name or path")
    parser.add_argument("--max_new_tokens", type=int, default=8, 
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--dataset", type=str, default="math500",
                        help="Dataset to use (e.g. 'math500', 'gsm8k')")
    parser.add_argument("--start_problem_index", type=int, default=0,
                        help="Starting index of problems to process")
    parser.add_argument("--end_problem_index", type=int, default=30, 
                        help="Ending index of problems to process (exclusive)")
    parser.add_argument("--generations_per_problem", type=int, default=5, 
                        help="Number of generations per problem")
    parser.add_argument("--output_dir", type=str, default="llama_model_results_batched",
                        help="Directory to store results")
    parser.add_argument("--prompt_template", type=str, 
                        default="{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n<think>\n", 
                        help="Prompt template. Use {problem} as placeholder for the problem text")
    parser.add_argument("--temperature", type=float, default=0.6, 
                        help="Temperature for generation (0 for deterministic)")
    parser.add_argument("--do_sample", action="store_true", default=True, 
                        help="Whether to use sampling for generation")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p value for nucleus sampling")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility. Set to a non-negative integer.")
    parser.add_argument("--hidden_layer", type=int, default=None,
                        help="Index into outputs.hidden_states to use as the TPV feature. "
                             "Negative counts from the end (-1 = final layer, the old behaviour). "
                             "Default: 25%% of depth, which measures substantially better than "
                             "the final layer (see resolve_hidden_layer in utils.py).")
    parser.add_argument("--use_chat_template", action="store_true", default=False,
                        help="Format the problem with the tokenizer chat template and thinking "
                             "enabled. Required for chat models such as Qwen3, which otherwise "
                             "emit </think> immediately and yield zero-length thinking.")
    
    args = parser.parse_args()

    prompt_template = args.prompt_template
    if args.use_chat_template:
        from transformers import AutoTokenizer
        prompt_template = build_prompt_template(
            AutoTokenizer.from_pretrained(args.model), use_chat_template=True)
        print(f"[INFO] Using chat template:\n{prompt_template}")

    current_seed = args.seed

    results_summary = extract_llm_hidden_states(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        dataset=args.dataset,
        start_problem_index=args.start_problem_index,
        end_problem_index=args.end_problem_index,
        generations_per_problem=args.generations_per_problem,
        output_dir=args.output_dir,
        prompt_template=prompt_template,
        temperature=args.temperature,
        do_sample=args.do_sample,
        top_p=args.top_p,
        seed=current_seed,
        hidden_layer=args.hidden_layer
    )
    # print("\nFinal results structure:")
    # print(json.dumps(results_summary, indent=2))
