# This software is for non-commercial use only.
# Commercial use requires a separate license.

import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import logging
from utils import load_problems, resolve_hidden_layer, resolve_intervention_module

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_with_intervention(
    model,
    tokenizer,
    prompt,
    max_new_tokens=1,
    intervention_vector=None,
    intervention_scale=1.0,
    target_module=None
):
    """Generates tokens with intervention by hooking into the generation process."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    hook_handles = []
    
    def forward_hook(module, input_args, output):
        # input_args is a tuple, output can be a tensor or a tuple
        # We are interested in modifying the first element of the output if it's a tuple (common for hidden states)
        # or the output itself if it's a tensor.
        
        original_output = output
        hidden_states_to_modify = None
        
        if isinstance(output, tuple):
            hidden_states_to_modify = output[0]
        elif isinstance(output, torch.Tensor):
            hidden_states_to_modify = output
        else:
            logging.warning(f"Unexpected output type from hooked layer: {type(output)}")
            return output

        if hidden_states_to_modify is None or not isinstance(hidden_states_to_modify, torch.Tensor):
            logging.warning("Hooked layer output does not contain a tensor of hidden states as expected.")
            return output

        modified_states = hidden_states_to_modify.clone()
        
        # Reshape intervention vector to [1, 1, hidden_dim] to broadcast correctly
        reshaped_vector = intervention_vector.view(1, 1, -1).to(hidden_states_to_modify.device)
        
        # Apply intervention to the last token position's hidden state
        modified_states[:, -1:, :] = modified_states[:, -1:, :] + (reshaped_vector * intervention_scale)
        
        if isinstance(original_output, tuple):
            return (modified_states,) + original_output[1:]
        else:
            return modified_states
    
    # Intervene where the TPV was trained. Defaults to model.model.norm (final
    # layer) only when no module is supplied, preserving the original behaviour.
    target_intervention_module = target_module if target_module is not None else model.model.norm
    
    if intervention_vector is not None:
        if not hasattr(target_intervention_module, 'register_forward_hook'):
            raise ValueError(f"Target module {type(target_intervention_module)} has no 'register_forward_hook'.")
        hook_handles.append(target_intervention_module.register_forward_hook(forward_hook))
        logging.debug(f"Intervention hook registered on model.model.norm ({type(target_intervention_module)}).")
        logging.debug(f"Intervention vector shape: {intervention_vector.shape}")
        logging.debug(f"Intervention scale: {intervention_scale}")
    
    try:
        # Generate with the hook in place
        with torch.no_grad(): # Standard practice for inference
            outputs = model.generate(
                inputs.input_ids, 
                max_new_tokens=max_new_tokens, 
                do_sample=False, # Greedy decoding for reproducibility
                num_beams=1,
                use_cache=True, # Use KV cache for efficiency
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    finally:
        for handle in hook_handles:
            handle.remove()
        if hook_handles:
            logging.debug("All intervention hooks removed.")
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Generate text with model intervention using a single alpha value.")
    parser.add_argument("--model_name_or_path", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                        help="Name or path of the pretrained model.")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None,
                        help="Name or path of the tokenizer. Defaults to model_name_or_path if not set.")
    parser.add_argument("--intervention_vector_path", type=str, default="tpv_model/tpv_linear_weights.npy",
                        help="Path to the .npy file containing the intervention vector.")
    parser.add_argument("--dataset", type=str, default="math500",
                        help="Base directory containing original problem text files.")
    parser.add_argument("--output_generations_dir", type=str, default="llama_intervention_responses",
                        help="Base directory to save generated responses.")
    parser.add_argument("--task_name", type=str, default="math500",
                        help="Task name, used for structuring output subdirectories (e.g., math500).")
    parser.add_argument("--problem_start_idx", type=int, default=30,
                        help="Starting index for problems to process.")
    parser.add_argument("--problem_end_idx", type=int, default=31,
                        help="Ending index (inclusive) for problems to process.")
    parser.add_argument("--alpha", type=float, default=100.0,
                        help="Single intervention scale (alpha) value.")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Maximum number of new tokens to generate.")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use ('cuda', 'cpu', or 'auto' to use model's device_map preference).")
    parser.add_argument("--hidden_layer", type=int, default=-1,
                        help="Hidden-states index the TPV was trained on. DEFAULT -1 (final "
                             "layer) and you almost certainly want to keep it. A final-layer "
                             "intervention edits only the current step logits; an intermediate "
                             "layer feeds attention in every later layer, so the edit is written "
                             "into the KV cache and leaks into all subsequent decoding steps "
                             "(paper Sec. 3.1/3.2). Intermediate layers are better for "
                             "MONITORING only.")
    parser.add_argument("--torch_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"],
                        help="Torch dtype for model loading (float16, bfloat16, float32).")

    args = parser.parse_args()

    if args.tokenizer_name_or_path is None:
        args.tokenizer_name_or_path = args.model_name_or_path

    logging.info(f"Loading tokenizer from: {args.tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.info("Set pad_token to eos_token.")

    logging.info(f"Loading model from: {args.model_name_or_path}")
    model_dtype = getattr(torch, args.torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        device_map=args.device if args.device != "auto" else "auto",
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True
    )
    
    # Determine the device the model ended up on
    effective_device = model.device 
    layer_idx = resolve_hidden_layer(model, args.hidden_layer)
    target_module = resolve_intervention_module(model, layer_idx)
    logging.info(f"Intervening on hidden_states[{layer_idx}] -> {type(target_module).__name__}")
    if layer_idx != model.config.num_hidden_layers:
        logging.warning(
            "Intervening at intermediate layer %d of %d. That output is consumed by the "
            "attention of every later layer, so the edit is written into the KV cache and "
            "persists across decoding steps; it does NOT isolate to a single step the way a "
            "final-layer intervention does. Use --hidden_layer -1 to reproduce the paper.",
            layer_idx, model.config.num_hidden_layers)
    logging.info(f"Model loaded. Effective device for intervention vector: {effective_device}")

    logging.info("Preparing intervention vector...")
    hidden_dim = model.config.hidden_size
    logging.info(f"Model hidden dimension: {hidden_dim}")

    intervention_vector = torch.Tensor(np.load(args.intervention_vector_path)).to(effective_device)
    
    # Reshape to a 1D vector of size hidden_dim
    intervention_vector = intervention_vector.squeeze()
    if intervention_vector.ndim != 1:
        raise ValueError(
            f"Intervention vector must be 1D after squeezing, but got shape {intervention_vector.shape}. "
            f"Original path: {args.intervention_vector_path}"
        )
    if intervention_vector.shape[0] != hidden_dim:
        raise ValueError(
            f"Intervention vector dimension ({intervention_vector.shape[0]}) "
            f"does not match model hidden dimension ({hidden_dim})."
        )
    logging.info(f"Intervention vector loaded and reshaped to: {intervention_vector.shape}")

    # Using only a single alpha value as requested
    intervention_scale = args.alpha
    logging.info(f"Using intervention scale (alpha): {intervention_scale}")

    # Construct the full path for output generations
    model_folder_name = args.model_name_or_path.split('/')[-1]
    generations_base_dir = os.path.join(args.output_generations_dir, model_folder_name, args.task_name, "intervention")

    problems = load_problems(args.dataset, args.problem_start_idx, args.problem_end_idx)

    for i, problem in enumerate(problems):
        problem_num = i
        logging.info(f"Processing problem {problem_num}...")
        
        problem_output_dir = os.path.join(generations_base_dir, f"problem_{problem_num}")
        os.makedirs(problem_output_dir, exist_ok=True)
        

        # Standard prompt suffix
        prompt = problem + "\nPlease reason step by step, and put your final answer within \\boxed{}.\n<think>\n"

        # Generate with single intervention scale
        output_with_intervention = generate_with_intervention(
            model, tokenizer, prompt, 
            max_new_tokens=args.max_new_tokens, 
            intervention_vector=intervention_vector,
            intervention_scale=intervention_scale,
            target_module=target_module
        )
        
        output_file_path = os.path.join(problem_output_dir, f"response_alpha_{args.alpha}.txt")
        try:
            with open(output_file_path, "w") as f:
                f.write(output_with_intervention)
            logging.info(f"Saved response for problem {problem_num}, alpha {args.alpha} to {output_file_path}")
        except IOError as e:
            logging.error(f"Failed to write output file {output_file_path}: {e}")
    
    logging.info("Processing complete.")

if __name__ == "__main__":
    main()
