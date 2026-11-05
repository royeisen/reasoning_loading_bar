# Thinking Progress Vectors (TPV) 

![Loading Bar and Overclocking](./readme_resources/new_teaser.jpg)
*Our Thinking Progress Vectors enable real-time monitoring of language model thinking progress as well as acceleration of thinking for more effective and efficient reasoning*

## Links

> **📖 Paper**: [Overclocking LLM Reasoning: Monitoring and Controlling Thinking Path Lengths in LLMs](https://arxiv.org/abs/2506.07240)

> **🌐 For more details and interactive examples visit our [**Project Page**](https://royeisen.github.io/OverclockingLLMReasoning-paper/)**

---

This repository contains scripts for training and utilizing a Thinking Progress Vectors (TPV) system. This system consists of two main parts:

1. **TPV Monitoring Pipeline**: A three-step process to generate data, prepare datasets, and train a TPV regressor model.
2. **TPV Intervention System**: A system that uses the trained TPV regressor to intervene in the generation process of language models.


## Part 1: TPV Monitoring Pipeline

![Diverse Prompting for TPV Training](./readme_resources/diverse_prompting.png)
*TPV regressor models enable robust progress monitoring across different problem-solving approaches.*

The monitoring pipeline consists of three sequential scripts that should be run in the following order:

### Step 1: Generate TPV Data

This script generates responses from a language model and extracts hidden states for TPV training.

> **Two things to get right before running this** (details in
> [Probe Layer Selection](#probe-layer-selection) and [Chat Models](#chat-models-qwen3-etc)):
> - **Token budget.** Generations that never reach `</think>` are silently dropped in Step 2.
>   At `--max_new_tokens 1024` only 18.7% of DeepSeek-R1-Distill-Llama-8B generations survive;
>   at 4096, 72.3% do. Check the surviving fraction before trusting any result.
> - **Chat models** (Qwen3 etc.) need `--use_chat_template`, or they emit `</think>`
>   immediately and produce a dataset with zero usable signal and no error.
>
> If you plan to run the **intervention** pipeline afterwards, add `--hidden_layer -1`.

```bash
python generate_tpv_data.py \
  --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
  --max_new_tokens 4096 \
  --dataset "math500" \
  --start_problem_index 0 \
  --end_problem_index 30 \
  --generations_per_problem 5 \
  --output_dir "hidden_states_output" \
  --temperature 0.6 \
  --do_sample True \
  --top_p 0.95 \
  --seed 42
```

This will generate model responses and extract hidden states, storing the results in the `hidden_states_output` directory.

### Step 2: Prepare TPV Dataset

This script processes the generated hidden states and creates training/testing datasets for the TPV regressor.

```bash
python prepare_tpv_dataset.py \
  --input_dir "hidden_states_output" \
  --output_dir "qwen_math_tpv_dataset" \
  --seed 42 \
  --train_split_ratio 0.8 \
  --end_of_thinking_token "</think>"
```

This will process the hidden states from the previous step and create train/test datasets in the `qwen_math_tpv_dataset` directory.

> **Note:** generations whose output never contains `end_of_thinking_token` are skipped
> without warning, as are problems left with no valid generations. The printed record
> counts are therefore the *surviving* data, not what you generated. A large drop here
> means your token budget was too small (see Step 1).

### Step 3: Train TPV Regressor

This script trains a linear regression model on the prepared datasets to predict token progress values.

```bash
python train_tpv.py \
  --input_dir "qwen_math_tpv_dataset" \
  --output_dir "qwen_math_tpv_model" \
  --epsilon 1e-10 \
  --device "cuda"
```

This will train a TPV regressor using linear regression, evaluate its performance, and save the model weights to the `qwen_math_tpv_model` directory.

> **Note:** this reports **MSE only**, and the weights it saves are refit on train+test,
> so the saved model is not the one whose held-out MSE is printed. For R^2 on held-out
> problems use `eval_r2.py` (see [Evaluation Utilities](#evaluation-utilities)).

## Part 2: TPV Intervention System

![Intervention vs Base Model](./readme_resources/progress_with_without.jpeg)
*Comparison of base model (blue) vs intervened model (red) prediction trajectories. The intervention system steers the model toward more efficient and focused reasoning paths.*

After training the TPV regressor, you can use it to intervene in the generation process of language models.

### Running TPV Intervention

This script uses the trained TPV regressor to modify the hidden states during the language model generation process.

```bash
python tpv_intervention.py \
  --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
  --intervention_vector_path "qwen_math_tpv_model/tpv_linear_weights.npy" \
  --dataset "math500" \
  --output_generations_dir "qwen_intervention_responses" \
  --task_name "math500" \
  --problem_start_idx 30 \
  --problem_end_idx 35 \
  --alpha 100.0 \
  --max_new_tokens 2048 \
  --device "cuda" \
  --torch_dtype "float16"
```

This will apply the trained intervention vector to the model's generation process with the specified alpha scale factor, and save the results in the `llama_intervention_responses` directory.

## Example End-to-End Pipeline

Here's an example of running the complete pipeline with consistent input/output directories:

```bash
# Step 1: Generate hidden states data.
# --max_new_tokens is REQUIRED here: it defaults to 8, which produces no usable data.
# --hidden_layer -1 because this example ends in an intervention (Step 4), which is only
# valid for a final-layer TPV. Drop it for a monitoring-only run to get the better default.
python generate_tpv_data.py --model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
  --max_new_tokens 4096 --end_problem_index 120 --hidden_layer -1 \
  --output_dir "llama_hidden_states"

# Step 2: Prepare the TPV dataset
python prepare_tpv_dataset.py --input_dir "llama_hidden_states" --output_dir "llama_math_tpv_dataset"

# Step 3: Train the TPV regressor (prints MSE; use eval_r2.py for R^2)
python train_tpv.py --input_dir "llama_math_tpv_dataset" --output_dir "tpv_model"
python eval_r2.py --input_dir "llama_math_tpv_dataset" --run_dir "llama_hidden_states" --per_generation

# Step 4: Use the trained TPV regressor for intervention
python tpv_intervention.py --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" --intervention_vector_path "tpv_model/tpv_linear_weights.npy" --alpha 100.0
```

## Probe Layer Selection

By default the TPV is now read from an **intermediate** layer (45% of depth) rather than
the final layer. Layer sweeps on held-out MATH-500 problems (4096-token budget, 120
problems, 5 generations each, 80/20 split by problem):

| probe layer | DeepSeek-R1-Distill-Llama-8B (32L) | Qwen3-1.7B (28L) |
|---|---|---|
| ~25% depth | R^2 0.766 (L8) | R^2 0.757 (L7) |
| ~45% depth | R^2 0.766 (L14) | **R^2 0.805** (L12) |
| final layer | R^2 0.602 (L32) | R^2 0.638 (L28) |

End-to-end through this pipeline at the 45% default:

| model | layer | test MSE | test R^2 |
|---|---|---|---|
| DeepSeek-R1-Distill-Llama-8B | 14 / 32 | 0.0195 | 0.766 |
| Qwen3-1.7B | 13 / 28 | 0.0153 | 0.816 |
| DeepSeek-R1-Distill-Llama-8B | 32 (previous default) | 0.0331 | 0.602 |

Use `--hidden_layer` to override; `-1` reproduces the original final-layer behaviour.

### Monitoring vs. intervention use different layers

This matters and is easy to get wrong. The final layer is the **correct** intervention
site, exactly as described in the paper (Sec. 3.1, 3.2): it sits downstream of all
attention, so an edit there changes only the current step's logits and is never written
into the KV cache. Editing an intermediate layer feeds attention in every later layer,
so the perturbation is cached and leaks into all subsequent decoding steps.

- **Monitoring only** -> use the intermediate-layer default (better accuracy).
- **Intervention** -> train the TPV with `--hidden_layer -1` and run
  `tpv_intervention.py` with its default `-1`. Weights trained at an intermediate layer
  are not valid inputs to the intervention pipeline.

## Chat Models (Qwen3, etc.)

The default prompt template appends a bare `<think>\n` and relies on the model
continuing from it. That works for the DeepSeek-R1 distills but **silently produces a
degenerate dataset for chat models**: without the `<|im_start|>assistant` scaffolding,
Qwen3-1.7B emits `</think>` as its first generated token, so every trajectory has
thinking length N=1. This passes the `</think>` filter and reports a 100% success rate
while containing no usable signal.

Pass `--use_chat_template` for such models to format via the tokenizer's chat template
with thinking enabled (Qwen3-1.7B: mean N goes from 1 to 2076).

```bash
python generate_tpv_data.py --model "Qwen/Qwen3-1.7B" --use_chat_template \
  --max_new_tokens 4096 --dataset math500 --output_dir qwen3_hidden_states
```

## Evaluation Utilities

```bash
# R^2 on a prepared dataset (train_tpv.py reports MSE only)
python eval_r2.py --input_dir <dataset_dir>

# ...plus R^2 computed inside each held-out generation. Needs the raw hidden-states
# dir because the .pkl files drop generation boundaries. Pooled R^2 can flatter a
# probe that only knows roughly how far into an average trace a state sits.
python eval_r2.py --input_dir <dataset_dir> --run_dir <hidden_states> --per_generation

# R^2 by probe depth; streams H^T H / H^T y so no hidden states are materialised
python layer_sweep.py --run_dir <hidden_states> --model <name>
```

`train_tpv.py` reports MSE only. Note it refits on train+test before saving weights, so
the saved model is not the one whose held-out MSE is printed.

A note on generation budget: `max_new_tokens` controls how many trajectories reach
`</think>`, and `prepare_tpv_dataset.py` silently discards those that do not. At 1024
tokens only 18.7% of Llama-8B generations survive (biasing the dataset toward short
reasoning); at 4096 tokens 72.3% survive. Check the surviving fraction before trusting
a result.

## Notes
- Ensure that the model names and data paths are correctly set according to your environment.
- The alpha parameter controls the strength of the intervention - higher values result in stronger interventions.
- alpha can be set to 0.0 to disable the intervention.

## Citation
If you find this work useful, please consider citing our paper:

```bibtex
@misc{eisenstadt2025overclockingllmreasoningmonitoring,
      title={Overclocking LLM Reasoning: Monitoring and Controlling Thinking Path Lengths in LLMs}, 
      author={Roy Eisenstadt and Itamar Zimerman and Lior Wolf},
      year={2025},
      eprint={2506.07240},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.07240}, 
}
```

## License
This repository is licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license and provided for non-commercial use only. For commercial use, you must obtain a commercial license by contacting Ramot(yair.eran@ramot.org).
