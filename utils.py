# This software is for non-commercial use only.
# Commercial use requires a separate license.

from datasets import load_dataset


DEFAULT_INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{}."

# Original raw-completion template. Works for models that continue from a bare
# "<think>\n" prefix (e.g. the DeepSeek-R1 distills).
DEFAULT_PROMPT_TEMPLATE = (
    "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n<think>\n"
)


def load_problems(dataset, starting_index=None, end_index=None):
    """
    Load a specified number of problems from a dataset starting from a given index.

    Args:
        dataset (str): The name of the dataset to load.
        starting_index (int): The index to start loading problems from.
        end_index (int): The index to stop at (exclusive).

    Returns:
        list: A list of problems loaded from the dataset.
    """
    # Load the dataset
    if dataset == "math500":
        if starting_index is None:
            starting_index = 30
        if end_index is None:
            end_index = 500
        problems = load_dataset("HuggingFaceH4/MATH-500")["test"]["problem"][starting_index:end_index]

    elif dataset == "gsm8k":
        if starting_index is None:
            starting_index = 30
        if end_index is None:
            end_index = 330
        problems = load_dataset("openai/gsm8k", "main")["train"]["question"][starting_index:end_index]

    else:
        raise ValueError("Unsupported dataset. Please use 'math500' or 'gsm8k'.")

    return problems


def resolve_hidden_layer(model, hidden_layer=None):
    """Resolve which entry of ``outputs.hidden_states`` to use as the TPV feature.

    ``hidden_states`` has ``num_hidden_layers + 1`` entries: index 0 is the embedding
    output and index ``i`` is the output of decoder layer ``i - 1``, so index ``n``
    (equivalently ``-1``) is the final layer.

    Earlier versions of this code hardcoded the final layer. Layer sweeps on held-out
    MATH-500 problems show the final layer is in fact the weakest useful probe target,
    because it is specialised for next-token prediction rather than carrying
    trajectory state.

    DeepSeek-R1-Distill-Llama-8B (32 layers, 434 generations):

        layer  2  R^2 0.673    layer 14  R^2 0.766  (44% depth)
        layer  6  R^2 0.733    layer 20  R^2 0.726
        layer  8  R^2 0.766    layer 32  R^2 0.602  <- final (previous default)
                  (25% depth)

    Qwen3-1.7B (28 layers, 418 generations):

        layer  4  R^2 0.758    layer 12  R^2 0.805  (43% depth) <- best
        layer  7  R^2 0.757    layer 14  R^2 0.804
                  (25% depth)  layer 18  R^2 0.789
        layer 10  R^2 0.773    layer 28  R^2 0.638  <- final

    Note the optimum is NOT a fixed 25% of depth: Llama-8B is flat between 25% and
    44%, but Qwen3-1.7B is clearly better at ~43% (0.805) than at 25% (0.757). We
    therefore default to 45% of depth, which is at or near the optimum for both.
    The plateau is broad, so the exact fraction is not critical -- but the final
    layer and the very early layers are both measurably bad.

    Args:
        model: a loaded HF model (uses ``model.config.num_hidden_layers``).
        hidden_layer (int | None): explicit index; negative counts from the end
            (``-1`` reproduces the old final-layer behaviour). ``None`` -> 25% depth.

    Returns:
        int: a non-negative index valid for ``outputs.hidden_states``.
    """
    n = model.config.num_hidden_layers
    if hidden_layer is None:
        return max(1, round(0.45 * n))
    if hidden_layer < 0:
        resolved = n + 1 + hidden_layer
    else:
        resolved = hidden_layer
    if not (0 <= resolved <= n):
        raise ValueError(
            f"hidden_layer={hidden_layer} resolves to {resolved}, outside [0, {n}] "
            f"for a model with {n} decoder layers."
        )
    return resolved


def resolve_intervention_module(model, layer_idx):
    """Return the module whose output corresponds to ``hidden_states[layer_idx]``.

    Monitoring and intervention must target the same site: a TPV trained on
    ``hidden_states[k]`` is only meaningful if the intervention is applied where
    those activations are produced.

    For the final layer this is ``model.model.norm`` (the pre-head norm, which is what
    ``hidden_states[-1]`` reflects). For layer ``k`` it is decoder layer ``k - 1``.

    IMPORTANT -- intervention is not symmetric with monitoring. The paper (Sec. 3.1,
    3.2) selects the final layer specifically because it sits downstream of every
    attention layer, so an edit there changes only the current step's logits and is
    never written into the KV cache. Editing an intermediate layer feeds attention in
    all later layers, so the perturbation is cached and leaks into every subsequent
    decoding step. Intermediate layers are the better MONITORING probe; the final layer
    remains the correct INTERVENTION site.
    """
    n = model.config.num_hidden_layers
    if layer_idx == n:
        return model.model.norm
    if layer_idx < 1:
        raise ValueError(
            f"Cannot hook layer_idx={layer_idx}; index 0 is the embedding output."
        )
    return model.model.layers[layer_idx - 1]


def build_prompt_template(tokenizer, use_chat_template=False, instruction=DEFAULT_INSTRUCTION):
    """Build a ``str.format``-ready prompt template containing a ``{problem}`` field.

    The raw-completion template appends a bare ``<think>\\n`` and relies on the model
    continuing from it. That holds for the DeepSeek-R1 distills but NOT for chat models
    such as Qwen3: without the ``<|im_start|>assistant`` scaffolding, Qwen3-1.7B emits
    ``</think>`` as its very first generated token, producing zero-length thinking
    (N=1 for every generation) and a silently degenerate dataset.

    Passing ``use_chat_template=True`` formats the problem through the tokenizer's own
    chat template with thinking enabled, after which the model opens ``<think>`` itself.

    Returns:
        str: template with a single ``{problem}`` placeholder; all other braces escaped.
    """
    if not use_chat_template:
        return DEFAULT_PROMPT_TEMPLATE

    placeholder = "__TPV_PROBLEM__"
    messages = [{"role": "user", "content": placeholder + instruction}]
    try:
        raw = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
    except TypeError:
        # tokenizers whose template does not accept enable_thinking
        raw = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    # escape every brace, then reintroduce the single {problem} field
    return raw.replace("{", "{{").replace("}", "}}").replace(placeholder, "{problem}")
