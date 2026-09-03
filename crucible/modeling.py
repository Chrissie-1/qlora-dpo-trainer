"""Model loading: 4-bit base, LoRA attachment, adapter reload.

Every phase loads the model through here so the quantisation settings that
training saw are exactly the ones evaluation and generation see. A QLoRA
adapter trained against nf4 weights and then evaluated against fp16 weights is
a different model.
"""

from __future__ import annotations

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from crucible.config import BASE_MODEL

# Attention projections only, as the plan specifies. Adding the MLP would lift
# quality but roughly triples trainable parameters, and the budget here is
# 8 GB of VRAM.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def quant_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_tokenizer(model_id: str = BASE_MODEL, *, for_generation: bool = False):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Decoder-only generation must pad on the left or the sampled continuation
    # starts after the padding. Training pads on the right and masks with -100.
    tok.padding_side = "left" if for_generation else "right"
    return tok


def load_base(
    model_id: str = BASE_MODEL,
    *,
    quantized: bool = True,
    for_training: bool = False,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config() if quantized else None,
        dtype=torch.bfloat16,
        device_map={"": 0} if torch.cuda.is_available() else "cpu",
    )
    if for_training:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model.config.use_cache = False
    return model


def attach_lora(model, *, r: int = 16, alpha: int = 32, dropout: float = 0.05):
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGETS,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def load_for_inference(adapter: str | None = None, *, quantized: bool = True):
    """Base model, optionally with an adapter merged in for generation."""
    model = load_base(quantized=quantized)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    model.config.use_cache = True
    return model
