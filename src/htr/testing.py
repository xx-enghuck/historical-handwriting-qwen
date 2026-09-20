"""Tiny, randomly initialized REAL Qwen model/processor for offline software smoke tests.

This is never selected implicitly by training or inference commands and produces
no meaningful recognition accuracy. No pretrained weights are downloaded.
"""

from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import (
    PreTrainedTokenizerFast,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    Qwen2VLImageProcessorPil,
    Qwen2VLVideoProcessor,
)


def tiny_components():
    specials = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
    ]
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {s: i for i, s in enumerate(specials + alphabet)}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token=specials[2],
        pad_token=specials[0],
        additional_special_tokens=specials,
        model_input_names=["input_ids", "attention_mask"],
    )
    template = """{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\n' }}{% for c in m['content'] %}{% if c['type'] == 'image' %}{{ '<|vision_start|><|image_pad|><|vision_end|>' }}{% else %}{{ c['text'] }}{% endif %}{% endfor %}{{ '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"""
    processor = Qwen2_5_VLProcessor(
        tokenizer=tokenizer,
        chat_template=template,
        video_processor=Qwen2VLVideoProcessor(),
        image_processor=Qwen2VLImageProcessorPil(
            patch_size=2, temporal_patch_size=2, merge_size=2, min_pixels=64, max_pixels=256
        ),
    )
    config = Qwen2_5_VLConfig(
        text_config={
            "vocab_size": len(vocab),
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "pad_token_id": 0,
            "eos_token_id": 2,
            "bos_token_id": 1,
            "rope_parameters": {
                "rope_type": "default",
                "mrope_section": [1, 1, 2],
                "rope_theta": 10000,
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "out_hidden_size": 32,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "window_size": 8,
            "fullatt_block_indexes": [0],
        },
        image_token_id=3,
        video_token_id=4,
        vision_start_token_id=5,
        vision_end_token_id=6,
    )
    return Qwen2_5_VLForConditionalGeneration(config), processor
