# Historical Handwriting Recognition with Qwen2.5-VL

An architecture prototype combining Qwen2.5-VL-3B, LoRA, StackMix-style
augmentation and minimum word error rate (MWER) training.
Full model training and recognition accuracy remain unverified.

## Model architecture

```mermaid
flowchart TD
    I[Handwritten line] --> C[Crop + image processor]
    P[Text prompt] --> T[Chat template + tokenizer]

    subgraph Q[Qwen2.5-VL-3B]
        V[Vision encoder - frozen] --> M[Visual merger]
        M --> F[Image + text tokens]
        F --> L[Language decoder - frozen base + LoRA]
    end

    C --> V
    T --> F
    L --> D[Greedy / beam search]
    D --> O[Transcription]
```

**SFT:** update LoRA + visual merger. **MWER:** update LoRA only; merger frozen.

[MIT](LICENSE). Model weights and datasets are not included.
