"""Model, data and training settings; paths are relative to project_root."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    image_root: str = "data/private"
    source_csv: str = "data/private/train.csv"
    validation_csv: str | None = None
    test_csv: str | None = None
    group_column: str | None = "writer_id"
    validation_fraction: float = 0.2
    split_dir: str = "artifacts/splits"


@dataclass
class CropConfig:
    enabled: bool = True
    threshold: int = 235
    padding: int = 8
    debug_dir: str | None = None


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    revision: str = "main"
    device: str = "auto"
    dtype: str = "auto"
    attn_implementation: str = "sdpa"
    min_pixels: int = 3136
    max_pixels: int = 401408
    max_sequence_length: int = 4096
    prompt: str = (
        "Transcribe the handwritten text exactly.\n"
        "Preserve the original spelling and punctuation.\n"
        "Output only the transcription."
    )


@dataclass
class LoraConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    targets: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class TrainConfig:
    output_dir: str = "runs/sft"
    epochs: int = 3
    batch_size: int = 1
    gradient_accumulation: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    early_stopping_patience: int | None = None
    resume: str | None = None
    checkpoint: str | None = None


@dataclass
class DecodeConfig:
    num_beams: int = 1
    num_return_sequences: int = 1
    max_new_tokens: int = 256
    length_penalty: float = 1.0
    input_csv: str | None = None
    output: str = "outputs/predictions.jsonl"


@dataclass
class NormalizeConfig:
    unicode_form: str | None = None
    lowercase: bool = False
    collapse_whitespace: bool = False
    strip: bool = False


@dataclass
class StackMixConfig:
    enabled: bool = False
    synthetic_ratio: float = 1.0
    bank_dir: str = "artifacts/stackmix"
    synthetic_dir: str = "data/private/synthetic"
    ctc_checkpoint: str = "artifacts/ctc/model.pt"
    ctc_epochs: int = 30
    ctc_learning_rate: float = 0.001
    height: int = 48
    min_quality: float = 0.5
    max_alignment_cer: float = 0.25
    spacing: int = 0
    word_spacing: int = 16
    style_consistent: bool = False
    min_words: int = 2
    max_words: int = 8


@dataclass
class MWERConfig:
    nbest_path: str = "artifacts/nbest/train.jsonl"
    num_beams: int = 4
    num_return_sequences: int = 4
    length_normalization: float = 0.0
    center_risks: bool = True
    lambda_ce: float = 0.1


@dataclass
class Config:
    project_root: str = "."
    seed: int = 42
    deterministic: bool = True
    data: DataConfig = field(default_factory=DataConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    normalization: NormalizeConfig = field(default_factory=NormalizeConfig)
    stackmix: StackMixConfig = field(default_factory=StackMixConfig)
    mwer: MWERConfig = field(default_factory=MWERConfig)

    def path(self, value: str) -> Path:
        return (Path(self.project_root) / Path(value).expanduser()).resolve()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if not 0 < self.data.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.train.batch_size < 1 or self.train.gradient_accumulation < 1:
            raise ValueError("batch_size and gradient_accumulation must be positive")
        if self.train.epochs < 1 or self.train.learning_rate <= 0:
            raise ValueError("epochs and learning_rate must be positive")
        if self.crop.padding < 0 or not 0 <= self.crop.threshold <= 255:
            raise ValueError("Invalid crop padding or threshold")
        if self.lora.bias != "none":
            raise ValueError("This frozen-base research design requires lora.bias: none")
        if not 1 <= self.decode.num_return_sequences <= self.decode.num_beams:
            raise ValueError("Require 1 <= num_return_sequences <= num_beams")
        if not 2 <= self.mwer.num_return_sequences <= self.mwer.num_beams:
            raise ValueError("MWER requires at least two candidates and enough beams")
        if self.mwer.lambda_ce < 0 or self.mwer.length_normalization < 0:
            raise ValueError("MWER lambda_ce and length_normalization must be nonnegative")
        if self.stackmix.synthetic_ratio < 0 or self.stackmix.height < 8:
            raise ValueError("Invalid StackMix ratio/height")
        if (
            not 0 <= self.lora.dropout < 1
            or self.lora.r < 1
            or self.lora.alpha <= 0
            or not self.lora.targets
        ):
            raise ValueError("Invalid LoRA rank, alpha, dropout or targets")
        if self.decode.max_new_tokens < 1 or self.model.max_sequence_length < 2:
            raise ValueError("Invalid generation/sequence length limit")
        if not 0 < self.model.min_pixels <= self.model.max_pixels:
            raise ValueError("Require 0 < min_pixels <= max_pixels")
        if not 0 <= self.stackmix.min_quality <= 1 or self.stackmix.max_alignment_cer < 0:
            raise ValueError("Invalid CTC alignment quality threshold")
        if self.stackmix.ctc_epochs < 1 or self.stackmix.ctc_learning_rate <= 0:
            raise ValueError("Invalid CTC training schedule")
        if self.stackmix.spacing < 0 or self.stackmix.word_spacing < 1:
            raise ValueError("Invalid StackMix spacing")
        if not 1 <= self.stackmix.min_words <= self.stackmix.max_words:
            raise ValueError("Invalid StackMix word count range")
        if self.normalization.unicode_form not in {None, "NFC", "NFD", "NFKC", "NFKD"}:
            raise ValueError("Unsupported Unicode normalization form")
        if (
            self.train.early_stopping_patience is not None
            and self.train.early_stopping_patience < 1
        ):
            raise ValueError("early_stopping_patience must be positive or null")
