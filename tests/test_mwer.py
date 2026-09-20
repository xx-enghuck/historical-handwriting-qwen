import pytest
import torch
from PIL import Image

from htr.config import LoraConfig, ModelConfig
from htr.decoding.rescore import rescore_candidates, sequence_logprobs
from htr.models.lora import add_lora, is_adapter, set_trainable, train_mode
from htr.models.qwen import QwenEncoder
from htr.testing import tiny_components
from htr.training.losses import mwer_loss, nbest_probabilities


def test_sequence_probability_shift_mask_and_length():
    probabilities = torch.tensor([[[0.2, 0.8], [0.3, 0.7], [0.6, 0.4], [0.1, 0.9]]])
    logits = probabilities.log().requires_grad_()
    labels = torch.tensor([[-100, -100, 1, 0]])
    expected = torch.log(torch.tensor(0.7 * 0.6))
    score = sequence_logprobs(logits, labels)
    assert score.item() == pytest.approx(expected.item())
    assert sequence_logprobs(logits, labels, 1).item() == pytest.approx(expected.item() / 2)
    score.sum().backward()
    assert logits.grad[0, 0].abs().sum() == 0
    assert logits.grad[0, 3].abs().sum() == 0


def test_toy_mwer_centering_and_risk_detach():
    scores = torch.tensor([0.0, torch.log(torch.tensor(3.0))], requires_grad=True)
    risks = torch.tensor([0.0, 1.0], requires_grad=True)
    assert nbest_probabilities(scores).tolist() == pytest.approx([0.25, 0.75])
    loss = mwer_loss(scores, risks)
    assert loss.item() == pytest.approx(0.25)
    centered_grad = torch.autograd.grad(loss, scores, retain_graph=True)[0]
    uncentered_grad = torch.autograd.grad(mwer_loss(scores, risks, False), scores)[0]
    assert torch.allclose(centered_grad, uncentered_grad)
    loss.backward()
    assert risks.grad is None
    assert scores.grad.tolist() == pytest.approx([-0.1875, 0.1875])
    assert nbest_probabilities(torch.tensor([-10000.0, -10001.0])).sum() == pytest.approx(1)


@pytest.mark.parametrize("checkpointed", [False, True])
def test_actual_lora_gradient_through_rescoring_not_wer(checkpointed):
    torch.set_num_threads(1)
    model, processor = tiny_components()
    model = add_lora(model, LoraConfig(r=2, alpha=4, dropout=0))
    set_trainable(model, "mwer")
    train_mode(model, "mwer")
    encoder = QwenEncoder(processor, ModelConfig(prompt="Transcribe."))
    candidates = [{"text": text, "token_ids": encoder.response_ids(text)} for text in ("a", "b")]
    image = Image.new("RGB", (32, 16), "white")
    sequence_logprob = rescore_candidates(
        model, encoder, image, candidates, checkpoint_forward=checkpointed
    )
    from htr.evaluation.metrics import sample_metrics

    risk = torch.tensor([sample_metrics("a", c["text"])["wer"] for c in candidates])
    assert risk.requires_grad is False
    assert sequence_logprob.requires_grad is True
    assert nbest_probabilities(sequence_logprob).sum().item() == pytest.approx(1)
    mwer_loss(sequence_logprob, risk).backward()
    lora = [p for n, p in model.named_parameters() if is_adapter(n)]
    assert all(p.grad is not None for p in lora)
    assert any(p.grad.abs().sum() > 0 for p in lora)
    assert all(p.grad is None for n, p in model.named_parameters() if not is_adapter(n))
