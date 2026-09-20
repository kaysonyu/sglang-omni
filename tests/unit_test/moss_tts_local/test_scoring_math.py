import torch

from sglang_omni.models.moss_tts_local.scoring_math import score_local_depth


class _LocalTransformer:
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size

    def step(self, hidden, depth):
        return hidden + float(depth + 1)


class _Model:
    dtype = torch.float32
    n_vq = 2

    def __init__(self):
        self.local_transformer = _LocalTransformer(hidden_size=4)
        self.local_text_lm_head = torch.nn.Linear(4, 3, bias=False)
        self.audio_heads = [
            torch.nn.Linear(4, 5, bias=False),
            torch.nn.Linear(4, 5, bias=False),
        ]
        self.audio_embeddings = [
            torch.nn.Embedding(5, 4),
            torch.nn.Embedding(5, 4),
        ]

    def _audio_head_weight(self, depth):
        return self.audio_heads[depth].weight

    def _audio_embedding_weight(self, depth):
        return self.audio_embeddings[depth].weight


def test_score_local_depth_returns_decision_and_code_logprobs():
    model = _Model()
    hidden = torch.ones(3, 4)
    decisions = torch.tensor([0, 1, 0])
    codes = torch.tensor([[1, 2], [2, 3], [3, 4]])

    decision_logprobs, code_logprobs = score_local_depth(
        model,
        hidden,
        decisions,
        codes,
        temperature=1.0,
        chunk_size=2,
    )

    assert decision_logprobs.shape == (3,)
    assert code_logprobs.shape == (3, 2)
    assert torch.isfinite(decision_logprobs).all()
    assert torch.isfinite(code_logprobs).all()
