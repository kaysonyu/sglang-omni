# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.models.moss_tts_local.sglang_model import (
    MossTTSLocalSGLangModel,
)


class _IdentityLocalTransformer(torch.nn.Module):
    def step(self, hidden_states: torch.Tensor, position: int) -> torch.Tensor:
        del position
        return hidden_states


def _tiny_model() -> MossTTSLocalSGLangModel:
    model = MossTTSLocalSGLangModel.__new__(MossTTSLocalSGLangModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        audio_vocab_size=3,
        audio_assistant_slot_token_id=1,
    )
    model.hidden_size = 2
    model.n_vq = 1
    model.embedding_list = torch.nn.ModuleList(
        [torch.nn.Embedding(4, 2), torch.nn.Embedding(4, 2)]
    )
    model.audio_lm_heads = torch.nn.ModuleList([torch.nn.Linear(2, 3, bias=False)])
    model.local_transformer = _IdentityLocalTransformer()
    model.local_text_lm_head = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.embedding_list[0].weight.zero_()
        model.embedding_list[0].weight[1] = torch.tensor([10.0, 20.0])
        model.embedding_list[1].weight.copy_(
            torch.tensor(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [5.0, 6.0],
                    [0.0, 0.0],
                ]
            )
        )
        # For hidden=[1, 0], the independent head selects code 2. The audio
        # embedding table would instead select code 1 if it were (incorrectly)
        # reused as the output projection.
        model.audio_lm_heads[0].weight.copy_(
            torch.tensor([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
        )
        model.local_text_lm_head.weight.zero_()
    model._sample_seeded_branchless = lambda logits, **_kwargs: logits.argmax(dim=-1)
    return model


def test_graphable_decode_uses_head_for_logits_and_embedding_for_feedback():
    model = _tiny_model()
    hidden = torch.tensor([[1.0, 0.0]])
    one = torch.ones(1)
    integer_one = torch.ones(1, dtype=torch.long)

    _, codes, feedback, _, _ = model._decode_frame_graphable(
        hidden_states=hidden,
        text_temperature=one,
        text_top_p=one,
        text_top_k=integer_one,
        audio_temperature=one,
        audio_top_p=one,
        audio_top_k=integer_one,
        seeds=torch.zeros(1, dtype=torch.long),
        base_positions=torch.zeros(1, dtype=torch.long),
    )

    assert codes.tolist() == [[2]]
    torch.testing.assert_close(feedback, torch.tensor([[15.0, 26.0]]))


def test_weight_loader_updates_independent_audio_head():
    model = _tiny_model()
    replacement = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])

    model.load_weights([("audio_lm_heads.0.weight", replacement)])

    torch.testing.assert_close(model.audio_lm_heads[0].weight, replacement)
    assert not torch.equal(
        model.audio_lm_heads[0].weight,
        model.embedding_list[1].weight[:3],
    )
