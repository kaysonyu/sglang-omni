"""Frozen Higgs teacher forcing, using the existing SGLang prefill scheduler."""

from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from typing import Any

import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.higgs_tts.engine_builder import HiggsTtsEngineBuilder
from sglang_omni.models.higgs_tts.rl_contract import model_identity
from sglang_omni.models.higgs_tts.scoring_protocol import HiggsScoreInput
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.generation_batch_policy import CudaGraphBackend
from sglang_omni.scheduling.types import ARRequestData


@dataclass
class HiggsScoreRequestData(ARRequestData):
    score: Any = None
    rows: Any = None
    req: Any = None
    stage_payload: Any = None
    score_result: Any = None
    admission_version: str | None = None
    synced: bool = False
    generation_steps: int = 0
    enforce_request_limits: bool = True


def build_score_request(payload, *, model):
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.runtime_context import get_serving
    from sglang.srt.sampling.sampling_params import SamplingParams

    score = HiggsScoreInput.model_validate(
        (payload.request.params or {}).get("higgs_score")
    )
    channels, vocab = model._num_codebooks, model._codebook_vocab_size
    if any(
        len(row) != channels or any(code < 0 or code >= vocab for code in row)
        for row in score.codes + score.reference_codes_delayed
    ):
        raise ValueError("Higgs scoring codes disagree with the model's vocabulary")
    text_vocab = int(model.config.get_text_config().vocab_size)
    if any(token >= text_vocab for token in score.prompt_token_ids):
        raise ValueError("Higgs scoring prompt exceeds the text vocabulary")
    prompt = torch.tensor(score.prompt_token_ids, dtype=torch.long)
    rows = torch.full(
        (len(prompt) + len(score.codes) - 1, channels + 1), -1, dtype=torch.long
    )
    rows[: len(prompt), 0] = prompt
    slots = prompt.eq(-100).nonzero().flatten()
    if len(slots):
        rows[slots, 1:] = torch.tensor(score.reference_codes_delayed, dtype=torch.long)
    rows[len(prompt) :, 0] = -100
    if len(score.codes) > 1:
        rows[len(prompt) :, 1:] = torch.tensor(score.codes[:-1], dtype=torch.long)
    token_ids = rows[:, 0].clamp_min(0).tolist()
    sampling = SamplingParams(max_new_tokens=0, temperature=1.0)
    sampling.normalize(None)
    sampling.verify(text_vocab)
    req = Req(
        rid=payload.request_id,
        origin_input_text="",
        origin_input_ids=token_ids,
        sampling_params=sampling,
        vocab_size=text_vocab,
    )
    req.tokenizer = None
    req._input_embeds_are_projected = True
    return HiggsScoreRequestData(
        input_ids=torch.tensor(token_ids),
        max_new_tokens=0,
        temperature=1.0,
        score=score,
        rows=rows,
        req=req,
        stage_payload=payload,
        admission_version=str(get_serving().weight_version),
    )


def score_result(data):
    if data.score_result is None or str(data.weight_version) != data.admission_version:
        raise RuntimeError(
            "Higgs teacher did not complete scoring under its admitted weight version"
        )
    return StagePayload(
        request_id=data.stage_payload.request_id,
        request=data.stage_payload.request,
        data={
            "modality": "text",
            "text": "",
            "omni_rollout": data.score_result,
            "weight_version": data.admission_version,
            "finish_reason": "stop",
        },
    )


class HiggsScoreRunner(ModelRunner):
    def custom_prefill_forward(self, forward_batch, schedule_batch, requests):
        if not schedule_batch.is_prefill_only or any(
            int(length) != len(request.data.rows)
            for length, request in zip(
                forward_batch.extend_seq_lens_cpu, requests, strict=True
            )
        ):
            raise ValueError(
                "Higgs scoring requires full prefill without prefix-cache reuse"
            )
        rows = torch.cat([request.data.rows for request in requests]).to(self.device)
        audio = rows[:, 0].eq(-100)
        text = self.model.backbone.model.embed_tokens(rows[:, 0].masked_fill(audio, 0))
        encoded = self.model.multimodal_embedding.modality_embedding_0(
            rows[:, 1:].clamp_min(0)
        )
        forward_batch.input_embeds = torch.where(audio[:, None], encoded, text)
        return None

    def post_prefill(self, result, forward_batch, schedule_batch, requests):
        hidden = result.logits_output.hidden_states
        if hidden.ndim != 2 or len(hidden) != sum(len(r.data.rows) for r in requests):
            raise RuntimeError("Higgs scoring requires all prefill hidden states")
        offset = 0
        for request in requests:
            data, score = request.data, request.data.score
            start = offset + len(score.prompt_token_ids) - 1
            selected = hidden[start : start + len(score.codes)]
            logits = (
                self.model.modality_head.generate(selected).float() / score.temperature
            )
            codes = torch.tensor(score.codes, device=logits.device, dtype=torch.long)
            values = logits.log_softmax(-1).gather(-1, codes[..., None]).squeeze(-1)
            if not torch.isfinite(values).all():
                raise ValueError("Non-finite Higgs teacher scores")
            data.score_result = dict(
                version=1,
                sample_id=score.sample_id,
                input_sha256=score.input_sha256(),
                code_logprobs=values.cpu().tolist(),
                temperature=score.temperature,
                weight_version=data.admission_version,
                teacher_weight_sha256=self.model._higgs_teacher_weight_sha256,
                model_identity=model_identity(),
                logprob_semantics="temperature_scaled_full_vocab_v1",
            )
            offset += len(data.rows)


class HiggsScoreEngineBuilder(HiggsTtsEngineBuilder):
    supports_breakable_prefill_cuda_graph = False

    def generation_defaults(self, *, dtype):
        values = super().generation_defaults(dtype=dtype)
        values.update(
            disable_cuda_graph=True,
            disable_radix_cache=True,
            chunked_prefill_size=-1,
            cuda_graph_backend_prefill=CudaGraphBackend.DISABLED,
            cuda_graph_bs_prefill=None,
            max_prefill_tokens=4096,
            max_total_tokens=8192,
            mem_fraction_static=0.2,
        )
        return values

    def setup_model(self, **kwargs):
        super().setup_model(**kwargs)
        self.model._higgs_score_only = True

    def make_model_runner(self, model_worker, output_proc):
        return HiggsScoreRunner(model_worker, output_proc)

    def make_adapters(self, model):
        return partial(build_score_request, model=model), score_result

    def make_abort_callback(self):
        return None

    def make_request_finished_callback(self):
        return None

    def post_scheduler_setup(self, scheduler, model_runner):
        from sglang_omni.model_runner.weight_checker import StrictWeightChecker

        class ParameterChecker(StrictWeightChecker):
            @staticmethod
            def _iter_named_tensors(model):
                return model.named_parameters()

        result = ParameterChecker(SimpleNamespace(model=self.model)).checksum()
        self.model._higgs_teacher_weight_sha256 = result["per_gpu_checksum"]


def create_score_engine(
    model_path,
    *,
    device="cuda:0",
    gpu_id=None,
    dtype="bfloat16",
    server_args_overrides=None,
    total_gpu_memory_fraction=None,
):
    return HiggsScoreEngineBuilder(
        max_new_tokens=None,
        max_running_requests=16,
        cuda_graph_max_bs=1,
        enable_async_decode=False,
        async_decode_min_batch_size=2,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
    ).build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )
