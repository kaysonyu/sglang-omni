# MOSS-TTS Local rollout generation

Start a dedicated RL instance with `examples/configs/moss_tts_local_rl.yaml`.
The pipeline-level `enable_rl` option defaults to `false`. RL instances prepare
frame-decode graphs that include selected-action logprobs during startup;
ordinary instances keep the existing graphs and skip that computation.
With frame graphs disabled or an uncaptured batch size, RL uses the same
branchless frame decoder eagerly. Teacher scoring runs in a separate service.

RL rollout is used with non-streaming requests (`stream=false`). The dedicated
configuration disables vocoder CUDA graphs because they only accelerate
streaming codec steps and otherwise reserve GPU memory at startup. This does
not disable the tts_engine frame-decode graphs.

It accepts a text prompt or a dictionary containing text and references.
Send `/generate` requests with `return_logprob=true` and
`return_omni_rollout=true`. Supply positive text/audio temperatures,
`top_p=1`, `top_k=-1`, and `repetition_penalty=1`. The first version supports
this temperature-only policy; unsupported sampling settings are rejected.
RL instances require rollout requests, and ordinary instances reject them.
Set an initial version with `--tts_engine.engine.weight_version rl-v1`, or
publish an explicit weight version before collecting trajectories; requests
that cross a weight-version change cannot return a valid rollout.

The returned `omni_rollout` uses schema version 2 and
`temperature_scaled_full_vocab_v1` logprob semantics. It contains:

- Original `prompt_rows`, model identity and sampling parameters.
- A decision action stream (`0=continue`, `1=stop`) and selected logprobs.
- A time-major audio-code action stream with one column per codebook.
- Finish reason and admission weight version.

For T audio frames, natural termination has T continue decisions and one stop;
length termination has T continue decisions. A first-step stop is a valid
zero-frame rollout with one decision and no audio, not an empty tensor sent to
the vocoder. Ordinary speech requests retain their existing empty-output error.

The decode journal snapshots graph outputs before reuse and records only
committed actions. Trace tensors stay on device until request completion;
there is no per-frame CPU probability readback. A fused CUDA kernel computes
selected logprobs in FP32 without materializing full-vocabulary logprobs.
Completed traces travel as JSON bytes between stages and become dictionaries
at the client boundary, avoiding recursive tensor routing over each scalar.
RL still adds probability computation, retained tensors, and serialization.
Graph outputs and probability buffers are rebuilt with frame graphs when
weight storage changes.

## Teacher scoring

Use `examples/configs/moss_tts_local_score.yaml` for a separate teacher service.
Set its `model_path` to the converted checkpoint used by the teacher, then run:

```bash
python -m sglang_omni.cli serve --config examples/configs/moss_tts_local_score.yaml
```

`POST /score_actions` accepts `{"samples": [...]}`. Build each sample from the
original rollout:

```python
sample = {
    "version": 1,
    "sample_id": "sample-1",
    "prompt_rows": rollout["replay_inputs"]["prompt_rows"],
    "decisions": rollout["action_streams"][0]["actions"],
    "codes": rollout["action_streams"][1]["actions"],
    "temperature": 1.0,
}
```

The response is `{"version": 1, "results": [...]}` in input order, with selected
`decision_logprobs` and `code_logprobs`, `sample_id`, `input_sha256`,
`teacher_weight_sha256`, `weight_version`, `model_identity`, and
`logprob_semantics`. The input hash covers canonical validated input excluding
`sample_id`. A single positive temperature applies to both action streams.
A batch accepts up to 64 samples; the scheduler may process them in separate
prefill batches. `scoring_batch_sha256` identifies the actual prefill batch.
Model vocabulary, channel counts and context limits come from the loaded model.

The teacher uses unchunked global prefill without prefix reuse and scores Local
frames in bounded batches (`tts_engine.factory.score_chunk_size`, default 128).
It does not load the audio tokenizer or vocoder. The caller keeps teacher weights
fixed for the service lifetime. Its startup version and parameter checksum are
reported with every result; no teacher-specific weight-update guards are added.

BF16 full-prefill scoring and incremental generation can produce different
logprobs even with the same weights. Keep the sampled behavior logprobs from
the rollout for training ratios; teacher scores do not replace them.

## Model discovery

The `tts_engine` entry in `/model_info` publishes `model_identity`,
`rollout_schema_versions`, `logprob_semantics`, `supports_action_scoring`, and
`supports_weight_update`. RL generation and scoring instances report schema v2;
ordinary generation instances report no enabled rollout schema.

Teacher scoring instances also publish their startup `teacher_weight_sha256`
and `weight_version`, matching each score response. They declare
`supports_action_scoring=true` and `supports_weight_update=false` for downstream
role discovery. This declaration relies on the caller keeping teacher weights
fixed; it does not add enforcement to the update endpoints.
