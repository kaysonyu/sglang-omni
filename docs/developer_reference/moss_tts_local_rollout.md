# MOSS-TTS Local rollout generation

Start a dedicated RL instance with `examples/configs/moss_tts_local_rl.yaml`.
The pipeline-level `enable_rl` option defaults to `false`. RL instances prepare
frame-decode graphs that include selected-action logprobs during startup;
ordinary instances keep the existing graphs and skip that computation.
With frame graphs disabled or an uncaptured batch size, RL uses the same
branchless frame decoder eagerly. No teacher scoring endpoint is added.

RL rollout is used with non-streaming requests (`stream=false`).
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
