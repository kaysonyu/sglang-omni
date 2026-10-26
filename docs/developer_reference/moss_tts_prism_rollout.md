# MOSS-TTS Prism rollout

Start `MossTTSPrismPipelineConfig` with `enable_rl: true`. The initial weight
version uses the framework default; training updates can publish a new version.
Ordinary serving defaults to RL disabled and
does not collect action scores. RL uses the existing non-streaming `/generate`
endpoint, full RVQ channels, TP1, and the checkpoint's complete execution schedule.

```json
{
  "prompt": {
    "modality": "speech",
    "script": "Please read this sentence."
  },
  "sampling_params": {
    "temperature": 0.7,
    "top_p": 1.0,
    "top_k": -1,
    "repetition_penalty": 1.0,
    "max_new_tokens": 64,
    "seed": 42
  },
  "stream": false,
  "output_modalities": ["audio"],
  "return_logprob": true,
  "return_omni_rollout": true,
  "return_audio": false
}
```

Both return flags are required on an RL instance. Temperature must be finite and
positive; top-p, top-k, and repetition penalty must have the neutral values above.
`max_new_tokens` counts emitted frames. Prism accepts the checkpoint's unified
prompt request through `/generate`:

- `modality`: `speech` (default) or `sing`.
- `script`: nonempty text or an array of `{"text": "...", "local_instruction": {...}}` segments.
- `global_instruction`: an attribute object. An explicit `Language` entry adds
  language guidance; reward metadata does not supply this condition.
- `references`: a mapping from named audio slots to local paths or audio data URIs.
  Slots may be noncontiguous, such as `audio3` and `audio10`.
- `reference_attributes`: explicit `timbre`, `accent`, `prosody`, and `emotion`
  bindings to a provided audio slot or `none`.
- `global_tokens`: an optional positive target frame count, independent of the
  generation limit.

The service validates the request and encodes its reference audio, then calls
`processor.build_user_message(request)`. The checkpoint processor owns attribute
normalization, ordering, rendering, and renderer revision. Checkpoints must
include `prompt_protocol.py` and preserve the typed input layout.

Prism does not adapt the old task-based prompt schema or `/v1/audio/speech`
shorthand. Use the unified request for inference and RL. Stringified instructions,
reference lists, and unknown prompt fields fail validation. Speech endpoints for
other model families retain their own contracts.

`return_audio: false` skips vocoder decoding and returns `audio: null` with the
complete trace. It does not unload the codec. With audio enabled, `response_format`
accepts `wav` (default) or `flac`; audio encoding uses the codec sample rate.

## Trace and stopping

`meta_info.omni_rollout` uses schema version 2 and model family `moss_tts_prism`.
Its `replay_inputs` has layout `moss_prism_typed_v1` and contains the unpadded,
pre-generation `prompt_rows`, `item_kind`, `audio_role`, `successor_audio_mask`,
`successor_audio_codes`, and `target_history_retention_mask`. The original target
anchor is preserved; no KV state is serialized.

With default threshold stopping, `action_streams` contains only `codes`:
`[frames, n_vq]` actions, selected FP32
`log_softmax(logits / temperature)` scores, and all-one action masks. The total
action count is `frames * n_vq`. Scores are collected in eager, full decode CUDA
Graph, and breakable prefill execution, using the live request's temperature and
seed. No full probability distribution is returned. These are behavior scores from the
actual serving forward pass. BF16 replay in another framework may produce different
logits; it should not replace the recorded behavior scores.

Stopping keeps the checkpoint's deterministic pre-frame threshold policy:
`softmax(stop_logits.float())[1] > 0.1`. The unmodified probability used for each
decision is returned in `non_action_outputs`:

```json
{"name": "stop_probabilities", "stage": "tts_engine", "values": [0.01, 0.02, 0.8]}
```

These are observations, not sampled actions or decision logprobs.
`stop_semantics` is `pre_frame_threshold_v1`. A natural stop has one more
observation than emitted frames. A length cap has exactly one observation per
frame, without an extra terminal query. Immediate stop is a valid zero-frame
rollout with one observation and no audio. RL does not force a first frame or
change the original length distribution.

Set `sampling_params.stop_sampling: true` to sample the binary stop/continue
decision instead. The request resolves this mode once; mixed batches apply it
per request. Stop sampling uses temperature 1, no vocabulary truncation, a seed
derived separately from the RVQ seed, and the emitted-frame counter. The same
seeded sampler handles RVQ and stop; enabling RL recording preserves the sampled
actions. Only sampled requests contribute stop actions and scores to the trace.

Stage overrides use `stage_sampling.tts_engine.stop_sampling` or
`stage_params.tts_engine.stop_sampling`. The latter takes precedence, followed by
stage sampling, then top-level sampling; an explicit `false` overrides `true`.
Speech requests can set this option through `stage_params.tts_engine` as well.

This mode declares `stop_semantics: pre_frame_bernoulli_v1` and adds a
`stop_decisions` stream containing actions `0` (continue) and `1` (stop), with
their selected behavior logprobs. Natural stopping records `F + 1` decisions;
the length limit records `F`. The total action count includes these decisions.
Immediate stop is a valid RL response with empty `[0, n_vq]` code actions,
`stop_decisions: [1]` and `audio: null`. Consumers should retain this sampled
outcome in their rewards and policy objective. Ordinary serving can use the
same sampling mode without recording scores; a zero-frame audio-only request
retains the existing retry error.

The trace includes effective sampling parameters, the resolved sampling seed,
finish reason, server request ID, model identity, and admission weight version.
`/model_info` exposes the same identity, hashing the original configuration,
modeling, processor, and prompt protocol files. A request whose final weight version differs from
its admission version fails instead of returning a successful trace.

## Frozen teacher scoring

Start a separate teacher with `examples/configs/moss_tts_prism_score.yaml`,
setting `model_path` to its checkpoint. It uses the same Prism model and execution
schedule, with one prefill-only engine and no codec, vocoder, or CUDA graphs.

Send up to 64 samples to `POST /score_actions`. Build each sample from a rollout:

```python
sample = {
    "version": 2,
    "sample_id": "sample-001",
    **trace["replay_inputs"],
    "codes": trace["action_streams"][0]["actions"],
    "finish_reason": trace["finish_reason"],
    "temperature": 0.7,
}
response = requests.post(teacher_url + "/score_actions", json={"samples": [sample]})
```

The teacher forces the supplied codes without sampling or applying its stop
threshold. Version 2 results contain selected `code_logprobs` of
shape `[frames, n_vq]`, plus `selected_stop_logprobs` and `other_stop_logprobs`,
each of shape `[observations]`.
The student actions are determined by the trajectory: one continue for every
frame, followed by a stop only when `finish_reason` is `stop`. The teacher
selects both the supplied action and the other binary action directly from the
same FP32 log-softmax. The pair preserves the complete stop distribution even
when the selected logprob rounds to zero or a probability underflows. Clients
computing stop forward KL must require both fields instead of reconstructing the
other logprob from the selected one. The additional `other_stop_logprobs` field
is additive within version 2; selected-score clients can ignore it.
Temperature scales only the RVQ logits. Natural stops include the final stop
observation; length-limited trajectories do not. Zero-frame natural stops return
an empty code score list and one entry in each stop logprob list.

The HTTP batch envelope remains version 1; each Prism sample and result uses
version 2. Upgrade the scoring client and teacher together. Version 1 Prism
requests are rejected rather than served with a different response shape.

Results preserve input order and carry `sample_id`, `input_sha256`, model identity,
and the teacher's weight version. The input hash covers canonical validated JSON
excluding `sample_id`. `/model_info` advertises scoring support and the teacher's
read-only role through an explicit capability flag. Teacher startup does not
compute a parameter checksum, and score responses do not carry one.

`tts_engine.factory.score_chunk_size` bounds the number of prediction states
projected through each scoring head at once (default 128). Scores come from the
teacher's BF16 replay and need not match the student's recorded behavior scores,
even for equal checkpoint weights, because the execution shapes differ.

## Student weight updates

Use the shared [RL admin endpoints](rl_admin_control.md) with checkpoints that
retain the same configuration and parameter shapes. Pause generation with
`mode: "abort"` before publishing weights.

For NCCL updates, send the complete checkpoint weight map in sorted HF key order.
Keep `keep_pause: true` and `flush_cache: true` on every bucket. Omit
`weight_version` on intermediate buckets and publish it only with the final
bucket. After all buckets succeed, confirm the version through `/model_info`
and call `/continue_generation`. The loader writes into existing parameter
storage, allowing the captured prefill and decode graphs to use the new weights.

The colocated CUDA IPC tensor endpoint additionally supports a tracked
`publication` manifest. That protocol requires the same explicit weight version
on every bucket and blocks resume until all buckets are acknowledged. See
[Colocated RL](colocated_rl.md) for its ordering and recovery contract.

Tensor loading can fail after some weights have changed. In that case, the stage
stays paused and does not publish the failed bucket's version. Restore a complete
checkpoint with `/update_weights_from_disk` or a complete NCCL update before
resuming; retain `keep_pause: true` during recovery. Same-shape disk reloads also
support `recapture_cuda_graph: false`.
