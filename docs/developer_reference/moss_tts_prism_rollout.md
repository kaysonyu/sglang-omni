# MOSS-TTS Prism rollout

Start `MossTTSPrismPipelineConfig` with `enable_rl: true`. The initial weight
version uses the framework default; training updates can publish a new version.
Ordinary serving defaults to RL disabled and
does not collect action scores. RL uses the existing non-streaming `/generate`
endpoint, full RVQ channels, TP1, and the checkpoint's complete execution schedule.

```json
{
  "prompt": {"script": "Please read this sentence."},
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
`max_new_tokens` counts emitted frames. `script` also accepts a list of
`{"text": "...", "local_instruction": "..."}` objects. Optional
`global_instruction` and ordered `references: [{"id": "audio1", "uri": "..."}]`
are rendered by the checkpoint processor. Reference URIs are local paths or audio
data URIs; their order determines audio1, audio2, and so on. Language guidance
continues to belong in the instruction, as in ordinary Prism serving.

`return_audio: false` skips vocoder decoding and returns `audio: null` with the
complete trace. It does not unload the codec. With audio enabled, `response_format`
accepts `wav` (default) or `flac`; audio encoding uses the codec sample rate.

## Trace and stopping

`meta_info.omni_rollout` uses schema version 2 and model family `moss_tts_prism`.
Its `replay_inputs` has layout `moss_prism_typed_v1` and contains the unpadded,
pre-generation `prompt_rows`, `item_kind`, `audio_role`, `successor_audio_mask`,
`successor_audio_codes`, and `target_history_retention_mask`. The original target
anchor is preserved; no KV state is serialized.

`action_streams` contains only `codes`: `[frames, n_vq]` actions, selected FP32
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

The trace includes effective sampling parameters, the resolved sampling seed,
finish reason, server request ID, model identity, and admission weight version.
`/model_info` exposes the same identity, hashing the original configuration,
modeling, and processor files. A request whose final weight version differs from
its admission version fails instead of returning a successful trace.

## Frozen teacher scoring

Start a separate teacher with `examples/configs/moss_tts_prism_score.yaml`,
setting `model_path` to its checkpoint. It uses the same Prism model and execution
schedule, with one prefill-only engine and no codec, vocoder, or CUDA graphs.

Send up to 64 samples to `POST /score_actions`. Build each sample from a rollout:

```python
sample = {
    "version": 1,
    "sample_id": "sample-001",
    **trace["replay_inputs"],
    "codes": trace["action_streams"][0]["actions"],
    "finish_reason": trace["finish_reason"],
    "temperature": 0.7,
}
response = requests.post(teacher_url + "/score_actions", json={"samples": [sample]})
```

The teacher forces the supplied codes without sampling or applying its stop
threshold. Each result contains selected `code_logprobs` of shape `[frames, n_vq]`
and the original `stop_probabilities`. Temperature scales only the RVQ logits.
Natural stops include the final stop observation; length-limited trajectories
do not. Zero-frame natural stops return an empty code score list and one stop
probability. Stop probabilities remain observations, not action logprobs.

Results preserve input order and carry `sample_id`, `input_sha256`, model identity,
and the teacher's weight version and parameter checksum. The input hash covers
canonical validated JSON excluding `sample_id`. `/model_info` advertises scoring
support and the same checksum reported by `/weights_checker?action=checksum`;
runtime caches are excluded. The teacher remains fixed during training.

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

Tensor loading can fail after some weights have changed. In that case, the stage
stays paused and does not publish the failed bucket's version. Restore a complete
checkpoint with `/update_weights_from_disk` or a complete NCCL update before
resuming; retain `keep_pause: true` during recovery. Same-shape disk reloads also
support `recapture_cuda_graph: false`.
