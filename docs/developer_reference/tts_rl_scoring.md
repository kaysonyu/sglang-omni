# Discrete TTS RL generation and teacher scoring

The MOSS Local model, processor invocation and native codec follow the supplied
snapshot reported as `92a53c268a167e153cbe7f36b1d916b8a72c0b4f`; source hashes are
recorded in `moss_local_reference.json`. RL interfaces are added on that Local
implementation. Serve the converted HF artifact with `gpt2_config` and native
Local parameter names. Original GPTNeoX weights must be converted offline;
there is no runtime NeoX repack or alternate hand-written prompt renderer.
The artifact's processor compatibility shim handles v1.5 call-site keywords.

RL clients request `/generate` with `return_logprob=true` and
`return_omni_rollout=true`. The v2 trace contains original conditioning, actual
sampled actions, selected behavior logprobs, finish reason, admission weight
version and model identity. The HTTP prompt may be a dictionary, allowing
Higgs reference audio/text to reach its normal frontend.

For the supported RL path, probabilities are
`temperature_scaled_full_vocab_v1`: positive temperature, no top-k/top-p
filtering, and no MOSS audio repetition penalty. MOSS preprocessing respects
explicit request sampling and the `tts_engine` stage overrides before it
prepares the prompt. Logprobs describe the actual selected policy actions.

Higgs has two masks: its existing `action_mask` identifies real audio, while
`sampled_action_mask` also includes sampled EOC/tail choices and excludes the
forced initial delay/BOC positions. Training must use the latter. Neither mask
changes audio decoding or sampling.

## Frozen teacher services

Launch `examples/configs/moss_tts_local_score.yaml` or
`examples/configs/higgs_tts_score.yaml` with a complete HF model directory.
These pipelines perform unchunked prefill with no prefix reuse and return
teacher-forced scores for supplied actions. They do not start a vocoder.

`POST /score_actions` accepts `{ "samples": [...] }` and returns
`{ "version": 1, "results": [...] }`. Every sample has a stable `sample_id`:

- MOSS: `version`, `sample_id`, `prompt_rows`, `decisions`, `codes`, `temperature`.
- Higgs: `version`, `sample_id`, `prompt_token_ids`,
  `reference_codes_delayed`, `codes`, `temperature`.

A result includes `sample_id`, `input_sha256`, `model_identity`,
`teacher_weight_sha256`, `weight_version`, `temperature`, `logprob_semantics`
and the selected-action scores in the model's native geometry. The input hash
is SHA256 of the canonical JSON of the validated sample excluding `sample_id`.
It binds the original conditioning and original student actions; teachers do
not substitute their own generations. Full-vocabulary logits are not returned.

Frozen scorer capabilities disable weight mutation. The common ModelWorker
rejects disk/tensor/distributed refit, update-group initialization and checker
`reset_tensors` when `supports_weight_update=false`. Normal student stages keep
the existing refit capability. Freeze and selected-action geometry are separate
concerns: the public envelope is shared, while the forward/scoring calculation
is implemented by each model family.

CPU contracts are in `tests/unit_test/higgs_tts/test_scoring_protocol.py` and
the MOSS Local pipeline tests. The linked slime TTS branch exercises real
MOSS/Higgs WER and two-route MOPD on H200. BF16 generation versus full-sequence
replay is not bitwise equivalent; consumers should keep actual behavior scores
and measure mismatch and policy-ratio statistics.
