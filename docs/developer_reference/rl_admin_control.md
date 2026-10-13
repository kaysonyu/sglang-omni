# RL Admin Control

SGLang-Omni exposes a small administrative API for inference-side RL workflows.
The contract follows the SGLang and Miles control surface while preserving the
Omni pipeline boundary:

```text
HTTP / router -> Client -> Coordinator -> Stage -> Scheduler -> ModelWorker
```

The control plane carries only metadata and small result summaries. Tensor
payloads and bulk checkpoint data must be moved through disk, a distributed
group, or another data plane.

## Authentication

Admin endpoints are unauthenticated by default for backward compatibility. They
require `Authorization: Bearer <key>` when either of these is set:

- `admin_api_key` passed to the worker/router `create_app(...)`
- `SGLANG_OMNI_ADMIN_KEY` in the environment

The external router also accepts `--admin-api-key`. The router forwards the
`Authorization` header to workers, so a deployment can use the same key at both
layers.

## Worker Endpoints

The worker server supports:

- `GET|POST /model_info`
- `POST /pause_generation`
- `POST /continue_generation`
- `POST /update_weights_from_disk`
- `POST /update_weights_from_tensor`
- `POST /release_memory_occupation`
- `POST /resume_memory_occupation`
- `POST /init_weights_update_group`
- `POST /destroy_weights_update_group`
- `POST /update_weights_from_distributed`
- `GET|POST /weights_checker`

`/update_weights_from_disk` pauses
the target scheduler, optionally aborts active requests, calls the underlying
SGLang model runner update method, optionally flushes cache, and resumes unless
`keep_pause=true`. From-disk updates run on the scheduler thread. If active
requests are present, the update is rejected unless the request sets
`abort_all_requests=true` or generation was already paused with `mode=retract`.

`/init_weights_update_group` and `/destroy_weights_update_group` manage the
SGLang/Miles distributed update process group. `/update_weights_from_distributed`
then sends metadata (`names`, `dtypes`, `shapes`, `group_name`, and optional
`load_format` / `weight_version`) through the admin control plane while the
actual tensors move over the distributed group. The distributed update path uses
the same scheduler-thread lifecycle as disk updates: active requests must be
aborted or safely retracted, cache is flushed by default, and the visible
`weight_version` is updated after a successful runner update. If the distributed
update fails, the scheduler remains paused because SGLang may have partially
updated the model weights; recover by reloading or otherwise repairing the
worker before calling `continue_generation`.

`/update_weights_from_tensor` accepts one serialized CUDA IPC payload per TP rank
at the worker. The actual tensor bytes stay on the local GPU; the control plane
forwards handles and metadata. Each rank opens only its own handles and
synchronizes the copy before acknowledging it. A failed tensor update leaves
the scheduler paused because tensor loading is non-transactional. The Python
router retains its HTTP 501 response for this route: colocated trainers address
their corresponding worker directly rather than broadcasting local handles.
An optional `publication` manifest adds ordered bucket acknowledgements and
prevents `/continue_generation` from accepting a partial or failed publication.
Its fields and complete-retry sequence are documented in
[Colocated RL](colocated_rl.md).

The memory occupation endpoints release and restore weights, KV cache and CUDA
graph allocations when `enable_memory_saver: true` is configured. See
[Colocated RL](colocated_rl.md) for the per-stage behavior, supported graph
configuration, and the pause/update/resume sequence.

## Stage and TP Behavior

The Coordinator sends one admin operation to each target stage and waits for
stage results. For TP stages, rank 0 fans the operation out to follower ranks,
collects one result per rank, and returns a stage-level aggregate result with
`rank_results`.

Stages without an admin-capable scheduler return a successful skipped result so
mixed pipelines can broadcast model info or pause commands without failing on
pre/post-processing stages.

## Router Behavior

The external router broadcasts admin requests to every non-dead worker. Update
and pause routes temporarily disable target workers from normal request routing
while the broadcast is in flight, then restore each worker's previous disabled
state.

The router serializes pause, distributed group lifecycle, and weight-update
broadcasts with an admin update lock. If another update holds the lock for too
long, the router returns HTTP 503 instead of blocking subsequent admin callers
indefinitely. If distributed group initialization fails or times out, the target
worker remains disabled until an operator explicitly re-enables it after
recovery.

## Weight Checker

`/weights_checker` supports `snapshot`, `reset_tensors`, `compare`, and
`checksum`. The Omni checker computes strict SHA256 digests from each tensor's
name, dtype, shape, and raw bytes, then derives a per-rank checksum from the
sorted tensor digests. Full-model SHA256 checks block inference on that worker
until the digest completes.
