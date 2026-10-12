# Colocated reinforcement learning

Omni accepts SGLang CUDA IPC weight buckets at a worker's
`POST /update_weights_from_tensor` endpoint. The trainer and worker must run on
the same host with access to the same physical GPU. Each tensor-parallel rank
receives its own serialized handle in `serialized_named_tensors`; these are
trusted administrative inputs using SGLang's multiprocessing serializer.

The endpoint forwards `load_format`, `weight_version`, `flush_cache`,
`abort_all_requests`, and `keep_pause` through the ordinary stage admin path.
Use `load_format: flattened_bucket` for SGLang flattened buckets. Target the AR
stage explicitly with `stages`. Each rank opens only its own handles, loads the
weights in place, and synchronizes the device before acknowledging the request,
so the producer can then release that bucket. Failed updates leave generation
paused. A multi-bucket publication should use `keep_pause: true` for every bucket
and call `/continue_generation` only after all buckets succeed.

Set `enable_memory_saver: true` on the pipeline to enable memory release. Engine
stages use SGLang's memory saver for weights, KV cache, and CUDA graphs. Other GPU
stages construct their models and persistent state in separate allocator regions
with CPU backups. The launcher installs the memory saver's subprocess allocator
hook before starting these workers. Memory offload cannot be combined with
`weight_share: on`.

`POST /release_memory_occupation` and `POST /resume_memory_occupation` accept
`tags: [weights, kv_cache, cuda_graph]` and optional `stages`. Omitting tags selects
all tags; omitting stages targets the pipeline. Memory release requires an idle
pipeline and blocks new requests. Restoring only weights keeps generation blocked
until KV and graph memory are restored. Non-AR stage allocations follow the
`weights` tag. Repeated release/resume calls for an already completed transition
are harmless. Each stage returns its remaining `offload_tags`.

A colocated iteration is:

1. Finish generation and drain the full pipeline.
2. Release rollout memory, then restore the learner and run its optimizer step.
3. Offload the learner, then restore rollout weights.
4. Publish tensor buckets while generation remains paused.
5. Commit the new weight version and restore KV and graph memory.
6. Generate the next rollout.

Prism's AR decode/prefill graphs retain their existing SGLang lifecycle. Its
separate vocoder uses eager execution when memory saving is enabled; explicitly
enabling vocoder CUDA graphs with memory saving is rejected. Its reference encoder
and vocoder weights are backed up to CPU during offload.
