<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
  <img alt="Views" src="https://lambda.348575.xyz/repo-view-counter?repo=vllm"/>
</p>

<p align="center">
  A fork of <a href="https://github.com/vllm-project/vllm">vllm-project/vllm</a> with support for KV cache preloading, break-even gating for KV cache ops, profiling.
</p>

## Changes

The stock vLLM `OffloadingConnector` only stores and loads KV on demand. Preload and break-even gating need the scheduler and connector to do things the API does not expose.

### Preload lookahead

The goal is to let the storage backend start reading a waiting request's KV *before* that request is scheduled, so its load is already in CPU memory when it runs, and will only require a much faster CPU->GPU copy.

1. At each scheduling step the scheduler sends the next N requests them to the offloading connector (`_notify_preload_candidates` / `_get_preload_candidate_requests` in the v1 scheduler).
2. The offloading connector then calculates if the request has a stored prefix (and how much), and forwards it to the vLLM worker as a preload hint using the `reqs_to_preload` field.
3. The vLLM worker then forwards the hint to the external kvcache calling `preload_async(preload_id, ...)` to start the speculative read. When the actual load for the request happens i.e. when the scheduler starts executing it, the worker calls `load_from_preload_async(...)` passing `preload_id`.

The connector does not check whether the blocks exist before starting a preload, this is up to external KV cache to check. The external KV cache has complete control and cancel or continue with the preload.

### Break-even gating

Through testing, a clear break-even point exists for kv caching, when the cost of loading is lower than re-computing the prefix. This break-even point is setup specific (GPU, LLM model, SSD). The vLLM scheduler uses the break-even data to determine when a load from either CPU DRAM or disk is worth it. If the prefix size is less than the break-even, then the load & store ops are declined. 

`scripts/pareto_measure.py` in [t348575/kvcache-experiments](https://github.com/t348575/kvcache-experiments) can be used to generate a pareto plot, and the break-even point for your setup.

`scripts/emit_break_even.py` can be used to generate the json break even data for vLLM to use.

### Profiling

Various portions have been profiled using [t348575/simple-profiler](https://github.com/t348575/simple-profiler/) for performance investigation.