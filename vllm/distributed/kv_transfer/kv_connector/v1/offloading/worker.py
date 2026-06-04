# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from collections import defaultdict
from dataclasses import replace

import torch
from simple_profiler import profiler

from vllm.config import get_layers_from_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMessageMetadata,
    OffloadingWorkerMetadata,
    ReqId,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
    OffloadingSpec,
)
from vllm.v1.kv_offload.worker.worker import (
    OffloadingWorker,
    TransferSpec,
)

logger = init_logger(__name__)


class OffloadingConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, spec: OffloadingSpec):
        self.spec = spec
        self.worker = OffloadingWorker()

        self.kv_connector_stats = OffloadingConnectorStats()
        # job_id -> req_id for in-flight loads.
        self._load_jobs: dict[int, ReqId] = {}
        # (job_id, transfer_spec, req_id) pending store submissions.
        self._unsubmitted_store_jobs: list[tuple[int, TransferSpec, ReqId]] = []
        self._connector_worker_meta = OffloadingWorkerMetadata()

        # Profiling: per-request trace track (req_id -> "req_N") + timings.
        self._req_profile_slot_end_ns: dict[int, int] = {}
        self._req_profile_tid: dict[ReqId, str] = {}
        self._store_queue_time_ns: dict[int, int] = {}
        self._store_job_req: dict[int, ReqId] = {}
        self._load_submit_time_ns: dict[int, int] = {}

    def _register_handlers(self, kv_caches: CanonicalKVCaches):
        for src_cls, dst_cls, handler in self.spec.get_handlers(kv_caches):
            self.worker.register_handler(src_cls, dst_cls, handler)

    def register_kv_caches(
        self, kv_caches: dict[str, torch.Tensor | list[torch.Tensor]]
    ):
        layer_names = list(kv_caches.keys())
        layers = get_layers_from_vllm_config(
            self.spec.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
            layer_names,
        )
        attn_backends = {
            layer_name: layers[layer_name].get_attn_backend()
            for layer_name in layer_names
            if layer_name in layers
        }

        num_blocks = self.spec.kv_cache_config.num_blocks

        # layer_name -> list of matching KV cache tensors
        # such that each tensor starts with the num_blocks dimension.
        # FlashAttention layers which use the (2, num_blocks, ...) layout
        # will possibly map to 2 tensors, one per K and one per V.
        # All other layers will probably map to a single tensor.
        tensors_per_block: dict[str, tuple[torch.Tensor, ...]] = {}
        # layer_name -> size of (un-padded) page in bytes
        unpadded_page_size_bytes: dict[str, int] = {}
        # layer_name -> size of page in bytes
        page_size_bytes: dict[str, int] = {}
        for kv_cache_group in self.spec.kv_cache_config.kv_cache_groups:
            group_layer_names = kv_cache_group.layer_names
            group_kv_cache_spec = kv_cache_group.kv_cache_spec
            if isinstance(group_kv_cache_spec, UniformTypeKVCacheSpecs):
                per_layer_specs = group_kv_cache_spec.kv_cache_specs
            else:
                per_layer_specs = {}
            for layer_name in group_layer_names:
                layer_kv_cache_spec = per_layer_specs.get(
                    layer_name, group_kv_cache_spec
                )
                if isinstance(layer_kv_cache_spec, AttentionSpec):
                    layer_kv_cache = kv_caches[layer_name]
                    assert isinstance(layer_kv_cache, torch.Tensor)
                    assert layer_kv_cache.storage_offset() == 0

                    # get the logical dimension for num_blocks
                    test_shape = attn_backends[layer_name].get_kv_cache_shape(
                        num_blocks=1234,
                        block_size=16,
                        num_kv_heads=1,
                        head_size=256,
                    )
                    num_blocks_logical_dim = test_shape.index(1234)

                    # sort the logical dimensions by stride (high to low)
                    # to get a physical-to-logical mapping:
                    # physical_to_logical[physical_pos] = logical_dim
                    logical_strides = layer_kv_cache.stride()
                    physical_to_logical = sorted(
                        range(len(logical_strides)),
                        key=lambda idx: logical_strides[idx],
                        reverse=True,
                    )

                    num_blocks_physical_dim = physical_to_logical.index(
                        num_blocks_logical_dim
                    )
                    if num_blocks_physical_dim == 0:
                        storage = layer_kv_cache.untyped_storage()
                        page = layer_kv_cache_spec.page_size_bytes
                        tensors_per_block[layer_name] = (
                            torch.tensor(
                                [],
                                dtype=torch.int8,
                                device=layer_kv_cache.device,
                            )
                            .set_(storage)
                            .view(num_blocks, page),
                        )
                        page_size_bytes[layer_name] = (
                            layer_kv_cache_spec.page_size_bytes
                        )
                        unpadded_page_size_bytes[layer_name] = (
                            layer_kv_cache_spec.real_page_size_bytes
                        )
                    else:
                        # Flash Attention case: (2, num_blocks, ...)
                        assert test_shape[0] == 2
                        assert physical_to_logical[0] == 0
                        assert num_blocks_physical_dim == 1

                        # unbind the tensor to separate K and V tensors
                        half_page_size = layer_kv_cache_spec.page_size_bytes // 2
                        storage = layer_kv_cache.untyped_storage()
                        raw = (
                            torch.tensor(
                                [],
                                dtype=torch.int8,
                                device=layer_kv_cache.device,
                            )
                            .set_(storage)
                            .view(2, num_blocks, half_page_size)
                        )
                        tensors_per_block[layer_name] = tuple(raw.unbind(0))

                        page_size_bytes[layer_name] = half_page_size
                        unpadded_page_size_bytes[layer_name] = (
                            layer_kv_cache_spec.real_page_size_bytes // 2
                        )

                elif isinstance(layer_kv_cache_spec, MambaSpec):
                    state_tensors = kv_caches[layer_name]
                    assert isinstance(state_tensors, list)

                    # re-construct the raw (num_blocks, page_size) tensor
                    # from the first state tensor
                    assert len(state_tensors) > 0
                    first_state_tensor = state_tensors[0]
                    assert first_state_tensor.storage_offset() == 0
                    tensor = (
                        torch.tensor(
                            [],
                            dtype=torch.int8,
                            device=first_state_tensor.device,
                        )
                        .set_(first_state_tensor.untyped_storage())
                        .view((num_blocks, layer_kv_cache_spec.page_size_bytes))
                    )
                    tensors_per_block[layer_name] = (tensor,)

                    page_size_bytes[layer_name] = layer_kv_cache_spec.page_size_bytes
                    unpadded_page_size_bytes[layer_name] = replace(
                        layer_kv_cache_spec, page_size_padded=None
                    ).page_size_bytes

                else:
                    raise NotImplementedError

        block_tensors: list[CanonicalKVCacheTensor] = []
        block_data_refs: dict[str, list[CanonicalKVCacheRef]] = defaultdict(list)
        for kv_cache_tensor in self.spec.kv_cache_config.kv_cache_tensors:
            # Filter to layers that were actually processed above.
            # _get_kv_cache_config_deepseek_v4 emits KVCacheTensor entries for
            # every (tuple_idx, page_size) slot; slots where no group has a
            # layer at that index produce an empty shared_by (reserved memory
            # with no corresponding model layer).
            tensor_layer_names = [
                n for n in kv_cache_tensor.shared_by if n in tensors_per_block
            ]
            if not tensor_layer_names:
                continue

            # verify all layers in the group reference the exact same tensors
            assert len({len(tensors_per_block[n]) for n in tensor_layer_names}) == 1
            assert (
                len({tensors_per_block[n][0].data_ptr() for n in tensor_layer_names})
                == 1
            )
            assert (
                len({tensors_per_block[n][0].stride() for n in tensor_layer_names}) == 1
            )

            # pick the first layer to represent the group
            first_layer_name = tensor_layer_names[0]
            for tensor in tensors_per_block[first_layer_name]:
                block_tensors.append(
                    CanonicalKVCacheTensor(
                        tensor=tensor,
                        page_size_bytes=page_size_bytes[first_layer_name],
                    )
                )

                curr_tensor_idx = len(block_tensors) - 1
                for layer_name in tensor_layer_names:
                    block_data_refs[layer_name].append(
                        CanonicalKVCacheRef(
                            tensor_idx=curr_tensor_idx,
                            page_size_bytes=(unpadded_page_size_bytes[layer_name]),
                        )
                    )

        group_data_refs: list[list[CanonicalKVCacheRef]] = []
        for kv_cache_group in self.spec.kv_cache_config.kv_cache_groups:
            group_refs: list[CanonicalKVCacheRef] = []
            for layer_name in kv_cache_group.layer_names:
                group_refs += block_data_refs[layer_name]
            group_data_refs.append(group_refs)

        canonical_kv_caches = CanonicalKVCaches(
            tensors=block_tensors,
            group_data_refs=group_data_refs,
        )

        self._register_handlers(canonical_kv_caches)

    def register_cross_layers_kv_cache(
        self, kv_cache: torch.Tensor, attn_backend: type[AttentionBackend]
    ):
        # verify that num_blocks is at physical position 0 in the cross-layers
        # tensor layout.
        test_shape = attn_backend.get_kv_cache_shape(
            num_blocks=1234, block_size=16, num_kv_heads=1, head_size=256
        )
        num_blocks_logical_dim = test_shape.index(1234) + 1
        physical_to_logical = attn_backend.get_kv_cache_stride_order(
            include_num_layers_dimension=True
        )
        num_blocks_physical_dim = physical_to_logical.index(num_blocks_logical_dim)
        assert num_blocks_physical_dim == 0

        kv_cache_groups = self.spec.kv_cache_config.kv_cache_groups
        assert len(kv_cache_groups) == 1
        kv_cache_spec = kv_cache_groups[0].kv_cache_spec
        num_layers = len(kv_cache_groups[0].layer_names)
        page_size_bytes = kv_cache_spec.page_size_bytes * num_layers

        assert kv_cache.storage_offset() == 0
        storage = kv_cache.untyped_storage()
        assert len(storage) % page_size_bytes == 0
        num_blocks = len(storage) // page_size_bytes
        tensor = (
            torch.tensor(
                [],
                dtype=torch.int8,
                device=kv_cache.device,
            )
            .set_(storage)
            .view(num_blocks, page_size_bytes)
        )
        kv_cache_tensor = CanonicalKVCacheTensor(
            tensor=tensor, page_size_bytes=page_size_bytes
        )
        # in cross layers layout, there's currently only a single group
        kv_cache_data_ref = CanonicalKVCacheRef(
            tensor_idx=0, page_size_bytes=page_size_bytes
        )
        canonical_kv_caches = CanonicalKVCaches(
            tensors=[kv_cache_tensor], group_data_refs=[[kv_cache_data_ref]]
        )

        self._register_handlers(canonical_kv_caches)

    def _get_or_alloc_req_tid(self, req_id: ReqId, start_ns: int) -> str:
        if req_id not in self._req_profile_tid:
            chosen = None
            for slot, end_ns in self._req_profile_slot_end_ns.items():
                if end_ns <= start_ns:
                    chosen = slot
                    break
            if chosen is None:
                chosen = len(self._req_profile_slot_end_ns)
                self._req_profile_slot_end_ns[chosen] = 0
            self._req_profile_tid[req_id] = f"req_{chosen}"
        return self._req_profile_tid[req_id]

    def _release_req_tid(self, req_id: ReqId, end_ns: int) -> None:
        tid = self._req_profile_tid.pop(req_id, None)
        if tid is None:
            return
        slot = int(tid.split("_")[1])
        self._req_profile_slot_end_ns[slot] = end_ns

    def set_req_profile_tids(self, tid_map: dict) -> None:
        self._req_profile_tid.update(tid_map)

    def _submit_store_jobs(self) -> None:
        for job_id, transfer_spec, req_id in self._unsubmitted_store_jobs:
            tid = self._req_profile_tid.get(req_id, "kv_store")
            success = self.worker.transfer_async(
                job_id, transfer_spec, profile_tid=tid, req_id=req_id
            )
            assert success
        self._unsubmitted_store_jobs.clear()

    def _submit_preloads(self, metadata: OffloadingConnectorMetadata) -> None:
        for req_id, src_spec in (metadata.reqs_to_preload or {}).items():
            preload_id = getattr(src_spec, "preload_id", None)
            if not preload_id:
                continue
            handler = self.worker.transfer_type_to_handler.get(
                (src_spec.medium(), GPULoadStoreSpec.medium())
            )
            if handler is None or not hasattr(handler, "preload_async"):
                continue
            num_blocks = len(
                getattr(
                    src_spec,
                    "block_hashes",
                    getattr(src_spec, "offload_keys", ()),
                )
            )
            start_ns = time.perf_counter_ns()
            tid = self._get_or_alloc_req_tid(req_id, start_ns)
            submitted = False
            try:
                submitted = handler.preload_async(
                    preload_id, src_spec, profile_tid=tid, req_id=req_id
                )
            except Exception:
                logger.warning(
                    "Failed to submit KV offload preload %s for request %s",
                    preload_id,
                    req_id,
                    exc_info=True,
                )
            if profiler._active:
                profiler.add_event(
                    name="offload_worker.submit_preload",
                    category="kv_offload",
                    start_ns=start_ns,
                    duration_ns=time.perf_counter_ns() - start_ns,
                    tid=tid,
                    args={
                        "req_id": req_id,
                        "preload_id": preload_id,
                        "submitted": submitted,
                        "num_blocks": num_blocks,
                    },
                )

    def handle_worker_message(self, metadata: OffloadingWorkerMessageMetadata) -> bool:
        return self.worker.handle_worker_message(metadata.message)

    def handle_preemptions(self, kv_connector_metadata: OffloadingConnectorMetadata):
        self._submit_preloads(kv_connector_metadata)
        self._submit_store_jobs()

        if kv_connector_metadata.jobs_to_flush:
            self.worker.wait(kv_connector_metadata.jobs_to_flush)

    def start_kv_transfers(self, metadata: OffloadingConnectorMetadata):
        self._submit_store_jobs()

        # Fallback for worker paths that reach start_kv_transfers without a
        # prior handle_preemptions. The storage handler de-duplicates preloads
        # by preload_id, so a repeat submission does not issue a second read.
        self._submit_preloads(metadata)

        for job_id, entry in metadata.load_jobs.items():
            req_id = entry.req_id
            self._load_jobs[job_id] = req_id
            load_start_ns = time.perf_counter_ns()
            self._load_submit_time_ns[job_id] = load_start_ns
            tid = self._get_or_alloc_req_tid(req_id, load_start_ns)
            src_spec, dst_spec = entry.transfer_spec
            success = False
            preload_id = getattr(src_spec, "preload_id", None)
            if preload_id:
                handler = self.worker.transfer_type_to_handler.get(
                    (src_spec.medium(), dst_spec.medium())
                )
                if handler is not None and hasattr(handler, "load_from_preload_async"):
                    try:
                        success = handler.load_from_preload_async(
                            job_id,
                            preload_id,
                            src_spec,
                            dst_spec,
                            profile_tid=tid,
                            req_id=req_id,
                        )
                    except Exception:
                        logger.warning(
                            "Failed to submit KV offload preload placement %s "
                            "for request %s",
                            preload_id,
                            req_id,
                            exc_info=True,
                        )
            if not success:
                success = self.worker.transfer_async(
                    job_id, entry.transfer_spec, profile_tid=tid, req_id=req_id
                )
            assert success

    def prepare_store_kv(self, metadata: OffloadingConnectorMetadata):
        for job_id, entry in metadata.store_jobs.items():
            # NOTE(orozery): defer the store to the beginning of the next
            # engine step, so that offloading starts AFTER transfers related
            # to token sampling, thereby avoiding delays to token generation.
            start_ns = time.perf_counter_ns()
            self._store_queue_time_ns[job_id] = start_ns
            self._store_job_req[job_id] = entry.req_id
            self._get_or_alloc_req_tid(entry.req_id, start_ns)
            self._unsubmitted_store_jobs.append(
                (job_id, entry.transfer_spec, entry.req_id)
            )

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """
        Returns:
            tuple of (finished_sending, finished_recving). Stores never
            emit finished_sending — the scheduler tracks store completion
            via kv_connector_worker_meta.completed_jobs and fences any
            block reuse via jobs_to_flush. Loads still emit
            finished_recving so the base scheduler can resume requests
            blocked on remote KV (and free aborted-during-load reqs).
        """
        finished_recving: set[str] = set()
        for transfer_result in self.worker.get_finished():
            # we currently do not support job failures
            job_id = transfer_result.job_id
            assert transfer_result.success
            now_ns = time.perf_counter_ns()
            if (
                transfer_result.transfer_time
                and transfer_result.transfer_size is not None
                and transfer_result.transfer_type is not None
            ):
                self.kv_connector_stats.record_transfer(
                    num_bytes=transfer_result.transfer_size,
                    time=transfer_result.transfer_time,
                    transfer_type=transfer_result.transfer_type,
                )

            self._connector_worker_meta.mark_completed(job_id)
            req_id = self._load_jobs.pop(job_id, None)
            if req_id is not None:
                finished_recving.add(req_id)
                submit_time_ns = self._load_submit_time_ns.pop(job_id, None)
                if submit_time_ns is not None and profiler._active:
                    profiler.add_event(
                        name=f"load_e2e(job={job_id})",
                        category="kv_offload",
                        start_ns=submit_time_ns,
                        duration_ns=now_ns - submit_time_ns,
                        tid=self._req_profile_tid.get(req_id, "kv_transfer"),
                        args={"req_id": req_id},
                    )
                self._release_req_tid(req_id, now_ns)
            else:
                store_req_id = self._store_job_req.pop(job_id, None)
                queue_time_ns = self._store_queue_time_ns.pop(job_id, None)
                if store_req_id is not None:
                    if queue_time_ns is not None and profiler._active:
                        profiler.add_event(
                            name=f"save_e2e(job={job_id})",
                            category="kv_offload",
                            start_ns=queue_time_ns,
                            duration_ns=now_ns - queue_time_ns,
                            tid=self._req_profile_tid.get(store_req_id, "kv_store"),
                            args={"req_id": store_req_id},
                        )
                    self._release_req_tid(store_req_id, now_ns)

        return set(), finished_recving

    def build_connector_worker_meta(self) -> OffloadingWorkerMetadata | None:
        """Return completed transfer job IDs since the last call."""
        if not self._connector_worker_meta.completed_jobs:
            return None
        meta = self._connector_worker_meta
        self._connector_worker_meta = OffloadingWorkerMetadata()
        return meta

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        """
        Get the KV transfer stats for the connector.
        """

        if self.kv_connector_stats.is_empty():
            return None
        # Clear stats for next iteration
        kv_connector_stats = self.kv_connector_stats
        self.kv_connector_stats = OffloadingConnectorStats()
        return kv_connector_stats

    def shutdown(self) -> None:
        self._unsubmitted_store_jobs.clear()
        self._load_jobs.clear()
        self._store_queue_time_ns.clear()
        self._store_job_req.clear()
        self._load_submit_time_ns.clear()
        self._req_profile_tid.clear()
        self._connector_worker_meta = OffloadingWorkerMetadata()
        self.worker.shutdown()
