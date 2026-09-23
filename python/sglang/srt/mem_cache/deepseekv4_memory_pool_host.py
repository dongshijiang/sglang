"""
Host (L2) KV cache pool for DeepSeek-V4 hierarchical cache (HiCache).

Index semantics
---------------
All indices exchanged with HiCacheController / HiRadixCache stay in the
**full token domain** (same as radix-tree node.value / node.host_value,
length-aligned to the radix page size), so the radix-tree bookkeeping logic
is unchanged. Internally the pool converts them to the compressed domains of
the V4 sub pools:

    c4 slot    = full_index // 4
    c128 slot  = full_index // 128
    indexer page = full_index // 256   (one page = 64 c4 slots)

Allocation granularity is one 256-full-token block, which is exactly
one full-domain allocator page == 64 c4 slots == 2 c128 slots == one indexer
page. This guarantees that every allocated segment is "group complete" in all
compressed domains, so no host slot is ever shared between two radix nodes.

Layouts
-------
- c4/c128 host buffers: linear, 584 bytes per slot (576B value + 8B scale),
  matching the CPU layout expected by hisparse_transfer kernels.
- indexer host buffers: page-layout mirror of the device pool
  (num_pages, page_bytes), transferred page-wise.

Phase 2 additionally mirrors the SWA KV pool and every compress-state pool,
page-granular in the host full-token domain like c128/indexer: the
shadow-slot identity mapping (swa_loc == full_loc) aligns swa pages with
full-domain pages, and compress-state rows are grouped ring_size-per-swa-page.

NUMA/CXL
--------
Memory is allocated through the shared HostKVCache allocator machinery
(--hicache-numa-node -> libnuma numa_alloc_onnode), followed by
cudaHostRegister pinning, so the pool can be placed on a CXL memory expander
exposed as a NUMA node.
"""

from __future__ import annotations

import logging
import threading
from functools import wraps
from typing import Optional

import psutil
import torch

from sglang.jit_kernel.deepseek_v4 import (
    hisparse_load_to_device,
    hisparse_offload_to_host,
)
from sglang.srt.mem_cache.memory_pool_host import (
    ALLOC_MEMORY_FUNCS,
    get_allocator_from_storage,
)
from sglang.srt.utils import get_bool_env_var, is_npu, is_xpu

if not (is_npu() or is_xpu()):
    from sgl_kernel.kvcacheio import (
        transfer_kv_all_layer_mla,
        transfer_kv_per_layer_mla,
    )

logger = logging.getLogger(__name__)

# Item layout shared by the c4/c128 pools: 576B value + 8B scale (see
# sgl_kernel/deepseek_v4/kvcacheio.cuh, kCPUItemBytes).
DSV4_KV_ITEM_BYTES = 584
# Host allocation granularity, in full-token domain.
DSV4_BLOCK_TOKENS = 256


def _synchronized(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return func(self, *args, **kwargs)

    return wrapper


class DeepSeekV4TokenToKVPoolHost:
    """Host-side KV pool mirroring the c4/c128/indexer sub pools of a
    DeepSeekV4TokenToKVPool. Duck-types the HostKVCache interface consumed by
    HiCacheController (alloc/free/backup/load/layout/page_size/...)."""

    def __init__(
        self,
        device_pool,
        full_size: int,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str = "layer_first",
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
        numa_node: Optional[int] = None,
    ):
        assert layout == "layer_first", (
            f"DeepSeek-V4 host pool only supports layer_first layout, got {layout!r}"
        )
        assert page_size == DSV4_BLOCK_TOKENS, (
            f"DeepSeek-V4 host pool requires page_size={DSV4_BLOCK_TOKENS}, "
            f"got {page_size}"
        )
        assert device_pool.c4_kv_pool.store_dtype == torch.uint8, (
            "DeepSeek-V4 host pool expects packed uint8 kv buffers"
        )

        self.device_pool = device_pool
        self.page_size = page_size
        self.layout = layout
        self.pin_memory = pin_memory
        self.device = device
        self.dtype = torch.uint8
        self.allocator = get_allocator_from_storage(allocator_type, numa_node)

        c4_pool = device_pool.c4_kv_pool
        c128_pool = device_pool.c128_kv_pool
        indexer_pool = device_pool.c4_indexer_kv_pool

        self.c4_layer_num = c4_pool.layer_num
        self.c128_layer_num = c128_pool.layer_num
        self.c4_slots_per_block = DSV4_BLOCK_TOKENS // 4
        self.c128_slots_per_block = DSV4_BLOCK_TOKENS // 128
        self.indexer_page_bytes = indexer_pool.index_k_with_scale_buffer[
            0
        ].shape[-1]
        # NOTE: the c128 device pool pages hold only 2 slots (1728B/page with
        # 576B alignment), so it CANNOT go through hisparse_transfer kernels
        # (they hard-code the 64-slot c4 page layout). The c128 host mirror is
        # therefore page-layout (like the indexer) and transferred page-wise.
        self.c128_page_bytes = c128_pool.bytes_per_page_padded

        # Host bytes per 256-full-token block across all mirrored sub pools.
        # (c4 / c128 / indexer mirrors scale with the host token capacity.)
        self.bytes_per_block = (
            DSV4_KV_ITEM_BYTES * self.c4_slots_per_block * self.c4_layer_num
            + self.c128_page_bytes * self.c128_layer_num
            + self.indexer_page_bytes * self.c4_layer_num
        )

        # Phase 2: SWA KV + compress states are mirrored too (they were
        # skipped in phase 1, which made restored prefixes lossy: local
        # needle retrieval survived on full-attention data while global
        # summarization died on the missing SWA window / compress states).
        # Both mirrors are page-granular in the host full-token domain, like
        # c128/indexer — NOT 1:1 device images (device swa slots are recycled
        # between backup and restore, so device-domain mirrors would go
        # stale):
        #   - SWA KV: with the shadow-slot identity mapping swa_loc ==
        #     full_loc, the device swa page of a 256-token block is exactly
        #     full_index // 256 (dev_pages); the host mirror is indexed by
        #     host_index // 256 (host_pages), which the radix tree keeps
        #     stable across evict/restore cycles.
        #   - compress states: state rows are grouped ring_size-per-swa-page
        #     (state_loc = swa_page * ring_size + swa_loc % ring_size), so
        #     one contiguous group of ring_size rows = one page's states.
        self.swa_layer_num = device_pool.swa_kv_pool.layer_num
        self.swa_page_bytes = device_pool.swa_kv_pool.kv_buffer[0].shape[1]
        self._state_mirrors_plan = []  # (pool, group_bytes)
        for pool in [
            *device_pool.compress_state_pools,
            *device_pool.indexer_compress_state_pools,
        ]:
            if pool is None:
                continue
            buf = pool.kv_score_buffer.kv_score
            row_bytes = buf.shape[1] * buf.element_size()
            self._state_mirrors_plan.append((pool, row_bytes * pool.ring_size))

        # Capacity in the full-token domain.
        if host_size > 0:
            host_bytes = int(host_size * 1e9)
            self.num_blocks = host_bytes // self.bytes_per_block
        else:
            full_size_target = int(full_size * host_to_device_ratio)
            self.num_blocks = full_size_target // DSV4_BLOCK_TOKENS
        assert self.num_blocks > 0, (
            "DeepSeek-V4 host pool is too small: increase --hicache-ratio or "
            "--hicache-size"
        )
        self.size = self.num_blocks * DSV4_BLOCK_TOKENS

        # Phase 2 mirror cost scales with host capacity (page-granular).
        self.swa_total_bytes = (
            self.num_blocks * self.swa_page_bytes * self.swa_layer_num
        )
        self.state_total_bytes = sum(
            self.num_blocks * group_bytes
            for _, group_bytes in self._state_mirrors_plan
        )

        assert self.size > full_size, (
            "The host memory should be larger than the device memory with the "
            "current protocol"
        )

        # Verify there is enough available host memory (keep 10GB free).
        host_mem = psutil.virtual_memory()
        requested_bytes = (
            self.num_blocks * self.bytes_per_block
            + self.swa_total_bytes
            + self.state_total_bytes
        )
        available_bytes = host_mem.available - 10 * (1024**3)
        if requested_bytes > available_bytes:
            raise ValueError(
                f"DeepSeek-V4 host pool requests {requested_bytes / 1e9:.2f} GB "
                f"but only {available_bytes / 1e9:.2f} GB host memory is "
                f"available (10 GB reserved)."
            )

        self._alloc_buffers()
        self.lock = threading.RLock()
        self.clear()

        # Debug probe (SGLANG_P2_VERIFY=1): byte-compare device vs host
        # windows right after every kernel copy, for the first few backups
        # and for all restores. Splits "transfer is unfaithful" from "data
        # semantics are wrong" when T3 similarity is below 1.0.
        self.verify = get_bool_env_var("SGLANG_P2_VERIFY")
        self._verify_backup_budget = 2
        self.verify_failures = 0

        logger.info(
            "DeepSeekV4TokenToKVPoolHost init: blocks=%d (size=%d full tokens), "
            "c4_slots=%d, c128_slots=%d, indexer_pages=%d, bytes/block=%d, "
            "phase2: swa_layers=%d swa_total=%.2fGB state_total=%.2fGB, "
            "allocator=%s, numa_node=%s",
            self.num_blocks,
            self.size,
            self.num_blocks * self.c4_slots_per_block,
            self.num_blocks * self.c128_slots_per_block,
            self.num_blocks,
            self.bytes_per_block,
            self.swa_layer_num,
            self.swa_total_bytes / 2**30,
            self.state_total_bytes / 2**30,
            type(self.allocator).__name__,
            numa_node,
        )

    def _alloc_buffers(self):
        alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
        dev = self.device_pool.device

        def _alloc(dims):
            return alloc_func(
                dims,
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )

        # Linear layout mirrors of the paged c4 device pool (584B per slot,
        # matching the CPU layout expected by hisparse_transfer kernels).
        self.c4_kv_buffer = [
            _alloc((self.num_blocks * self.c4_slots_per_block, DSV4_KV_ITEM_BYTES))
            for _ in range(self.c4_layer_num)
        ]
        # Page-layout mirrors of the c128 / indexer device pools.
        self.c128_kv_buffer = [
            _alloc((self.num_blocks, self.c128_page_bytes))
            for _ in range(self.c128_layer_num)
        ]
        self.indexer_buffer = [
            _alloc((self.num_blocks, self.indexer_page_bytes))
            for _ in range(self.c4_layer_num)
        ]

        # Phase 2: SWA KV page mirror (same page domain as c128) and one
        # page-group image per compress-state pool, both indexed by the
        # tree-bound host page.
        self.swa_kv_buffer = [
            _alloc((self.num_blocks, self.swa_page_bytes))
            for _ in range(self.swa_layer_num)
        ]
        self.state_mirrors = []  # (pool, host_tensor, group_bytes)
        self._state_mirror_map = {}  # id(pool) -> (host_tensor, group_bytes)
        for pool, group_bytes in self._state_mirrors_plan:
            host = _alloc((self.num_blocks, group_bytes))
            self.state_mirrors.append((pool, host, group_bytes))
            self._state_mirror_map[id(pool)] = (host, group_bytes)

        def _ptrs(tensors):
            return torch.tensor(
                [t.data_ptr() for t in tensors], dtype=torch.uint64, device=dev
            )

        self.c4_host_ptrs = _ptrs(self.c4_kv_buffer)
        self.c4_device_ptrs = _ptrs(self.device_pool.c4_kv_pool.kv_buffer)
        self.c128_host_ptrs = _ptrs(self.c128_kv_buffer)
        self.c128_device_ptrs = _ptrs(self.device_pool.c128_kv_pool.kv_buffer)
        self.indexer_host_ptrs = _ptrs(self.indexer_buffer)
        self.indexer_device_ptrs = _ptrs(
            self.device_pool.c4_indexer_kv_pool.index_k_with_scale_buffer
        )
        self.swa_host_ptrs = _ptrs(self.swa_kv_buffer)
        self.swa_device_ptrs = _ptrs(self.device_pool.swa_kv_pool.kv_buffer)

    # ------------------------------------------------------------------
    # Slot management (block granularity, full-token domain interface)
    # ------------------------------------------------------------------

    @_synchronized
    def clear(self):
        self.free_blocks = torch.arange(self.num_blocks, dtype=torch.int64)
        # Memo of the last index-split result (load path reuses it per layer).
        self._split_cache_key = None
        self._split_cache_val = None

    @_synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        """Allocate `need_size` full-token domain indices (page aligned).

        Returns a tensor of full-domain host indices whose length equals
        `need_size`, expandable to complete 4/128/256-token groups.
        """
        assert need_size % self.page_size == 0, (
            "The requested size should be a multiple of the page size."
        )
        num_blocks = need_size // DSV4_BLOCK_TOKENS
        if num_blocks > len(self.free_blocks):
            return None
        blocks = self.free_blocks[:num_blocks]
        self.free_blocks = self.free_blocks[num_blocks:]
        # Expand block ids to full-domain indices: block b -> [b*256, b*256+256)
        offsets = torch.arange(DSV4_BLOCK_TOKENS, dtype=torch.int64)
        return (
            blocks.view(-1, 1) * DSV4_BLOCK_TOKENS + offsets.view(1, -1)
        ).reshape(-1)

    @_synchronized
    def free(self, full_indices: torch.Tensor) -> int:
        """Return the blocks covered by `full_indices` (full-token domain)."""
        if full_indices.numel() == 0:
            return 0
        blocks = torch.unique(full_indices.cpu() // DSV4_BLOCK_TOKENS)
        self.free_blocks = torch.cat([self.free_blocks, blocks])
        return len(full_indices)

    def available_size(self) -> int:
        return len(self.free_blocks) * DSV4_BLOCK_TOKENS

    # ------------------------------------------------------------------
    # Index conversion (full-token domain -> compressed domains)
    # ------------------------------------------------------------------

    def _split_indices(self, host_indices, device_indices):
        """Convert page-aligned full-domain index pairs to per-subpool slots.

        Both inputs are full-token domain tensors of equal length, aligned to
        a 256-token boundary at every segment start (guaranteed by the radix
        page semantics), so strided sampling yields one representative per
        c4 group / c128 group / indexer page.

        NOTE: deliberately NOT memoized — the cache controller's index
        tensors are freed and reallocated between evictions/restores, so a
        (data_ptr, numel) key can collide across different requests and
        would silently apply a previous request's slot mapping.
        """
        dev_c4 = device_indices[0::4] // 4
        host_c4 = host_indices[0::4] // 4
        dev_c128 = device_indices[0::128] // 128
        host_c128 = host_indices[0::128] // 128
        dev_pages = device_indices[0::256] // 256
        host_pages = host_indices[0::256] // 256
        return dev_c4, host_c4, dev_c128, host_c128, dev_pages, host_pages

    def _split_indices_cached(self, host_indices, device_indices):
        return self._split_indices(host_indices, device_indices)

    def _verify_windows(self, tag, dev_flat, host_flat, dev_pages, host_pages, item):
        """Debug probe: byte-compare the copied windows on both sides."""
        if dev_pages.numel() == 0:
            return
        dev_off = dev_pages.to(torch.int64) * item
        host_off = (host_pages.to(torch.int64) * item).cpu()
        d = dev_flat[dev_off[:, None] + torch.arange(item, device=dev_flat.device)]
        h = host_flat[host_off[:, None] + torch.arange(item)]
        if not torch.equal(d.cpu(), h):
            bad = (d.cpu() != h).any(dim=1).nonzero().flatten()
            logger.error(
                "P2-VERIFY %s MISMATCH: %d/%d windows differ (first dev_page=%d)",
                tag,
                bad.numel(),
                dev_pages.numel(),
                dev_pages[bad[0]].item() if bad.numel() else -1,
            )
            self.verify_failures += 1

    def _verify_backup(self, device_pool, dev_pages, host_pages):
        """Debug probe: verify all page-domain mirrors after a D2H backup."""
        if not (self.verify and self._verify_backup_budget > 0):
            return
        self._verify_backup_budget -= 1
        if self.c128_layer_num > 0:
            for l in range(self.c128_layer_num):
                self._verify_windows(
                    f"c128-backup-l{l}",
                    device_pool.c128_kv_pool.kv_buffer[l].reshape(-1),
                    self.c128_kv_buffer[l].reshape(-1),
                    dev_pages,
                    host_pages,
                    self.c128_page_bytes,
                )
        for l in range(self.swa_layer_num):
            self._verify_windows(
                f"swa-backup-l{l}",
                device_pool.swa_kv_pool.kv_buffer[l].reshape(-1),
                self.swa_kv_buffer[l].reshape(-1),
                dev_pages,
                host_pages,
                self.swa_page_bytes,
            )
        for pool, host_buf, group_bytes in self.state_mirrors:
            self._verify_windows(
                f"state-backup-{id(pool)}",
                pool.kv_score_buffer.kv_score.view(torch.uint8).reshape(-1),
                host_buf.reshape(-1),
                dev_pages,
                host_pages,
                group_bytes,
            )
        logger.info(
            "P2-VERIFY backup done (failures=%d, budget left=%d)",
            self.verify_failures,
            self._verify_backup_budget,
        )

    # ------------------------------------------------------------------
    # Transfers
    # ------------------------------------------------------------------

    def backup_from_device_all_layer(self, device_pool, host_indices, device_indices, io_backend):
        """D2H: back up c4/c128/indexer KV of all layers (one shot)."""
        if io_backend != "kernel":
            raise ValueError(
                "DeepSeek-V4 host pool only supports the 'kernel' io backend, "
                f"got {io_backend!r}"
            )
        dev_c4, host_c4, _, _, dev_pages, host_pages = (
            self._split_indices_cached(host_indices, device_indices)
        )
        if dev_c4.numel() > 0:
            hisparse_offload_to_host(
                gpu_ptrs=self.c4_device_ptrs,
                cpu_ptrs=self.c4_host_ptrs,
                gpu_indices=dev_c4,
                cpu_indices=host_c4,
            )
        if dev_pages.numel() > 0:
            # c128: one 256-token block == one c128 device page.
            if self.c128_layer_num > 0:
                transfer_kv_all_layer_mla(
                    src_layers=self.c128_device_ptrs,
                    dst_layers=self.c128_host_ptrs,
                    src_indices=dev_pages,
                    dst_indices=host_pages,
                    item_size=self.c128_page_bytes,
                    num_layers=self.c128_layer_num,
                )
            # indexer: one block == one indexer page (c4 layers only).
            transfer_kv_all_layer_mla(
                src_layers=self.indexer_device_ptrs,
                dst_layers=self.indexer_host_ptrs,
                src_indices=dev_pages,
                dst_indices=host_pages,
                item_size=self.indexer_page_bytes,
                num_layers=self.c4_layer_num,
            )
        if self.swa_layer_num > 0 and dev_pages.numel() > 0:
            # SWA KV: the shadow-slot identity mapping (swa_loc == full_loc)
            # makes the device swa page of a block exactly dev_pages, so this
            # mirrors the c128 transfer one-for-one.
            transfer_kv_all_layer_mla(
                src_layers=self.swa_device_ptrs,
                dst_layers=self.swa_host_ptrs,
                src_indices=dev_pages,
                dst_indices=host_pages,
                item_size=self.swa_page_bytes,
                num_layers=self.swa_layer_num,
            )
        self._backup_states(dev_pages, host_pages)
        self._verify_backup(device_pool, dev_pages, host_pages)

    def _restore_swa_layer(
        self, device_pool, host_indices, device_indices, swa_layer_id
    ):
        """H2D: restore one SWA layer's KV (page-granular).

        The device swa page of a 256-token block is full_index // 256 under
        the shadow-slot identity mapping, so this mirrors the c128 restore.
        """
        if self.swa_layer_num == 0:
            return
        _, _, _, _, dev_pages, host_pages = self._split_indices_cached(
            host_indices, device_indices
        )
        if dev_pages.numel() == 0:
            return
        transfer_kv_per_layer_mla(
            src=self.swa_kv_buffer[swa_layer_id],
            dst=device_pool.swa_kv_pool.kv_buffer[swa_layer_id],
            src_indices=host_pages,
            dst_indices=dev_pages,
            item_size=self.swa_page_bytes,
        )
        if self.verify:
            self._verify_windows(
                f"swa-restore-l{swa_layer_id}",
                device_pool.swa_kv_pool.kv_buffer[swa_layer_id].reshape(-1),
                self.swa_kv_buffer[swa_layer_id].reshape(-1),
                dev_pages,
                host_pages,
                self.swa_page_bytes,
            )

    def _backup_states(self, dev_pages, host_pages):
        """D2H: back up per-page compress-state groups of all layers.

        State rows are grouped ring_size-per-swa-page, so one contiguous
        group of ring_size rows travels per 256-token block. The device side
        is indexed by the block's device page, the host mirror by its
        tree-bound host page.
        """
        if dev_pages.numel() == 0:
            return
        for pool, host_buf, group_bytes in self.state_mirrors:
            transfer_kv_per_layer_mla(
                src=pool.kv_score_buffer.kv_score.view(torch.uint8),
                dst=host_buf,
                src_indices=dev_pages,
                dst_indices=host_pages,
                item_size=group_bytes,
            )

    def _restore_state_pool(self, pool, host_indices, device_indices):
        """H2D: restore ONE layer's compress-state pool (per-layer pools)."""
        if pool is None or id(pool) not in self._state_mirror_map:
            return
        host_buf, group_bytes = self._state_mirror_map[id(pool)]
        _, _, _, _, dev_pages, host_pages = self._split_indices_cached(
            host_indices, device_indices
        )
        if dev_pages.numel() == 0:
            return
        transfer_kv_per_layer_mla(
            src=host_buf,
            dst=pool.kv_score_buffer.kv_score.view(torch.uint8),
            src_indices=host_pages,
            dst_indices=dev_pages,
            item_size=group_bytes,
        )
        if self.verify:
            self._verify_windows(
                f"state-restore-{id(pool)}",
                pool.kv_score_buffer.kv_score.view(torch.uint8).reshape(-1),
                host_buf.reshape(-1),
                dev_pages,
                host_pages,
                group_bytes,
            )

    def load_to_device_per_layer(self, device_pool, host_indices, device_indices, layer_id, io_backend):
        """H2D: restore one model layer (by its layer_mapping entry)."""
        if io_backend != "kernel":
            raise ValueError(
                "DeepSeek-V4 host pool only supports the 'kernel' io backend, "
                f"got {io_backend!r}"
            )
        # `layer_id` is the PP-rank-local layer index (HiCacheController loops
        # over range(mem_pool_device.layer_num)), while `layer_mapping` is
        # indexed by the GLOBAL layer id (the device-pool getters receive
        # layer.layer_id). Translate local -> global.
        compress_ratio, compress_layer_id, _ = device_pool.layer_mapping[
            layer_id + device_pool.start_layer
        ]
        if compress_ratio == 0:
            # SWA-only layer (phase 2): restore the SWA KV window.
            self._restore_swa_layer(
                device_pool, host_indices, device_indices, compress_layer_id
            )
            return

        if compress_ratio == 4:
            dev_c4, host_c4, _, _, dev_pages, host_pages = (
                self._split_indices_cached(host_indices, device_indices)
            )
            if dev_c4.numel() > 0:
                hisparse_load_to_device(
                    gpu_ptrs=self.c4_device_ptrs[
                        compress_layer_id : compress_layer_id + 1
                    ],
                    cpu_ptrs=self.c4_host_ptrs[
                        compress_layer_id : compress_layer_id + 1
                    ],
                    gpu_indices=dev_c4,
                    cpu_indices=host_c4,
                )
            if dev_pages.numel() > 0:
                transfer_kv_per_layer_mla(
                    src=self.indexer_buffer[compress_layer_id],
                    dst=device_pool.c4_indexer_kv_pool.index_k_with_scale_buffer[
                        compress_layer_id
                    ],
                    src_indices=host_pages,
                    dst_indices=dev_pages,
                    item_size=self.indexer_page_bytes,
                )
                if self.verify:
                    self._verify_windows(
                        f"indexer-restore-l{compress_layer_id}",
                        device_pool.c4_indexer_kv_pool.index_k_with_scale_buffer[
                            compress_layer_id
                        ].reshape(-1),
                        self.indexer_buffer[compress_layer_id].reshape(-1),
                        dev_pages,
                        host_pages,
                        self.indexer_page_bytes,
                    )
            global_layer_id = layer_id + device_pool.start_layer
            # Per-layer state pools: this layer's own c4 state + indexer state.
            self._restore_state_pool(
                device_pool.compress_state_pools[global_layer_id],
                host_indices,
                device_indices,
            )
            self._restore_state_pool(
                device_pool.indexer_compress_state_pools[global_layer_id],
                host_indices,
                device_indices,
            )
        elif compress_ratio == 128:
            _, _, _, _, dev_pages, host_pages = self._split_indices_cached(
                host_indices, device_indices
            )
            if dev_pages.numel() > 0:
                transfer_kv_per_layer_mla(
                    src=self.c128_kv_buffer[compress_layer_id],
                    dst=device_pool.c128_kv_pool.kv_buffer[compress_layer_id],
                    src_indices=host_pages,
                    dst_indices=dev_pages,
                    item_size=self.c128_page_bytes,
                )
                if self.verify:
                    self._verify_windows(
                        f"c128-restore-l{compress_layer_id}",
                        device_pool.c128_kv_pool.kv_buffer[compress_layer_id].reshape(
                            -1
                        ),
                        self.c128_kv_buffer[compress_layer_id].reshape(-1),
                        dev_pages,
                        host_pages,
                        self.c128_page_bytes,
                    )
            global_layer_id = layer_id + device_pool.start_layer
            self._restore_state_pool(
                device_pool.compress_state_pools[global_layer_id],
                host_indices,
                device_indices,
            )
        else:
            raise ValueError(
                f"Unsupported compression ratio: {compress_ratio} "
                f"(layer {layer_id})"
            )
