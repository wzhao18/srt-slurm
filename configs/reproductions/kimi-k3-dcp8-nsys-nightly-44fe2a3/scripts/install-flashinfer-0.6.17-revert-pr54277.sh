#!/bin/bash

set -euo pipefail

/configs/reproductions/kimi-k3-dcp8-nsys-nightly-44fe2a3/scripts/install-flashinfer-0.6.17.sh

site_packages="/usr/local/lib/python3.12/dist-packages"
test -f "${site_packages}/vllm/v1/worker/gpu/spec_decode/speculator.py"
command -v patch >/dev/null

patch --batch --forward --directory "${site_packages}" --strip 1 <<'PATCH'
diff --git a/vllm/v1/attention/backends/mla/flashinfer_mla.py b/vllm/v1/attention/backends/mla/flashinfer_mla.py
--- a/vllm/v1/attention/backends/mla/flashinfer_mla.py
+++ b/vllm/v1/attention/backends/mla/flashinfer_mla.py
@@ -120,7 +120,6 @@ class FlashInferMLAMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
     query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM
     # Non-causal DSpark blocks are flattened to single-token rows in forward_mqa.
     supports_non_causal_multi_token_decode: ClassVar[bool] = True
-    supports_non_causal_multi_token_dcp: ClassVar[bool] = True
 
     def __init__(
         self,
@@ -289,8 +288,8 @@ class FlashInferMLAImpl(MLACommonImpl[MLACommonMetadata]):
         seq_lens = attn_metadata.decode.seq_lens
 
         if not attn_metadata.causal:
-            # FlashInfer decode has no causal flag. Flatten each non-causal
-            # query block into independent single-token rows.
+            # FlashInfer decode has no causal flag. For TP-only DSpark, flatten
+            # each non-causal query block into independent single-token rows.
             query_len = attn_metadata.num_decode_tokens // attn_metadata.num_decodes
             q = q.unsqueeze(1)
             if query_len > 1:
@@ -327,10 +326,6 @@ class FlashInferMLAImpl(MLACommonImpl[MLACommonMetadata]):
             assert self.cp_kv_cache_interleave_size == 1
             causal_seqlens_kv_global = attn_metadata.decode.dcp_tot_seq_lens
             assert causal_seqlens_kv_global is not None
-            if not attn_metadata.causal and query_len > 1:
-                causal_seqlens_kv_global = causal_seqlens_kv_global.repeat_interleave(
-                    query_len
-                )
             extra_kwargs.update(
                 enable_dcp=True,
                 cp_world=self.dcp_world_size,
diff --git a/vllm/v1/kv_cache_interface.py b/vllm/v1/kv_cache_interface.py
--- a/vllm/v1/kv_cache_interface.py
+++ b/vllm/v1/kv_cache_interface.py
@@ -570,11 +570,6 @@ class MLAAttentionSpec(FullAttentionSpec):
             "All attention layers in the same KV cache group must use the same "
             "quantization method, tokens per state, and model version."
         )
-        non_causal_mtd_set = {spec.non_causal_multi_token_decode for spec in specs}
-        assert len(non_causal_mtd_set) == 1, (
-            "All attention layers in the same KV cache group must agree on "
-            "non_causal_multi_token_decode."
-        )
         merged_spec = cls(
             block_size=specs[0].block_size,
             num_kv_heads=specs[0].num_kv_heads,
@@ -587,7 +582,9 @@ class MLAAttentionSpec(FullAttentionSpec):
             cache_dtype_str=cache_dtype_str_set.pop(),
             tokens_per_state=tokens_per_state_set.pop(),
             model_version=model_version_set.pop(),
-            non_causal_multi_token_decode=non_causal_mtd_set.pop(),
+            non_causal_multi_token_decode=any(
+                spec.non_causal_multi_token_decode for spec in specs
+            ),
         )
         for spec in specs:
             for f in fields(AttentionSpec):
diff --git a/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py b/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py
--- a/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py
+++ b/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py
@@ -18,7 +18,7 @@ from vllm.v1.attention.backends.utils import PAD_SLOT_ID
 from vllm.v1.kv_cache_interface import KVCacheConfig
 from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
 from vllm.v1.worker.gpu.block_table import BlockTables
-from vllm.v1.worker.gpu.cp_utils import cp_local_slot
+from vllm.v1.worker.gpu.cp_utils import cp_local_slot, prepare_dcp_local_seq_lens
 from vllm.v1.worker.gpu.dp_utils import DPSyncState, dispatch_cg_and_sync_dp
 from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
 from vllm.v1.worker.gpu.model_states.interface import ModelState
@@ -291,6 +291,16 @@ class DFlashSpeculator(DraftModelSpeculator):
         if not self.draft_attn_layer_names:
             return None
         assert num_query_per_req is None  # Omitted for DFlash, read from self instead
+        if dcp_local_seq_lens is None and self.block_tables.cp_size > 1:
+            prepare_dcp_local_seq_lens(
+                self.input_buffers.dcp_local_seq_lens,
+                self.input_buffers.seq_lens,
+                num_reqs,
+                self.block_tables.cp_size,
+                self.block_tables.cp_rank,
+                self.block_tables.cp_interleave,
+            )
+            dcp_local_seq_lens = self.input_buffers.dcp_local_seq_lens
         return super()._build_draft_attn_metadata(
             num_reqs,
             num_reqs_padded,
diff --git a/vllm/v1/worker/gpu/spec_decode/speculator.py b/vllm/v1/worker/gpu/spec_decode/speculator.py
--- a/vllm/v1/worker/gpu/spec_decode/speculator.py
+++ b/vllm/v1/worker/gpu/spec_decode/speculator.py
@@ -21,7 +21,6 @@ from vllm.v1.worker.gpu.attn_utils import (
     init_attn_backend,
 )
 from vllm.v1.worker.gpu.block_table import BlockTables
-from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
 from vllm.v1.worker.gpu.dp_utils import DPSyncState
 from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
 from vllm.v1.worker.gpu.model_states.interface import ModelState
@@ -293,18 +292,6 @@ class DraftModelSpeculator(BaseSpeculator):
             out=draft_seq_lens_cpu_upper_bound[:num_reqs],
         )
         draft_seq_lens_cpu_upper_bound[:num_reqs].clamp_(max=self.max_model_len)
-        if dcp_local_seq_lens is None and self.block_tables.cp_size > 1:
-            # Draft steps advance and rewind their own global sequence lengths,
-            # so the target model's DCP-local lengths may already be stale.
-            prepare_dcp_local_seq_lens(
-                self.input_buffers.dcp_local_seq_lens,
-                self.input_buffers.seq_lens,
-                num_reqs,
-                self.block_tables.cp_size,
-                self.block_tables.cp_rank,
-                self.block_tables.cp_interleave,
-            )
-            dcp_local_seq_lens = self.input_buffers.dcp_local_seq_lens
         attn_metadata = build_attn_metadata(
             attn_groups=self.attn_groups,
             num_reqs=num_reqs_padded,
PATCH

find "${site_packages}/vllm/v1" -type d -name __pycache__ -prune -exec rm -rf {} +

if grep -q "supports_non_causal_multi_token_dcp: ClassVar\[bool\] = True" \
    "${site_packages}/vllm/v1/attention/backends/mla/flashinfer_mla.py"; then
    echo "PR #54277 reverse patch did not apply" >&2
    exit 1
fi
echo "vLLM PR #54277 reverted in the job-local container overlay"

# Do not let one node enter the distributed rendezvous while another node is
# still downloading and installing its runtime overlay.
ready_dir="/logs/.runtime-setup/pr54277-revert-ready"
node_name="${SLURMD_NODENAME:-${HOSTNAME:-local}}"
expected_nodes="${SLURM_JOB_NUM_NODES:-1}"
mkdir -p "${ready_dir}"
touch "${ready_dir}/${node_name}"

deadline=$((SECONDS + 1800))
while (( $(find "${ready_dir}" -maxdepth 1 -type f | wc -l) < expected_nodes )); do
    if (( SECONDS >= deadline )); then
        echo "Timed out waiting for ${expected_nodes} patched runtime nodes" >&2
        exit 1
    fi
    sleep 2
done

echo "All ${expected_nodes} nodes completed the patched runtime setup"
