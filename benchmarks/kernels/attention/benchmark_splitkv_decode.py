# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Time BF16 paged attention for Qwen3.5 TP2 small-query shapes on one GPU."""

import argparse
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.vllm_flash_attn import flash_attn_varlen_func


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=544)
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[4096, 32768, 131072, 256000]
    )
    parser.add_argument("--queries", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--segments", type=int, nargs="+", default=[16, 32, 64, 128])
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_default_device("cuda")
    torch.manual_seed(42)
    raw_config = args.model_config.read_bytes()
    config = json.loads(raw_config)
    config = config.get("text_config", config)
    heads = config["num_attention_heads"] // args.tp
    kv_heads = config["num_key_value_heads"] // args.tp
    dim, block_size = config["head_dim"], args.block_size
    rows = []
    metadata = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "q_heads_per_rank": heads,
        "kv_heads_per_rank": kv_heads,
        "head_dim": dim,
        "dtype": "bfloat16",
        "block_size": block_size,
        "kv_layout": "packed [blocks, kv_heads, block_size, 2 * head_dim]",
        "tp": args.tp,
        "model_config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "timer": "FlashInfer CUPTI, CUDA Graph, cold L2",
        "atol": 0.005,
        "rtol": 0.03,
        "scope": "One rank, attention kernels only; not end-to-end TP latency",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_diff": subprocess.check_output(["git", "diff", "HEAD"], text=True),
    }
    for length in args.lengths:
        blocks = (length + block_size - 1) // block_size
        cache = torch.randn(blocks, kv_heads, block_size, 2 * dim, dtype=torch.bfloat16)
        k, v = cache.transpose(1, 2).split(dim, dim=-1)
        table = torch.arange(blocks, dtype=torch.int32).unsqueeze(0)
        kv_lens = torch.tensor([length], dtype=torch.int32)
        for queries in args.queries:
            q = torch.randn(queries, heads, dim, dtype=torch.bfloat16)
            out = torch.empty_like(q)
            cu_q = torch.tensor([0, queries], dtype=torch.int32)
            common = {
                "q": q,
                "k": k,
                "v": v,
                "out": out,
                "cu_seqlens_q": cu_q,
                "seqused_k": kv_lens,
                "max_seqlen_q": queries,
                "max_seqlen_k": length,
                "softmax_scale": dim**-0.5,
                "softcap": 0,
                "q_descale": None,
                "k_descale": None,
                "v_descale": None,
                "causal": True,
                "block_table": table,
            }
            unified_attention(**common, window_size=(-1, -1))
            reference = out.clone()
            methods = [("fa2", None), ("triton-2d", 0)] + [
                (f"split-{n}", n) for n in args.segments
            ]
            if (length + queries) % 2:
                methods.reverse()
            for method, segments in methods:
                if segments is None:

                    def run(common=common):
                        flash_attn_varlen_func(**common, fa_version=2)

                else:
                    scratch = {}
                    if segments:
                        scratch = {
                            "seq_threshold_3D": queries,
                            "num_par_softmax_segments": segments,
                            "softmax_segm_output": torch.empty(
                                queries, heads, segments, dim, dtype=torch.float32
                            ),
                            "softmax_segm_max": torch.empty(
                                queries, heads, segments, dtype=torch.float32
                            ),
                            "softmax_segm_expsum": torch.empty(
                                queries, heads, segments, dtype=torch.float32
                            ),
                        }

                    def run(common=common, scratch=scratch):
                        unified_attention(**common, **scratch, window_size=(-1, -1))

                run()
                torch.accelerator.synchronize()
                torch.testing.assert_close(out, reference, atol=0.005, rtol=0.03)
                error = (out.float() - reference.float()).abs().max().item()
                times = bench_gpu_time_with_cupti(
                    run,
                    use_cuda_graph=True,
                    cold_l2_cache=True,
                    dry_run_time_ms=25,
                    repeat_time_ms=100,
                )
                ms = statistics.median(times)
                # Best-case traffic: K/V each read once, plus query/output tensors.
                moved = 2 * length * kv_heads * dim * 2 + 2 * queries * heads * dim * 2
                row = {
                    "context": length,
                    "query_tokens": queries,
                    "method": method,
                    "median_us": ms * 1000,
                    "estimated_gbps": moved / (ms * 1e6),
                    "tflops": 4 * queries * heads * length * dim / (ms * 1e9),
                    "max_abs_error_vs_2d": error,
                    "samples_ms": times,
                }
                rows.append(row)
                args.output.write_text(
                    json.dumps({"metadata": metadata, "rows": rows}, indent=2) + "\n"
                )
                print(
                    json.dumps(
                        {key: val for key, val in row.items() if key != "samples_ms"}
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
