"""
Extract PIM v1 and PIM v2 latency from the MNSIM reports and edge_os.csv.

Per transformer layer (ns):

  non-attention (both designs)   max(Q, K, V) + O + FFN up + FFN down       MNSIM 'total latency'
  PIM v1 attention               Total_Attention_Cycles x TPU_CYCLE_NS      edge_os.csv (TPU)
  PIM v2 attention               2 x xbar(l x d/h) + 2 x h x (buffer read + 10 x buffer write)
                                 values of the attention layer in the MNSIM report; heads compute in
                                 parallel, but each head reads/writes the buffers, hence x h

Per token: latency x n_layers, tokens/s = 1 / (n_layers x latency).

Output: results/latency.csv, one row per (model, context length), with the raw MNSIM attention
values next to the derived ones so every number can be traced back to a report.

    python iscas2026/extract_latency.py
"""
import pandas as pd

import config as cfg
from mnsim_parser import parse_layers, split_transformer_layers, qkv_parallel_sum


def pim_v2_attention_latency(attn, heads):
    """(xbar part, buffer part) of the PIM v2 attention latency in ns.

    The heads compute in parallel (xbar counted once), but each head reads and writes the buffers.
    """
    xbar = cfg.ATTN_MATMULS * attn[cfg.ATTN_COMPUTE_FIELD]
    buffer_heads = heads if cfg.BUFFER_PER_HEAD else 1
    buffer = cfg.ATTN_MATMULS * buffer_heads * (attn['buf_read'] + cfg.BUFFER_WRITE_FACTOR * attn['buf_write'])
    return xbar, buffer


def load_edge_os():
    edge_os = pd.read_csv(cfg.EDGE_OS_CSV)
    return {(row.Model, row.Context_Length): row for row in edge_os.itertuples()}


def extract():
    edge_os = load_edge_os()
    rows = []
    for name, m in cfg.MODELS.items():
        for l in cfg.CONTEXT_LENGTHS:
            report = cfg.FULL_RUNS_DIR / f'{name}_{l}.txt'
            attn, non_attn = split_transformer_layers(parse_layers(report), cfg.ATTN_FC_INDEX)
            tpu = edge_os[(m['edge_os'], l)]

            attn_xbar, attn_buffer = pim_v2_attention_latency(attn, m['heads'])
            row = {
                'model': name,
                'display': m['display'],
                'context_length': l,
                'n_layers': m['n_layers'],
                'heads': m['heads'],
                'mnsim_report': report.name,
                # non-attention PIM latency (shared by v1 and v2), ns per transformer layer
                'non_attention_ns': qkv_parallel_sum(non_attn, 'total'),
                'non_attention_computing_ns': qkv_parallel_sum(non_attn, 'computing'),
                'non_attention_buffer_ns': qkv_parallel_sum(non_attn, 'buffer'),
                'non_attention_digital_ns': qkv_parallel_sum(non_attn, 'digital'),
                'non_attention_communication_ns': qkv_parallel_sum(non_attn, 'intra') + qkv_parallel_sum(non_attn, 'inter'),
                # raw MNSIM values of the attention layer (one l x d/h matmul)
                'mnsim_attention_computing_ns': attn['computing'],
                'mnsim_attention_buffer_read_ns': attn['buf_read'],
                'mnsim_attention_buffer_write_ns': attn['buf_write'],
                'mnsim_attention_total_ns': attn['total'],
                # PIM v2 attention
                'v2_attention_xbar_ns': attn_xbar,
                'v2_attention_buffer_ns': attn_buffer,
                'v2_attention_ns': attn_xbar + attn_buffer,
                # PIM v1 attention (TPU)
                'tpu_attention_cycles': int(tpu.Total_Attention_Cycles),
                'v1_attention_ns': int(tpu.Total_Attention_Cycles) * cfg.TPU_CYCLE_NS,
                # TPU-only baseline, for reference
                'tpu_only_ns': int(tpu.Total_Compute_Cycles) * cfg.TPU_CYCLE_NS,
            }
            row['v1_layer_ns'] = row['non_attention_ns'] + row['v1_attention_ns']
            row['v2_layer_ns'] = row['non_attention_ns'] + row['v2_attention_ns']
            for design in ('v1', 'v2'):
                row[f'{design}_token_latency_s'] = row[f'{design}_layer_ns'] * m['n_layers'] * 1e-9
                row[f'{design}_tokens_per_s'] = 1 / row[f'{design}_token_latency_s']
            row['tpu_only_tokens_per_s'] = 1 / (row['tpu_only_ns'] * m['n_layers'] * 1e-9)
            row['speedup_v2_vs_v1'] = row['v2_tokens_per_s'] / row['v1_tokens_per_s']
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    cfg.RESULTS_DIR.mkdir(exist_ok=True)
    df = extract()
    out = cfg.RESULTS_DIR / 'latency.csv'
    df.to_csv(out, index=False)
    print(f'wrote {out}')
    print(df.pivot(index='display', columns='context_length', values='speedup_v2_vs_v1').round(2).to_string())
