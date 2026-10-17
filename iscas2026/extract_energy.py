"""
Extract PIM v1 and PIM v2 energy from the MNSIM reports.

MNSIM reports the buffer read/write energy only for the whole network, so the attention layer is
isolated with two extra runs per model:

  {model}_layer0.txt            -NN transformers_layer0     dummy conv only
  {model}_attention_{l}.txt     -NN transformers_attention  dummy conv + attention layer (l x d/h)

  attention X = X(conv + attention run) - X(conv-only run)     for hardware, read and write buffer

PIM v2 attention energy per transformer layer (nJ):

  non_buffer = h x (hardware - (read buffer + write buffer))
  new_buffer = h x (2 x read buffer + 10 x write buffer)
  attention  = non_buffer + new_buffer                      (every head computes and accesses the buffers)

Non-attention energy (both designs) = per-layer MNSIM energy of Q, K, V, O, FFN up, FFN down.
Its component split (crossbar, DAC, ADC, buffer, other digital) = full run - (conv + attention) run.

PIM v1 attention energy = TPU attention latency x TPU_POWER_W (per transformer layer).

Per token: x n_layers, tokens/J = 1 / energy per token.

Output: results/energy.csv, one row per (model, context length).

    python iscas2026/extract_energy.py
"""
import math

import pandas as pd

import config as cfg
from mnsim_parser import ENERGY_COMPONENTS, parse_energy_totals, parse_layers, split_transformer_layers


def isolated_attention(name, l):
    """Hardware, read and write buffer energy (nJ) of the attention layer alone."""
    layer0 = parse_energy_totals(cfg.ISOLATION_RUNS_DIR / f'{name}_layer0.txt')
    both = parse_energy_totals(cfg.ISOLATION_RUNS_DIR / f'{name}_attention_{l}.txt')
    return {key: both[key] - layer0[key] for key in ('hardware', 'buf_read', 'buf_write')}, both


def pim_v2_attention_energy(iso, heads):
    """(non_buffer, new_buffer) of the PIM v2 attention energy in nJ."""
    n = heads if cfg.ATTN_ENERGY_PER_HEAD else 1
    non_buffer = n * (iso['hardware'] - (iso['buf_read'] + iso['buf_write']))
    new_buffer = n * (cfg.BUFFER_READ_ENERGY_FACTOR * iso['buf_read'] + cfg.BUFFER_WRITE_ENERGY_FACTOR * iso['buf_write'])
    return non_buffer, new_buffer


def extract(latency):
    """latency: the DataFrame from extract_latency (for the TPU attention latency)."""
    tpu_attention_ns = latency.set_index(['model', 'context_length'])['v1_attention_ns']
    rows = []
    for name, m in cfg.MODELS.items():
        for l in cfg.CONTEXT_LENGTHS:
            full_report = cfg.FULL_RUNS_DIR / f'{name}_{l}.txt'
            attn, non_attn = split_transformer_layers(parse_layers(full_report), cfg.ATTN_FC_INDEX)
            iso, conv_plus_attn = isolated_attention(name, l)

            # the isolated attention layer must be the same layer as in the full run
            if not math.isclose(iso['hardware'], attn['energy'], rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError(f'{name} l={l}: isolated attention {iso["hardware"]} nJ != full run {attn["energy"]} nJ')

            # non-attention energy and its components
            non_attention = sum(x['energy'] for x in non_attn)
            full = parse_energy_totals(full_report)
            components = {k: full[k] - conv_plus_attn[k] for k in ENERGY_COMPONENTS}
            if not math.isclose(sum(components.values()), non_attention, rel_tol=1e-6):
                raise ValueError(f'{name} l={l}: energy components do not add up to the non-attention energy')

            non_buffer, new_buffer = pim_v2_attention_energy(iso, m['heads'])
            v1_attention = tpu_attention_ns[(name, l)] * 1e-9 * cfg.TPU_POWER_W * 1e9  # nJ per transformer layer

            row = {
                'model': name,
                'display': m['display'],
                'context_length': l,
                'n_layers': m['n_layers'],
                # non-attention PIM energy (shared by v1 and v2), nJ per transformer layer
                'non_attention_nJ': non_attention,
                **{f'non_attention_{k}_nJ': v for k, v in components.items()},
                # raw values of the isolated attention layer
                'mnsim_attention_hardware_nJ': iso['hardware'],
                'mnsim_attention_buffer_read_nJ': iso['buf_read'],
                'mnsim_attention_buffer_write_nJ': iso['buf_write'],
                # PIM v2 attention
                'v2_attention_non_buffer_nJ': non_buffer,
                'v2_attention_new_buffer_nJ': new_buffer,
                'v2_attention_nJ': non_buffer + new_buffer,
                # PIM v1 attention (TPU)
                'v1_attention_nJ': v1_attention,
            }

            n = m['n_layers']
            v1_attention_per_token = v1_attention * (1 if cfg.OLD_ATTN_ENERGY_AS_PUBLISHED else n)
            row['v1_token_energy_J'] = (non_attention * n + v1_attention_per_token) * 1e-9
            row['v2_token_energy_J'] = (non_attention + row['v2_attention_nJ']) * n * 1e-9
            row['tpu_only_token_energy_J'] = latency.set_index(['model', 'context_length']).loc[(name, l), 'tpu_only_ns'] * 1e-9 * cfg.TPU_POWER_W * n
            for design in ('v1', 'v2', 'tpu_only'):
                row[f'{design}_tokens_per_J'] = 1 / row[f'{design}_token_energy_J']
            row['tokens_per_J_improvement_v2_vs_v1_pct'] = (row['v2_tokens_per_J'] / row['v1_tokens_per_J'] - 1) * 100
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    import extract_latency

    cfg.RESULTS_DIR.mkdir(exist_ok=True)
    df = extract(extract_latency.extract())
    out = cfg.RESULTS_DIR / 'energy.csv'
    df.to_csv(out, index=False)
    print(f'wrote {out}')
    print(df.pivot(index='display', columns='context_length', values='tokens_per_J_improvement_v2_vs_v1_pct').round(1).to_string())
