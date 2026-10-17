"""
Compact, IEEE-column-sized figures for the ISCAS 2026 paper (vector PDF).

    python iscas2026/make_paper_figures.py      (after extract_latency.py and extract_energy.py)

Writes into iscas2026/paper_figures/:
  throughput_energy.pdf     (a) speedup v2/v1 vs l  (b) tokens/J improvement vs l  (c) tokens/s at l=2048
  breakdown.pdf             latency breakdown at l=256 and l=8192, energy breakdown at l=8192
BitNet-8B is left out: same architecture as Llama3-8B, so the same throughput and energy.
"""
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

import config as cfg

OUT = cfg.ISCAS_DIR / 'paper_figures'
MODELS = [m for m in cfg.MODELS if m != 'bitnet_8B']
SHORT = {'bitnet_07': 'BitNet-0.7B', 'llama32_1B': 'Llama3-1B', 'gemma2': 'Gemma2-2B',
         'llama32_3B': 'Llama3-3B', 'phi3': 'Phi3-3.8B', 'llama3_8B': 'Llama3-8B'}
ORDER = ['bitnet_07', 'llama32_1B', 'gemma2', 'llama32_3B', 'phi3', 'llama3_8B']
MODEL_COLORS = dict(zip(ORDER, ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02']))
MARKERS = dict(zip(ORDER, ['o', 's', '^', 'D', 'v', 'P']))
TPU_COLOR, V1_COLOR, V2_COLOR = '#BBBBBB', '#457B9D', '#E63946'

# Times-like font: TeX Gyre Termes ships with TeX Live; falls back to whatever serif is installed
for path in glob.glob('/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-*.otf') + \
        glob.glob('/usr/share/texlive/texmf-dist/fonts/opentype/public/tex-gyre/texgyretermes-*.otf'):
    font_manager.fontManager.addfont(path)

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['TeX Gyre Termes', 'Times New Roman', 'Times', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7,
    'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6,
    'axes.linewidth': 0.6, 'lines.linewidth': 1.0, 'lines.markersize': 3,
    'pdf.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.01,
})


def load():
    latency = pd.read_csv(cfg.RESULTS_DIR / 'latency.csv')
    energy = pd.read_csv(cfg.RESULTS_DIR / 'energy.csv')
    df = latency.merge(energy, on=['model', 'display', 'context_length', 'n_layers'])
    return df[df['model'].isin(MODELS)]


def throughput_energy(df):
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.75), gridspec_kw={'width_ratios': [1, 1, 1.25]})
    ls = cfg.CONTEXT_LENGTHS

    for ax, col, ylabel, title in (
        (axes[0], 'speedup_v2_vs_v1', 'Tokens/s speedup (v2/v1)', '(a) Throughput'),
        (axes[1], 'tokens_per_J_improvement_v2_vs_v1_pct', 'Tokens/J improvement (%)', '(b) Energy efficiency'),
    ):
        for m in ORDER:
            d = df[df['model'] == m].sort_values('context_length')
            ax.plot(d['context_length'], d[col], marker=MARKERS[m], color=MODEL_COLORS[m], label=SHORT[m])
        ax.set_xscale('log', base=2)
        ax.set_xticks(ls)
        ax.set_xticklabels([str(l) for l in ls])
        ax.set_xlabel('Context length $l$')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle=':', linewidth=0.5)
    axes[0].set_yscale('log')
    axes[0].set_ylim(0.9, 20)
    axes[0].set_yticks([1, 2, 5, 10, 20])
    axes[0].set_yticklabels(['1', '2', '5', '10', '20'])
    axes[1].axhline(0, color='black', linewidth=0.6)
    axes[0].legend(ncol=2, frameon=False, loc='upper left', handlelength=1.5, columnspacing=0.8)

    ax = axes[2]
    d = df[df['context_length'] == 2048].set_index('model').loc[ORDER]
    x = np.arange(len(ORDER))
    w = 0.27
    for i, (col, label, color) in enumerate((('tpu_only_tokens_per_s', 'TPU-LLM', TPU_COLOR),
                                              ('v1_tokens_per_s', 'PIM-LLM v1', V1_COLOR),
                                              ('v2_tokens_per_s', 'PIM-LLM v2', V2_COLOR))):
        ax.bar(x + (i - 1) * w, d[col], w, label=label, color=color, edgecolor='black', linewidth=0.4)
    for xi, v in zip(x, d['v2_tokens_per_s']):
        ax.text(xi + w, v * 1.15, f'{v:.0f}', ha='center', va='bottom', fontsize=5)
    ax.set_yscale('log')
    ax.set_ylim(0.3, 1000)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in ORDER], rotation=30, ha='right')
    ax.set_ylabel('Tokens/s')
    ax.set_title('(c) Throughput at $l=2048$')
    ax.legend(ncol=3, frameon=False, loc='upper center', handlelength=1.2, columnspacing=0.8)
    ax.grid(True, axis='y', linestyle=':', linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout(w_pad=1.0)
    fig.savefig(OUT / 'throughput_energy.pdf')
    plt.close(fig)


LAT_PARTS = [
    ('non_attention_computing_ns', 'Xbar+DAC+ADC', '#F4A261', 'both'),
    ('non_attention_communication_ns', 'Communication', '#E9C46A', 'both'),
    ('non_attention_buffer_ns', 'Buffer', '#1D3557', 'both'),
    ('non_attention_digital_ns', 'Digital', '#2A9D8F', 'both'),
    ('v1_attention_ns', 'Attn: systolic array (v1)', '#8DA9C4', 'v1'),
    ('v2_attention_xbar_ns', 'Attn: PIM xbar (v2)', '#E63946', 'v2'),
    ('v2_attention_buffer_ns', 'Attn: PIM buffer (v2)', '#9B2226', 'v2'),
]
EN_PARTS = [
    ('non_attention_crossbar_nJ', 'Crossbar', '#B5838D', 'both'),
    ('non_attention_DAC_nJ', 'DAC', '#FFCDB2', 'both'),
    ('non_attention_ADC_nJ', 'ADC', '#6D6875', 'both'),
    ('non_attention_buffer_nJ', 'Buffer', '#52B788', 'both'),
    ('non_attention_other_digital_nJ', 'Other digital', '#CDB4DB', 'both'),
    ('v1_attention_nJ', 'Attn: systolic array (v1)', '#8DA9C4', 'v1'),
    ('v2_attention_nJ', 'Attn: PIM (v2)', '#E63946', 'v2'),
]


def stacked(ax, df, l, parts, title):
    d = df[df['context_length'] == l].set_index('model').loc[ORDER]
    labels, rows = [], []
    for m in ORDER:
        for v in ('v1', 'v2'):
            vals = np.array([d.loc[m, c] if who in ('both', v) else 0.0 for c, _, _, who in parts])
            rows.append(vals / vals.sum() * 100)
            labels.append(f'{SHORT[m]} {v}')
    rows = np.array(rows)
    y = np.arange(len(rows))[::-1] + np.repeat(np.arange(len(ORDER))[::-1], 2) * 0.5
    left = np.zeros(len(rows))
    handles = []
    for j, (_, name, color, _) in enumerate(parts):
        h = ax.barh(y, rows[:, j], left=left, height=0.85, color=color, edgecolor='black', linewidth=0.3, label=name)
        for yi, li, wi in zip(y, left, rows[:, j]):
            if wi >= 12:
                ax.text(li + wi / 2, yi, f'{wi:.0f}', ha='center', va='center', fontsize=5, color='white')
        left += rows[:, j]
        handles.append(h)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=5)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Share (%)')
    ax.set_title(title)
    ax.tick_params(axis='y', length=0)
    return handles


def breakdown(df):
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.2), sharey=True)
    h_lat = stacked(axes[0], df, 256, LAT_PARTS, '(a) Latency, $l=256$')
    stacked(axes[1], df, 8192, LAT_PARTS, '(b) Latency, $l=8192$')
    h_en = stacked(axes[2], df, 8192, EN_PARTS, '(c) Energy, $l=8192$')
    fig.legend(h_lat, [p[1] for p in LAT_PARTS], loc='lower center', ncol=7, frameon=False,
               bbox_to_anchor=(0.53, -0.07), handlelength=1.0, columnspacing=0.8)
    fig.legend(h_en, [p[1] for p in EN_PARTS], loc='lower center', ncol=7, frameon=False,
               bbox_to_anchor=(0.53, -0.13), handlelength=1.0, columnspacing=0.8)
    fig.text(0.075, -0.027, 'Latency (a, b):', fontsize=6, fontweight='bold', va='center')
    fig.text(0.075, -0.081, 'Energy (c):', fontsize=6, fontweight='bold', va='center')
    plt.tight_layout(w_pad=0.6)
    fig.savefig(OUT / 'breakdown.pdf')
    plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    df = load()
    throughput_energy(df)
    breakdown(df)
    print(f'paper figures written to {OUT}')
