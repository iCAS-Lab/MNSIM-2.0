"""
Paper figures in the style of the PIM-LLM paper (same colors, fonts, bar labels and ratio lines),
comparing PIM-LLM (v1, attention on TPU) with PIM-LLM v2 (attention on PIM).

    python iscas2026/make_paper_figures.py      (after extract_latency.py and extract_energy.py)

Writes into iscas2026/paper_figures/, one PNG per context length l:
  tokens_per_second_{l}.png      bars + speedup line            (PIM-LLM Fig. 8 style)
  tokens_per_joule_{l}.png       bars + improvement (%) line    (PIM-LLM Fig. 10 style)
  words_per_battery_{l}.png      bars                           (PIM-LLM Fig. 11 style)
  latency_distribution_{l}.png   stacked % bars + zoom inset    (PIM-LLM Fig. 9 style)
BitNet-8B is left out, as in PIM-LLM: same architecture as Llama3-8B, so the same throughput and energy.
"""
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import config as cfg

OUT = cfg.ISCAS_DIR / 'paper_figures'
# model order and names as in the PIM-LLM paper
ORDER = ['bitnet_07', 'gemma2', 'llama32_1B', 'llama32_3B', 'phi3', 'llama3_8B']
NAMES = {'bitnet_07': 'BitNet - 0.7B', 'gemma2': 'Gemma2 - 2B', 'llama32_1B': 'Llama3 - 1B',
         'llama32_3B': 'Llama3 - 3B', 'phi3': 'Phi3 - 3.8B', 'llama3_8B': 'Llama3 - 8B'}
V1, V2 = 'PIM-LLM', 'PIM-LLM v2'

plt.rcParams['font.sans-serif'] = ['Liberation Sans', 'Arial', 'Nimbus Sans', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'


def load():
    latency = pd.read_csv(cfg.RESULTS_DIR / 'latency.csv')
    energy = pd.read_csv(cfg.RESULTS_DIR / 'energy.csv')
    df = latency.merge(energy, on=['model', 'display', 'context_length', 'n_layers'])
    df['v1_words'] = df['v1_tokens_per_J'] * cfg.BATTERY_J / cfg.TOKENS_PER_WORD
    df['v2_words'] = df['v2_tokens_per_J'] * cfg.BATTERY_J / cfg.TOKENS_PER_WORD
    return df


def at(df, l):
    return df[df['context_length'] == l].set_index('model').loc[ORDER].reset_index()


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, dpi=300, bbox_inches='tight')
    plt.close(fig)


def tokens_per_second(df, l):
    d = at(df, l)
    d['Speedup'] = d['v2_tokens_per_s'] / d['v1_tokens_per_s']

    fig, ax1 = plt.subplots(figsize=(14, 8))
    bar_width = 0.35
    r1 = np.arange(len(d))
    r2 = r1 + bar_width
    bars1 = ax1.bar(r1, d['v1_tokens_per_s'], color='#A9C0E5', width=bar_width, label=V1, alpha=0.9, edgecolor='black', linewidth=2)
    bars2 = ax1.bar(r2, d['v2_tokens_per_s'], color='#4169E1', width=bar_width, label=V2, alpha=0.9, edgecolor='black', linewidth=2)
    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, height / 2, f'{height:.2f}',
                     ha='center', va='center', fontsize=25, fontweight='bold', color='black', rotation='vertical')

    ax1.set_ylabel('Tokens per Second', fontweight='bold', fontsize=25)
    ax1.set_yscale('log')
    ax1.set_ylim(bottom=0.1)
    ax1.set_xticks(r1 + bar_width / 2)
    ax1.set_xticklabels(d['model'].map(NAMES), fontsize=25, rotation=45, ha='right')
    ax1.tick_params(axis='y', labelsize=25)

    ax2 = ax1.twinx()
    ax2.plot(r1 + bar_width / 2, d['Speedup'], color='red', marker='o', linewidth=2, label='Speedup')
    ax2.set_ylabel('Speedup', color='red', fontweight='bold', fontsize=25)
    ax2.tick_params(axis='y', labelcolor='red', labelsize=25)
    ax2.set_ylim(bottom=1)
    for i, s in enumerate(d['Speedup']):
        ax2.annotate(f'{s:.2f}x', (r1[i] + bar_width / 2, s), xytext=(0, 5), textcoords='offset points',
                     ha='center', va='bottom', color='red', fontweight='bold', fontsize=25)

    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, fontsize=25, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, facecolor='#F0F0F0', edgecolor='black')
    ax1.grid(axis='y', linestyle='--', alpha=0.5, color='#CCCCCC')
    ax2.grid(False)
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)
    plt.tight_layout()
    save(fig, f'tokens_per_second_{l}.png')


def tokens_per_joule(df, l):
    d = at(df, l)
    d['Improvement (%)'] = d['tokens_per_J_improvement_v2_vs_v1_pct']

    fig, ax1 = plt.subplots(figsize=(14, 8))
    bar_width = 0.35
    r1 = np.arange(len(d))
    r2 = r1 + bar_width
    ax1.set_axisbelow(True)
    ax1.grid(axis='y', linestyle='--', alpha=0.5, color='#CCCCCC', zorder=0)
    bars1 = ax1.bar(r1, d['v1_tokens_per_J'], color='#00CED1', width=bar_width, label=V1, alpha=0.9, edgecolor='black', linewidth=2, zorder=2)
    bars2 = ax1.bar(r2, d['v2_tokens_per_J'], color='#4682B4', width=bar_width, label=V2, alpha=0.9, edgecolor='black', linewidth=2, zorder=2)
    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, height / 2, f'{height:.0f}',
                     ha='center', va='center', rotation='vertical', fontsize=25, fontweight='bold', color='black')

    ax1.set_ylabel('Tokens per Joule', fontweight='bold', fontsize=25)
    ax1.set_yscale('log')
    ax1.set_xticks(r1 + bar_width / 2)
    ax1.set_xticklabels(d['model'].map(NAMES), fontsize=25, rotation=45, ha='right')
    ax1.set_ylim(bottom=1)
    ax1.tick_params(axis='y', labelsize=25)

    ax2 = ax1.twinx()
    imp = d['Improvement (%)']
    ax2.plot(r2, imp, color='red', marker='o', linestyle='-', linewidth=2, markersize=8, zorder=3)
    ax2.set_ylabel('Improvement (%)', color='red', fontweight='bold', fontsize=25)
    ax2.tick_params(axis='y', labelcolor='red', labelsize=25)
    ax2.axhline(0, color='red', linewidth=1, linestyle=':')
    span = max(imp.max() - imp.min(), 1.0)
    ax2.set_ylim(imp.min() - 0.15 * span, imp.max() + 0.25 * span)
    ax2.set_axisbelow(True)
    ax2.grid(axis='y', alpha=0.5, color='#CCCCCC')
    for i, v in enumerate(imp):
        ax2.text(r2[i], v, f'{v:.2f}%', color='red', fontweight='bold', ha='center', va='bottom', fontsize=25)

    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, fontsize=25, loc='upper center', facecolor='#F0F0F0', edgecolor='none', bbox_to_anchor=(0.5, 1.15), ncol=3)
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.set_facecolor('#F8F8F8')
    plt.tight_layout()
    save(fig, f'tokens_per_joule_{l}.png')


def words_per_battery(df, l):
    d = at(df, l)
    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax1.set_facecolor('#F8F8F8')
    bar_width = 0.35
    r1 = np.arange(len(d))
    r2 = r1 + bar_width
    bars1 = ax1.bar(r1, d['v1_words'], color='#3C763D', width=bar_width, label=V1, alpha=0.9, edgecolor='black', linewidth=2, zorder=2)
    bars2 = ax1.bar(r2, d['v2_words'], color='#36454F', width=bar_width, label=V2, alpha=0.9, edgecolor='black', linewidth=2, zorder=2)
    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, height / 3, re.sub(r'E\+0?', 'E', f'{height:.1E}'),
                     ha='center', va='center', rotation='vertical', fontsize=30, fontweight='bold', color='white')
    ax1.set_ylabel('Words per Battery Life', fontweight='bold', fontsize=25)
    ax1.set_yscale('log')
    ax1.set_xticks(r1 + bar_width / 2)
    ax1.set_xticklabels(d['model'].map(NAMES), fontsize=25, rotation=45, ha='right')
    ax1.set_ylim(bottom=1e4)
    ax1.tick_params(axis='y', labelsize=25)
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, fontsize=25, loc='upper center', facecolor='#F0F0F0', edgecolor='none', bbox_to_anchor=(0.5, 1.15), ncol=3)
    ax1.spines['top'].set_visible(False)
    plt.tight_layout()
    save(fig, f'words_per_battery_{l}.png')


# latency components, colors as in PIM-LLM Fig. 9 plus the two PIM attention parts of v2
LAT_COLORS = {
    'non_attention_computing_ns': ('Xbar + DAC + ADC', '#F4A261'),
    'non_attention_communication_ns': ('Communication', '#E9C46A'),
    'non_attention_buffer_ns': ('Buffer', '#1D3557'),
    'non_attention_digital_ns': ('Digital Circuitry', '#2A9D8F'),
    'v1_attention_ns': ('Systolic Array', '#457B9D'),
    'v2_attention_xbar_ns': ('Attention Xbar', '#E63946'),
    'v2_attention_buffer_ns': ('Attention Buffer', '#9B2226'),
}
NON_ATTN = ['non_attention_computing_ns', 'non_attention_communication_ns', 'non_attention_buffer_ns', 'non_attention_digital_ns']


def latency_distribution(df, l):
    d = at(df, l)
    cols = list(LAT_COLORS)
    rows, labels = [], []
    for _, row in d.iterrows():
        for tag, attn in (('v1', ['v1_attention_ns']), ('v2', ['v2_attention_xbar_ns', 'v2_attention_buffer_ns'])):
            used = NON_ATTN + attn
            values = pd.Series({c: (row[c] if c in used else 0.0) for c in cols})
            rows.append(values / values.sum() * 100)
            labels.append(f"{NAMES[row['model']]} ({tag})")
    pct = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, 9))
    pct.plot(kind='barh', stacked=True, ax=ax, color=[LAT_COLORS[c][1] for c in cols], edgecolor='black', linewidth=2, width=0.8)
    ax.set_xlabel('Percentage (%)', fontsize=25, fontweight='bold')
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim(0, 100)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], [LAT_COLORS[c][0] for c in cols][::-1], bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=25)
    for c in ax.containers:
        ax.bar_label(c, labels=[f'{v.get_width():.1f}' if v.get_width() >= 3 else '' for v in c],
                     label_type='center', color='white', fontweight='bold', fontsize=18)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=20)
    ax.tick_params(axis='x', labelsize=25)
    ax.invert_yaxis()

    axins = inset_axes(ax, width='30%', height='30%', loc='lower center', bbox_to_anchor=(0.85, -0.03, 1, 1), bbox_transform=ax.transAxes)
    pct.plot(kind='barh', stacked=True, ax=axins, color=[LAT_COLORS[c][1] for c in cols], legend=False, edgecolor='black', linewidth=1)
    axins.set_xlim(0, 1)
    axins.set_yticks([])
    axins.invert_yaxis()
    axins.set_xlabel('Percentage (%)', fontsize=25, fontweight='bold')
    axins.tick_params(axis='x', labelsize=25)
    axins.set_title('Zoom (0-1%)', fontsize=25, fontweight='bold')
    save(fig, f'latency_distribution_{l}.png')


if __name__ == '__main__':
    df = load()
    for l in cfg.CONTEXT_LENGTHS:
        tokens_per_second(df, l)
        tokens_per_joule(df, l)
        words_per_battery(df, l)
        latency_distribution(df, l)
    print(f'paper figures written to {OUT}')
