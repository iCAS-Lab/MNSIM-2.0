"""
PIM v1 vs PIM v2 figures from results/latency.csv and results/energy.csv.

    python iscas2026/make_figures.py

Writes results/pim_v1_vs_v2_summary.csv and, for every context length, into figures/:

  tokens_per_second/        bars + speedup (v2 / v1)
  tokens_per_joule/         bars + improvement (%)
  words_per_battery_life/   bars (BATTERY_J x tokens/J / TOKENS_PER_WORD)
  latency_distribution/     stacked % bars, one v1 and one v2 bar per model
  energy_distribution/      same for energy
  latency_pies/<l>/         v1 and v2 pies per model
  energy_pies/<l>/          same for energy
  attention_latency_v1_vs_v2.png   attention latency vs context length, per model
"""
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg

plt.rcParams['font.sans-serif'] = ['Liberation Sans', 'Arial', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'

V1_COLOR, V2_COLOR = '#457B9D', '#E63946'

LATENCY_PARTS = {
    'non_attention_computing_ns': ('Xbar + DAC + ADC', '#F4A261'),
    'non_attention_communication_ns': ('Communication', '#E9C46A'),
    'non_attention_buffer_ns': ('Buffer', '#1D3557'),
    'non_attention_digital_ns': ('Digital Circuitry', '#2A9D8F'),
    'v1_attention_ns': ('Attention: Systolic Array', '#457B9D'),
    'v2_attention_xbar_ns': ('Attention: PIM Xbar', '#E63946'),
    'v2_attention_buffer_ns': ('Attention: PIM Buffer', '#9B2226'),
}
NON_ATTN_LATENCY = ['non_attention_computing_ns', 'non_attention_communication_ns', 'non_attention_buffer_ns', 'non_attention_digital_ns']
V1_LATENCY = NON_ATTN_LATENCY + ['v1_attention_ns']
V2_LATENCY = NON_ATTN_LATENCY + ['v2_attention_xbar_ns', 'v2_attention_buffer_ns']

ENERGY_PARTS = {
    'non_attention_crossbar_nJ': ('Crossbar', '#2A9D8F'),
    'non_attention_DAC_nJ': ('DAC', '#F4A261'),
    'non_attention_ADC_nJ': ('ADC', '#E9C46A'),
    'non_attention_buffer_nJ': ('Buffer', '#A8DADC'),
    'non_attention_other_digital_nJ': ('Other Digital', '#6D597A'),
    'v1_attention_nJ': ('Attention: Systolic Array', '#457B9D'),
    'v2_attention_nJ': ('Attention: PIM', '#E63946'),
}
NON_ATTN_ENERGY = ['non_attention_crossbar_nJ', 'non_attention_DAC_nJ', 'non_attention_ADC_nJ', 'non_attention_buffer_nJ', 'non_attention_other_digital_nJ']
V1_ENERGY = NON_ATTN_ENERGY + ['v1_attention_nJ']
V2_ENERGY = NON_ATTN_ENERGY + ['v2_attention_nJ']


def load_summary():
    latency = pd.read_csv(cfg.RESULTS_DIR / 'latency.csv')
    energy = pd.read_csv(cfg.RESULTS_DIR / 'energy.csv')
    summary = latency.merge(energy, on=['model', 'display', 'context_length', 'n_layers'], validate='one_to_one')
    summary['v1_words_per_battery'] = summary['v1_tokens_per_J'] * cfg.BATTERY_J / cfg.TOKENS_PER_WORD
    summary['v2_words_per_battery'] = summary['v2_tokens_per_J'] * cfg.BATTERY_J / cfg.TOKENS_PER_WORD
    return summary


def at_context_length(summary, l):
    return summary[summary['context_length'] == l].sort_values('v2_token_latency_s').reset_index(drop=True)


def save(fig, path, dpi=300):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_attention_latency(summary):
    fig, axes = plt.subplots(2, 4, figsize=(28, 12), sharex=True)
    for ax, model in zip(axes.flat, cfg.MODELS):
        d = summary[summary['model'] == model].sort_values('context_length')
        ax.plot(d['context_length'], d['v1_attention_ns'] / 1e3, color=V1_COLOR, marker='o', linewidth=3, label='PIM v1 (attention on TPU)')
        ax.plot(d['context_length'], d['v2_attention_ns'] / 1e3, color=V2_COLOR, marker='s', linewidth=3, label='PIM v2 (attention on PIM)')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_title(cfg.MODELS[model]['display'], fontsize=25, fontweight='bold')
        ax.tick_params(axis='both', labelsize=18)
        ax.grid(True, linestyle='--', alpha=0.5)
    for ax in axes[:, 0]:
        ax.set_ylabel('Attention Latency (us)', fontsize=22, fontweight='bold')
    for ax in axes[-1]:
        ax.set_xlabel('Context Length', fontsize=22, fontweight='bold')
    axes.flat[-1].axis('off')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, fontsize=25, loc='center')
    plt.tight_layout()
    save(fig, cfg.FIGURES_DIR / 'attention_latency_v1_vs_v2.png', dpi=200)


def plot_bars(summary, v1_col, v2_col, ylabel, folder, ylim_bottom, value_fmt, ratio=None):
    """Two bars per model (PIM v1, PIM v2) for every context length, with an optional ratio line."""
    for l in cfg.CONTEXT_LENGTHS:
        df = at_context_length(summary, l)

        fig, ax1 = plt.subplots(figsize=(14, 8))
        bar_width = 0.35
        r1 = np.arange(len(df))
        r2 = r1 + bar_width

        ax1.set_axisbelow(True)
        ax1.grid(axis='y', linestyle='--', alpha=0.5, color='#CCCCCC', zorder=0)
        bars1 = ax1.bar(r1, df[v1_col], color=V1_COLOR, width=bar_width, label='PIM v1', alpha=0.9, edgecolor='black', linewidth=2, zorder=2)
        bars2 = ax1.bar(r2, df[v2_col], color=V2_COLOR, width=bar_width, label='PIM v2', alpha=0.9, edgecolor='black', linewidth=2, zorder=2)
        for bars in (bars1, bars2):
            for bar in bars:
                height = bar.get_height()
                # middle of the bar on the log axis
                ax1.text(bar.get_x() + bar.get_width() / 2, np.sqrt(height * ylim_bottom), value_fmt(height),
                         ha='center', va='center', rotation='vertical', fontsize=22, fontweight='bold', color='white')

        top = max(df[v1_col].max(), df[v2_col].max())
        # leave room above the bars for the ratio line
        ax1.set_ylim(ylim_bottom, top * (top / ylim_bottom) ** (0.6 if ratio else 0.15))
        ax1.set_ylabel(ylabel, fontweight='bold', fontsize=25)
        ax1.set_yscale('log')
        ax1.set_xticks(r1 + bar_width / 2)
        ax1.set_xticklabels(df['display'], fontsize=25, rotation=45, ha='right')
        ax1.tick_params(axis='y', labelsize=25)
        ax1.spines['top'].set_visible(False)

        if ratio:
            ratio_col, ratio_label, ratio_fmt = ratio
            ax2 = ax1.twinx()
            ax2.plot(r1 + bar_width / 2, df[ratio_col], color='black', marker='o', linewidth=2, zorder=3)
            lo, hi = df[ratio_col].min(), df[ratio_col].max()
            span = max(hi - lo, abs(hi) * 0.1, 1e-9)
            ax2.set_ylim(lo - 2.4 * span, hi + 0.35 * span)
            ticks = plt.MaxNLocator(5).tick_values(lo, hi)
            ax2.set_yticks([t for t in ticks if lo - 0.1 * span <= t <= hi + 0.35 * span and t >= 0])
            ax2.set_ylabel(ratio_label, fontweight='bold', fontsize=25)
            ax2.tick_params(axis='y', labelsize=25)
            ax2.spines['top'].set_visible(False)
            for i, v in enumerate(df[ratio_col]):
                ax2.annotate(ratio_fmt(v), (r1[i] + bar_width / 2, v), xytext=(0, 6), textcoords='offset points',
                             ha='center', va='bottom', color='black', fontweight='bold', fontsize=20)

        lines, labels = ax1.get_legend_handles_labels()
        ax1.legend(lines, labels, fontsize=25, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, facecolor='#F0F0F0', edgecolor='black')
        plt.tight_layout()
        save(fig, cfg.FIGURES_DIR / folder / f'{folder}_{l}.png')


def plot_distribution(summary, parts, v1_cols, v2_cols, folder, title):
    """Stacked percentage bars, one PIM v1 and one PIM v2 bar per model, for every context length."""
    cols = list(dict.fromkeys(v1_cols + v2_cols))
    for l in cfg.CONTEXT_LENGTHS:
        df = at_context_length(summary, l)
        rows, labels = [], []
        for _, row in df.iterrows():
            for version, vcols in (('v1', v1_cols), ('v2', v2_cols)):
                values = pd.Series({c: (row[c] if c in vcols else 0.0) for c in cols})
                rows.append(values / values.sum() * 100)
                labels.append(f"{row['display']} {version}")
        pct = pd.DataFrame(rows)

        fig, ax = plt.subplots(figsize=(16, 14))
        pct.plot(kind='barh', stacked=True, ax=ax, color=[parts[c][1] for c in cols], edgecolor='black', linewidth=1.5, width=0.8)
        for c in ax.containers:
            ax.bar_label(c, labels=[f'{v.get_width():.1f}' if v.get_width() >= 5 else '' for v in c],
                         label_type='center', color='white', fontweight='bold', fontsize=14)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=18)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel('Percentage (%)', fontsize=25, fontweight='bold')
        ax.set_title(f'{title} (Context Length: {l})', fontsize=25, fontweight='bold')
        ax.tick_params(axis='x', labelsize=20)
        ax.xaxis.grid(True, linestyle='--', alpha=0.7)
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, [parts[c][0] for c in cols], bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=20)
        plt.tight_layout()
        save(fig, cfg.FIGURES_DIR / folder / f'{folder}_{l}.png')


def plot_pies(summary, parts, v1_cols, v2_cols, folder, unit_title):
    """For each context length and model, PIM v1 and PIM v2 pies side by side."""
    cols = list(dict.fromkeys(v1_cols + v2_cols))
    for l in cfg.CONTEXT_LENGTHS:
        for _, row in at_context_length(summary, l).iterrows():
            fig, axes = plt.subplots(1, 2, figsize=(18, 9))
            for ax, (version, vcols) in zip(axes, (('PIM v1', v1_cols), ('PIM v2', v2_cols))):
                _, _, autotexts = ax.pie(
                    row[vcols].astype(float),
                    autopct=lambda pct: f'{pct:.1f}%' if pct >= 3 else '',
                    startangle=140,
                    colors=[parts[c][1] for c in vcols],
                    textprops={'fontsize': 20, 'fontweight': 'bold'},
                    wedgeprops={'edgecolor': 'black', 'linewidth': 2},
                )
                for autotext in autotexts:
                    autotext.set_color('white')
                ax.set_title(version, fontsize=25, fontweight='bold')
            handles = [plt.Rectangle((0, 0), 1, 1, facecolor=parts[c][1], edgecolor='black') for c in cols]
            fig.legend(handles, [parts[c][0] for c in cols], loc='lower center', ncol=4, fontsize=18, bbox_to_anchor=(0.5, -0.08))
            fig.suptitle(f"{row['display']} - {unit_title} (Context Length: {l})", fontsize=25, fontweight='bold', y=1.06)
            plt.tight_layout()
            save(fig, cfg.FIGURES_DIR / folder / str(l) / f"pie_{row['display'].replace(' ', '_')}.png", dpi=200)


def exp_fmt(v):
    return re.sub(r'E\+0?', 'E', f'{v:.1E}')


if __name__ == '__main__':
    summary = load_summary()
    summary.to_csv(cfg.RESULTS_DIR / 'pim_v1_vs_v2_summary.csv', index=False)

    plot_attention_latency(summary)
    plot_bars(summary, 'v1_tokens_per_s', 'v2_tokens_per_s', 'Tokens per Second', 'tokens_per_second', 1,
              value_fmt=lambda v: f'{v:.1f}',
              ratio=('speedup_v2_vs_v1', 'Speedup (v2 / v1)', lambda v: f'{v:.2f}x'))
    plot_bars(summary, 'v1_tokens_per_J', 'v2_tokens_per_J', 'Tokens per Joule', 'tokens_per_joule', 10,
              value_fmt=lambda v: f'{v:.0f}',
              ratio=('tokens_per_J_improvement_v2_vs_v1_pct', 'Improvement (%)', lambda v: f'{v:.1f}%'))
    plot_bars(summary, 'v1_words_per_battery', 'v2_words_per_battery', 'Words per Battery Life', 'words_per_battery_life', 1e5,
              value_fmt=exp_fmt)
    plot_distribution(summary, LATENCY_PARTS, V1_LATENCY, V2_LATENCY, 'latency_distribution', 'Latency Breakdown')
    plot_distribution(summary, ENERGY_PARTS, V1_ENERGY, V2_ENERGY, 'energy_distribution', 'Energy Breakdown')
    plot_pies(summary, LATENCY_PARTS, V1_LATENCY, V2_LATENCY, 'latency_pies', 'Latency')
    plot_pies(summary, ENERGY_PARTS, V1_ENERGY, V2_ENERGY, 'energy_pies', 'Energy')
    print(f'figures written to {cfg.FIGURES_DIR}')
