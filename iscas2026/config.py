"""
Shared configuration for the PIM v1 vs PIM v2 experiments.

Every script in this folder imports from here, so models, context lengths and the
formula constants are defined in one place.

Run as a script to print the MNSIM commands used by run_simulations.sh:
    python iscas2026/config.py
"""
from pathlib import Path

# ----------------------------------------------------------------------------- paths
ISCAS_DIR = Path(__file__).resolve().parent
REPO_DIR = ISCAS_DIR.parent

EDGE_OS_CSV = ISCAS_DIR / 'data' / 'edge_os.csv'               # ScaleSim TPU results (PIM v1 attention)
FULL_RUNS_DIR = ISCAS_DIR / 'simulations' / 'full'              # {model}_{l}.txt
ISOLATION_RUNS_DIR = ISCAS_DIR / 'simulations' / 'isolation'    # {model}_layer0.txt, {model}_attention_{l}.txt
RESULTS_DIR = ISCAS_DIR / 'results'
FIGURES_DIR = ISCAS_DIR / 'figures'

# ----------------------------------------------------------------------------- models
# d: embedding dimension, dff: feed-forward dimension, heads: attention heads (published configs),
# n_layers: transformer layers, edge_os: model name in edge_os.csv
MODELS = {
    'bitnet_07':  {'d': 1536, 'dff': 4096,  'heads': 24, 'n_layers': 24, 'edge_os': 'BITNET-0.7B', 'display': 'BitNet - 0.7B'},
    'gemma2':     {'d': 2048, 'dff': 16384, 'heads': 8,  'n_layers': 18, 'edge_os': 'GEMMA-2B',    'display': 'Gemma2 - 2B'},
    'llama32_1B': {'d': 2048, 'dff': 8192,  'heads': 32, 'n_layers': 16, 'edge_os': 'LLAMA3-1B',   'display': 'Llama3 - 1B'},
    'llama32_3B': {'d': 3072, 'dff': 8192,  'heads': 24, 'n_layers': 28, 'edge_os': 'LLAMA3-3B',   'display': 'Llama3 - 3B'},
    'phi3':       {'d': 3072, 'dff': 8192,  'heads': 32, 'n_layers': 32, 'edge_os': 'PHI3-3.8B',   'display': 'Phi3 - 3.8B'},
    'bitnet_8B':  {'d': 4096, 'dff': 14336, 'heads': 32, 'n_layers': 32, 'edge_os': 'BITNET-8B',   'display': 'BitNet - 8B'},
    'llama3_8B':  {'d': 4096, 'dff': 14336, 'heads': 32, 'n_layers': 32, 'edge_os': 'LLAMA3-8B',   'display': 'Llama3 - 8B'},
}
CONTEXT_LENGTHS = [256, 512, 1024, 2048, 4096, 8192]
WEIGHT_BIT = 2

# MNSIM network layout (-NN transformers): conv (dummy), then fc layers
#   Q, K, V, attention (l x d/h), O, FFN up, FFN down
ATTN_FC_INDEX = 3  # position of the attention layer among the fc layers

# ----------------------------------------------------------------------------- PIM v2 attention formulas
# latency = ATTN_MATMULS x xbar latency + ATTN_MATMULS x (buffer read + BUFFER_WRITE_FACTOR x buffer write)
ATTN_MATMULS = 2            # Q.K^T and score.V
BUFFER_WRITE_FACTOR = 10
ATTN_COMPUTE_FIELD = 'computing'  # MNSIM field for the l x d/h term: 'computing' (xbar + DAC + ADC) or 'layer'

# energy = (hardware - buffer) + BUFFER_READ_ENERGY_FACTOR x read + BUFFER_WRITE_ENERGY_FACTOR x write
BUFFER_READ_ENERGY_FACTOR = 2
BUFFER_WRITE_ENERGY_FACTOR = 10

# ----------------------------------------------------------------------------- PIM v1 (attention on TPU)
TPU_CYCLE_NS = 10
TPU_POWER_W = 4.0713e-3
# The original micro_2b.ipynb added the TPU attention energy once per token instead of once per
# transformer layer. False counts it per layer (consistent with latency); True reproduces the old numbers.
OLD_ATTN_ENERGY_AS_PUBLISHED = False

# ----------------------------------------------------------------------------- words per battery life
BATTERY_J = 18000
TOKENS_PER_WORD = 1.5


def mnsim_commands(python='python'):
    """Every MNSIM run needed by the extraction scripts, as (output file, command) pairs."""
    base = f'{python} main.py --disable_accuracy_simulation -Weights None -b {WEIGHT_BIT}'
    runs = []
    for name, m in MODELS.items():
        args = f"-d {m['d']} -dff {m['dff']} -H {m['heads']}"
        # dummy conv (layer 0) alone: l does not change it
        runs.append((ISOLATION_RUNS_DIR / f'{name}_layer0.txt', f'{base} -NN transformers_layer0 {args} -l 256'))
        for l in CONTEXT_LENGTHS:
            runs.append((FULL_RUNS_DIR / f'{name}_{l}.txt', f'{base} -NN transformers {args} -l {l}'))
            runs.append((ISOLATION_RUNS_DIR / f'{name}_attention_{l}.txt', f'{base} -NN transformers_attention {args} -l {l}'))
    return runs


if __name__ == '__main__':
    import sys
    for out, cmd in mnsim_commands(sys.argv[1] if len(sys.argv) > 1 else 'python'):
        print(f'{cmd} > {out} 2>&1')
