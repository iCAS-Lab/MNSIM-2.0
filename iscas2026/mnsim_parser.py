"""
Parse an MNSIM 2.0 text report (the stdout of main.py).

The report has two parts:

1. Latency results, one block per layer:

       Layer 4  type: fc
       total latency: 5873.14
       layer latency: 5233.14
       Buffer latency of layer 4 : 1440 ( 24.52 %)
            read buffer latency of layer 4 : 800 ( 13.62 %)
            write buffer latency of layer 4 : 640 ( 10.90 %)
       Computing latency of layer 4 : 226.56 ( 3.86 %)
       Digital part latency of layer 4 : 81.98 ( 1.40 %)
       Intra tile communication latency of layer 4 : 3.0 ( 0.05 %)
       Inter tile communication latency of layer 4 : 4121.6 ( 70.18 %)

2. Energy results: network totals first, then one value per layer:

       Hardware energy: 83509.02 nJ            <- whole network
               crossbar energy: 337.42 nJ
               DAC energy: ...   ADC energy: ...
               Buffer energy: ...
                       |---read buffer energy: ...
                       |---write buffer energy: ...
               Other digital part energy: ...
       Layer 0 :
            Hardware energy: 498.58 nJ          <- per layer
       ...

All latencies are in ns and all energies in nJ, as reported by MNSIM.
"""
import re

LAYER_LATENCY_PATTERNS = {
    'total': re.compile(r'total latency: ([\d.]+)'),
    'layer': re.compile(r'layer latency: ([\d.]+)'),
    'buffer': re.compile(r'Buffer latency of layer \d+ : ([\d.]+)'),
    'buf_read': re.compile(r'read buffer latency of layer \d+ : ([\d.]+)'),
    'buf_write': re.compile(r'write buffer latency of layer \d+ : ([\d.]+)'),
    'computing': re.compile(r'Computing latency of layer \d+ : ([\d.]+)'),
    'digital': re.compile(r'Digital part latency of layer \d+ : ([\d.]+)'),
    'intra': re.compile(r'Intra tile communication latency of layer \d+ : ([\d.]+)'),
    'inter': re.compile(r'Inter tile communication latency of layer \d+ : ([\d.]+)'),
}

ENERGY_TOTAL_PATTERNS = {
    'hardware': re.compile(r'^Hardware energy: ([\d.]+) nJ', re.M),
    'crossbar': re.compile(r'crossbar energy:\s*([\d.]+) nJ'),
    'DAC': re.compile(r'DAC energy:\s*([\d.]+) nJ'),
    'ADC': re.compile(r'ADC energy:\s*([\d.]+) nJ'),
    'buffer': re.compile(r'Buffer energy:\s*([\d.]+) nJ'),
    'buf_read': re.compile(r'read buffer energy:\s*([\d.]+) nJ'),
    'buf_write': re.compile(r'write buffer energy:\s*([\d.]+) nJ'),
    'other_digital': re.compile(r'Other digital part energy:\s*([\d.]+) nJ'),
}

# components that add up to the network hardware energy (pooling and NoC are 0 here)
ENERGY_COMPONENTS = ['crossbar', 'DAC', 'ADC', 'buffer', 'other_digital']


def _read(path):
    with open(path) as f:
        content = f.read()
    if 'Hardware energy:' not in content:
        raise ValueError(f'{path}: no energy results, the MNSIM run probably failed')
    return content


def parse_layers(path):
    """Per-layer list of dicts: 'type', the latency fields above (ns) and 'energy' (nJ)."""
    content = _read(path)
    latency_part, energy_part = content.split('Hardware energy:', 1)

    layers = []
    for block in re.split(r'\nLayer \d+\s+type: ', latency_part)[1:]:
        layer = {'type': block.split()[0]}
        for key, pattern in LAYER_LATENCY_PATTERNS.items():
            m = pattern.search(block)
            layer[key] = float(m.group(1)) if m else 0.0
        layers.append(layer)

    per_layer_energy = [float(e) for e in re.findall(r'Layer \d+ :\s+Hardware energy: ([\d.]+)', energy_part)]
    if len(per_layer_energy) != len(layers):
        raise ValueError(f'{path}: {len(layers)} latency blocks but {len(per_layer_energy)} layer energies')
    for layer, energy in zip(layers, per_layer_energy):
        layer['energy'] = energy
    return layers


def parse_energy_totals(path):
    """Network-level energy totals (nJ): hardware, the components and the buffer read/write split."""
    content = _read(path)
    return {key: float(pattern.search(content).group(1)) for key, pattern in ENERGY_TOTAL_PATTERNS.items()}


def split_transformer_layers(layers, attn_fc_index):
    """(attention layer, non-attention fc layers) of a -NN transformers run; the dummy conv is dropped."""
    fc = [x for x in layers if x['type'] != 'conv']
    return fc[attn_fc_index], fc[:attn_fc_index] + fc[attn_fc_index + 1:]


def qkv_parallel_sum(layers, key):
    """max over the parallel Q, K, V projections + sum of the remaining layers (as in micro_2b.ipynb)."""
    return max(x[key] for x in layers[:3]) + sum(x[key] for x in layers[3:])
