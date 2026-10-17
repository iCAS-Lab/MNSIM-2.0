# PIM-LLM: PIM v1 vs PIM v2 (ISCAS 2026)

This branch extends [MNSIM 2.0](MNSIM_README.md) to evaluate LLM decoding on a processing-in-memory
(PIM) accelerator, and compares two designs:

| Design | Q/K/V/O projections + FFN | Attention (Q·Kᵀ and score·V) |
|---|---|---|
| **PIM v1** | PIM crossbars (MNSIM) | TPU systolic array (ScaleSim, `edge_os.csv`) |
| **PIM v2** | PIM crossbars (MNSIM) | **PIM crossbars (MNSIM)** |

Both designs share the same non-attention PIM latency and energy; only attention differs.
Everything for the paper lives in [`iscas2026/`](iscas2026); the rest of the repository is the MNSIM simulator.

## Results at a glance (edge, 2-bit weights)

Speedup (tokens/s) and tokens/J change of PIM v2 over PIM v1, range over the 7 models:

| Context length | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|
| Speedup v2 / v1 | 1.1–1.5× | 1.1–1.9× | 1.0–2.3× | 2.0–4.2× | 3.9–8.1× | 7.7–15.8× |
| Tokens/J change | +0.9 to +2.6% | +0.4 to +2.4% | -8.6 to +3.2% | -13.4 to +5.9% | -18.3 to +10.5% | -22.5 to +17.6% |

Throughput always improves. Energy efficiency improves at short contexts; at long contexts only the
8-head Gemma2-2B gains, because the per-head KV-cache buffer traffic grows with the number of heads.

At context length 2048:

| Model | v1 tokens/s | v2 tokens/s | Speedup | v1 tokens/J | v2 tokens/J | Change |
|---|---|---|---|---|---|---|
| Gemma2 - 2B | 17.56 | 74.08 | 4.22× | 685.6 | 725.8 | +5.9% |
| Llama3 - 1B | 15.46 | 30.81 | 1.99× | 1089.9 | 983.6 | -9.7% |
| BitNet - 0.7B | 13.79 | 27.60 | 2.00× | 1384.8 | 1199.6 | -13.4% |
| Llama3 - 3B | 7.01 | 21.33 | 3.04× | 349.0 | 331.2 | -5.1% |
| Phi3 - 3.8B | 5.77 | 14.71 | 2.55× | 301.6 | 270.4 | -10.4% |
| BitNet - 8B | 4.55 | 13.54 | 2.98× | 157.0 | 151.4 | -3.6% |
| Llama3 - 8B | 4.55 | 13.54 | 2.98× | 157.0 | 151.4 | -3.6% |

All numbers: [`iscas2026/results/pim_v1_vs_v2_summary.csv`](iscas2026/results/pim_v1_vs_v2_summary.csv).
All figures: [`iscas2026/figures/`](iscas2026/figures).

## Quick start

```bash
conda activate <env with torch, pandas, matplotlib>
bash iscas2026/run_all.sh            # 91 MNSIM runs (~7 min on 4 cores) -> CSVs -> figures
SKIP_SIM=1 bash iscas2026/run_all.sh # reuse the committed MNSIM reports, only extract + plot
```

Or step by step (each step only reads the outputs of the previous one):

```bash
bash iscas2026/run_simulations.sh [jobs]   # MNSIM reports  -> iscas2026/simulations/
python iscas2026/extract_latency.py        # reports + edge_os.csv -> iscas2026/results/latency.csv
python iscas2026/extract_energy.py         # reports -> iscas2026/results/energy.csv
python iscas2026/make_figures.py           # CSVs -> iscas2026/results/pim_v1_vs_v2_summary.csv, iscas2026/figures/
```

Use `PYTHON=python3 bash ...` if your interpreter is not called `python`. On a SLURM cluster:
`sbatch --wrap "bash iscas2026/run_simulations.sh 16"` after activating the environment.

## Layout

```
MNSIM/ main.py SimConfig.ini techfile.txt   MNSIM 2.0 simulator (with the changes below)
MNSIM_README.md, MNSIM_manual.pdf           original MNSIM documentation
iscas2026/
  config.py            models, context lengths, formula constants, MNSIM commands (single source of truth)
  run_simulations.sh   runs every MNSIM simulation in parallel
  mnsim_parser.py      parses an MNSIM text report (per-layer latency/energy, network energy totals)
  extract_latency.py   PIM v1 / v2 latency and tokens/s           -> results/latency.csv
  extract_energy.py    PIM v1 / v2 energy and tokens/J            -> results/energy.csv
  make_figures.py      PIM v1 vs v2 figures                       -> figures/
  run_all.sh           all of the above
  data/edge_os.csv     ScaleSim TPU results (attention cycles for PIM v1, TPU-only baseline)
  simulations/full/       {model}_{l}.txt              full transformer layer
  simulations/isolation/  {model}_layer0.txt           dummy conv only
                          {model}_attention_{l}.txt    dummy conv + attention layer
  results/  figures/   outputs
```

## Models

From the published configurations; `l` is swept over 256, 512, 1024, 2048, 4096 and 8192.

| Model | d | d_ff | heads h | layers | d/h |
|---|---|---|---|---|---|
| BitNet - 0.7B | 1536 | 4096 | 24 | 24 | 64 |
| Gemma2 - 2B | 2048 | 16384 | 8 | 18 | 256 |
| Llama3 - 1B | 2048 | 8192 | 32 | 16 | 64 |
| Llama3 - 3B | 3072 | 8192 | 24 | 28 | 128 |
| Phi3 - 3.8B | 3072 | 8192 | 32 | 32 | 96 |
| BitNet - 8B | 4096 | 14336 | 32 | 32 | 128 |
| Llama3 - 8B | 4096 | 14336 | 32 | 32 | 128 |

## MNSIM changes

* `main.py`: new options `-l` (context length) and `-H` (number of heads), next to `-d`, `-dff`, `-b`.
* `MNSIM/Interface/network.py`, `-NN transformers` (one transformer layer):

  ```
  conv 1x1 (dummy) -> view -> fc Q (d→d) -> fc K (d→d) -> fc V (d→d)
                           -> fc attention (d/h → l, one head)        <- new
                           -> fc O (d→d, input from V) -> fc FFN up (d→d_ff) -> fc FFN down (d_ff→d)
  ```
  The ReLUs between the fc layers were removed (latency unchanged). The attention layer holds Kᵀ of
  one head (`l x d/h` weights); heads run in parallel on separate crossbars, so one head is simulated.
* Two variants used only to isolate attention energy: `-NN transformers_layer0` (dummy conv only) and
  `-NN transformers_attention` (dummy conv + attention layer).
* `MNSIM/Interface/quantize.py`: fc layers with `slice_input` read only their first `in_features`
  inputs, so the `d/h` attention layer can follow a `d`-wide layer (one head's slice).
* `MNSIM/Interface/interface.py`: passes `l` and `h` to the network.

## How latency is extracted (`extract_latency.py`)

From each full report, the dummy conv is dropped and the fc layers are split into the attention
layer (4th fc) and the six non-attention layers. Per transformer layer, in ns:

| Quantity | Formula | Source |
|---|---|---|
| Non-attention (v1 and v2) | max(Q, K, V) + O + FFN up + FFN down | MNSIM `total latency` |
| PIM v1 attention | `Total_Attention_Cycles` × 10 ns | `edge_os.csv` |
| PIM v2 attention | 2 × xbar + 2 × h × (buffer read + 10 × buffer write) | attention layer: `Computing latency`, `read/write buffer latency` |
| Per token | layer latency × `n_layers`; tokens/s = 1 / per-token latency | |

Q, K and V run in parallel, hence the max. The h heads compute in parallel on separate crossbars
(one head is simulated), but each head reads and writes the buffers, so the buffer term is counted h times. The latency breakdown figures apply the same rule to
MNSIM's computing, buffer, digital and intra + inter tile communication latencies.

## How energy is extracted (`extract_energy.py`)

MNSIM gives per-layer hardware energy, but buffer read/write energy only for the whole network, so
the attention layer is isolated by difference:

```
attention X = X(conv + attention run) - X(conv-only run)        X = hardware, read buffer, write buffer
```

Per transformer layer, in nJ:

| Quantity | Formula |
|---|---|
| Non-attention (v1 and v2) | sum of MNSIM per-layer energy of Q, K, V, O, FFN up, FFN down |
| Non-attention components | full run − (conv + attention) run, for crossbar, DAC, ADC, buffer, other digital |
| PIM v2 attention | h × [(hardware − buffer) + 2 × read buffer + 10 × write buffer] (every head computes and accesses the buffers) |
| PIM v1 attention | TPU attention latency × 4.0713 mW |
| Per token | × `n_layers`; tokens/J = 1 / per-token energy |
| Words per battery | tokens/J × 18000 J / 1.5 tokens per word |

Built-in checks stop the script if the isolated attention energy differs from the attention layer of
the full run, or if the components do not add up to the non-attention energy.

## Checking a number

Every row of `results/latency.csv` and `results/energy.csv` keeps the raw MNSIM values next to the
derived ones (`mnsim_attention_*` columns) and names the report it came from (`mnsim_report`).
For example, Phi3 at l = 256 (`simulations/full/phi3_256.txt`, `Layer 4`):

```
Computing latency 226.56, read buffer latency 880, write buffer latency 720 (ns), h = 32
v2 attention = 2 × 226.56 + 2 × 32 × (880 + 10 × 720) = 517573.1 ns   -> v2_attention_ns
```

## Notes and limitations

* **Attention latency levels off from l ≈ 1024.** MNSIM spreads the `l` output columns over more
  crossbars in parallel: computing latency stays constant and buffer read/write latency stops growing.
  Only inter-tile communication of the attention layer keeps growing with `l`, and the formula does
  not include it (switch `ATTN_COMPUTE_FIELD` to `'layer'` in `config.py` to include the whole layer).
* **Per-head scaling** is set by `BUFFER_PER_HEAD` (latency buffer term × h) and `ATTN_ENERGY_PER_HEAD`
  (attention energy × h) in `config.py`; set both to `False` for the earlier single-head numbers.
* **Energy formula** has no × 2 for the two matmuls (unlike latency), as specified.
* **PIM v1 energy** counts the TPU attention energy per transformer layer. Set
  `OLD_ATTN_ENERGY_AS_PUBLISHED = True` in `config.py` to reproduce the original notebook, which added it
  once per token.
* **Weight bits:** `-b` sets the weight bit of the batch-norm layer only; the fc layers use the 2-bit
  quantisation set in `network.py`.
* **Model simplifications:** one head is simulated (heads in parallel); SwiGLU's gate projection,
  grouped-query attention (smaller K/V), softmax, LayerNorm and residual adds are not modelled.
