# FedECAI E1-E7 validation decision packet

All values are development-validation evidence. The fixed 665-sample test was not evaluated.

Completed evidence available: E1, E2, E3, E4, E5, E6, E7.

## Primary main-scenario comparison

| Exp | Method | Macro-F1 | Worst accent | Accent gap | Accent probe | Train sec | Latency ms | Size MB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E1 | Centralized ResNet18 baseline | 0.621835 | 0.518403 | 0.128591 | 0.712096 | 158.99 | 3.5130 | 42.719 |
| E2 | Centralized ResNet18 + unconditional GRL | 0.610973 | 0.455529 | 0.189409 | 0.693879 | 265.17 | 3.6858 | 42.971 |
| E3 | Centralized ResNet18 + emotion-conditioned GRL | 0.587763 | 0.543859 | 0.059016 | 0.652572 | 184.65 | 3.4587 | 42.973 |
| E4 | Federated ResNet18 FedAvg | 0.646339 | 0.552033 | 0.129835 | 0.693218 | 799.54 | 2.8599 | 42.719 |
| E5 | Federated ResNet18 FedProx | 0.648969 | 0.588607 | 0.069624 | 0.684325 | 735.78 | 4.7959 | 42.719 |
| E6 | Federated ResNet18 FedAvg + unconditional GRL | 0.643101 | 0.406917 | 0.263270 | 0.712058 | 549.45 | 3.0689 | 42.973 |
| E7 | FedECAI ResNet18 | 0.642503 | 0.551200 | 0.112855 | 0.748564 | 10264.39 | 4.8964 | 42.975 |

E3 reduces accent gap and probe performance relative to E1 but exceeds the planned utility-loss budget. E5 is the strongest primary federated result in both Overall and Worst-accent Macro-F1. E6 substantially worsens primary accent disparity. E7 is close to E4 on utility/worst-accent and reduces the gap slightly, but its accent-probe score increases; therefore this run does not establish accent invariance.

## Matched deltas

| Comparison | Δ Macro-F1 | Δ Worst accent | Δ Accent gap | Δ Accent probe |
|---|---:|---:|---:|---:|
| E2 - E1 | -0.010862 | -0.062874 | 0.060819 | -0.018217 |
| E3 - E1 | -0.034072 | 0.025456 | -0.069574 | -0.059524 |
| E3 - E2 | -0.023210 | 0.088330 | -0.130393 | -0.041308 |
| E5 - E4 | 0.002630 | 0.036574 | -0.060210 | -0.008892 |
| E6 - E4 | -0.003238 | -0.145116 | 0.133435 | 0.018840 |
| E7 - E4 | -0.003836 | -0.000833 | -0.016979 | 0.055347 |
| E7 - E6 | -0.000598 | 0.144283 | -0.150414 | 0.036506 |

No final lambda is selected. The controlled value 0.10 requires later RQ6 sensitivity analysis.

## Per-emotion and per-accent evidence

Exact precision/recall/F1/support and confusion matrices are retained in each `metrics.json`; consolidated source tables are `per_emotion_comparison.csv` and `per_accent_comparison.csv`.

## Resource and communication cost

Parameter counts, model size, training time, latency and measured federated state-transfer bytes are in `federated_resource_comparison.csv` and each experiment's `resource_metrics.json`. E7 recorded an unusually long wall-clock duration; it is preserved rather than normalized manually.

## Lineage

E1 is the immutable `b6_resnet18_mel3_finetune_seed391` baseline. E2-E7 initialize the ResNet18 emotion state from that checkpoint.

- E1: checkpoint `e9d76399d2becadb87641d9a13624ad2f612994353fcc3a3f8a3bbb056bfbc0b`; predictions `e9a55570092e8dbc352bbd2daff3cdf35c5914e401930c217623ec869ca54789`.
- E2: checkpoint `30a838d6d0fb67179af2d179a05bc0d3a04df4ad8c2560e8dcc1afe78101252a`; predictions `6abb40db7f5558e90f367ceba6bf8a87788ea91e8c46bb951d1d954c44602bc0`.
- E3: checkpoint `f78c5042d9c82783b4e738e859f7e0b7e2c427dd28807d7bae2bba08dcb6a395`; predictions `bcf78dc4bb8ba816c32c71308aa72e47e157ce59f5216e5db93abad3a834ab0d`.
- E4: checkpoint `151bbd85ef9667ff10ea6ac41ef086d43e1bb6dc79cb1d80935aa5948d5a12e7`; predictions `2261bf0ff67a243cab78045da251f8b7c75eec8ead45a3516cd4e84e53dc6049`.
- E5: checkpoint `064340a81eaea40422b974ce02cfadbce17108315c8cc19f6cbb3b8bd10c4581`; predictions `3f1da7ba604d37be51aac57a8414a0f3121ab66bae7f283ce7af11cea2c9e4cf`.
- E6: checkpoint `e3c90e6eae7134ceb2a19af5833e1b37cc1e22e0502c9188c65ed0f77e195358`; predictions `bc870b4f9629e70b772f35f0c514bda74a75971ee5522d06da29f16781c9d390`.
- E7: checkpoint `e4644e2c85b3e9950e122cf67159203cb305599aa64f34f5d3164fc57ba90fec`; predictions `3cee0c2a86191432be47bda5af98e722b4fb057bf428dd364d5b41f3a3f759c5`.

## Execution failures

No E2-E7 smoke/full run failed. All checkpoint replay gates passed. Intermediate metric fluctuations are retained in training/round histories.

## Limitations

The active sample-level protocol permits speaker overlap. Accent results and probes are descriptive and are not speaker-independent or causal accent effects. Lambda 0.10 is the controlled initial ladder setting, not an RQ6-selected optimum.

Source tables: `e1_e7_full_ladder.csv`, `e1_e7_deltas.csv`, `per_emotion_comparison.csv`, `per_accent_comparison.csv`, `accent_probe_comparison.csv`, and `federated_resource_comparison.csv`.
