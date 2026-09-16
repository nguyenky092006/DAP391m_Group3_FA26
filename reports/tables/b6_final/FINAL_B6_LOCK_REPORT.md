# Final locked B6 centralized benchmark

All values are validation-only saved evidence. The fixed 665-sample test set remains untouched.

Report-ready table image: `reports/figures/b6/final_centralized_b6_ranking.png`
(`Accent Gap ↓` included; generated directly from the canonical CSV).

| Rank | Model | Experiment ID | Macro-F1 | UAR | Worst-accent F1 | Latency median ms | Size MB |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | resnet18_mel3 | `b6_resnet18_mel3_finetune_seed391` | 0.621835 | 0.621631 | 0.518403 | 3.513 | 42.719 |
| 2 | xgboost | `b6_xgboost_seed391` | 0.604658 | 0.600488 | 0.513889 | 1.934 | 3.608 |
| 3 | svm_rbf | `b6_svm_rbf_seed391` | 0.590991 | 0.592056 | 0.539906 | 2.648 | 8.890 |
| 4 | pitch_fusion_paper_aligned | `b6_pitch_fusion_paper_aligned_seed391` | 0.573992 | 0.569248 | 0.319444 | 15.483 | 541.291 |
| 5 | wav2vec2_raw | `b6_wav2vec2_raw_seed391` | 0.557707 | 0.555241 | 0.426487 | 11.676 | 362.658 |
| 6 | mobilenet_v3_mel3 | `b6_mobilenet_v3_mel3_finetune_seed391` | 0.536118 | 0.543582 | 0.455514 | 7.555 | 5.930 |
| 7 | vit_b16_mel3 | `b6_vit_b16_mel3_seed391` | 0.509377 | 0.513624 | 0.165278 | 10.179 | 330.308 |
| 8 | cnn2d_logmel | `b6_cnn2d_logmel_seed391` | 0.493006 | 0.502607 | 0.249457 | 0.977 | 0.101 |

## Human decision

- `FEDERATED_BACKBONE = resnet18_mel3`
- `BACKBONE_TRAINING_POLICY = full_finetune`
- `EFFICIENCY_FALLBACK = mobilenet_v3_mel3`
- ResNet18 is the strongest finalized neural model, retains measurable accent disparity, and is materially smaller/faster than Pitch-Fusion while remaining practical for simulated FL.

## Lock status

```text
B6_STATUS = LOCKED
B6_ARCHIVE_STATUS = COMPLETE
CENTRALIZED_HYPERPARAMETER_SEARCH = CLOSED
CENTRALIZED_RETRAINING = DISABLED_BY_HUMAN_DECISION
FEDERATED_BACKBONE_STATUS = SELECTED_RESNET18
EFFICIENCY_FALLBACK_STATUS = SELECTED_MOBILENETV3
TEST_SET_STATUS = UNTOUCHED
FEDERATED_STAGE = READY_TO_START
FEDECAI_STATUS = NOT_STARTED
```
