# FedECAI — Trạng thái project đã xác minh

Cập nhật ngày 2026-09-15 sau khi hoàn tất ladder ResNet18 E1–E7 trên tập
validation.

## Dữ liệu và protocol

- Dataset: `visec_clean_eligible_v1`, 4.992 waveform duy nhất, 147 speaker.
- Split: `emotion_stratified_v1`, seed 391.
- Train/validation/test: 3.681/646/665 mẫu.
- Speaker có thể trùng giữa các split và chỉ được báo cáo như thông tin mô tả.
- Trùng waveform hash và file identity giữa các split: 0.
- Tập test 665 mẫu vẫn chưa được truy cập.

Vì protocol là sample-level và có speaker overlap, các kết quả hiện tại không
chứng minh khả năng tổng quát hóa tới speaker hoàn toàn mới.

## B6 và backbone

B6 đã khóa tám centralized model. ResNet18 3-channel Mel với full fine-tuning
được người dùng chọn làm backbone; MobileNetV3 là phương án hiệu quả dự phòng.
E1 dùng lại đúng checkpoint B6
`b6_resnet18_mel3_finetune_seed391`, không train lại.

## Kết quả ladder E1–E7 trên validation

E2–E7 đã hoàn tất bằng cùng 646 sample ID validation. Mỗi run có config,
prediction, metric, checkpoint, history, chi phí tài nguyên và kiểm tra replay.
Final integrity audit trả về `PASS`; mọi run ghi `test_samples_loaded=0`.

| Experiment | Phương pháp | Macro-F1 | Worst-accent F1 | Accent gap | Accent probe F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| E1 | Centralized ResNet18 | 0.621835 | 0.518403 | 0.128591 | 0.712096 |
| E2 | Centralized unconditional GRL | 0.610973 | 0.455529 | 0.189409 | 0.693879 |
| E3 | Centralized conditional GRL | 0.587763 | 0.543859 | 0.059016 | 0.652572 |
| E4 | FedAvg | 0.646339 | 0.552033 | 0.129835 | 0.693218 |
| E5 | FedProx | 0.648969 | 0.588607 | 0.069624 | 0.684325 |
| E6 | FedAvg + unconditional GRL | 0.643101 | 0.406917 | 0.263270 | 0.712058 |
| E7 | FedECAI conditional GRL | 0.642503 | 0.551200 | 0.112855 | 0.748564 |

E5 có Macro-F1 và Worst-accent Macro-F1 cao nhất trong nhóm federated chính.
E7 chưa chứng minh accent invariance: so với E4, utility và Worst-accent F1 hơi
thấp hơn, accent gap nhỏ hơn nhưng accent-probe F1 lại cao hơn. Đây chỉ là kết
quả validation, không phải kết luận final-test.

## Trạng thái B-step

| Bước | Trạng thái hiện tại |
| --- | --- |
| B1–B5 | DONE theo protocol hiện hành |
| B6 | LOCKED; tám centralized model được giữ nguyên |
| B7 | PARTIAL; so sánh validation E1–E7 hoàn tất, final test chưa chạy |
| B8 | PARTIAL; run kiểm soát dùng lambda 0.10, RQ6 lambda sweep chưa bắt đầu |
| B9 | Chưa được thực hiện trong task này |
| B10 | PARTIAL; decision packet đã có, báo cáo cuối chưa hoàn tất |

## Bằng chứng chính

- Kết quả E1–E7: `reports/tables/fedecai/e1_e7_full_ladder.csv`
- So sánh RQ4: `reports/tables/fedecai/rq4_e1_e2_e3.csv`
- So sánh RQ5: `reports/tables/fedecai/rq5_e4_e5_e6_e7.csv`
- Decision packet: `reports/tables/fedecai/FEDECAI_E1_E7_DECISION_PACKET.md`
- Integrity audit: `artifacts/fedecai/final_integrity_audit.json`
- Run evidence: `logs/runs/fedecai_e1_e7_20260915/`
- Registry: `logs/run_index.csv`, `logs/experiment_registry.csv`

## Chưa được phép tự động thực hiện

RQ6 lambda sweep vẫn là `NOT_STARTED`; chưa có lambda cuối được chọn. Fixed test
vẫn là `UNTOUCHED`. E8 Wav2Vec2 FedECAI chỉ là future work. Không được tự động
train thêm, mở test, hoặc diễn giải validation metric thành final result.
