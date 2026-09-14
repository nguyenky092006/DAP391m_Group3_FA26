# FedECAI — Trạng thái đã xác minh

Cập nhật ngày 2026-09-14. Đây là tài liệu trạng thái dành cho người đọc, không phải báo cáo kết quả cuối cùng.

## Phạm vi và dữ liệu

FedECAI nghiên cứu nhận diện bốn cảm xúc Angry, Happy, Neutral và Sad trên ViSEC, đồng thời đánh giá chênh lệch giữa ba nhóm giọng North, Central và South.

- Raw ViSEC có 5.280 bản ghi và 147 speaker ID.
- Sau B2 có 4.992 mẫu hợp lệ; 288 mẫu bị loại đều có lý do lưu trong manifest.
- Split chuẩn là `speaker_disjoint_v2`: train 3.681 mẫu/103 speaker, validation 646 mẫu/22 speaker, test 665 mẫu/22 speaker.
- Speaker overlap giữa train, validation và test bằng 0; test cố định chưa được dùng để tạo các kết quả phát triển hiện có.
- B4 đã tạo đầy đủ feature cho 4.992 mẫu, gồm MFCC, eGeMAPS, pitch, log-Mel và frozen Wav2Vec2 embedding. Mel 3-channel cho các image backbone được tạo on-load từ log-Mel.

Chi tiết đã hợp nhất theo bước tại `docs/research/B1_SPEC.md` đến `docs/research/B6_SPEC.md`. Audit takeover đầy đủ nằm ở `docs/PROJECT_STATUS_AUDIT.md`.

## Trạng thái B1–B6

| Bước | Trạng thái đã xác minh |
| --- | --- |
| B1 | DONE |
| B2 | DONE |
| B3 | DONE |
| B4 | DONE |
| B5 | DONE |
| B6 | PARTIAL — 8/8 development validation complete; final test/report/Pareto chưa thực hiện |

Benchmark centralized bắt buộc gồm đúng 8 model: SVM, XGBoost, CNN 2D, ResNet-18, MobileNetV3, ViT, raw-waveform frozen Wav2Vec2 và Pitch-Fusion. ViT vẫn được giữ vì thuộc protocol đã chốt; không được xóa chỉ dựa trên một kết quả validation thấp.

Infrastructure của cả 8 path đã pass smoke test với nhãn `SMOKE_TEST_ONLY`; smoke metric không phải benchmark result. Cả 8 model hiện có checkpoint, 646 prediction validation và metrics JSON chuẩn. Macro-F1 validation lần lượt là SVM 0.4663, XGBoost 0.4236, CNN2D 0.3430, ResNet-18 0.3518, MobileNetV3 0.3702, ViT frozen-pretrained 0.4142, raw-waveform frozen Wav2Vec2 0.4193 và Pitch-Fusion 0.6023. Đây vẫn là development evidence, không phải final-test result.

Bảng dễ đọc được sinh trực tiếp từ các metrics JSON tại `artifacts/metrics/b6_centralized_validation_summary.csv`. Chi tiết per-class, per-accent, confusion matrix và chi phí của từng model nằm trong tám file `artifacts/metrics/b6_*_seed391_validation.json`.

## Runtime và hard gate

Benchmark dùng đúng interpreter:

`D:\PROJECT\PVC_FedOrion\DAP391m_Group3_FA26\DAP301m\Scripts\python.exe`

Environment đã xác minh: Python 3.12.13, Torch 2.11.0+cu126, torchvision 0.26.0+cu126, Transformers 4.57.6, CUDA 12.6 và NVIDIA GeForce RTX 4060 Laptop GPU. Torchaudio không có nhưng không phải dependency của 8 runner hiện tại.

`artifacts/audit/benchmark_preflight.json` hiện báo `CENTRALIZED_INFRASTRUCTURE_STATUS = READY`. Development validation 8 model đã hoàn tất; bước còn thiếu của B6 là đánh giá đúng một lần trên fixed test sau khi đóng băng cấu hình, tạo report/Pareto và chờ người dùng chọn backbone. Chưa cho phép GRL, FedAvg, FedProx, federated learning hoặc FedECAI và chưa có backbone nào được chọn.

## Evidence và tổ chức repository

- Tài liệu nghiên cứu chủ đạo: 6 file `docs/research/B1_SPEC.md` … `B6_SPEC.md`.
- Prediction/checkpoint/metrics hiện hành: `artifacts/predictions/`, `artifacts/checkpoints/`, `artifacts/metrics/`.
- Run kiểm định hạ tầng đầy đủ mới nhất: `logs/runs/20260914_155444_centralized_infrastructure_seed391/`.
- Audit 8 bộ artifact validation: `logs/runs/20260914_093211_b6_validation_artifact_audit_seed391/`.
- Lịch sử các run hạ tầng cũ: `logs/infrastructure_run_history.csv` và `artifacts/archive/infrastructure_runs_pre_155544.zip`.
- Log terminal B4–B6 cũ đã hợp nhất tại `logs/legacy_terminal_runs.log`.
- Report FedECAI preliminary cũ đã archive tại `artifacts/archive/legacy_fedecai_reports.zip`; chúng không phải bằng chứng centralized hiện hành.
- Quyết định dọn file và khả năng khôi phục: `artifacts/audit/repository_cleanup_manifest.csv`.

Fixed-test benchmark, report-ready figures, Pareto frontier và backbone decision đều chưa tồn tại. Không được tự tạo hoặc diễn giải smoke/development metrics thành kết quả cuối cùng.
