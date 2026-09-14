# FedECAI Project Status Audit

## 1. Phạm vi và kết luận tổng quan

- Thời điểm chụp trạng thái: **2026-09-13, Asia/Bangkok**.
- Git branch/HEAD: `main` tại commit `675568845895309c157442b85fb2c88af3f9b8a3` (`Complete B6 modelling and unify dependencies`).
- Worktree đã có nhiều thay đổi và file chưa được track trước audit. Audit giữ nguyên toàn bộ các thay đổi đó. Đáng chú ý, `PROJECT_RULES.md`, `PLAN_FEDECAI.md` và bốn log `log_b4_*.log`, `log_b5_split.log`, `log_b6_models.log` đang untracked tại thời điểm kiểm tra.
- Audit chỉ đọc code/dữ liệu/artifact và chạy test nhẹ; **không chạy model training**.
- Hai file được tạo bởi audit này là `docs/PROJECT_STATUS_AUDIT.md` và `doc.md`.

Kết luận chính: repository đã có nền tảng dữ liệu, feature và split khá chắc cho B2 và B5; có EDA B3, feature B4 và các kết quả validation B6 có thể kiểm chứng. Tuy nhiên implementation hiện hữu được xây theo specification cũ gồm **5 model**, không phải benchmark **8 model** hiện hành. ResNet-18, MobileNetV3 và ViT chưa có code/config/result; Mel 3-channel chưa được định nghĩa; FedECAI hiện dùng CNN 1-channel thay vì backbone ResNet-18/MobileNetV3. Vì vậy project chưa thể chuyển sang final evaluation, Pareto, tuning cuối, deployment hoặc report cuối.

Các kiểm tra độc lập trong audit:

- Raw Parquet tồn tại, kích thước `366955512` byte và SHA-256 khớp `bfc7697b3a591cc6cf185c61178815a35363dae4f2c43276636d68eb72cd4a3e`.
- `split_manifest_v2_enriched.csv`: 4.992 dòng hợp lệ, 288 dòng excluded, 147 speaker; train/validation/test lần lượt 3.681/646/665 mẫu và 103/22/22 speaker.
- Speaker overlap train-validation, train-test và validation-test đều bằng 0; duplicate audio hash trong tập hợp lệ bằng 0.
- 103/103 training speaker có đúng một assignment cho cả federated main và stress; không có non-train speaker trong client assignment.
- Feature table hiện tại có 4.992 dòng, 951 cột, 942 cột numeric feature hữu hạn; 4.992/4.992 đường dẫn log-Mel, pitch và Wav2Vec2 artifact tồn tại.
- Tất cả 17 prediction CSV B6 có đúng 646 sample ID thuộc validation v2, không có test sample; metric tính lại từ prediction khớp report.
- `python -m unittest discover -s tests -v`: **77/77 test pass** trong environment `DAP301m` (Python 3.12.13, Torch 2.11.0+cu126, CUDA khả dụng).

### Quy ước trạng thái

| Trạng thái | Ý nghĩa trong audit |
| --- | --- |
| DONE | Đạt requirement và có bằng chứng trực tiếp phù hợp. |
| PARTIAL | Có implementation/evidence nhưng chưa đủ Definition of Done hiện hành. |
| NOT STARTED | Không tìm thấy implementation hoặc artifact phù hợp. |
| BLOCKED | Chưa thể hoàn tất an toàn vì dependency trước đó chưa đạt. |
| UNCERTAIN | Có dấu hiệu nhưng bằng chứng không đủ để xác nhận. |

## 2. Bảng trạng thái B1-B10

| Bước | Trạng thái tổng thể | Kết luận ngắn |
| --- | --- | --- |
| B1 | PARTIAL | Có problem statement, scope và RQ1-RQ3 theo bản cũ; thiếu RQ4-RQ6, H1-H6, owners, evidence map đầy đủ và policy 8 model/backbone mới. |
| B2 | DONE | Audit và cleaning 5.280 record có code, manifest, report, lý do exclusion và test; tên/vị trí một số deliverable khác master plan nhưng nội dung cốt lõi đạt DoD. |
| B3 | PARTIAL | Có EDA uni/bi/multivariate, notebook, SQL và quantitative takeaways; mới có 4 query thay vì đủ 8 nhóm SQL bắt buộc và vẫn tham chiếu split v1 trong pipeline/notebook mặc định. |
| B4 | PARTIAL | Năm feature family cũ đã extract đầy đủ và có 3 visualization/dashboard; thiếu 3-channel Mel cho ResNet/MobileNet/ViT và thiếu speaker-disjoint accent probe. |
| B5 | DONE | Split v2 speaker-disjoint, fixed test, grouped CV và cả main/stress client partition tồn tại, deterministic và đã được audit trực tiếp. |
| B6 | PARTIAL | Có 5-model validation benchmark cũ và ladder E1-E7 bằng CNN; thiếu 3 model mới, backbone policy, test results và reproducibility metadata/checkpoint đầy đủ. |
| B7 | PARTIAL | Có validation predictions và một phần metric/per-accent; final test 8 model, Pareto 4 chiều, probe, statistics, error analysis và federated client metrics chưa có. |
| B8 | PARTIAL | Có grouped CV/tuning cho SVM/XGBoost và grouped development protocol cho deep models; lambda grid, utility budget, failed-trial registry và locked final configs chưa có. |
| B9 | NOT STARTED | Không tìm thấy FastAPI, Docker, Cloud Run, Pub/Sub, Dialogflow hoặc Streamlit/Dash application service. B4 Plotly dashboard chỉ là feature EDA tĩnh. |
| B10 | BLOCKED | Chưa có final report, RQ1-RQ6 conclusions, final figures/tables, reproducibility README hay audit logs; phụ thuộc B6-B9 chưa hoàn tất. |

## 3. Đối chiếu requirement chi tiết và bằng chứng DONE

### B1 — Problem Definition and Research Scope

| Requirement | Trạng thái | Bằng chứng/thiếu hụt |
| --- | --- | --- |
| Problem statement rõ ràng | DONE | `docs/research/B1_SPEC.md`. |
| Research gap có evidence | PARTIAL | Có mô tả nguy cơ accent disparity/speaker leakage nhưng không có mục research gap hoàn chỉnh gắn với literature hiện hành. |
| Expected contributions | PARTIAL | Có objectives/intended outputs; chưa có bộ expected contributions theo master plan. |
| RQ1-RQ6 | PARTIAL tại thời điểm audit | Hiện đã được hợp nhất vào `docs/research/B1_SPEC.md`; dòng này giữ nguyên kết luận lịch sử của takeover audit. |
| H1-H6 | NOT STARTED | Không tìm thấy hypothesis H1-H6 trong tài liệu B1. |
| Baseline literature: ViSEC/Pitch-Fusion, DANN/GRL, FedAvg, FedProx | PARTIAL | ViSEC/Pitch-Fusion có provenance/citation; DANN, FedAvg, FedProx chưa có literature baseline đầy đủ. |
| Scope/out-of-scope | DONE | Bằng chứng hiện được hợp nhất trong `docs/research/B1_SPEC.md`. |
| Owners | NOT STARTED | Không tìm thấy ownership map. |
| 6/6 RQ-to-evidence mappings | PARTIAL | Chỉ có mapping RQ1-RQ3. |
| RQ3 liệt kê đúng 8 model | NOT STARTED | Tài liệu cũ ghi 5 model. |
| RQ5 có ResNet-18/MobileNetV3 policy | NOT STARTED | RQ5 chưa có trong artifact B1. |
| `B1_SPEC.md` và human docs cập nhật | PARTIAL | `B1_SPEC.md` thiếu; `doc.md` chỉ được tạo trong audit để mô tả trạng thái, không làm B1 mới đạt DoD. |

### B2 — Data Understanding and Quality Audit

| Requirement | Trạng thái | Bằng chứng |
| --- | --- | --- |
| Provenance, pinned revision, license và raw checksum | DONE | `docs/research/B1_SPEC.md`, raw Parquet và `raw_audit.json`. |
| Row/audio/path/schema/missing audit | DONE | 5.280 rows, 0 embedded-audio missing, 0 unreadable, 0 missing metadata; `audit_visec.py`, `raw_manifest.csv`, `raw_audit.json`. |
| Unique speaker/label/accent/gender distributions | DONE | 147 speaker; raw và eligible distributions trong audit/cleaning reports. |
| Duration/sample rate/channels/corruption/silence/clipping | DONE | 16 kHz mono toàn bộ; duration quantiles, 80 clipping flags, 408 high-silence flags trong `docs/research/B2_SPEC.md`/`raw_audit.json`. |
| Duplicate/path/waveform/label-conflict audit | DONE | 277 duplicate waveform groups/555 rows, 10 conflicting groups; `duplicate_report.csv`. |
| Speaker mapped to multiple accents/genders | DONE | Sáu speaker accent-conflict và hai speaker gender-conflict được ghi nhận và xử lý traceably. |
| Every exclusion has a reason | DONE | 288 exclusions, không có exclusion thiếu `exclusion_reason`; `clean_manifest.csv`, `cleaning_report.json`. |
| Clean eligible set has unique waveforms and valid labels | DONE | 4.992 eligible rows; 0 duplicate eligible hashes; validation flags đều true. |
| Script/config-driven, raw immutable | DONE | `audit_visec.py`, `clean_visec_manifest.py`; raw hash khớp pinned source. |
| Deliverable path/name đúng master plan | PARTIAL | Nội dung có nhưng nằm dưới `data/manifests/` thay vì một số path `data/processed/`/`artifacts/` được master plan nêu. |

### B3 — EDA and Advanced SQL

| Requirement | Trạng thái | Bằng chứng/thiếu hụt |
| --- | --- | --- |
| Univariate EDA | DONE | Duration summary/figure. |
| Bivariate EDA | DONE | Emotion composition within accent. |
| Multivariate EDA | DONE | Accent × emotion duration/quality heatmap/table. |
| Samples/speakers per accent | PARTIAL | Có accent sample totals và speaker concentration/top-5, chưa có query đầy đủ đúng yêu cầu. |
| Emotion distribution within accent | DONE | `rq2_emotion_distribution_by_accent`. |
| Speaker ranking bằng `RANK()` | DONE | `rq1_rq2_speaker_concentration`. |
| Least-represented emotion per accent | DONE | Query emotion distribution xuất toàn bộ rank và zero-complete grid, đủ suy ra nhóm ít nhất. |
| Speaker overlap across splits bằng SQL | NOT STARTED | Không có named query này. Audit độc lập xác nhận overlap 0 nhưng đó không thay thế deliverable B3. |
| Federated-client distributions bằng SQL | NOT STARTED | Không có named query này. |
| Manifest + quality JOIN | NOT STARTED | Database dùng manifest đã chứa quality fields; không có query JOIN được yêu cầu. |
| CTE/window emotion proportions | DONE | Có CTE, window sum và rank. |
| Mọi query map tới RQ hiện hành | PARTIAL | Chỉ map RQ1/RQ2/evaluation protocol theo specification cũ. |
| Quantitative takeaway cho chart/table chính | DONE | `docs/research/B3_SPEC.md` và `reports/tables/b3/`. |
| Notebook/pipeline dùng canonical split hiện tại | PARTIAL | Notebook và default `b3_eda_sql.py` vẫn trỏ `split_manifest.csv` v1; descriptive results vẫn hữu ích nhưng split-coverage output là lịch sử v1. |

### B4 — Feature Engineering and Representation Analysis

| Requirement | Trạng thái | Bằng chứng/thiếu hụt |
| --- | --- | --- |
| MFCC/eGeMAPS cho SVM/XGBoost | DONE | 4.992 × (80 MFCC + 88 eGeMAPS), finite, indexed artifacts. |
| Log-Mel configurable | PARTIAL | 80-band, 16 kHz, FFT/hop/frequency config có; current artifacts/model là 1-channel và duration policy chưa được đóng thành contract 3-channel. |
| 3-channel Mel cho ResNet/MobileNet/ViT và `mel_3ch_mode` | NOT STARTED | Không có config/code/model nào chứa policy này. |
| Pitch contour + summary | DONE | 4.992 contours và 6-value summaries; 78 all-unvoiced cases có sentinel được document/test. |
| Wav2Vec2 frozen baseline representation | DONE | 4.992 finite 768-D embeddings, pinned checkpoint/revision, resume/chunk lineage. |
| Speaker-disjoint accent probe + majority/permutation/bootstrap | NOT STARTED | Chỉ thấy `planned_consumer`; không có code/output probe. |
| Ít nhất 3 advanced visualizations | DONE | PCA, embedding-shift heatmap, acoustic radar cùng JSON insights. |
| Interactive visualization | DONE | `reports/dashboard/b4_feature_dashboard.html`. |
| Feature families cho đủ 8 model | PARTIAL | Feature nền cho 5 model cũ hoàn tất; input contract mới cho ResNet/MobileNet/ViT chưa có. |
| Reproducible lineage hiện tại | PARTIAL tại thời điểm audit | Status drift lịch sử đã được xử lý khi hợp nhất tài liệu vào `docs/research/B4_SPEC.md`; machine report hiện hành vẫn là nguồn bằng chứng. |

### B5 — Dataset Partition and Federated Clients

| Requirement | Trạng thái | Bằng chứng |
| --- | --- | --- |
| Versioned speaker-disjoint global split | DONE | `split_manifest_v2.csv`, enriched manifest, `split_report_v2.json`, assignment hash `da5610...`. |
| Zero speaker leakage | DONE | Direct audit: cả ba pair overlap = 0; unit tests pass. |
| Fixed test set/all models reuse split | DONE | v2 frozen, 665 test samples; current B6 configs đều khai báo `speaker_disjoint_v2`. |
| No duplicate waveform leakage | DONE | 4.992 eligible hashes đều unique. |
| Grouped CV by speaker | DONE | 5 folds trên 103 training speakers; mỗi speaker đúng một fold; non-train rows không có fold. |
| Main moderate-skew partition | DONE | `main_mixed_accent_v1`, ba clients, 103 speakers assigned once. |
| Stress strong-skew partition | DONE | `stress_accent_domain_v1`, ba accent-domain clients. |
| Per-client sample/speaker/accent/emotion distributions | DONE | `b5_federated_scenarios_v1.json/csv` và `b5_federated_client_summary.csv`. |
| Augmentation only train | DONE | Chưa có augmentation artifact; policy/config cấm fit/apply ngoài train. |

Lưu ý: DONE ở đây là đúng theo released `speaker_id`. Dataset source không chứng minh ID `0` là một người thật thay vì catch-all ID.

### B6 — Modelling

| Requirement | Trạng thái | Bằng chứng/thiếu hụt |
| --- | --- | --- |
| SVM | DONE | Code/config/CV/validation predictions/report; metric kiểm tra lại khớp. Model `.joblib` hiện thiếu. |
| XGBoost | DONE | Code/config/CV/validation predictions/report; metric kiểm tra lại khớp. Model `.joblib` hiện thiếu. |
| CNN 2D | DONE | Code/config/checkpoint/report/validation predictions có. |
| ResNet-18 3-channel Mel | NOT STARTED | Không tìm thấy code/config/checkpoint/result. |
| MobileNetV3 3-channel Mel | NOT STARTED | Không tìm thấy code/config/checkpoint/result. |
| ViT 3-channel Mel | NOT STARTED | Không tìm thấy code/config/checkpoint/result. |
| Wav2Vec2 | PARTIAL | Có frozen-embedding MLP, không phải raw-waveform/fine-tuned classifier; report ghi rõ boundary. |
| Pitch-Fusion | PARTIAL | Có gated-summary ablation và paper-aligned raw-waveform reproduction; bản paper-aligned không exact vì dùng pYIN và masked pooling. |
| Default results trước tuning | PARTIAL | Có validation results một seed cho 5 model cũ; chưa có đủ 8 model và chưa có final test. |
| ResNet/MobileNet input consistency | NOT STARTED | Hai model chưa tồn tại; `mel_3ch_mode` chưa có. |
| E1 baseline/E2 unconditional/E3 conditional | PARTIAL | Đã chạy bằng compact CNN và có report/predictions; không dùng backbone được specification cho phép. |
| E4 FedAvg/E5 FedProx/E6 FedAvg+GRL/E7 FedECAI | PARTIAL | Report/predictions có cho main và stress, 20 rounds; dùng CNN, thiếu checkpoints/lineage chuẩn. |
| Evidence-based backbone decision | NOT STARTED | Không có ResNet/MobileNet benchmark, latency hoặc model-size comparison. |
| Experiment reproducibility | PARTIAL | Seeds/config/split có rải rác; không report `git_commit`, không experiment registry, thiếu nhiều checkpoints và một số input hashes không còn khớp. |

### B7 — Evaluation and Pareto Analysis

| Requirement | Trạng thái | Bằng chứng/thiếu hụt |
| --- | --- | --- |
| Macro-F1/UAR/Accuracy/per-class | PARTIAL | Có trên validation cho model cũ; Weighted-F1 và final test chưa có. |
| Per-accent/worst-accent/gap | PARTIAL | Có trên validation; chưa có final 8-model test. |
| Same fixed test set across 8 models | BLOCKED | B6 chưa có đủ 8 model; test chưa được dùng. |
| Saved predictions | PARTIAL | Validation predictions đầy đủ; test predictions chưa có. |
| Pareto 4 chiều cho 8 model | NOT STARTED | Không có Pareto table/plot. |
| Latency protocol và measurements | NOT STARTED | Không có warm-up/timed-runs/batch/precision protocol hay result. |
| Consistent model size/parameter count | NOT STARTED | Không có comparison artifact; file size quan sát trong audit không thay thế protocol. |
| Accent probe/baselines | NOT STARTED | Không có. |
| Federated global/mean/median/worst-client, communication/convergence | PARTIAL | Có global validation-by-round history; không có client evaluation, rounds-to-target hay communication cost. |
| 3+ seeds, mean±std, speaker-bootstrap CI, effect size | NOT STARTED | Results hiện dùng một seed. |
| Confusion matrices và error analysis | NOT STARTED | Không tìm thấy artifact. |

### B8 — Tuning and Cross-Validation

| Requirement | Trạng thái | Bằng chứng/thiếu hụt |
| --- | --- | --- |
| Classical grouped CV | DONE | SVM/XGBoost dùng predefined grouped folds và train-fold scaler. |
| Deep development/grouped protocol | PARTIAL | CNN/frozen-embedding/pitch ablation có grouped CV; paper-aligned Pitch-Fusion và E1-E7 chọn checkpoint trực tiếp trên validation. |
| Federated development partitions | DONE | main/stress training-client partitions có version và chỉ dùng training speakers. |
| Search spaces documented | PARTIAL | Classical grids có config; deep/FedECAI chủ yếu một cấu hình. |
| Lambda grid 0.00–1.00 | NOT STARTED | Chỉ có `adversarial_lambda=0.1`. |
| Lambda ablation | NOT STARTED | Không có trial/result grid. |
| Utility budget | NOT STARTED | Không có enforcement artifact. |
| Failed trials logged | NOT STARTED | Không có trial registry/log chuẩn. |
| Final configs frozen trước test | BLOCKED | 8-model benchmark/backbone/lambda chưa hoàn tất; test vẫn untouched. |

### B9 — Pipeline, API, Cloud Services, Application

| Requirement | Trạng thái | Bằng chứng/thiếu hụt |
| --- | --- | --- |
| FastAPI `/health` và `/predict` | NOT STARTED | Không có service code. |
| Docker | NOT STARTED | Không có Dockerfile/deployment config. |
| GCP Cloud Run | NOT STARTED | Không có config/evidence deployment. |
| Pub/Sub + webhook/email | NOT STARTED | Không có integration/demo. |
| Dialogflow 2+ intents | NOT STARTED | Không có agent/config/docs. |
| Streamlit hoặc Dash app gọi real endpoint | NOT STARTED | B4 Plotly HTML là EDA dashboard, không phải application client. |
| Secrets/input/CORS/rate-limit/no-audio-retention controls | NOT STARTED | Không có service implementation/security docs. |
| Architecture/deployment docs và service metrics | NOT STARTED | Không có. |

### B10 — Final Report, Conclusions, Reproducibility, AI Reflection

| Requirement | Trạng thái | Bằng chứng/thiếu hụt |
| --- | --- | --- |
| RQ1-RQ6 evidence-backed traceability | BLOCKED | RQ set cũ chỉ có RQ1-RQ3; required experiments còn thiếu. |
| Final report đúng 7 top-level sections | NOT STARTED | Không tìm thấy paper source/PDF. |
| Required final results coverage | BLOCKED | B6-B8 chưa hoàn tất. |
| Human-readable `doc.md` hoàn chỉnh theo B10 | PARTIAL | File được tạo trong audit chỉ ghi trạng thái đã xác minh; chưa thể có final results/Pareto/cloud/reproduction hoàn chỉnh. |
| Final figures/tables | NOT STARTED | Chỉ có B3/B4/B6 preliminary artifacts. |
| Reproducibility README | PARTIAL | `README.md` có nhiều command cũ nhưng environment path/status không còn nhất quán. |
| Project/decision/experiment/AI audit logs | NOT STARTED | Không có thư mục `logs/` hay bốn file bắt buộc. |
| 0 placeholders/fake values/leaked credentials | UNCERTAIN | Không thấy fabricated metrics hoặc hard-coded secrets trong phạm vi đã đọc, nhưng chưa có final report/secret scan gate để kết luận toàn project. |

## 4. Requirement còn thiếu quan trọng nhất

1. Đồng bộ B1 sang 6 RQ/6 hypotheses và 8-model benchmark.
2. Tạo các `B1_SPEC.md` … `B10_SPEC.md`; hiện cả 10 file đều thiếu.
3. Định nghĩa và triển khai Mel 3-channel có `mel_3ch_mode` cho ResNet-18, MobileNetV3, ViT.
4. Triển khai speaker-disjoint accent probe với majority baseline, permutation baseline và speaker-bootstrap CI.
5. Bổ sung ResNet-18, MobileNetV3, ViT và lựa chọn FL backbone bằng evidence.
6. Chạy lại ladder E1-E7 trên backbone hợp lệ; kết quả CNN cũ chỉ là preliminary/legacy evidence.
7. Hoàn thiện evaluation: fixed test cho đủ 8 model, saved predictions, four-axis Pareto, latency/size protocol, error/statistical/client analysis.
8. Hoàn thiện B8 lambda grid/utility budget/locked configs trước khi mở test.
9. Toàn bộ B9 service stack.
10. Final report, `logs/*`, experiment registry và end-to-end reproducibility record.

## 5. Xung đột với specification FedECAI hiện tại

- Tại thời điểm takeover audit, `README.md`, các tài liệu B1/B6 cũ và `run_b6_all_models.py` đều quy định/triển khai **5 model**. Tài liệu hiện hành đã được hợp nhất vào các `B*_SPEC.md`; specification vẫn yêu cầu đúng 8 model.
- Không có ResNet-18, MobileNetV3, ViT hoặc 3-channel Mel; vì vậy RQ3 và federated-backbone policy hiện tại chưa được thực thi.
- E1-E7 dùng compact 1-channel `AccentInvariantCNN`. Đây là một prototype hợp lệ để chứng minh GRL/FedAvg/FedProx chạy được, nhưng không phải FedECAI backbone theo policy mới.
- Tài liệu B1 chỉ có RQ1-RQ3; master plan yêu cầu RQ1-RQ6 và H1-H6.
- B4 tự đánh dấu “complete” theo checklist cũ dù thiếu accent probe và 3-channel Mel theo master plan mới.
- Các tài liệu B4 cũ từng mâu thuẫn về trạng thái full Wav2Vec2 extraction và feature-table hash; chúng đã được hợp nhất vào `docs/research/B4_SPEC.md`, lấy machine report hiện hành làm nguồn chuẩn.
- B3 notebook/pipeline mặc định vẫn dùng split v1, trong khi v2 là canonical cho modeling.
- Không có experiment registry và không report git commit, trái quy tắc “mọi experiment phải record config, seed, split version, git commit, outputs”.
- `README.md` hướng dẫn `.venv`/`.venv-gpu`; hai environment này không tồn tại trong checkout hiện tại. Environment dùng được là `DAP301m` Python 3.12.13/Torch 2.11.0+cu126, trong khi một số B4 docs ghi Python 3.14.3/Torch 2.14.0 CPU.
- Các deliverable `B*_SPEC.md`, `doc.md` lịch sử và `logs/*` được master plan yêu cầu nhưng không có trước audit.

## 6. Nguy cơ kỹ thuật và data leakage

### Đã kiểm soát tốt

- Split v2 không có speaker overlap và không có duplicate waveform trong eligible set.
- B6 prediction files không chứa test IDs; source runners lọc train/validation và report `test_samples_loaded=0`.
- Classical scaler nằm trong CV pipeline; deep scalers fit train/fold-train; log-Mel target frames lấy percentile từ train.
- B4 feature extraction là per-sample deterministic; không reuse PCA/full-data standardization trong model code.
- Augmentation chưa được triển khai, nên chưa có augmentation leakage.

### Nguy cơ còn lại

- Speaker-disjoint chỉ đúng theo metadata phát hành. Speaker ID `0` chiếm 2.217/4.992 mẫu (44,41%) và chưa được nhà phát hành xác nhận là một speaker thật hay catch-all ID.
- Speaker `0` làm một CV fold và một main client cực lớn; kết quả mean fold/client có thể bị méo dù không phải leakage trực tiếp.
- Central rất nhỏ. Validation Central-Angry chỉ có 1 mẫu/1 speaker, Central-Sad 3 mẫu/1 speaker; test Central-Sad 1 mẫu/1 speaker và Central-Angry 7 mẫu/1 speaker. Worst-accent/per-class estimates sẽ bất ổn; speaker-bootstrap CI là bắt buộc và có thể rất rộng.
- E1-E7 và paper-aligned Pitch-Fusion chọn best checkpoint dựa trên cùng validation set dùng để báo preliminary metric. Đây không phải test leakage, nhưng validation score có selection bias và không được dùng như final generalization result.
- B4 PCA/radar/heatmap được fit trên toàn eligible data. Hiện chúng chỉ dùng minh họa; nếu reuse cho feature/model selection sẽ gây leakage.
- Main client assignment rất lệch do speaker `0` (main client 01 có 2.858/3.681 training samples). Cần report sample-weighted và macro-client riêng.
- B6 source join feature table với toàn manifest trước khi filter train/validation ở một số runner. Audit không thấy test samples được đưa vào loader/training, nhưng regression tests nên khóa invariant này.
- Không có credential scan/repository security gate; kết luận “không có leaked secret” hiện chỉ là chưa phát hiện trong phạm vi audit, không phải chứng nhận.

## 7. Khả năng tái lập và mức an toàn của result/experiment cũ

### Có thể tiếp tục sử dụng an toàn

- Raw Parquet pinned checksum, `raw_manifest.csv`, `clean_manifest.csv`, audit/duplicate/quality reports.
- Split v2/enriched manifest và federated main/stress assignments. Đây nên tiếp tục là canonical data contract; không tạo lại split nếu không có protocol-change decision chính thức.
- B4 MFCC/eGeMAPS/log-Mel/pitch/Wav2Vec2 artifacts và feature table hiện tại cho các consumer cũ, sau khi dùng hash hiện tại `89ed3...` làm lineage mới. Các artifact này không thay thế 3-channel Mel contract.
- B3 descriptive findings không phụ thuộc model và B4 visualizations với nhãn **exploratory only**.
- Validation prediction CSVs B6 và metric reports như **preliminary one-seed validation evidence**: sample IDs đúng validation v2 và metric đã được recompute khớp.
- CNN/Wav2Vec2-MLP/pitch-ablation checkpoints và paper-aligned Pitch-Fusion checkpoint tồn tại; chỉ nên reuse sau khi ghi bổ sung checkpoint hash/config/environment/git lineage.

### Chỉ giữ làm lịch sử, không dùng làm final evidence

- 5-model consolidated comparison: thiếu ba model bắt buộc, latency/size và final test.
- E1-E7 CNN reports: hữu ích để chứng minh pipeline/GRL/federated mechanics, nhưng sai backbone policy mới, một seed, không accent probe.
- B4 full-data PCA/radar/heatmap: không được reuse như train transform hoặc evidence unseen-speaker.
- B3 split-coverage result v1: chỉ giải thích lý do tạo v2, không phải current canonical coverage.

### Chưa tái lập được đầy đủ

- SVM/XGBoost model artifacts được report tham chiếu nhưng `artifacts/b6/classical/svm_rbf.joblib` và `xgboost.joblib` hiện không tồn tại.
- Classical reports ghi feature hash cũ `22c98...`; current feature table là `89ed3...`. Config hash trong classical reports `7f93...` cũng không khớp current file hash `0c6ac...`.
- Toàn bộ E1-E7 checkpoints/progress files dưới `artifacts/b6/fedecai/` không tồn tại; report cũng không chứa config/manifest/checkpoint hashes, seed, split version hoặc git commit.
- Deep reports có input hashes nhưng không có checkpoint hash/path và git commit; exact binary-to-report binding chưa được chứng minh.
- Root logs không tạo thành chuẩn `logs/*`, không phải complete command/environment/commit registry; `log_b4_wav2vec.log` còn kết thúc giữa chunk 3 dù aggregate report nói full extraction hoàn tất.

## 8. Những phần không nên làm lại

- Không download hoặc sửa raw ViSEC; checksum đã đúng.
- Không audit/clean lại 5.280 rows nếu cleaning policy không đổi.
- Không overwrite split v1/v2 và không random-split audio; tiếp tục dùng frozen speaker-disjoint v2.
- Không extract lại 4.992 MFCC/eGeMAPS/log-Mel/pitch/Wav2Vec2 artifacts chỉ để “làm mới”; trước hết sửa lineage/docs và chỉ tạo feature mới cần cho 3-channel consumers.
- Không làm lại B3 charts hiện hữu; chỉ bổ sung các SQL requirement còn thiếu và chuyển canonical split reference sang v2 theo versioned change.
- Không xóa 5-model/E1-E7 validation evidence cũ; giữ với nhãn legacy/preliminary.
- Không mở hoặc tune trên 665-sample test set trước khi đủ 8 model, lambda/config được khóa và evaluation protocol hoàn tất.

## 9. Bước tiếp theo được đề xuất

Thực hiện một bước **B1 protocol reconciliation** trước mọi feature/model training khác:

1. Chuyển B1 từ 3 RQ/5 model sang đúng 6 RQ/8 model và ghi rõ H1-H6, research gap, contributions, owners, evidence mapping.
2. Ghi decision rằng các run CNN/E1-E7 hiện tại là legacy preliminary evidence, không phải benchmark/FedECAI cuối.
3. Khởi tạo experiment registry và log policy trước khi có run mới.
4. Chưa sửa model code trong bước này; sau khi protocol được review mới mở task B4 cho 3-channel Mel/accent probe rồi B6 cho ba model còn thiếu.

## 10. Chính xác các file nên sửa/tạo ở bước tiếp theo

Chỉ package B1 protocol reconciliation đề xuất:

- **Tạo:** `docs/research/B1_SPEC.md`
- **Đã hợp nhất sau audit:** `docs/research/B1_SPEC.md`, gồm scope, RQ/evidence và provenance đã xác minh.
- **Sửa:** `README.md` để loại bỏ phát biểu “five models” như current specification và ghi đúng status legacy/current.
- **Sửa:** `doc.md` chỉ nếu protocol reconciliation thay đổi phần mô tả human-visible; không thêm result chưa chạy.
- **Tạo:** `logs/project_log.md`
- **Tạo:** `logs/decision_log.md`
- **Tạo:** `logs/experiment_registry.csv` với schema trong master plan và chỉ backfill trường có bằng chứng; trường không biết để trống.
- **Tạo:** `logs/ai_audit_log.md`

Không nên sửa source/config/model/result trong bước kế tiếp này. Danh sách file cho task B4/B6 sau đó phải được xác định riêng sau khi B1 protocol reconciliation được duyệt.
