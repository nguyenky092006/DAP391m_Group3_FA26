# B5 Speaker Disjoint Split V2

The frozen `speaker_disjoint_v2` manifest is the common partition for M1-M4.
It contains train, validation, test, and excluded records. Train, validation,
and test speakers are disjoint, and seed 391 is recorded in the manifest.

Training speakers also have deterministic grouped-CV labels. M1 uses these
folds only for diagnostic Macro-F1 and UAR under a fixed SVM configuration.
Hyperparameter tuning remains a B8 activity. M2-M4 use the frozen train and
validation partitions and do not use test samples during B6.

The historical enriched CSV may retain a `federated_client_id` column from the
previous repository direction. M1-M4 never read that column; it is retained
only to avoid rewriting the frozen split artifact.
