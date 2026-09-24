# B1 Research Question Evidence Matrix

| Research question | Required comparison | Main evidence | Decision boundary |
|---|---|---|---|
| RQ1 baseline reproduction | Published setup versus reproduced M3 where comparable | M3 Macro-F1, UAR, Accuracy, per-class F1, reproduction differences | Report gaps; do not silently change the baseline |
| RQ2 feature gain | M3 pitch-only versus M4 five-channel multi-prosody | Same split, seed, Wav2Vec2, Conv1D, fusion, classifier, loss, optimizer, batch size, and epochs | Attribute the difference only to the prosody input |
| RQ3 architecture gain | M4 versus a later human-approved M5 | Performance, parameter count, runtime, and latency | Do not implement M5 before human approval |

Accent, gender, and duration breakdowns are supporting evidence only. They do
not create new research questions and are not model inputs in the main study.
