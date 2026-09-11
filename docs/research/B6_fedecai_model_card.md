# B6 FedECAI Preliminary Model Card

## Intended use

The E1-E7 ladder compares centralized, adversarial, and simulated federated
Vietnamese speech-emotion classifiers under the frozen B5 speaker-disjoint
protocol. It is research evidence only and must not be used for psychological
or clinical assessment.

## Evidence status

- 3 centralized experiments ran for 30 epochs.
- 8 federated experiment/scenario combinations ran for 20 rounds.
- Every report contains 646 validation predictions.
- The 665-sample test partition remains untouched.
- Results currently use one seed and are preliminary.

## Current finding

At the default adversarial lambda of 0.10, FedECAI does not improve FedAvg
Macro-F1 or worst-accent Macro-F1 in both client scenarios. This is a negative
preliminary result, not a failure of the pipeline and not a final rejection of
the hypothesis. B8 must tune lambda within the predeclared utility budget.

## Limitations

- Virtual clients simulate federated learning; they are not physical devices.
- Speaker 0 contributes 2,217 samples and causes unavoidable client-size skew.
- The shared CNN is a compact log-Mel baseline, not a Wav2Vec2 FedECAI model.
- Validation is used for checkpoint selection; test is reserved for final use.
- Accent invariance has not yet been measured with a frozen accent probe.
- Bootstrap confidence intervals, effect sizes, and multi-seed variance are pending.
- Federated learning alone does not guarantee privacy; no differential privacy
  or secure aggregation is implemented.
