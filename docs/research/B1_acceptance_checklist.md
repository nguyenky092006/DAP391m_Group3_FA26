# B1 acceptance checklist

Status: **FROZEN — team-approved on 2026-09-08**

Use this checklist before starting B2 implementation decisions.

## Research framing

- [x] Working English and Vietnamese titles are defined.
- [x] The target is five-class Vietnamese speech emotion classification.
- [x] The three linked problems—leakage, view stability, and federated
  heterogeneity—are stated.
- [x] The proposed solution is connected directly to those problems.
- [x] The problem statement does not reduce the project to selecting the model
  with the highest Accuracy.

## Research questions

- [x] Three Basic RQs are defined.
- [x] Three Extension RQs are defined.
- [x] Every RQ has a falsifiable hypothesis.
- [x] Every RQ maps to one named experiment.
- [x] Every RQ has primary metrics and required evidence.
- [x] Protocol sensitivity is separated from main clean-protocol performance.
- [x] Extra processed data is separated from the proposed PVC mechanism through
  a four-way ablation.

## Contributions and novelty

- [x] Three scientific contributions are defined.
- [x] Course requirements are not mislabeled as scientific contributions.
- [x] The project does not claim to be the first federated SER study.
- [x] The research gap is labeled as working rather than proven novelty.
- [x] The literature matrix contains at least three papers.
- [x] The current matrix includes primary sources for VESC, FL baselines,
  federated SER, and three comparison architectures.
- [ ] A systematic search log and explicit inclusion/exclusion criteria are
  completed before the final novelty claim.

## Scope and claim safety

- [x] Regional SER is future work.
- [x] Formal privacy is future work.
- [x] Cross-dataset/real-world validation is future work.
- [x] Medical-diagnosis language is prohibited.
- [x] `_vocals` is described conservatively as a processed/denoised view.
- [x] OrionNet-LM is described as an adaptation.
- [x] Neutral's lack of paired families is disclosed in pair claims.
- [x] FedBN requires an explicit test-time BN policy.

## Approval gate

- [x] The project owner approves the exact six RQ wordings.
- [x] The project owner approves the three hypotheses.
- [x] The project owner approves the in-scope/out-of-scope boundary.
- [ ] The supervisor/instructor has no requested framing changes.
- [x] B1 is marked `FROZEN` with the approval date.

Do not silently rewrite a frozen RQ after seeing model results. If an RQ must
change, record the date, reason, expected impact, and approving reviewers.
