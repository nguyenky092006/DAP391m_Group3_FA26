# B4 Multi Prosody Feature Specification

The M3 and M4 temporal feature extractor uses mono 16 kHz waveform input with
a 1,024-sample frame and 160-sample hop.

| Channel | Definition | M3 | M4 |
|---|---|---:|---:|
| log-F0 | Natural logarithm of interpolated positive pYIN F0 | Yes | Yes |
| Voicing confidence | pYIN voiced probability, clipped to 0 through 1 | No | Yes |
| log-energy | Natural logarithm of frame RMS energy | No | Yes |
| delta log-F0 | First temporal delta of log-F0 | No | Yes |
| delta log-energy | First temporal delta of log-energy | No | Yes |

Artifacts are extracted independently per utterance. Channel mean and standard
deviation are fitted from training frames only and then applied unchanged to
validation and test. The stored tensor order is `[5, T_p]`; batching adds the
leading batch dimension. Before fusion, every channel is aligned to the
waveform timeline and passed through the same temporal Conv1D encoder used by
M3. M4 changes only the number of input channels.
