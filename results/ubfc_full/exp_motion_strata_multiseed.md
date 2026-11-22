# Multi-Seed Mouth-Motion Stratified Evaluation (UBFC Test Split, Seeds 42, 43, 44)

| Variant                         | Stratum           |   Subjects | Mean_Pearson    | Mean_HR_MAE   | Mean_SNR_dB   |
|:--------------------------------|:------------------|-----------:|:----------------|:--------------|:--------------|
| SKIA-v2 Static Prior            | Low Mouth Motion  |          5 | 0.8725 ± 0.0384 | 0.27 ± 0.02   | 5.67 ± 0.05   |
| SKIA-v2 Static Prior            | High Mouth Motion |          5 | 0.8506 ± 0.0205 | 0.10 ± 0.02   | 7.12 ± 0.06   |
| SKIA-v2 Gauss Off, λ=0          | Low Mouth Motion  |          5 | 0.8920 ± 0.0236 | 0.26 ± 0.04   | 5.58 ± 0.12   |
| SKIA-v2 Gauss Off, λ=0          | High Mouth Motion |          5 | 0.8584 ± 0.0380 | 0.10 ± 0.04   | 7.16 ± 1.01   |
| SKIA-v2 Gauss Off, λ=0.05       | Low Mouth Motion  |          5 | 0.8836 ± 0.0233 | 0.26 ± 0.04   | 5.61 ± 0.11   |
| SKIA-v2 Gauss Off, λ=0.05       | High Mouth Motion |          5 | 0.8507 ± 0.0059 | 0.11 ± 0.01   | 7.10 ± 0.80   |
| SKIA-v2 Gauss Off, λ=0.10       | Low Mouth Motion  |          5 | 0.8891 ± 0.0224 | 0.23 ± 0.02   | 5.71 ± 0.05   |
| SKIA-v2 Gauss Off, λ=0.10       | High Mouth Motion |          5 | 0.8816 ± 0.0125 | 0.08 ± 0.01   | 7.91 ± 0.12   |
| SKIA-v2 Gauss On, λ=0           | Low Mouth Motion  |          5 | 0.8657 ± 0.0344 | 0.24 ± 0.05   | 5.55 ± 0.25   |
| SKIA-v2 Gauss On, λ=0           | High Mouth Motion |          5 | 0.8492 ± 0.0200 | 0.11 ± 0.03   | 7.27 ± 1.05   |
| SKIA-v2 Gauss On, λ=0.05        | Low Mouth Motion  |          5 | 0.8830 ± 0.0222 | 0.26 ± 0.05   | 5.65 ± 0.04   |
| SKIA-v2 Gauss On, λ=0.05        | High Mouth Motion |          5 | 0.8758 ± 0.0089 | 0.10 ± 0.02   | 7.66 ± 0.57   |
| SKIA-v2 Gauss On, λ=0.10 (Full) | Low Mouth Motion  |          5 | 0.8792 ± 0.0152 | 0.28 ± 0.09   | 5.59 ± 0.13   |
| SKIA-v2 Gauss On, λ=0.10 (Full) | High Mouth Motion |          5 | 0.8703 ± 0.0173 | 0.08 ± 0.00   | 7.55 ± 0.79   |
| SKIA-v2 Upper-Face Only (K=3)   | Low Mouth Motion  |          5 | 0.8744 ± 0.0105 | 0.24 ± 0.01   | 5.70 ± 0.03   |
| SKIA-v2 Upper-Face Only (K=3)   | High Mouth Motion |          5 | 0.8708 ± 0.0080 | 0.09 ± 0.01   | 7.55 ± 0.26   |
