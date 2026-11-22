# Multi-Seed Ablation Condition-Stratified Cross-Dataset Robustness

| Dataset   | Condition               | Variant                         |   Sessions | Mean_Pearson     | Oracle_Lag_Pearson   | Mean_HR_MAE   | Mean_SNR_dB   |
|:----------|:------------------------|:--------------------------------|-----------:|:-----------------|:---------------------|:--------------|:--------------|
| MCD-rPPG  | Resting (before)        | SKIA-v2 Static Prior            |          5 | 0.5508 ± 0.0166  | 0.6418 ± 0.0018      | 1.10 ± 0.01   | 1.54 ± 0.16   |
| MCD-rPPG  | Post-Exercise (after)   | SKIA-v2 Static Prior            |          5 | 0.5617 ± 0.0070  | 0.6612 ± 0.0104      | 8.76 ± 1.55   | -1.40 ± 0.17  |
| MMPD      | Stationary (Low Motion) | SKIA-v2 Static Prior            |          4 | -0.0604 ± 0.0250 | 0.4764 ± 0.0024      | 1.38 ± 0.17   | -1.14 ± 0.18  |
| MMPD      | Rotation (Head Motion)  | SKIA-v2 Static Prior            |          4 | 0.0660 ± 0.0203  | 0.3417 ± 0.0084      | 31.08 ± 4.88  | -4.10 ± 0.71  |
| MMPD      | Talking + LED-low       | SKIA-v2 Static Prior            |          4 | 0.0294 ± 0.0043  | 0.3674 ± 0.0048      | 28.95 ± 1.98  | -5.72 ± 0.28  |
| MMPD      | Talking + LED-high      | SKIA-v2 Static Prior            |          4 | 0.0205 ± 0.0571  | 0.4663 ± 0.0239      | 5.21 ± 1.22   | -2.72 ± 0.31  |
| MCD-rPPG  | Resting (before)        | SKIA-v2 Gauss Off, λ=0          |          5 | 0.5777 ± 0.0310  | 0.6475 ± 0.0148      | 1.17 ± 0.09   | 1.39 ± 0.27   |
| MCD-rPPG  | Post-Exercise (after)   | SKIA-v2 Gauss Off, λ=0          |          5 | 0.5736 ± 0.0246  | 0.6667 ± 0.0133      | 7.47 ± 0.14   | -1.12 ± 0.62  |
| MMPD      | Stationary (Low Motion) | SKIA-v2 Gauss Off, λ=0          |          4 | -0.0318 ± 0.0227 | 0.4740 ± 0.0049      | 1.28 ± 0.11   | -1.23 ± 0.20  |
| MMPD      | Rotation (Head Motion)  | SKIA-v2 Gauss Off, λ=0          |          4 | 0.0781 ± 0.0078  | 0.3495 ± 0.0126      | 27.07 ± 2.81  | -4.23 ± 0.28  |
| MMPD      | Talking + LED-low       | SKIA-v2 Gauss Off, λ=0          |          4 | 0.0491 ± 0.0270  | 0.3710 ± 0.0156      | 30.40 ± 0.12  | -5.79 ± 0.45  |
| MMPD      | Talking + LED-high      | SKIA-v2 Gauss Off, λ=0          |          4 | 0.0471 ± 0.0319  | 0.4603 ± 0.0220      | 7.37 ± 2.02   | -3.00 ± 0.29  |
| MCD-rPPG  | Resting (before)        | SKIA-v2 Gauss Off, λ=0.05       |          5 | 0.5685 ± 0.0221  | 0.6434 ± 0.0016      | 1.16 ± 0.08   | 1.49 ± 0.23   |
| MCD-rPPG  | Post-Exercise (after)   | SKIA-v2 Gauss Off, λ=0.05       |          5 | 0.5741 ± 0.0370  | 0.6598 ± 0.0177      | 8.69 ± 1.75   | -1.37 ± 0.71  |
| MMPD      | Stationary (Low Motion) | SKIA-v2 Gauss Off, λ=0.05       |          4 | -0.0334 ± 0.0031 | 0.4845 ± 0.0044      | 1.20 ± 0.09   | -1.14 ± 0.31  |
| MMPD      | Rotation (Head Motion)  | SKIA-v2 Gauss Off, λ=0.05       |          4 | 0.0745 ± 0.0081  | 0.3534 ± 0.0125      | 29.02 ± 5.63  | -4.66 ± 0.81  |
| MMPD      | Talking + LED-low       | SKIA-v2 Gauss Off, λ=0.05       |          4 | 0.0513 ± 0.0292  | 0.3893 ± 0.0195      | 25.25 ± 4.53  | -5.61 ± 0.11  |
| MMPD      | Talking + LED-high      | SKIA-v2 Gauss Off, λ=0.05       |          4 | 0.0462 ± 0.0265  | 0.4654 ± 0.0210      | 6.67 ± 2.78   | -2.88 ± 0.29  |
| MCD-rPPG  | Resting (before)        | SKIA-v2 Gauss Off, λ=0.10       |          5 | 0.5675 ± 0.0124  | 0.6432 ± 0.0062      | 1.17 ± 0.08   | 1.51 ± 0.07   |
| MCD-rPPG  | Post-Exercise (after)   | SKIA-v2 Gauss Off, λ=0.10       |          5 | 0.5522 ± 0.0213  | 0.6462 ± 0.0163      | 8.52 ± 0.83   | -1.85 ± 0.42  |
| MMPD      | Stationary (Low Motion) | SKIA-v2 Gauss Off, λ=0.10       |          4 | -0.0286 ± 0.0090 | 0.4809 ± 0.0111      | 1.29 ± 0.08   | -1.14 ± 0.27  |
| MMPD      | Rotation (Head Motion)  | SKIA-v2 Gauss Off, λ=0.10       |          4 | 0.0555 ± 0.0088  | 0.3327 ± 0.0205      | 30.93 ± 4.45  | -5.35 ± 0.30  |
| MMPD      | Talking + LED-low       | SKIA-v2 Gauss Off, λ=0.10       |          4 | 0.0359 ± 0.0102  | 0.3759 ± 0.0065      | 23.74 ± 3.09  | -5.73 ± 0.24  |
| MMPD      | Talking + LED-high      | SKIA-v2 Gauss Off, λ=0.10       |          4 | 0.0470 ± 0.0122  | 0.4737 ± 0.0167      | 5.74 ± 3.16   | -2.74 ± 0.20  |
| MCD-rPPG  | Resting (before)        | SKIA-v2 Gauss On, λ=0           |          5 | 0.5734 ± 0.0333  | 0.6584 ± 0.0176      | 1.02 ± 0.11   | 1.43 ± 0.54   |
| MCD-rPPG  | Post-Exercise (after)   | SKIA-v2 Gauss On, λ=0           |          5 | 0.5584 ± 0.0374  | 0.6671 ± 0.0140      | 9.90 ± 1.55   | -1.18 ± 0.57  |
| MMPD      | Stationary (Low Motion) | SKIA-v2 Gauss On, λ=0           |          4 | -0.0242 ± 0.0156 | 0.4800 ± 0.0040      | 1.17 ± 0.09   | -1.24 ± 0.36  |
| MMPD      | Rotation (Head Motion)  | SKIA-v2 Gauss On, λ=0           |          4 | 0.0764 ± 0.0410  | 0.3550 ± 0.0167      | 22.71 ± 2.01  | -4.69 ± 1.21  |
| MMPD      | Talking + LED-low       | SKIA-v2 Gauss On, λ=0           |          4 | 0.0453 ± 0.0191  | 0.3842 ± 0.0039      | 27.47 ± 2.16  | -5.61 ± 0.20  |
| MMPD      | Talking + LED-high      | SKIA-v2 Gauss On, λ=0           |          4 | 0.0533 ± 0.0119  | 0.4755 ± 0.0183      | 5.26 ± 1.08   | -2.83 ± 0.31  |
| MCD-rPPG  | Resting (before)        | SKIA-v2 Gauss On, λ=0.05        |          5 | 0.5382 ± 0.0382  | 0.6317 ± 0.0151      | 1.11 ± 0.10   | 1.30 ± 0.57   |
| MCD-rPPG  | Post-Exercise (after)   | SKIA-v2 Gauss On, λ=0.05        |          5 | 0.5509 ± 0.0321  | 0.6513 ± 0.0054      | 8.76 ± 1.52   | -1.78 ± 0.33  |
| MMPD      | Stationary (Low Motion) | SKIA-v2 Gauss On, λ=0.05        |          4 | -0.0407 ± 0.0167 | 0.4733 ± 0.0076      | 1.34 ± 0.14   | -1.28 ± 0.25  |
| MMPD      | Rotation (Head Motion)  | SKIA-v2 Gauss On, λ=0.05        |          4 | 0.0645 ± 0.0144  | 0.3408 ± 0.0077      | 25.41 ± 4.40  | -4.45 ± 0.35  |
| MMPD      | Talking + LED-low       | SKIA-v2 Gauss On, λ=0.05        |          4 | 0.0392 ± 0.0159  | 0.3710 ± 0.0068      | 27.40 ± 2.22  | -5.76 ± 0.27  |
| MMPD      | Talking + LED-high      | SKIA-v2 Gauss On, λ=0.05        |          4 | 0.0309 ± 0.0346  | 0.4713 ± 0.0230      | 5.18 ± 1.03   | -2.65 ± 0.21  |
| MCD-rPPG  | Resting (before)        | SKIA-v2 Gauss On, λ=0.10 (Full) |          5 | 0.5799 ± 0.0212  | 0.6568 ± 0.0071      | 1.09 ± 0.06   | 1.63 ± 0.11   |
| MCD-rPPG  | Post-Exercise (after)   | SKIA-v2 Gauss On, λ=0.10 (Full) |          5 | 0.5563 ± 0.0302  | 0.6643 ± 0.0130      | 9.82 ± 1.61   | -1.30 ± 0.33  |
| MMPD      | Stationary (Low Motion) | SKIA-v2 Gauss On, λ=0.10 (Full) |          4 | -0.0326 ± 0.0121 | 0.4686 ± 0.0100      | 1.32 ± 0.07   | -1.37 ± 0.28  |
| MMPD      | Rotation (Head Motion)  | SKIA-v2 Gauss On, λ=0.10 (Full) |          4 | 0.0774 ± 0.0113  | 0.3446 ± 0.0141      | 24.95 ± 5.28  | -4.61 ± 0.98  |
| MMPD      | Talking + LED-low       | SKIA-v2 Gauss On, λ=0.10 (Full) |          4 | 0.0373 ± 0.0230  | 0.3741 ± 0.0147      | 33.22 ± 7.42  | -5.79 ± 0.40  |
| MMPD      | Talking + LED-high      | SKIA-v2 Gauss On, λ=0.10 (Full) |          4 | 0.0289 ± 0.0360  | 0.4727 ± 0.0138      | 5.25 ± 1.15   | -2.68 ± 0.08  |
| MCD-rPPG  | Resting (before)        | SKIA-v2 Upper-Face Only (K=3)   |          5 | 0.5408 ± 0.0379  | 0.6374 ± 0.0160      | 1.09 ± 0.09   | 1.41 ± 0.47   |
| MCD-rPPG  | Post-Exercise (after)   | SKIA-v2 Upper-Face Only (K=3)   |          5 | 0.5339 ± 0.0388  | 0.6620 ± 0.0200      | 11.43 ± 5.45  | -1.41 ± 0.69  |
| MMPD      | Stationary (Low Motion) | SKIA-v2 Upper-Face Only (K=3)   |          4 | -0.0702 ± 0.0302 | 0.4748 ± 0.0023      | 1.33 ± 0.10   | -1.20 ± 0.18  |
| MMPD      | Rotation (Head Motion)  | SKIA-v2 Upper-Face Only (K=3)   |          4 | 0.0437 ± 0.0365  | 0.3407 ± 0.0140      | 27.27 ± 5.38  | -4.51 ± 0.61  |
| MMPD      | Talking + LED-low       | SKIA-v2 Upper-Face Only (K=3)   |          4 | 0.0089 ± 0.0344  | 0.3661 ± 0.0064      | 28.97 ± 2.21  | -5.86 ± 0.43  |
| MMPD      | Talking + LED-high      | SKIA-v2 Upper-Face Only (K=3)   |          4 | 0.0120 ± 0.0553  | 0.4637 ± 0.0314      | 5.08 ± 1.06   | -2.71 ± 0.38  |
