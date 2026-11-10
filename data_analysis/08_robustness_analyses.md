# Robustness analyses (08)

## Leave-out analyses

| Variant | Removed med/cos | Prevalence diff pp (95% CI) | JSD (terms) | JSD p |
|---|---:|---:|---:|---:|
| without r/migraine | 4,801/58 | -0.04 (-1.15 to +1.08) | 0.408 (208) | 0.0004998 |
| without r/30plusskincare | 240/3,122 | -1.13 (-2.27 to +0.01) | 0.365 (206) | 0.0004998 |
| without r/noburp | 2,588/9 | -8.14 (-9.19 to -7.08) | 0.273 (191) | 0.0004998 |
| without r/plasticsurgery | 366/2,117 | -1.70 (-2.81 to -0.60) | 0.378 (209) | 0.0004998 |
| without r/tmj | 1,628/100 | -2.37 (-3.43 to -1.30) | 0.388 (216) | 0.0004998 |
| without r/botoxsupportcommunity | 189/921 | -0.25 (-1.31 to +0.81) | 0.386 (210) | 0.0004998 |
| without r/askdocs | 983/76 | -1.86 (-2.93 to -0.80) | 0.379 (214) | 0.0004998 |
| without r/analfissures | 795/4 | -3.71 (-4.76 to -2.65) | 0.364 (212) | 0.0004998 |
| without r/hyperhidrosis | 732/5 | -2.57 (-3.63 to -1.52) | 0.375 (220) | 0.0004998 |
| without r/skincareaddiction | 45/423 | -3.02 (-4.09 to -1.96) | 0.375 (217) | 0.0004998 |
| without r/beauty | 37/325 | -2.86 (-3.92 to -1.80) | 0.373 (219) | 0.0004998 |
| without r/dystonia | 358/0 | -2.78 (-3.83 to -1.73) | 0.372 (221) | 0.0004998 |
| without r/diyaesthetics | 28/268 | -2.95 (-4.01 to -1.90) | 0.374 (220) | 0.0004998 |
| without r/chronicpain | 262/1 | -2.57 (-3.62 to -1.52) | 0.372 (221) | 0.0004998 |
| without r/pelvicfloor | 253/0 | -2.83 (-3.88 to -1.78) | 0.371 (219) | 0.0004998 |
| without r/cosmeticsurgery | 34/218 | -2.73 (-3.79 to -1.68) | 0.371 (221) | 0.0004998 |
| without r/interstitialcystitis | 251/1 | -2.83 (-3.88 to -1.78) | 0.370 (220) | 0.0004998 |
| without r/vaginismus | 247/0 | -2.89 (-3.94 to -1.84) | 0.372 (219) | 0.0004998 |
| without r/diycosmeticprocedures | 24/223 | -2.90 (-3.95 to -1.84) | 0.372 (219) | 0.0004998 |
| without r/45plusskincare | 13/204 | -3.05 (-4.11 to -2.00) | 0.372 (221) | 0.0004998 |
| without harm-focused support community (r/BotoxSupportCommunity) | 189/921 | -0.25 (-1.31 to +0.81) | 0.386 (210) | 9.999e-05 |
| without DIY-injection communities (r/DIYAesthetics, r/DIYCosmeticProcedures) | 52/491 | -3.07 (-4.13 to -2.01) | 0.373 (218) | 9.999e-05 |
| without user-profile pages | 104/120 | -2.82 (-3.88 to -1.77) | 0.372 (219) | 9.999e-05 |
| without r/noburp and r/BotoxSupportCommunity together | 2,777/930 | -5.71 (-6.77 to -4.64) | 0.286 (180) | 9.999e-05 |

## Author-clustered bootstrap

23,110 clusters (22,953 named authors), 2,000 replicates.
Prevalence difference -2.79 pp, clustered 95% CI -4.08 to -1.44; design effect 1.59; clustered P 3.62e-05.
JSD 0.372 (221 terms), clustered 95% CI 0.363 to 0.403.
Per-term: 159 of 221 significant at BH q<0.05 with clustered SEs; 162 with a clustered percentile CI excluding zero.

## Random one-submission-per-author draws

200 draws of 23,110 records: prevalence difference median -1.93 pp (2.5-97.5%: -2.29 to -1.54); negative in 100.0% of draws; CI excludes zero in 100.0%. JSD median 0.367 (0.363 to 0.373).

## Calendar periods

| Variant | Med/cos n | Prevalence diff pp (95% CI) | JSD (terms) | JSD p |
|---|---:|---:|---:|---:|
| 2015-2019 | 1,748/348 | -0.87 (-6.19 to +4.04) | 0.383 (9) | 0.0004998 |
| 2020-2022 | 4,539/1,443 | -2.82 (-5.67 to -0.01) | 0.347 (49) | 0.0004998 |
| 2023-2025 | 14,524/10,925 | -2.37 (-3.55 to -1.18) | 0.369 (196) | 0.0004998 |
| before July 2023 | 7,668/2,464 | -3.14 (-5.30 to -1.01) | 0.361 (77) | 0.0004998 |
| from July 2023 | 13,143/10,252 | -2.24 (-3.47 to -1.01) | 0.369 (186) | 0.0004998 |

## Within-site contrasts

| Variant | Med/cos n | Prevalence diff pp (95% CI) | JSD (terms) | JSD p |
|---|---:|---:|---:|---:|
| chronic migraine vs forehead or glabellar lines | 6,857/3,956 | -22.05 (-23.88 to -20.22) | 0.297 (140) | 0.0004998 |
| blepharospasm vs lateral canthal lines | 33/607 | no term in at least 10 reports in both groups | | |
| temporomandibular disorder vs masseter or jawline | 3,344/1,593 | -11.61 (-14.52 to -8.71) | 0.123 (82) | 0.0004998 |
| cervical dystonia vs neck or platysma | 328/147 | -0.80 (-10.01 to +7.86) | 0.202 (5) | 0.004498 |

## Vocabulary variants

| Variant | Med/cos n | Prevalence diff pp (95% CI) | JSD (terms) | JSD p |
|---|---:|---:|---:|---:|
| synonym merges | 20,811/12,716 | -2.79 (-3.84 to -1.74) | 0.367 (180) | 9.999e-05 |
| synonym merges and symmetric exclusions | 20,811/12,716 | -3.41 (-4.44 to -2.38) | 0.338 (162) | 9.999e-05 |
