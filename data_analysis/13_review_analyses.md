# Analyses for the 2026-10-23 review

## Self-use decision against the human ground truth

- stage1: sensitivity 86.1% (334/388), PPV 93.3% (334/358), specificity 96.9%, NPV 93.3%
- stage3: sensitivity 98.7% (383/388), PPV 99.0% (383/387), specificity 99.5%, NPV 99.4%

## Runs scored against the verified ground truth

- stage4 (1168 records): self-use accuracy 98.3%, presence accuracy 95.6% (n=383), indication accuracy 87.5% (n=384); medical label right 201/205, cosmetic label right 110/112
- stage2 (1165 records): self-use accuracy 93.3%, presence accuracy 88.9% (n=380), indication accuracy 81.4% (n=381); medical label right 184/193, cosmetic label right 110/113
- stage3 (1168 records): self-use accuracy 99.2%, presence accuracy 96.1% (n=383), indication accuracy 91.4% (n=384); medical label right 205/205, cosmetic label right 115/115
- stage1 (1168 records): self-use accuracy 93.3%, presence accuracy 92.7% (n=383), indication accuracy 82.6% (n=384); medical label right 189/194, cosmetic label right 110/112

## Missed values in optional fields (null judged wrong)

- Products mentioned: 0/0 (n/a)
- Indication subtype: 28/67 (41.8%)
- Side-effect list: 5/247 (2.0%)
- Attitude toward side effect: 3/247 (1.2%)
- Treatment area: 8/159 (5.0%)
- Dose (units): 3/341 (0.9%)
- Provider type: 16/314 (5.1%)
- First or repeat treatment: 16/234 (6.8%)
- Prior treatment count: 1/334 (0.3%)
- Time to onset: 5/314 (1.6%)
- Duration status: 7/258 (2.7%)
- Sought medical attention: 1/332 (0.3%)
- Discontinuation intent: 21/346 (6.1%)
- Age range: 1/294 (0.3%)
- Sex: 9/328 (2.7%)

## Unknown-indication stratum

- 5,856 submissions, 2,735 with a side effect; r/noburp 1,775 (1,389 with a side effect); outside r/noburp 33.0%
- verified split: {'noburp': {'stays_unknown': 13, 'medical': 8}, 'other': {'medical': 7, 'stays_unknown': 19, 'undetermined': 3, 'cosmetic': 14}}
- r/noburp unknowns to medical: +0.78 pp (-0.27 to +1.82)
- verified reallocation: median -1.14 pp (-1.88 to -0.37)

## Yield under the cleaned vocabulary

- any side effect 14,120 -> 13,165; multiple 7,886 -> 7,276
- reports whose only terms were excluded, by arm: {'medical': 545, 'cosmetic': 254, 'unknown': 156}

## Divergence bootstrap

- JSD 0.372; bootstrap mean 0.382; percentile 0.363-0.403; basic 0.342-0.381

## Author re-reading (private adjudication file)

- false positives re-read: 9; reclassified as true positives: {}
- PPV medical 90.4%, cosmetic 95.0%; corrected difference -6.38 pp (-11.32 to -1.68)
- unknown split {'noburp': {'stays_unknown': 13, 'medical': 8}, 'other': {'medical': 8, 'stays_unknown': 20, 'cosmetic': 14, 'undetermined': 1}}; reallocation median -1.15 pp (-1.90 to -0.38)
