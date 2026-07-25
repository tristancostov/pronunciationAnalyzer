# Text-independent nucleus detection: V3 progress

## Data-integrity correction

The original 3local file mixed two states: the first 0–17.35 s had been
manually nucleus-reviewed, while later automatic points had not. Earlier V3
experiments ran before that distinction was encoded, so their JSON files were
renamed with `_PROVISIONAL_PSEUDOLABEL` and are not reported as measured
accuracy. The corrected partial ground truth keeps
94 reviewed points as `N` and marks the 95 later automatic points as `?`, which
all evaluators ignore.

## Intended evaluation protocol

- Metric: event precision/recall/F1 with a ±50 ms matching tolerance.
- Data after correction: 6fori and 7fori are fully reviewed; the reviewed
  0–17.35 s region of 3local is used and the rest is ignored.
- Split: nested leave-one-recording/speaker-out. Event thresholds and fusion
  distance are selected only on the other speakers.
- 8fori was sealed until the four-recording configuration and model artifact
  were frozen. Its predictions were written before its reviewed TextGrid was
  first read.

## Verified results after the 3local correction

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| Frozen acoustic V2 | 91.24% | 76.85% | 83.42% |
| First V2-priority + classified Sylber fusion | 86.69% | 89.60% | 88.12% |
| Retuned text-independent Sylber SVM + V2 | 87.66% | 90.60% | **89.11%** |
| Multi-scale Sylber candidate consensus | 87.42% | 88.59% | 88.00% |
| Top-three complete-detector ensemble | 88.37% | 89.26% | 88.81% |
| Automatic-ASR hard word quota | 91.40% | 85.57% | 88.39% |
| **Automatic-ASR conservative over-count pruning** | **89.40%** | **90.60%** | **90.00%** |

The verified development set contains 94 manually reviewed 3local nuclei plus
the reviewed 6fori and 7fori nuclei, 298 events in total. Evaluation remains
nested leave-one-recording-out at ±50 ms.

The selected 90.00% mode did not use user-provided or expected text. It first
runs the acoustic detector, then uses the application's own ASR word times only
to remove the lowest-quality event when a word contains more events than its
recognized spelling's syllable count plus one. Events outside recognized words
are preserved. This mode is therefore direct recognition but is reported
separately from the strictly text-independent 89.11% result.

The three outer folds independently selected zero word-boundary padding and one
extra allowed nucleus. Their F1 scores were 88.89%, 90.23% and 90.73%.
This remains the historical three-recording development result.

## New-speaker round and frozen blind result

The manually reviewed 17.43-second 2fori chunk added 60 independently checked
nuclei, taking development data to four recordings and 358 events. It exposed
that old ASR JSON files assign `syllableCount = 1` to consonant-only tokens such
as `в` and `с`. The corrected pruner counts Russian vowels in the application's
own recognized token, gives vowel-less tokens a zero quota, and requires no
user transcript. Nested leave-one-recording-out results are:

| Four-recording method | Precision | Recall | F1 |
|---|---:|---:|---:|
| Acoustic V3 before ASR pruning | 87.88% | 89.11% | 88.49% |
| ASR-token vowel-count pruning | **92.63%** | 87.71% | **90.10%** |

All four outer folds selected zero boundary padding and zero extra events. The
held-out 2fori fold improved from 83.61% to 85.71% after the vowel-less-token
fix. The final model was then trained on all four development recordings and
frozen with SHA-256
`70fc22d9a92f63e49c19b38f46fdb4d90d3f280a7f4070ba50675875fb192717`.

The one-shot 8fori prediction was saved before reading its manual labels. The
blind result at ±50 ms was TP=84, FP=20, FN=19, precision 80.77%, recall
81.55%, and **F1 81.16%**. At ±80 ms F1 was 87.92%; predicted and reference
counts were 104 and 103. This shows that the development result did not fully
generalize and that center localization plus independently positioned labels
remain the main limitation. 8fori is no longer a blind holdout and must not be
used for another "blind" claim.

## Reusable V3 code

- `acoustic_nuclei_v3.py`: train/detect CLI and optional conservative ASR prune.
- `models/nuclei_v3_svm.joblib`: frozen model trained on four reviewed
  development recordings; 8fori is excluded from training and selection.
- `models/nuclei_v3_svm_3speakers.joblib`: archived earlier artifact.
- `requirements-research.txt`: explicit V3 research dependencies.
- `experiment_nuclei_v3_consensus.py`,
  `experiment_nuclei_v3_model_ensemble.py`,
  `experiment_nuclei_v3_asr_assisted.py`, and
  `experiment_nuclei_v3_asr_pruner.py`: reproducible positive and negative
  experiments.

## Superseded provisional results (old 3local pseudo-label positions)

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| Frozen acoustic V2 | 91.0% | 81.0% | 85.7% on prior 7fori holdout |
| Permissive peak-shape classifier | 89.3% | 81.5% | 85.3% |
| Sylber zero-shot | 85.2% | 88.6% | 86.8% |
| Sylber segment + acoustic ExtraTrees | 88.1% | 89.3% | 88.7% |
| Mel/MFCC temporal frame regression | 78.7% | 80.5% | 79.6% |
| V2-priority + classified Sylber fusion | 88.8% | 90.6% | 89.7% |

These values remain only as experiment history and cannot establish that the
detector reached 89.7%. The verified replacement score is 88.12% above.

## Literature ideas tested

- de Jong & Wempe: intensity peaks, dips and voicing as a transparent baseline.
- Landsiedel et al.: temporally contextualized perceptual/cepstral features and
  a smooth frame target.
- Yarra et al.: classify full peak/mode shape instead of peak height alone.
- Russian vowel-centred work: 250–2000 Hz energy, spectral entropy and ZCR.
- Sylber: textless self-supervised syllabic regions, fused with Russian acoustic
  evidence rather than copied as a complete solution.

## Next data round

Nine existing recordings have been preannotated: five files named `local` and
four named `fori`, about 12.2 minutes and 2486 proposed points. These filenames
must not be treated as native-language labels; speaker background needs explicit
metadata from the user. Additional development data should be manually centered
instead of accepting unchanged automatic timestamps, because auto-seeded labels
can make ±50 ms evaluation optimistic. 8fori may now be used for error analysis,
but a different, untouched speaker recording is required for the next final
blind test.
