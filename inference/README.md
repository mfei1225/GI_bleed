# GI-bleed inference

This folder scores a study that has already been through `preprocessing`. It writes one row per slice. Patient scores are computed by `aggregation`. It does not register, segment, or train.

The checkpoints are not in git. Put them here before running:

```
weights/maxvit/maxvit_fold0.pth … maxvit_fold4.pth      (~117 MB each)
weights/detector/detector_fold0.pth … detector_fold4.pth (~466 MB each)
```

Inference loads every `*.pth` in each folder in sorted order (fold 0–4).

```bash
pip install -r requirements.txt
```

Needs a GPU. Python 3.10+.

## What it does

Two models score every axial slice. This folder saves those slice scores. `aggregation` keeps the GI slices when it builds the patient score.

### MaxViT

One axial slice at a time, `maxvit_tiny_tf_512` at 512×512. The three channels are the non-contrast difference used in training:

- channel 0 = (venous − non-contrast + 1) / 2
- channel 1 = venous
- channel 2 = (arterial − non-contrast + 1) / 2

Pixel values are the soft-tissue window, scaled to [0, 1]. Mid-gray on a difference channel means that phase looks like non-contrast. If non-contrast or arterial is missing, that difference channel is zero.

The network is an 11-class CT injury head. The bleed score is class 1 (extravasation), after a sigmoid. The five fold checkpoints are averaged. Each scored slice keeps that probability. Turning the slices into one patient number is done in `aggregation`.

### Detector

Faster R-CNN with a ResNet-50 feature pyramid. Nine channels, in this order:

```
venous z, venous z-1, venous z+1,
noncontrast z, noncontrast z-1, noncontrast z+1,
arterial z, arterial z-1, arterial z+1
```

Each channel is the same soft-tissue window. The body is cropped from the venous slice, that crop is applied to every channel, the result is padded to a square, and then resized to 512. Differences between phases are computed inside the network (venous−noncontrast, arterial−noncontrast, arterial−venous).

Every box the model keeps is saved. Boxes scoring below 0.1 are already dropped. A slice score, used later for the patient number, is still the highest box score on that slice, averaged over the five folds.

The detector reads `slices/`, not the volume cache. The cache is already resized to 512 without the body crop, so it is the wrong input for this model.

## Input

Point `--data` at the preprocessing output:

```
<preprocessed>/
  volcache/s512/<Subject>/<Accession>/
    Venous.npy
    NonCon.npy
    Arterial.npy
  slices/
    <Subject>/<Accession>/
      Venous/0000.dcm
      NonCon/0000.dcm
      Arterial/0000.dcm
```

Each `.npy` is `uint8`, shape `(nz, 512, 512)`. 0 is −160 HU or lower, 255 is 240 HU or higher. The MaxViT divides by 255.

## How to run it on a new dataset

Preprocess first, then:

```bash
python infer.py \
  --data /data/site/preprocessed \
  --out /data/site/results
```

The weights next to this script are used unless you pass `--weights` pointing at a folder that contains `maxvit/` and `detector/`. Pass `--subjects` to score a few ids.

## Speed

Defaults match the full run: every slice, all five folds, both models, MaxViT batches of 8, detector batches of 1. On a GPU the forward pass uses fp16. These flags change that:

| Flag | Default | What to raise |
|---|---|---|
| `--batch` | 8 | MaxViT slices per forward. Try 16 or 32 on a 40 GB GPU |
| `--detector-batch` | 1 | Detector slices per forward. This is the slow one. Try 4, then 8 |
| `--stride` | 1 | `2` scores every other slice |
| `--n-folds` | all | `1` loads only the first checkpoint |
| `--models` | `both` | `maxvit` or `detector` skips the other network |
| `--no-amp` | fp16 on | Turns off fp16 |

If the process runs out of GPU memory, lower `--batch` or `--detector-batch`. `--stride` and `--n-folds` change which slices and checkpoints are scored, so the patient number will not match a full run.

## Output

`--out` is one folder:

```
results/
  slices.csv
  boxes.csv
```

`aggregation` reads that folder and adds `scores.csv`.

`slices.csv`, one row per scored slice:

| Column | Meaning |
|---|---|
| `Subject` | Patient id |
| `Accession` | Study that was registered |
| `nz` | Slices in the venous volume |
| `z` | Slice index, the same index as `0000.dcm` |
| `maxvit_prob` | Fold-averaged MaxViT extravasation probability |
| `detector_score` | Fold-averaged highest box score on that slice |

`boxes.csv` has one row per detection:

| Column | Meaning |
|---|---|
| `Subject`, `Accession`, `z` | Which slice |
| `fold` | Which of the five checkpoints, 0–4 |
| `label` | Class id. 1 is the bleed class |
| `score` | Box score |
| `x_min`, `y_min`, `x_max`, `y_max` | Box corners in pixels of that venous DICOM slice |

The corners are on the original slice, after undoing the body crop and the resize to 512. `x` runs across the image and `y` runs down it, the same axes as the DICOM pixel array.

Pass this folder to `aggregation`. It reads `slices.csv` and writes `scores.csv` beside it. `boxes.csv` is not used for the patient score.
