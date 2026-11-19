# Detectron2 — Renal Tubule & Immune Cell Segmentation

Training script for instance segmentation of **renal tubules and immune cells in immunostained
kidney biopsies**, built on [Detectron2](https://github.com/facebookresearch/detectron2)'s
Mask R-CNN. Part of a clinical AI diagnostic tool for **lupus nephritis** developed at the
**University of Houston Mohan Lab** under Dr. Chandra Mohan and Dr. Anto Sam Crosslee.

The trained model reached **~90% segmentation accuracy on 3,000+ labeled kidney biopsy images**.
Presented at **ASN Kidney Week 2025**; manuscript in progress for the *Journal of Imaging*.

> **Scope of this repo.** This is the model training and evaluation script only.
> The biopsy dataset, trained weights, and full quantitative results belong to the lab and
> are not published here. See **[TRIMS-results](https://github.com/SuchayK/TRIMS-results)** for
> the method write-up, or reach me directly: suchayk@utexas.edu

## Why lupus nephritis

Lupus nephritis is kidney inflammation caused by systemic lupus erythematosus. Grading it means
a pathologist counting immune-cell infiltration and assessing tubule damage across a biopsy
slide — slow, and variable between annotators. Automating the count is the point of the tool.

The task is framed as **instance** segmentation rather than semantic segmentation because the
clinical question is *how many* cells and *which tubule* they sit in. A semantic mask merges a
cluster of adjacent immune cells into one region, which destroys exactly the count that matters.
Mask R-CNN emits one mask per object, so touching cells stay separable.

## How it works

| | |
|---|---|
| **Model** | Mask R-CNN, ResNet-50 FPN backbone (`COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x`) |
| **Init** | COCO-pretrained weights from the Detectron2 model zoo, fine-tuned on the biopsy set |
| **Data** | COCO-format instance annotations (`json_annotation_train.json` / `_val.json`) registered via `register_coco_instances`. The ROI head's class count is read back from the registered metadata, so the head always matches the annotations. |
| **Augmentation** | Applied during training for robustness to stain and illumination variation across slides |
| **Training** | 1000 iterations · base LR 2.5e-4 · 2 images/batch · 128 ROIs per image |
| **Inference** | Score threshold 0.7 for the qualitative overlay |
| **Evaluation** | `COCOEvaluator` on the held-out validation split (COCO AP / AP50 / AP75), at score threshold 0.05 — COCO AP integrates precision across the full recall range, so evaluating at the 0.7 display threshold would truncate the PR curve and understate AP |
| **Environment** | GPU runtime. Runs anywhere; the inline Colab preview is used only if `google.colab` is importable, otherwise the overlay is written to `output/prediction.png` |

Fine-tuning from COCO rather than training from scratch matters when the annotated set is a few
thousand images: the backbone already knows edges and blobs, so only the head has to learn what
a tubule and a lymphocyte look like.

## Running it

Requires a GPU runtime with Detectron2 installed:

```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

The script expects a COCO-format dataset laid out as:

```
cells/
  train/  images + json_annotation_train.json
  val/    images + json_annotation_val.json
```

Then point it at your data directory:

```bash
python main.py --data-root /path/to/cells
```

`--data-root` defaults to Colab's `/content/cells`. Training hyperparameters are also
flags (`--max-iter`, `--base-lr`, `--ims-per-batch`, `--roi-batch-size`); the defaults
are the values in the table above.

Training writes checkpoints to `./output/`, saves a prediction overlay to
`output/prediction.png`, and the final `COCOEvaluator` pass prints AP metrics for the
validation split.

## Related

- **[TRIMS-results](https://github.com/SuchayK/TRIMS-results)** — method write-up, evaluation
  approach, and publication status for this work.
