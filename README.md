# Detectron2 — Immune Cell Instance Segmentation

Training script for instance segmentation of immune cells in lymph node microscopy
images, built on [Detectron2](https://github.com/facebookresearch/detectron2)'s
Mask R-CNN. Research conducted with the University at Buffalo Visual Computing Lab;
related work was presented at **ASN Kidney Week 2025**.

> **Scope of this repo.** This is the model training and evaluation script only.
> The dataset, trained weights, and full quantitative results belong to the lab and
> are not published here. If you're evaluating this work, the numbers and sample
> segmentations are the part you want — I'm happy to walk through them directly:
> suchay.kommisetty@gmail.com

## How it works

| | |
|---|---|
| **Model** | Mask R-CNN, ResNet-50 FPN backbone (`COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x`) |
| **Init** | COCO-pretrained weights from the Detectron2 model zoo, fine-tuned on the cell dataset |
| **Data** | COCO-format instance annotations (`json_annotation_train.json` / `_val.json`) registered via `register_coco_instances`. The ROI head's class count is read back from the registered metadata, so the head always matches the annotations. |
| **Task** | Instance segmentation — separating touching and overlapping cells, which is where box-only detectors fail |
| **Training** | 1000 iterations · base LR 2.5e-4 · 2 images/batch · 128 ROIs per image |
| **Inference** | Score threshold 0.7 for the qualitative overlay |
| **Evaluation** | `COCOEvaluator` on the held-out validation split (COCO AP / AP50 / AP75), at score threshold 0.05 — COCO AP integrates precision across the full recall range, so evaluating at the 0.7 display threshold would truncate the PR curve and understate AP |
| **Environment** | GPU runtime. Runs anywhere; the inline Colab preview is used only if `google.colab` is importable, otherwise the overlay is written to `output/prediction.png` |

Instance segmentation was chosen over semantic segmentation because the downstream
question is *how many* cells and *where each one is*, not just which pixels are cell
versus background. Mask R-CNN gives a per-cell mask, so adjacent cells stay distinct.

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

- **TRIMS** — results summary, figures, and the ASN Kidney Week 2025 poster *(publishing soon)*
