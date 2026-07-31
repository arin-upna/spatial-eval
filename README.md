# LiDAR Spatial Eval

[![arXiv](https://img.shields.io/badge/arXiv_preprint-2603.22420-b31b1b.svg)](https://arxiv.org/abs/2603.22420)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code accompanying the preprint:

> Alex Salvatierra, José Antonio Sanz, Christian Gutiérrez, and Mikel Galar.  
> **Spatially-Aware Evaluation Framework for Aerial LiDAR Point Cloud Semantic
> Segmentation: Distance-Based Metrics on Challenging Regions.**  
> **Preprint:** [arXiv:2603.22420](https://arxiv.org/abs/2603.22420)

This repository implements the proposed distance-based metrics and hard-points
evaluation for aerial LiDAR semantic segmentation, together with conventional
classification metrics. It contains only the evaluation code, the
configurations used in the preprint, and a compact tutorial. Ensemble code is
not included.

## Installation

```bash
git clone https://github.com/arin-upna/lidar-spatial-eval.git
cd lidar-spatial-eval
python -m pip install -r requirements.txt
```

## Methodology

Let $X=\{p_i\}_{i=1}^{N}$ be a point cloud, where $y_i$ is the
ground-truth label of point $p_i$ and $\hat{y}_i^{(m)}$ is the prediction
made by model $m$.

### Hard points

Given a set of compared models $\mathcal{M}$, the hard-points subset contains
every point misclassified by at least one model:

$$
\mathcal{H} =
\{p_i \in X : \exists m \in \mathcal{M},
\hat{y}_i^{(m)} \neq y_i\}.
$$

All models are evaluated on the same subset $\mathcal{H}$. This removes
points that every model classifies correctly and focuses the comparison on
regions where their predictions differ.

### Distance-based metrics

Let $X_c=\{p_i\in X:y_i=c\}$ denote the ground-truth points of class
$c$, and let $X_{\hat{y}=c}=\{p_i\in X:\hat{y}_i=c\}$ denote the points
predicted as class $c$.

For a point predicted as class $c=\hat{y}_i$, its raw distance error is the
Euclidean distance to the nearest ground-truth point of class $c$:

$$
d_i^{\mathrm{raw}} =
\min_{p_j \in X:\,y_j=\hat{y}_i}
\lVert p_i-p_j \rVert_2.
$$

A correctly classified point has distance zero because it belongs to the
ground-truth set of its predicted class. To prevent isolated extreme errors
from dominating the results, the distance is clipped using the class-specific
threshold $\tau_c$:

$$
d_i = \min\{d_i^{\mathrm{raw}},\tau_{\hat{y}_i}\}.
$$

The Mean Distance Error for class $c$ averages the clipped distances of all
points predicted as that class:

$$
\mathrm{MDE}_c =
\frac{1}{\lvert X_{\hat{y}=c}\rvert}
\sum_{p_i \in X_{\hat{y}=c}} d_i,
\qquad
\mathrm{mMDE} =
\frac{1}{N_c}\sum_{c=1}^{N_c}\mathrm{MDE}_c.
$$

The spatial distribution of the errors is summarized with:

$$
X_{\hat{y}=c}^{\mathrm{distant}} =
\{p_i \in X : \hat{y}_i=c,\ y_i\neq c,
d_i^{\mathrm{raw}}>\tau_c\},
$$

$$
\rho_c = 100\,
\frac{\lvert X_{\hat{y}=c}^{\mathrm{distant}}\rvert}
{\lvert X_c\rvert},
$$

which measures the class-normalized proportion of distant errors, and

$$
X_{\hat{y}=c}^{\mathrm{near}} =
\{p_i \in X : \hat{y}_i=c,\ y_i\neq c,
d_i^{\mathrm{raw}}\leq\tau_c\},
$$

$$
\mu_c =
\frac{1}{\lvert X_{\hat{y}=c}^{\mathrm{near}}\rvert}
\sum_{p_i\in X_{\hat{y}=c}^{\mathrm{near}}} d_i,
$$

where $X_{\hat{y}=c}^{\mathrm{near}}$ contains the misclassified points
predicted as $c$ whose raw distance is at most $\tau_c$. Thus,
$\mu_c$ describes the typical deviation of non-distant errors.

When these metrics are computed on $\mathcal{H}$, only hard points contribute
to the averages and counts, but nearest ground-truth neighbors are still
searched in the complete point cloud $X$.

## Usage

Ground-truth and prediction directories must contain corresponding LAS/LAZ
files. Labels are read from the `classification` dimension.

Evaluate one model:

```bash
python metrics/metrics.py \
  --gt_dir data/ground_truth \
  --pred_dir data/model_a \
  --config config/dales.yaml \
  --output_dir results/model_a
```

Evaluate several models on their shared hard-points subset:

```bash
python metrics/metrics.py \
  --gt_dir data/ground_truth \
  --pred_dir data/model_a \
  --pred_dir data/model_b \
  --pred_dir data/model_c \
  --config config/dales.yaml \
  --union_errors \
  --output_dir results/hard_points
```

A brief example using synthetic data is available in
[notebooks/tutorial_metrics.ipynb](notebooks/tutorial_metrics.ipynb).

## Citing the preprint

If you use this repository, please cite the arXiv preprint:

```bibtex
@misc{salvatierra2026spatiallyaware,
  title         = {Spatially-Aware Evaluation Framework for Aerial LiDAR Point
                   Cloud Semantic Segmentation: Distance-Based Metrics on
                   Challenging Regions},
  author        = {Salvatierra, Alex and Sanz, Jos{\'e} Antonio and
                   Guti{\'e}rrez, Christian and Galar, Mikel},
  year          = {2026},
  eprint        = {2603.22420},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2603.22420},
  url           = {https://arxiv.org/abs/2603.22420}
}
```

## License

This code is released under the [MIT License](LICENSE).
