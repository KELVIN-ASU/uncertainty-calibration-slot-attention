````markdown
# Uncertainty Calibration for Slot-Attention-Based Generative Factor Prediction

Empirical study of uncertainty calibration for Slot Attention-based generative factor prediction on the Causal3DIdent dataset.

This repository contains the experimental code used for a master's thesis on uncertainty-aware generative factor prediction from frozen Slot Attention representations. The experiments evaluate five downstream predictor architectures, two training objectives, and several post-hoc calibration methods under in-distribution and corruption-based out-of-distribution settings.

## Overview

This repository contains the code used for the empirical section of the thesis.

The study follows a two-stage pipeline. First, a Slot Attention autoencoder is trained for image reconstruction. Second, the trained Slot Attention backbone is frozen and used to extract object-centric slot representations from images. These slot representations are then used by probabilistic predictor heads to predict the ten ground-truth generative factors of the Causal3DIdent dataset.

Each predictor outputs both a mean and a variance for every generative factor. This makes it possible to evaluate not only prediction accuracy, but also uncertainty calibration, interval coverage, predictive sharpness, and robustness under distribution shift.

The experiments compare five downstream predictor architectures:

- MeanPoolLinear
- MeanPoolMLP
- FlatMLP
- DeepSets
- FactorQuery

Each architecture is trained with two objectives:

- Gaussian negative log-likelihood (NLL)
- Gaussian negative log-likelihood with residual-variance regularization (RESVAR)

The trained predictors are evaluated with several post-hoc calibration methods:

- scalar temperature scaling
- per-factor temperature scaling
- isotonic CDF / PIT calibration
- absolute-error conformal calibration
- Gaussian split conformal calibration

The evaluation includes:

- in-distribution predictive accuracy
- uncertainty calibration
- predictive sharpness
- empirical interval coverage
- per-factor calibration diagnostics
- PIT histograms
- corruption-based OOD robustness
- representation decodability using linear probes
- FactorQuery attention diagnostics
- matched-pair search and stability diagnostics

## Repository structure

```text
uncertainty-calibration-slot-attention/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── 1_Baseline_slotAE.ipynb
├── 2_calibration_experiments.ipynb
│
├── datasets/
│   ├── trainset.tar
│   └── testset.tar
│
├── SlotAttentionCheckpoints/
│   ├── slot_attention_epoch_10.pth
│   ├── slot_attention_epoch_20.pth
│   ├── slot_attention_epoch_30.pth
│   ├── slot_attention_epoch_40.pth
│   └── slot_attention_epoch_50.pth
│
└── Experiment_Results/
    ├── checkpoints/
    └── logs/
        ├── all_conformal.csv
        ├── all_factor_query_attention.csv
        ├── all_identifiability.csv
        ├── all_matched_pair_stability.csv
        ├── all_metrics.csv
        ├── all_ood_shift.csv
        ├── all_ood_slices.csv
        ├── all_pair_search.csv
        ├── summary_identifiability_ci.csv
        ├── summary_metrics_ci.csv
        │
        ├── pair_cache/
        │
        └── by_architecture/
            ├── DeepSets/
            │   ├── NLL/
            │   │   ├── seed_45/
            │   │   ├── seed_46/
            │   │   └── seed_47/
            │   │       ├── attention/
            │   │       ├── identifiability/
            │   │       ├── matched_pairs/
            │   │       ├── metrics/
            │   │       ├── ood/
            │   │       └── plots/
            │   └── RESVAR/
            │       ├── seed_45/
            │       ├── seed_46/
            │       └── seed_47/
            │
            ├── FactorQuery/
            │   ├── NLL/
            │   └── RESVAR/
            │
            ├── FlatMLP/
            │   ├── NLL/
            │   └── RESVAR/
            │
            ├── MeanPoolLinear/
            │   ├── NLL/
            │   └── RESVAR/
            │
            └── MeanPoolMLP/
                ├── NLL/
                └── RESVAR/
````

## Experiments

### 1. Slot Attention Autoencoder Training

The first notebook trains the Slot Attention autoencoder on Causal3DIdent images:

```text
1_Baseline_slotAE.ipynb
```

The Slot Attention autoencoder consists of:

* CNN encoder
* Slot Attention module
* spatial broadcast decoder
* reconstruction loss based on mean squared error

The model uses:

```text
Number of slots: 4
Slot dimension: 64
Image resolution: 64 x 64
Training objective: reconstruction MSE
```

The notebook saves checkpoints every 10 epochs to:

```text
SlotAttentionCheckpoints/
```

The checkpoint used by the downstream experiments is:

```text
SlotAttentionCheckpoints/slot_attention_epoch_50.pth
```

This notebook also includes:

* reconstruction visualization
* slot mask visualization
* slot latent vector visualization
* Hungarian-matched slot sensitivity diagnostics under input noise

### 2. Calibration and Generative Factor Prediction Experiments

The second notebook runs the main empirical study:

```text
2_calibration_experiments.ipynb
```

This notebook loads the pretrained Slot Attention checkpoint, freezes the Slot Attention backbone, extracts deterministic slot representations, and trains probabilistic downstream predictor heads.

The main predictor architectures are:

#### MeanPoolLinear

Raw slot vectors are averaged across slots, then passed directly to a linear probabilistic head.

#### MeanPoolMLP

Raw slot vectors are averaged across slots, passed through an MLP, and then used for probabilistic factor prediction.

#### FlatMLP

All slot vectors are concatenated and passed through an MLP. This architecture preserves slot-specific information but is not permutation-invariant.

#### DeepSets

Each slot is processed by a shared function, followed by symmetric aggregation. This architecture is permutation-invariant and respects the set-like structure of Slot Attention outputs.

#### FactorQuery

Each generative factor has a learned query that attends over the slot set. This architecture tests whether different factors rely on different slot combinations.

## Training objectives

### 1. Gaussian NLL

The standard probabilistic objective is Gaussian negative log-likelihood:

```math
\mathcal{L}_{\mathrm{NLL}}
=
\frac{1}{2}
\left[
\log \sigma^2
+
\frac{(y-\mu)^2}{\sigma^2}
\right].
```

Each predictor outputs a mean (\mu) and variance (\sigma^2) for each generative factor.

### 2. Residual-variance regularization

The RESVAR objective augments Gaussian NLL with a residual-variance alignment term:

```math
\mathcal{L}_{\mathrm{RESVAR}}
=
\mathcal{L}_{\mathrm{NLL}}
+
\lambda_{\mathrm{var}}
\left|
(y-\mu)^2 - \sigma^2
\right|
+
\lambda_1 \|z\|_1.
```

This term encourages predicted variance to align with squared prediction error. It is used as an in-training calibration-aware regularizer.

## Post-hoc calibration methods

The experiments evaluate the following post-hoc calibration methods.

### 1. Scalar temperature scaling

A single scalar temperature rescales all predicted variances:

```math
\sigma^2_{\mathrm{cal}}
=
T^2 \sigma^2.
```

### 2. Per-factor temperature scaling

Each generative factor receives its own temperature:

```math
\sigma^2_{\mathrm{cal},j}
=
T_j^2 \sigma_j^2.
```

This is useful because different generative factors can have different residual scales and calibration behavior.

### 3. Isotonic CDF / PIT calibration

The Probability Integral Transform (PIT) is computed from the predictive Gaussian distribution. Isotonic regression is then used to calibrate the empirical CDF behavior.

### 4. Absolute-error conformal calibration

Per-factor absolute residual quantiles are computed on the calibration split and used to form conformal prediction intervals.

### 5. Gaussian split conformal calibration

Gaussian predictive intervals are expanded using split conformal scores computed on the calibration split.

## Dataset

This project uses the Causal3DIdent dataset.

The dataset can be downloaded from Zenodo:

```text
https://zenodo.org/records/4784282
```

Required files:

```text
trainset.tar.gz
testset.tar.gz
```

After downloading the files, decompress them:

```bash
gunzip trainset.tar.gz
gunzip testset.tar.gz
```

Then place the resulting `.tar` files in the `datasets/` directory:

```text
datasets/
├── trainset.tar
└── testset.tar
```

The notebooks automatically extract the dataset into:

```text
causal3DIdent/
├── trainset/
└── testset/
```

## Reproducibility

All main experiments are run over three random seeds:

```text
45, 46, 47
```

Main experimental settings:

```text
Batch size: 512
Training epochs: 10
Learning rate: 1e-3
Number of slots: 4
Slot dimension: 64
Number of generative factors: 10
Validation fraction: 0.10
Calibration split: 0.50 of the validation split
```

The Slot Attention backbone is frozen during the downstream prediction experiments. Slot representations are extracted deterministically using fixed slot initialization noise so that all predictor architectures are evaluated on the same slot representations.

Out-of-distribution evaluation uses deterministic image corruptions:

```text
gaussian_blur: severity 1, 3, 5
brightness: severity 1, 3, 5
gaussian_noise: severity 1, 3, 5
```

The experiments were run in Google Colab using an A100 GPU runtime. Runtime values may differ across hardware platforms.

## Installation

Install the required packages with:

```bash
pip install -r requirements.txt
```

Main dependencies:

```text
numpy
pandas
matplotlib
scipy
scikit-learn
torch
torchvision
pillow
tqdm
jupyter
```

## Running the notebooks

Run the notebooks in the following order:

```text
1_Baseline_slotAE.ipynb
2_calibration_experiments.ipynb
```

The first notebook trains the Slot Attention autoencoder and saves checkpoints.

The second notebook loads the pretrained Slot Attention checkpoint and runs the downstream uncertainty calibration experiments.

Expected checkpoint path:

```text
SlotAttentionCheckpoints/slot_attention_epoch_50.pth
```

Each notebook saves output files, summary tables, diagnostic CSV files, and plots.

## Output files

The main experiment outputs are saved under:

```text
Experiment_Results/logs/
```

Important aggregated output files include:

```text
all_metrics.csv
all_conformal.csv
all_ood_slices.csv
all_ood_shift.csv
all_identifiability.csv
all_factor_query_attention.csv
all_matched_pair_stability.csv
all_pair_search.csv
summary_metrics_ci.csv
summary_identifiability_ci.csv
```

### Aggregated CSV files

* `all_metrics.csv`
  In-distribution RMSE, MAE, NLL, sharpness, Reg-ECE, and coverage results.

* `all_conformal.csv`
  Absolute-error conformal interval results.

* `all_ood_slices.csv`
  Factor-slice evaluation results for low/high quantile regions of each generative factor.

* `all_ood_shift.csv`
  OOD corruption results across corruption types, severity levels, architectures, and calibration variants.

* `all_identifiability.csv`
  Linear-probe decodability results for set representations and learned latent representations.

* `all_factor_query_attention.csv`
  Mean factor-to-slot attention weights for the FactorQuery architecture.

* `all_matched_pair_stability.csv`
  Matched-pair stability diagnostics.

* `all_pair_search.csv`
  Matched-pair construction and filtering diagnostics.

* `summary_metrics_ci.csv`
  Summary table with confidence intervals for the main predictive and calibration metrics.

* `summary_identifiability_ci.csv`
  Summary table with confidence intervals for representation decodability.

### Per-architecture results

Detailed results are organized as:

```text
Experiment_Results/logs/by_architecture/<Architecture>/<Objective>/seed_<Seed>/
```

For example:

```text
Experiment_Results/logs/by_architecture/DeepSets/NLL/seed_47/
```

Each seed folder contains:

```text
attention/
identifiability/
matched_pairs/
metrics/
ood/
plots/
```

* `metrics/` contains per-run metrics and training histories.
* `ood/` contains OOD corruption and factor-slice results.
* `plots/` contains coverage curves, PIT histograms, per-factor coverage plots, and per-factor PIT plots.
* `identifiability/` contains linear-probe representation decodability results.
* `attention/` contains FactorQuery attention diagnostics when applicable.
* `matched_pairs/` contains matched-pair search and stability diagnostics.

## Summary of empirical findings

The in-distribution results show that `FlatMLP_RESVAR+pfT` is the strongest overall configuration. It achieves the best RMSE, MAE, NLL, and Reg-ECE among the evaluated models. This indicates that preserving slot-specific information through slot concatenation is beneficial when the Slot Attention backbone is frozen.

DeepSets shows substantial improvement when trained with residual-variance regularization and emerges as the strongest permutation-invariant alternative. Although it does not match FlatMLP in raw prediction accuracy, it better respects the unordered structure of Slot Attention representations.

Residual-variance regularization improves point prediction for several architectures, especially FlatMLP and DeepSets. However, it does not consistently improve calibration by itself. RESVAR often sharpens predictive distributions but can also cause under-coverage before post-hoc calibration.

Per-factor temperature scaling is the most consistent post-hoc temperature calibration method. It improves or preserves NLL, reduces Reg-ECE in most cases, and restores empirical coverage close to 0.90 for several under-covered RESVAR variants.

Conformal and CDF-based interval calibration recover near-nominal 90% in-distribution coverage across architectures. However, PIT histograms show residual distributional miscalibration, demonstrating that interval coverage alone is not sufficient to conclude that a predictive distribution is fully calibrated.

The OOD corruption results show that calibration fitted on clean data does not transfer reliably under severe brightness and Gaussian noise corruptions. Gaussian blur at low severity is relatively mild, while high-severity Gaussian noise produces the strongest degradation.

FactorQuery variants show stronger OOD likelihood and calibration behavior than FlatMLP, even though FlatMLP is stronger in-distribution. This indicates that the best in-distribution predictor is not necessarily the most reliable under distribution shift.

Representation decodability is highest for `FlatMLP_RESVAR` at the learned latent representation level, while `DeepSets_RESVAR` has the strongest set-level representation. FactorQuery attention is factor-dependent but generally diffuse, so it does not provide strong evidence of one-to-one factor-slot alignment.

Matched-pair stability analysis is inconclusive because the strict pair-construction protocol produces too few valid pairs for most factors. The matched-pair results are therefore presented as diagnostics of pair-construction difficulty rather than as evidence for causal invariance.

## Notes on large files

The fdataset folders can become large and are usually not committed


For reproducibility, this repository provides the code and instructions needed to regenerate these files. Large checkpoints, extracted datasets, and full experiment outputs can be shared separately when needed.

## Citation

If you use this work in your research, please cite the thesis:

```bibtex
@misc{ekuri2026slotattentioncalibration,
      title={Uncertainty Calibration for Slot-Attention-Based Generative Factor Prediction},
      author={Kelvin Asu Ekuri and Bader Rasheed},
      year={2026},
      eprint={XXXX.XXXXX},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/XXXX.XXXXX},
}
```

Replace `XXXX.XXXXX` with the arXiv identifier after the thesis is uploaded.

If you use the Causal3DIdent dataset, please also cite:

```bibtex
@inproceedings{vonkugelgen2021self,
  title={Self-Supervised Learning with Data Augmentations Provably Isolates Content from Style},
  author={von K{\"u}gelgen, Julius and Sharma, Yash and Gresele, Luigi and Brendel, Wieland and Sch{\"o}lkopf, Bernhard and Besserve, Michel and Locatello, Francesco},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```

## License

This project is licensed under the MIT License. See `LICENSE` for details.

**Key points (MIT):**

* You may use, copy, modify, and distribute the code.
* You may use the code for research or commercial purposes.
* The code is provided without warranty.
* The original license notice should be preserved.

```
```
