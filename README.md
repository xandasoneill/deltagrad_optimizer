# DeltaGrad: Towards Robust Deep Learning via Adaptive Gradient

This repository contains the official implementation of **DeltaGrad**, an adaptive optimizer designed to mitigate noise memorization and gradient instability in non-convex optimization. [cite_start]By introducing a dynamic **Reliability Metric ($R_t$)** [cite_start], the framework modulates updates based on instantaneous gradient coherence.

---

## 📈 Experimental Results Summary

[cite_start]Compared to the Adam baseline on CIFAR-100, DeltaGrad demonstrates superior stability and noise resilience:

| Metric | Adam | DeltaGrad | Performance Gain |
| :--- | :--- | :--- | :--- |
| **Learning Rate Stability** | Baseline | **10x Standard** | [cite_start]Enhanced Robustness |
| **Global Mean Variance** | $6.78 \times 10^{-4}$ | **$5.90 \times 10^{-5}$** | [cite_start]10x Lower Instability |
| **Comp. Overhead** | 0.00% | **0.38%** | [cite_start]Negligible |
| **Training Time/Epoch** | 36.96s | **37.10s** | [cite_start]Scalable |

---

## 📂 Project Organization

The repository is modularly structured to ensure experimental transparency and extensibility:

* [cite_start]**`DeltaGrad.py`**: Implementation of the $R_t$ metric and Windowed Inertia logic.
* **`engine.py`**: Standardized training and evaluation logic across all test regimes.
* [cite_start]**`model.py`**: Compact ConvNet architecture used for all benchmarks.
* **`tune_hyperparams.py`**: Automated Hyperparameter Optimization suite using **Optuna**.
* **`final_benchmark.py`**: Primary script to execute Learning Rate Stress, Batch Size, and Data Noise tests.
* **📂 `best_params/`**: Serialized optimal hyperparameters from the Optuna studies.
* **📂 `results/`**: Comprehensive storage for raw experimental results and generated performance analytics.

---

## 💻 Environment Requirements

To ensure reproducibility of the benchmarks, the following environment is recommended:

* **Python**: 3.8 or higher.
* **Core Libraries**: 
    * `torch >= 1.10.0`
    * `torchvision >= 0.11.0`
    * `numpy >= 1.21.0`
* **Optimization & Analysis**: 
    * [cite_start]`optuna >= 2.10.0` (for hyperparameter tuning) 
    * `matplotlib` & `pandas` (for result visualization and log processing).
* [cite_start]**Dataset**: CIFAR-100 (automatically handled via `torchvision`).

---

## 📚 Citation

If you utilize this implementation or the DeltaGrad framework in your research, please cite:

```latex
@article{oneill2026deltagrad,
  title={DeltaGrad: Towards Robust Deep Learning via Adaptive Gradient},
  author={Alexandre de Abreu O'Neill Mendes},
  journal={GitHub Repository},
  year={2026},
  note={Preprint in submission},
  url={[https://github.com/xandasoneill/deltagrad_optimizer.git](https://github.com/xandasoneill/deltagrad_optimizer.git)}
}