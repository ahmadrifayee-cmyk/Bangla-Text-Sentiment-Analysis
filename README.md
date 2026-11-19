[README.md](https://github.com/user-attachments/files/35066218/README.md)
# Bangla Text Sentiment Analysis: Probability-Level Fusion of Machine Learning, Deep Learning and Transformer Models

Code for the paper **"Bangla Text Sentiment Analysis: An Evaluation of a Hybrid Approach Using Machine Learning, Deep Learning, and Transformer Models"**.

The study evaluates five-class (strongly negative to strongly positive) Bangla sentiment classification. Three independently trained branches are fused at the probability level with a stacked logistic-regression meta-learner, and the fusion gain is analysed with a full branch-subset ablation, bootstrap confidence intervals and an ordinal-aware error analysis.

## Overview

| Branch | Features / model |
|---|---|
| Classical ML | Random Forest + XGBoost on word (1,2)-gram and character (3,5)-gram TF-IDF |
| Deep learning | CNN-BiLSTM-Attention on 300-d Bangla FastText (`wiki.bn.vec`) embeddings |
| Transformer | Fine-tuned [BanglaBERT](https://huggingface.co/csebuetnlp/banglabert) |
| Fusion | Logistic regression on the concatenated 15-d probability vector (meta-learner chosen by 5-fold cross-validation on the validation set only) |

All branches use the same preprocessing and one shared stratified 80/10/10 split (seed 42).

## Main results (test set, 7,000 samples, seed 42)

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Random Forest | 0.4856 | 0.4743 |
| XGBoost | 0.5017 | 0.4961 |
| CNN-BiLSTM-Attention | 0.5280 | 0.5162 |
| BanglaBERT | 0.6676 | 0.6640 |
| **Hybrid fusion (LR)** | **0.6763** | **0.6708** |

The hybrid improves on standalone BanglaBERT by 0.7 macro-F1 points (paired bootstrap 95% CI [0.0009, 0.0128]; Wilcoxon p = 0.0029). About half of this gain comes from recalibrating BanglaBERT's probabilities on the validation set. Full tables, per-class results, source-wise results and error analysis are in the paper.

## Repository structure

```
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── bangla_sentiment_fusion_seed42.ipynb   # full pipeline (data -> branches -> fusion -> evaluation -> figures)
└── results/
    └── seed_42/                                # JSON/CSV outputs and figures produced by the notebook
```

## Data

The experiments use the publicly available **Kaggle release** of SentiGOLD (70,000 texts, five classes, six source categories):
<https://www.kaggle.com/datasets/joy11117/sentigold-data-for-fine-tuning>

This is a third-party copy. The full, officially annotated SentiGOLD corpus (with its 30+ topic domains) is distributed by its authors under their own license (Islam et al., KDD 2023). Comparisons with the original benchmark should therefore be read as indicative only. The dataset is **not** redistributed in this repository; please download it from Kaggle and respect its license.

The notebook also downloads the pretrained Bangla FastText vectors (`wiki.bn.vec`) from the official fastText servers on first run.

## Reproducing the results

The code was run in a Kaggle notebook on one NVIDIA Tesla T4 GPU (Python 3, Hugging Face Transformers 4.40.0).

1. Create a new Kaggle notebook and enable a GPU accelerator and internet access.
2. Add the Kaggle dataset linked above to the notebook.
3. Upload `notebooks/bangla_sentiment_fusion_seed42.ipynb` and, if the dataset path differs, edit the `RAW` path in the setup cell (it points to `.../sentigold-data-for-fine-tuning/sentigold_sample.csv`).
4. Keep `RUN_SEED = 42` and choose **Run All**.
5. Outputs (metrics as JSON/CSV and all figures) are written to `outputs/seed_<RUN_SEED>/`.

To run locally instead, install the dependencies with `pip install -r requirements.txt` and adjust the paths accordingly. Exact numbers can differ slightly across hardware and library versions.

## Limitations

- One data split and one training seed.
- Hyperparameters followed common practice and were not tuned.
- Source-wise analysis uses the source-category column of the Kaggle copy, not the original topic domains.

## Citation

If you use this code, please cite the paper. A `CITATION.cff` file is provided, which GitHub uses to show a "Cite this repository" button. The bibliographic details of the paper will be added here once it is published.

## License

The code is released under the MIT License (see `LICENSE`). The datasets and pretrained models used here have their own licenses.

## Authors

Ahmad Rifayee Chowdhury, Adil Rashid, Md. Badiuzzaman Biplob (corresponding author), Tanveer Ahsan
Department of Computer Science and Engineering, International Islamic University Chittagong, Chattogram, Bangladesh.

For questions, please open a GitHub issue.
