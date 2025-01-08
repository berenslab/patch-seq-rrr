# Sparse reduced-rank regression for exploratory visualization of paired multivariate data

![Sparse reduced-rank regression for Patch-seq data](https://github.com/berenslab/patch-seq-rrr/blob/master/figures/sketch.png)

This repository holds all the code and all the data for the following manuscript:

* Kobak D, Bernaerts Y, Weis MA, Scala F, Tolias AS, Berens P (2021), [Sparse reduced-rank regression for exploratory visualisation of paired multivariate data](https://rss.onlinelibrary.wiley.com/doi/10.1111/rssc.12494). *Journal of the Royal Statistical Society: Series C (Applied Statistics),* 70 (4), 980--1000.

BioRxiv link (first version from 2018): https://www.biorxiv.org/content/10.1101/302208v3

### Jan 2025: switching from `glmnet_py` to `scikit_learn`
All analysis in the 2021 paper was done using the sRRR implementation based on the [glmnet_py](https://github.com/bbalasub1/glmnet_python) package. This package can cause conflicts with many versions of `scipy` and gets increasingly difficult to work with. We therefore now provide an equivalent sRRR implementation based on the `MultiTaskElasticNet` class from `scikit-learn`, and recommend this version for all users from now on. Please find it in `sparseRRR_scikit.py` and `demo_scikit.ipynb`.

In this implementation, we adopted the `alpha` and `l1_ratio` nomenclature from `scikit-learn`, meaning that `alpha` controls the strength of the overall group lasso + ridge regularization and `l1_ratio` controls how much lasso vs. ridge penalty is used (`l1_ratio=1` corresponds to pure group lasso).
