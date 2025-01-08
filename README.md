# Sparse reduced-rank regression for exploratory visualization of paired multivariate data

![Sparse reduced-rank regression for Patch-seq data](https://github.com/berenslab/patch-seq-rrr/blob/master/figures/sketch.png)

This repository holds all the code and all the data for the following manuscript:

* Kobak D, Bernaerts Y, Weis MA, Scala F, Tolias AS, Berens P (2021), [Sparse reduced-rank regression for exploratory visualisation of paired multivariate data](https://rss.onlinelibrary.wiley.com/doi/10.1111/rssc.12494). *Journal of the Royal Statistical Society: Series C (Applied Statistics),* 70 (4), 980--1000.

BioRxiv link (first version from 2018): https://www.biorxiv.org/content/10.1101/302208v3.article-info

### Note on the usage of glmnet_py
The results in the 2021 paper listed above were derived with the [glmnet_py](https://github.com/bbalasub1/glmnet_python) package. This package currently however causes conflicts with many versions of _scipy_ and seems increasingly difficult for MAC users to work with. We therefore included `sparseRRR_scikit.py` as well as `demo_scikit.ipynb` that runs sparse reduced-rank regression without the need for _glmnet_py_ but uses _scikit-learn_. We believe this will help interested users in deploying sparse reduced-rank regression on their own data without the need for sophisticated environments or complicated dependencies.
<br><br>
Importantly, we adopted the _alpha_ and _l1_ratio_ nomenclature from _scikit-learn_ in that demo, meaning that _alpha_ controls the strength of overall group lasso + ridge regularization and _l1_ratio_ controls how much lasso vs ridge penalty you use (e.g. _l1_ratio_ equal to 1 corresponds to pure group lasso).
