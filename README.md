# Transcriptional Burst Kinetics
[![Build Status](https://travis-ci.org/vanheeringen-lab/Transcriptional_Burst_Kinetics.svg?branch=master)](https://travis-ci.org/vanheeringen-lab/Transcriptional_Burst_Kinetics)
[![Maintainability](https://api.codeclimate.com/v1/badges/6885ade32f6bb232bd7a/maintainability)](https://codeclimate.com/github/vanheeringen-lab/Transcriptional_Burst_Kinetics/maintainability)
[![Test Coverage](https://api.codeclimate.com/v1/badges/6885ade32f6bb232bd7a/test_coverage)](https://codeclimate.com/github/vanheeringen-lab/Transcriptional_Burst_Kinetics/test_coverage)

Private effort of bundling different transcriptional burst kinetics methods into a single (Python!) repository.

## Markovian Modeling of Gene-Product Synthesis
The 'original' markov model of Peccoud & Ycart where a gene can either be in an active state (A), or inactive state (I). When a gene is in an active state, it produces gene-products (P), which then get broken down (Ø):

<p align="center">
    <img src="imgs/markov.jpg?sanitize=true">
</p>

In [gene](https://github.com/vanheeringen-lab/Transcriptional_Burst_Kinetics/blob/master/tbk/gene.py), [product](https://github.com/vanheeringen-lab/Transcriptional_Burst_Kinetics/blob/master/tbk/product.py), and [run](https://github.com/vanheeringen-lab/Transcriptional_Burst_Kinetics/blob/master/tbk/run.py) the code is implemented to run this markovian model. Estimating the parameters with their first three moments is implemented in [inference](https://github.com/vanheeringen-lab/Transcriptional_Burst_Kinetics/blob/master/tbk/inference.py).

## Beta-Poisson model for single-cell RNA-seq data analyses
The problem with the moment-based inference of the parameters is that parameters often get unreasonable values (e.g. negative values). As it turns out, when the 'markov model' is in steady state the distribution of gene products follows a Beta-Poisson distribution, which can relatively easily be fit and won't give unreasonable values.

Currently the beta-poisson 3 parameter and beta-poisson 4 parameter models are implemented, of which the relevant code can be found in [bp](https://github.com/vanheeringen-lab/Transcriptional_Burst_Kinetics/blob/master/tbk/bp.py), [inference](https://github.com/vanheeringen-lab/Transcriptional_Burst_Kinetics/blob/master/tbk/inference.py).

## Genomic encoding of transcriptional burst kinetics
The sandberg-lab made an addition to the beta-poisson 3 model with which you can test the inferred parameters for two conditions with a [wald-test](https://en.wikipedia.org/wiki/Wald_test)(they use it to compare maternal and paternal expression). The wald-test can be found in [inference](https://github.com/vanheeringen-lab/Transcriptional_Burst_Kinetics/blob/master/tbk/inference.py).

## Examples
Take a look at our [examples](https://github.com/vanheeringen-lab/Transcriptional_Burst_Kinetics/tree/master/examples) on how to run the code.

## Notes
Different groups use different notations for the parameters. The name of the parameter we use is always the name of its respective paper, however (for consistency) they are always passed to every function in the same order.
