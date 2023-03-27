# TPS S3E10 <!-- omit in toc -->

## Nonlinearity <!-- omit in toc -->

## Table of Contents <!-- omit in toc -->

- [Overview](#overview)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Baseline](#baseline)
- [Final Model](#final-model)

## Overview

This is the 6th place solution for [episode 10 of the third season of the Tabular Playground Series](https://www.kaggle.com/competitions/playground-series-s3e10/overview). The dataset this round had 118_000 observations and 9 features synthesized from pulsar observations. The goal was binary classification to predict if an observation was a pulsar, with log loss as the evaluation metric. The TPS competitions often have smaller datasets than their monetary cousins to make it more accessible, particularly towards newcomers as tabular data is easier to handle than a large image dataset for example. That said, there is also some sort of theme or challenge unique to the competition which competitors must figure out. The challenge to hurdle in this episode was nonlinearity between the features and target. To try and lessen the affect of confident wrong predictions being penalized by log loss, my final model was an ensemble of a few different types of models: Generalized Additive Model and a couple forest models.

## Exploratory Data Analysis

Since we are allowed to use the original data if we want to, one of the first things investigated was if the two datasets were similar enough. As we can see in [the feature distributions](data/images/features_distribution.png), the synthetic data is nearly identical whilst the original is different. Other Kagglers performed Adversarial Validation to double check if the original was different enough to dump and the results were in agreement. In [the features by class distribution](data/images/features_vs_class_dist.png) there is a stark contrast how the target behaves. Some features the two classes are roughly similar and dense while others the positive class is easily distinguishable by how flat dissimilar it is from the negative. There were some very long tails which present the opportunity to log transform some features. Class imbalance was present at 90% to 10% negative and positive classes, respectively. As seen in the following pie chart:

![class imbalance pie chart](data/images/plot_count_Train.png)

There were also some strong [correlations](data/images/plot_heatmap_Train Correlation.png) among the features as well as the target. Applying polynomial expansion and feature selection might be worthwhile. Lastly, the data's dimensions were reduced by PCA, TSNE and UMAP to [visualize in 3D](notebooks/1.EDA.ipynb#3d-visualization) which showed mostly separated classes with only a few of each class throughout the other. This should make it fairly easy to get a decent score, but much harder to get a good one.

## Baseline


## Final Model

