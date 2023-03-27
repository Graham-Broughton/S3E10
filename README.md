# TPS S3E10 <!-- omit in toc -->

## Nonlinearity <!-- omit in toc -->

## Table of Contents <!-- omit in toc -->

- 

## Overview

This is the 6th place solution for [episode 10 of the third season of the Tabular Playground Series](https://www.kaggle.com/competitions/playground-series-s3e10/overview). The dataset this round had 118_000 observations and 9 features synthesized from pulsar observations. The goal was binary classification to predict if an observation was a pulsar, with log loss as the evaluation metric. The TPS competitions often have smaller datasets than their monetary cousins to make it more accessible, particularly towards newcomers as tabular data is easier to handle than a large image dataset for example. That said, there is also some sort of theme or challenge unique to the competition which competitors must figure out. The challenge to hurdle in this episode was nonlinearity between the features and target. To try and lessen the affect of confident wrong predictions being penalized by log loss, my final model was an ensemble of a few different types of models: Generalized Additive Model and a couple forest models.

## Exploratory Data Analysis


