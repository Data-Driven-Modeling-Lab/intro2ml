---
title: "Classification Lab: Metrics, Thresholds, and Softmax"
layout: note
category: "Jupyter Notebook"
permalink: /materials/notebooks/classification_metrics_lab/
notebook_source: "classification_metrics_lab.ipynb"
colab_url: "https://colab.research.google.com/github/Data-Driven-Modeling-Lab/intro2ml/blob/main/materials/notebooks/classification_metrics_lab.ipynb"
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Data-Driven-Modeling-Lab/intro2ml/blob/main/materials/notebooks/classification_metrics_lab.ipynb)

[Download the notebook]({{ 'materials/notebooks/classification_metrics_lab.ipynb' | relative_url }})

This is the hands-on starter for Session 8. It is a fill-in notebook: the scaffolding is there, and you write the lines marked `# TODO`. Open it in Colab (button above) or download and run it locally; you only need `numpy`, `matplotlib`, and `scikit-learn`.

By the end you will be able to:
- Load a real dataset and inspect its class balance before fitting anything
- Make an honest, stratified train/test split and fit logistic regression
- Read the confusion matrix, precision, and recall instead of trusting accuracy
- Move the decision threshold and see precision trade against recall
- Extend the same idea to many classes with softmax, and read a per-class report

The lab starts with the Breast Cancer Wisconsin dataset (binary), then generalises to a ten-class problem with the handwritten digits dataset. A final section points you at other multi-class datasets to try (wine, Palmer penguins, iris).
