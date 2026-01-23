# Machine learning modularity
[![arXiv](https://img.shields.io/badge/arXiv-2601.01779-b31b1b.svg)](https://arxiv.org/abs/2601.01779)

This repository provides the official implementation for the paper **"Machine learning modularity"**. The project explores the application of machine learning architectures to identify and simplify complex mathematical structures, specifically focusing on Möbius transformations, $q-\theta$ functions, and elliptic $\Gamma$ functions.

## Project Architecture

The codebase is organized into three primary modules, each corresponding to a specific section of the research:

### 1. Domain Reduction

**Focus:** Machine Learning Möbius Transformations.

* **Objective:** Given a complex point  located outside the fundamental domain, the model predicts a matrix  such that the modular action  maps the point back into the fundamental domain.
* **Location:** `/domain_reduction`

### 2. $q-\theta$ Simplification

**Focus:** Machine Learning for $q-\theta$ Functions.

* **Objective:** The model identifies patterns within symbolic expressions of - products and reduces them to their minimal, simplified forms.
* **Location:** `/q_theta_simplify`

### 3. Elliptic Gamma Simplification

**Focus:** Machine Learning the Elliptic Gamma Function.

* **Objective:** This module handles elliptic gamma expressions following specific identities, training the model to transform them into a canonical or simplified representation.
* **Location:** `/elliptic_gamma_simplify`

---

## Quick Start Guide

### 1. Environment Setup

Install the necessary dependencies using pip:

```bash
pip install -r requirements.txt
```

### 2. Model Weights

To run the inference or evaluation scripts, you must download the pre-trained model weights:

1. **Download:** Access the weight files via [Google Drive](https://drive.google.com/drive/folders/1nuG2LzysKkp-cRiFss3Iy2qOH4pVGW7k?usp=drive_link).
2. **Extraction:** After downloading, extract the archives.
3. **Placement:** Move the extracted model files into their respective directory structures. 

### 3. Execution

Navigate to the desired module folder to begin:

* **quick start:** Open and run the `demo.ipynb` notebooks for interactive demonstrations.

