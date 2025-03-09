# GCN-PFAS-Docking-Score-Prediction

# Uncovering the Mechanism of Hepatotoxicity of PFAS Targeting L-FABP Using GCN and Computational Modeling

## Overview

This repository contains the code, data, and results related to my thesis titled Uncovering the Mechanism of Hepatotoxicity ofPFAS Targeting L-FABP Using GCN and Computational Modeling.

## Table of Contents

- [Introduction](#introduction)
- [Usage](#usage)
- [Data](#data)
- [Methodology](#methodology)
- [Results](#results)

## Introduction

Per- and polyfluoroalkyl substances (PFAS) are persistent environmental pollutants with known toxicity and bioaccumulation issues. Their widespread industrial use and resistance to degradation have led to global environmental contamination and significant health concerns. While a minority of PFAS have been extensively studied, the toxicity of many PFAS remains poorly understood due to limited direct toxicological data. This study advances the predictive modeling of PFAS toxicity by combining semi-supervised graph convolutional networks (GCNs) with molecular descriptors and fingerprints. We propose a novel approach to enhance PFAS binding affinity predictions by utilizing molecular fingerprints to construct graph representations, where nodes encode molecular descriptors. This approach specifically captures the structural, physicochemical, and topological features of PFAS without overfitting due to an abundance of features. Unsupervised clustering then identifies representative compounds for detailed binding studies. Our results offer improved accuracy in estimating PFAS hepatotoxicity, aiding chemical discovery of new PFAS and the formulation of safety regulations.

## Usage

The core code consists of the graph_construction.py and model.py scripts. The former is used to create the graph structure to then to be implemented to train a GCN using the model.py script. Detailed description for running the scripts can be found commented within them.

## Data

- **Source:** Labeled binding affinites of PFAS towards the nonportal region of LFABP is provided by J. Zhao et al. in the study: Hepatotoxicity assessment investigations on PFASs targeting L-FABP using binding affinity data and machine learning-based QSAR model”. Unlabeled PFAS come from the OECD PFAS global database.

## Methodology

This study employs a semi-supervised GCN with a self-constructed graph from the cosine similarity of the indivual PFAS' molecular fingerprint. The approach includes:

- **Graph Construction**: Created by the graph_consruction.py script using the sortedFeat_impDesc.csv and AP2D_count_stand.csv files.
- **GCN Model Training**: The model training is implemented in the model.py script and uses the created graph file from the graph_consruction.py script.
- **Modeling/Analysis**: Predicted binding affinites towards LFABP of the test and unlabeled PFAS were then calculated with trained model. The resulting PFAS were then clustered to comare their topologies with their binding affinities. Representative PFAS from each cluster were then modeled with molecular dynamic (MD) simulations to uncover their binding mechanisms.
- **Evaluation Metrics**: Results from this study can be found in the presentation and the Supporting Figures and Results folder.

## Results

Summary of findings:

- Proposed semi-supervised GCN solves limited training data problem while utilizing the maximum information contained within a molecule.

- Larger PFAS consistently interact with LFABP and thus have a higher chance of forming bonds with the protein.

- Docking inside the LFABP is largely driven by hydrophobic CF chains lodging inside the hydrophobic channel of LFABP.


## Contact

For any questions, please contact Lucas Jividen at jividelh@mail.uc.edu.

