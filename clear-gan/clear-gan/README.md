# clear-gan

CLEAR-GAN: Locality-aware Hybrid Resampling for Ordinal Multiclass Imbalance



CLEAR-GAN is a data-centric hybrid resampling framework developed to jointly address class imbalance and class overlap in ordinal multiclass datasets. The method constrains overlap analysis to adjacent class pairs and integrates boundary-preserving undersampling with conditional generative oversampling to improve minority discrimination while maintaining decision boundary stability.



This repository contains the implementation and reproducibility resources for the manuscript



**Why CLEAR-GAN?**



Most resampling methods treat imbalance and overlap independently, often leading to boundary distortion or redundant synthesis. CLEAR-GAN reframes overlap in ordinal settings as a local phenomenon, enabling targeted data conditioning that improves classifier generalisation across heterogeneous datasets.



**Key properties:**



-Locality-aware overlap detection



-Boundary-preserving neighbourhood cleaning



-Conditional tabular GAN synthesis



-Classifier-agnostic design



-Scalable to multiclass settings



**Experimental Protocol:**



The implementation follows the protocol described in the manuscript:



-Stratified 5-fold cross-validation



-Hyperparameter tuning via GridSearchCV on training folds only



-Evaluation using: G-Mean, Mean Sensitivity, Class-balanced Accuracy, and Confusion Entropy



-Statistical comparison using Friedman tests with Holm post-hoc correction.



Random seeds are fixed to ensure reproducibility.



**Datasets:**



Experiments are conducted on five public datasets, alongside a proprietary dataset.

For questions, data access requests, or collaboration inquiries, please contact the corresponding author listed in the manuscript.

