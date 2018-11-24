# Email-Classification
Compare SVM &amp; Naive Bayes and Neural Network on the task of email classification

## Prerequisites
- Python 3.x.
- Skilearn, numpy installed.

## Content

### Set up:
Training dataset includes 702 emails, the first half is ham, the second half is spam. 
The test dataset includes 260 emails with the same proportion between ham and spam.

### Feature representation
Bag-of-words model is to represent the features. The feature matrix consists of vectors of length 3000.

### Models: 
Three models: **Support Vector Machine**, **Naive Bayes** and **Neural Network** are trained and tested on two variants of the dataset: one with the original feature space and one with reduced feature space by **PCA**. We use **PCA** to reduce dimensionality such that more than 85% of the variance is explained. 



