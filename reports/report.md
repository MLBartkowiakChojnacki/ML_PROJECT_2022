# Abstract

## Problem statement

The aim of this project was to investigate a number of models to find one with as high performance as possible. After a preliminary study of the provided data set, it became apparent that the problem fell under the imbalanced binary classification set with both classes being important. Thus, the study could be evaluated by two certain metrics: balanced accuracy score and weighted F1 score. Due to the class distribution, the major discovery was that not every classification model was suitable for this kind of problem.  

## Metrics

*The balanced accuracy* in binary classification is a metric used to deal with imbalanced dataset. Is calculated as the average of the proportion corrects of each class individually.  

$Balanced Accuracy$ = $(Sensitivity + Specificity) \over 2$  

Where *sensitivity* is the true positive rate, true positive ratio to all positive classified, and *specificity* is the true negative rate, true negative to all negative classified respectively.  

*F1 score* is defined as the harmonic mean of *precision* and *recall*. Normally is used to evaluate model performance when the true class is important. I this case *weighted* parametr is apropriate due to both classes importance. The average calculated by this metric is weighted by the *support*, which is the number of samples with a given label.  

$F1 score$ = $2 * Precision * Recall \over Precision + Recall$  

Where *precision* is the ratio of true positives to all labeled positives, and *recall* is true positives to all positives ratio.  

# Analysis

## Data exploration

### Features

A collection of data intended for this project is consisting of training set of 3750 samples  associated with a label from 2 classes and test set of 1250 samples. Each set has 10000 features, floating point type. Data set is complete and does not show any duplicates. Contamination by outliers is very low, less than 1% in the entire dataset, the highest indicator for columns reaches 2.8% and for rows 0.56%. Due to the low contamination, outliers do not need to be removed from the data.  

Most features present signs of Gaussian distribution, close to 95% passed the normal distribution test, what can be perfectly shown on the chart below after standardization.  

![Feature distribution](figures/feature_distribution.jpg "Feature distribution")  

There are no significant correlations between features. Only one feature reaches positive correlation indicator above 0.7, the rest is neutral, oscilating below 0.1. There is no negative correlation at all.  

### Target

The target contains two classes: "-1" and "1" with 1:9 ratio, as shown in the chart below.  

![Class distribution](figures/target_distribution.jpg "Class distribution")  

Features to the target also do not show any relevant correlation, any result can be found between -0.07 and 0.07.  

# Techniques and methodology

## Baseline

Searching for best model must be preceded by creating a baseline. For this purpose a `DummyClassifier` has been chosen. It is the simplest classificator that makes predictions absolutely ignoring the input features. It is working on one of five strategies:  

- *most_frequent* - always returns most frequent class label as a prediction,
- *prior* - similarly to *most_frequent*, always returns most frequent class label as a prediction, 
- *stratified* - returns the label, which got probability one in the one-hot vector of `predict_proba`,
- *uniform* - each class has equal probability, thus prediction is generated randomly,
- *constant* - constant label provided by the user is a prediction

## Data preprocessing

At the beginning the idea was to standardize the data by `StandardScaler` to bring the various numbers down to a common level and perform better scores. It standardize each feature independently by removing the mean and scaling to unit variance by following formula:  

$Z$ = $(x - u)\over s$  

Where $x$ is a sample, $u$ is the mean of the training samples and $s$ is the standard deviation.  

Removing outliers using *Interquartile range* method was a subsequent step to minimize the risk of misjudged results. This is a measure of statistical dispersion, which is the spread od the data. To calculate the IQR, the data set is divided into quartiles:  

- Q1 - known as lower quartile, 25th percentile,
- Q2 - median, 50th percentile,
- Q3 - upper quartile, 75th percentile,
  
and then by removing lower quartile from the upper quartile:  

$IQR$ = $Q3 - Q1$  

In the end new data set is created through selecting samples, which fit to the range between $(Q1 - 1.5*IQR)$ and $(Q3 + 1.5*IQR)$.  

The last step was reduction of dimensions based on ANOVA test, which is a statistical technique used to check if the means of two or more groups significanlty differ. After these steps it turned out that clusters are indistinguishable, as can be seen in the attached diagram.  

![Cluster analysis](figures/cluster_analysis.jpg "Cluster analysis")  

Thus, every step taken had to be rethink. First of all was to look once more into the data and check if clusters can be seen without any preprocessing. For this purpose elbow method and PCA with 1 and 2 components was used. First diagram clearly shows that the labels overlap in a certain area, there is no significant gap in distribution of label "1" (yellow color). Second graph is more optimistic, two structured clusters can be seen on the chart, but the clouds belong to one class. After applying the colors, second cluster emerges between two strong appearances of the first class.  

![Supervised clustering](figures/clustering.jpg "Supervised clustering")  

Feature selection consisted in retain the data, which explained 95% of the variance. PCA with 0.95 components has been used to achieve this goal.  

## Finding best model

### Dataset split

Splitting the data set is the most important part of the entire machine learning process. The more varied the training set the better the model performance. It also raises the chance of avoiding overfitting. Imbalanced classification requires every precaution to be taken. `RepeatedStratifiedKFold` seemd to be the best option for this challenge, for the sake of several functionalities it offers:  

- splitting entire data into $n$-assigned folds,
- maintaining a proper data balance,
- paying attention to data randomization,
- repetitiveness.  

### Classifiers selection

The longest and the hardes part of the entire project was selection algorithms, which will be able to catch regularities among this imbalanced set of data. After insightful analysis a few algorithms have been chosen. 

ALGORITHMS IN A TABLE WITH PARAMS

EVERY ALGORITHM WITH PERFORMANCE ON A CHART

# Conclusions

Porównanie z baseline
metryki
confusion matrix
classisifaction report