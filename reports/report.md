# Abstract

The aim of this project was to investigate a number of models to find one with a high performance. After a preliminary study of the provided data set, it became apparent that the problem fell under the imbalanced binary classification set with both classes being important. Thus, the study could be evaluated by two certain metrics: balanced accuracy score and F1 score. Due to the class distribution, the major discovery was that not every classification model was suitable for this kind of problem.

# Dataset description

A collection of data intended for this project is consisting of training set of 3750 samples and test set of 1250 samples, associated with a label from 2 classes. Each set has 10000 features, floating point type. Data set is complete and does not show any duplicates. Contamination by outliers is very low, less than 1% in entire dataset, (MAX ROW, MAX COL) Due to the low contamination, outliers do not need to be removed from the data.

[Chart min/max/mean]

There are no significant correlations between features. [above 0.7 / weak positive corr / neutral / weak negative corr / under -0.7]

The target (or class) distribution is 1:9 labeled by 1 or -1 (which had to be renamed for better performance), as shown in the chart below.

[Target distribution chart](!pic of target!)

Features to the target also does not show any relevant correlation, any result can be found between -0.07 and 0.07.

Most features presents signs of Gaussian distribution, close to 95% passed the normal distribution test, what can be perfectly shown on the chart below after standardization. [standard scaler chart]

# Technics and workflow

At the beginning the idea was to standardize the data by StandardScaler to bring the various numbers down to a common level and perform better scores. Removing outliers using IQR method was a subsequent step to minimize the risk of erroneous results. And the last step was reduction dimensions based on ANOVA test, which is a statistical technique that is used to check ( or: the impact of one or more factors by comparing the means of different samples. ) if the means of two or more groups are significanlty different from each other. After these steps it turned out that clusters are indistinguishable, as can be seen in the attached diagram.

[PCA diagram]

That surprise made us rethink every taken step. First of all was to look once more into the data and check if clusters can be seen without any preprocessing. For this purpose was used PCA with 1 and 2 components. First diagram clearly shows that the labels overlaps in a certain area, there is not any significant gap in label "1" distribution. Second graph is more optimistic, two structured clusters can be seen on the chart, but these clouds belong to one class. After applying the colors, second cluster emerges between two strong appearances of the first class. 

[PCA 3 diagrams]

This discovery zapoczątkowało SMOTE, ale po smote model overfittował, więc wycofaliśy się ze smote i było spoko, dobrze sobie radził KNN 