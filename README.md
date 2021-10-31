# Imbalanced-Classification

1. Imbalanced Classification Problems
2. Causes of Class Imbalance
3. Challenge of Imbalanced Classification
4. Examples of Imbalanced Classification
5. [How to solve imbalance dataset problem](#1)

[Source]('https://machinelearningmastery.com/what-is-imbalanced-classification/')

## 1. Imbalanced Classification Problems

The number of examples that belong to each class may be referred to as the class distribution.

Imbalanced classification refers to a classification predictive modeling problem where the number of examples in the training dataset for each class label is not balanced.

That is, where the class distribution is not equal or close to equal, and is instead biased or skewed.

Imbalanced Classification: A classification predictive modeling problem where the distribution of examples across the classes is not equal.

For example, we may collect measurements of flowers and have 80 examples of one flower species and 20 examples of a second flower species, and only these examples comprise our training dataset. This represents an example of an imbalanced classification problem.

`An imbalance occurs when one or more classes have very low proportions in the training data as compared to the other classes.`
There are other less general names that may be used to describe these types of classification problems, such as:

  *  Rare event prediction.
  *  Extreme event prediction.
  *  Severe class imbalance.

## 2. Causes of Class Imbalance

The imbalance to the class distribution in an imbalanced classification predictive modeling problem may have many causes.

There are perhaps two main groups of causes for the imbalance we may want to consider; they are data sampling and properties of the domain.

It is possible that the imbalance in the examples across the classes was caused by the way the examples were collected or sampled from the problem domain. This might involve biases introduced during data collection, and errors made during data collection.

*    Biased Sampling.
*    Measurement Errors.

## 3. Challenge of Imbalanced Classification

The imbalance of the class distribution will vary across problems.

A classification problem may be a little skewed, such as if there is a slight imbalance. Alternately, the classification problem may have a severe imbalance where there might be hundreds or thousands of examples in one class and tens of examples in another class for a given training dataset.

*    **Slight Imbalance:** An imbalanced classification problem where the distribution of examples is uneven by a small amount in the training dataset (e.g. 4:6).
*    **Severe Imbalance:** An imbalanced classification problem where the distribution of examples is uneven by a large amount in the training dataset (e.g. 1:100 or more).

Most of the contemporary works in class imbalance concentrate on **imbalance ratios ranging from 1:4 up to 1:100.** […] `In real-life applications such as fraud detection or cheminformatics we may deal with problems with imbalance ratio ranging from 1:1000 up to 1:5000.`

Some examples of Imbalance Dataset

*    Fraud Detection.
*    Claim Prediction
*    Default Prediction.
*    Churn Prediction.
*    Spam Detection.
*    Anomaly Detection.
*    Outlier Detection.
*    Intrusion Detection
*    Conversion Prediction.

# 5. 7 Techniques to Handle Imbalanced Data 
[Source]('https://www.kdnuggets.com/2017/06/7-techniques-handle-imbalanced-data.html')

## 1. Use the right evaluation metrics

 
Applying inappropriate evaluation metrics for model generated using imbalanced data can be dangerous. Imagine our training data is the one illustrated in graph above. If accuracy is used to measure the goodness of a model, a model which classifies all testing samples into “0” will have an excellent accuracy (99.8%), but obviously, this model won’t provide any valuable information for us.

In this case, other alternative evaluation metrics can be applied such as:

    Precision/Specificity: how many selected instances are relevant.
    Recall/Sensitivity: how many relevant instances are selected.
    F1 score: harmonic mean of precision and recall.
    MCC: correlation coefficient between the observed and predicted binary classifications.
    AUC: relation between true-positive rate and false positive rate. 

 
## 2. Resample the training set

 
Apart from using different evaluation criteria, one can also work on getting different dataset. Two approaches to make a balanced dataset out of an imbalanced one are under-sampling and over-sampling.

2.1. Under-sampling

Under-sampling balances the dataset by reducing the size of the abundant class. This method is used when quantity of data is sufficient. By keeping all samples in the rare class and randomly selecting an equal number of samples in the abundant class, a balanced new dataset can be retrieved for further modelling.

2.2. Over-sampling

On the contrary, oversampling is used when the quantity of data is insufficient. It tries to balance dataset by increasing the size of rare samples. Rather than getting rid of abundant samples, new rare samples are generated by using e.g. repetition, bootstrapping or SMOTE (Synthetic Minority Over-Sampling Technique) [1].

Note that there is no absolute advantage of one resampling method over another. Application of these two methods depends on the use case it applies to and the dataset itself. A combination of over- and under-sampling is often successful as well.

 
## 3. Use **K-fold Cross-Validation** in the right way

 
It is noteworthy that cross-validation should be applied properly while using over-sampling method to address imbalance problems.

Keep in mind that over-sampling takes observed rare samples and applies bootstrapping to generate new random data based on a distribution function. If cross-validation is applied after over-sampling, basically what we are doing is overfitting our model to a specific artificial bootstrapping result. That is why cross-validation should always be done before over-sampling the data, just as how feature selection should be implemented. Only by resampling the data repeatedly, randomness can be introduced into the dataset to make sure that there won’t be an overfitting problem.

 
## 4. Ensemble different resampled datasets

 
The easiest way to successfully generalize a model is by using more data. The problem is that out-of-the-box classifiers like logistic regression or random forest tend to generalize by discarding the rare class. One easy best practice is building n models that use all the samples of the rare class and n-differing samples of the abundant class. Given that you want to ensemble 10 models, you would keep e.g. the 1.000 cases of the rare class and randomly sample 10.000 cases of the abundant class. Then you just split the 10.000 cases in 10 chunks and train 10 different models.

Imbalanced data image

This approach is simple and perfectly horizontally scalable if you have a lot of data, since you can just train and run your models on different cluster nodes. Ensemble models also tend to generalize better, which makes this approach easy to handle.

 
## 5. Resample with different ratios

 
The previous approach can be fine-tuned by playing with the ratio between the rare and the abundant class. The best ratio  heavily depends on the data and the models that are used. But instead of training all models with the same ratio in the ensemble, it is worth trying to ensemble different ratios.  So if 10 models are trained, it might make sense to have a model that has a ratio of 1:1 (rare:abundant) and another one with 1:3, or even 2:1. Depending on the model used this can influence the weight that one class gets.

Imbalanced data image

 
## 6. Cluster the abundant class

 
An elegant approach was proposed by Sergey on Quora [2]. Instead of relying on random samples to cover the variety of the training samples, he suggests clustering the abundant class in r groups, with r being the number of cases in r. For each group, only the medoid (centre of cluster) is kept. The model is then trained with the rare class and the medoids only.

 
## 7. Design your own models

 
All the previous methods focus on the data and keep the models as a fixed component. But in fact, there is no need to resample the data if the model is suited for imbalanced data. The famous XGBoost is already a good starting point if the classes are not skewed too much, because it internally takes care that the bags it trains on are not imbalanced. But then again, the data is resampled, it is just happening secretly.

By designing a cost function that is penalizing wrong classification of the rare class more than wrong classifications of the abundant class, it is possible to design many models that naturally generalize in favour of the rare class. For example, tweaking an SVM to penalize wrong classifications of the rare class by the same ratio that this class is underrepresented.

Imbalanced data image

 
### Final Remarks

 
This is not an exclusive list of techniques, but rather a starting point to handle imbalanced data. There is no best approach or model suited for all problems and it is strongly recommended to try different techniques and models to evaluate what works best. Try to be creative and combine different approaches. It is also important, to be aware that in many domains (e.g. fraud detection, real-time-bidding), where imbalanced classes occur, the “market-rules” are constantly changing. So, check if past data might have become obsolete.
