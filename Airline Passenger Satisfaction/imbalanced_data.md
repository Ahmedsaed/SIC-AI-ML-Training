# Imbalanced Data

A classification data set with skewed class proportions is called imbalanced. Classes that make up a large proportion of the data set are called majority classes. Those that make up a smaller proportion are minority classes.

What counts as imbalanced? The answer could range from mild to extreme, as the table below shows.

| Degree of imbalance | Proportion of Minority Class |
| --- | --- |
| Mild | 20-40% of the data set |
| Moderate | 1-20% of the data set |
| Extreme	| <1% of the data set |

<img src="https://developers.google.com/static/machine-learning/data-prep/images/distribution-true-v2.svg" width="400em" />

Why would this be problematic? With so few positives relative to negatives, the training model will spend most of its time on negative examples and not learn enough from positive ones. For example, if your batch size is 128, many batches will have no positive examples, so the gradients will be less informative.

# Techniques for handling imbalanced data

1. Downsampling refers to removing records from majority classes in order to create a more balanced dataset. The simplest way of downsampling majority classes is by randomly removing records from that category. Let’s walk through an example.

```python
from sklearn.utils import resample
downsample = resample(major_class,
             replace=True,
             n_samples=len(minor_class),
             random_state=42)
```

2. Upsampling refers to manually adding data samples to the minority classes in order to create a more balanced dataset. Upsampling By Copying Minority Class Instances

```python
from sklearn.utils import resample
upsample = resample(minor_class,
             replace=True,
             n_samples=len(major_class),
             random_state=42)

```

Upsampling by simply copying records may lead to overfitting when you train machine learning models. Techniques have been developed that add instances to dataset which are not exactly the copy of existing instances but are very similar to the original instances.

3. Upsampling with SMOTE (Synthetic Minority Over-sampling Technique)

```python
%pip install imbalanced-learn

from imblearn.over_sampling import SMOTE

su = SMOTE(random_state=42)
X_su, y_su = su.fit_resample(X, y)
```

