---
layout: post
title: "Performing Linear Regression Analysis (Ordinary Least Square) Using Python Statsmodels"
info: #
tech: "python, jupyter notebook, pandas, numpy, scipy, statsmodels, matplotlib, seaborn"
type: project, linear regression
---


The objective of this project is to perform linear regression analysis (ordinary least square technique) using Statsmodels (a Python library) to predict the car price, based on the automobile dataset from [UCI Machine Learning repository](https://archive.ics.uci.edu/ml/datasets/automobile), which is a common dataset for regression analysis. The automobile dataset is from the year 1985 which is quite old, but it's suitable for the learning purposes of this project. You can find the project's jupyter notebook and the dataset (if you want to skip extracting step) on my [GitHub repository](https://github.com/mohammad-agus/linear_regression_ordinary_least_square).

### Import Library


```python
import numpy as np
import pandas as pd

import scipy.stats as stats
import statsmodels.api as sm

import seaborn as sns
import matplotlib.pyplot as plt

# download dataset description from the source
from urllib import request 

import warnings
warnings.filterwarnings('ignore')
```

# Data Preprocessing & Exploration

### Extract the Dataset

* Download and read the dataset description.


```python
data_desc_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.names"
response = request.urlretrieve(data_desc_url, "dataset/data_desc.names")

with open("dataset/data_desc.names") as data_desc:
    print(data_desc.read())
```

<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/linear-regression-ordinary-least-square-using-python-statsmodels-ipynb/v/1&cellId=6&hideInput=true" title="Jovian Viewer" height="800" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>
    
<br/>
* Read the dataset and assign the column header.


```python
# url source dataset
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data" 
df = pd.read_csv(data_url)
df.head()
```


<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/linear-regression-ordinary-least-square-using-python-statsmodels-ipynb/v/1&cellId=8&hideInput=true" title="Jovian Viewer" height="348" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>


As shown above, the dataset doesn't contain the column header. Here is the column list that has been created manually based on the dataset descriptions.


```python
column = [  'symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration',
            'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'wheel-base',
            'length', 'width', 'height', 'curbweight', 'engine-type', 'num-of-cylinders',
            'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 
            'peak-rpm', 'city-mpg', 'highway-mpg', 'price' ]

data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
df.columns = column
df.head()
```




<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/linear-regression-ordinary-least-square-using-python-statsmodels-ipynb/v/1&cellId=10&hideInput=true" title="Jovian Viewer" height="362" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>



### Data Cleaning

* Check the null value.
<br/>
The null values denoted by '?' mark.


```python
print(df.shape)
print(df.isin(['?']).sum())
```

<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/linear-regression-ordinary-least-square-using-python-statsmodels-ipynb/v/1&cellId=13&hideInput=true" title="Jovian Viewer" height="731" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>
    
<br/>
* Handle the missing values.
<br/>
All the null values will be removed, starting with normalized-losses column (otherwise all 40 rows will be removed) along with the symboling column because there is no enough information about this feature. For the rest of the missing values, replace '?' mark with a null value (numpy.nan) then perform dropna().


```python
df.drop(['normalized-losses','symboling'], axis=1, inplace=True)
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)
df.head()
```




<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/linear-regression-ordinary-least-square-using-python-statsmodels-ipynb/v/1&cellId=15&hideInput=true" title="Jovian Viewer" height="362" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>



```python
df.shape
```




    (192, 24)


<br/>
After the null values were removed, now the dataset consist of 192 rows and 24 features (columns).
<br/>
* Check the data type of df dataset columns.


```python
df.info()
```
<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/linear-regression-ordinary-least-square-using-python-statsmodels-ipynb/v/1&cellId=19&hideInput=true" title="Jovian Viewer" height="800" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>

The df.info() shows that there are some columns hasn't have the correct data type.
<br/>
* Convert the columns datatype.


```python
# to float
to_float = ['bore', 'stroke', 'horsepower', 'peak-rpm', 'price']

for i in to_float:
    df[i] = df[i].astype(float)
    
df.info()
```
<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/linear-regression-ordinary-least-square-using-python-statsmodels-ipynb/v/1&cellId=22&hideInput=true" title="Jovian Viewer" height="800" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>

Now, the columns have the correct data type and it's ready for some calculations.
<br/>
* Histogram of the Car Price.
<br/>
The value that will be predicted (Price) is called the Target or Dependent Variable and the predictor is called the Features or Independent Variables.


```python
plt.figure(figsize=(12,8))
sns.displot(df['price'], kde=True)
plt.title('Car Price Distribution Data')
plt.show()
```
<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/linear-regression-ordinary-least-square-using-python-statsmodels-ipynb/v/1&cellId=25&hideInput=true" title="Jovian Viewer" height="440" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>
    


As shown in the plot above, most of the car prices are concentrated between the range 5000 to 20000 and the data has right-skewed distribution.
<br/>
* Price range based on categorical variables.


```python
cat_var = df.loc[: , df.columns!='make'].select_dtypes(include=['object'])

plt.figure(figsize=(12,8))

# enumerate(df.columns): return list of tuple(index column, name of column)
for i in enumerate(cat_var.columns):
    
    # 3 rows, 3 columns, index enumerate() + 1
    plt.subplot(3,3, i[0]+1) 
    
    # x: name of the column
    sns.boxplot(data=df,  x=i[1], y='price', palette="husl") 
    plt.xticks(rotation = 45)
    plt.tight_layout()
```


<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/linear-regression-ordinary-least-square-using-python-statsmodels-ipynb/v/1&cellId=28&hideInput=true" title="Jovian Viewer" height="481" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>
    


There are some outliers data based on the plots, but it's permissible for several reasons. Their occurrence is natural and the error can't be identified or corrected. So, in this step, the outliers won't be removed.
<br/>
* Summary of the numerical value.


```python
num_var = df.loc[: , df.columns!='make'].select_dtypes(include=['float','int64'])
num_var.describe() # or df.describe
```

<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/linear-regression-ordinary-least-square-using-python-statsmodels-ipynb/v/1&cellId=31&hideInput=true" title="Jovian Viewer" height="355" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>


<br/>
* Plot a heatmap based on the correlation value of each numerical variables.


```python
plt.figure(figsize=(12,8))
sns.heatmap(num_var.corr(), annot=True, cmap='Blues')
plt.title('Correlation Heatmap of Numerical Variables')
plt.show()
```

 <iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/linear-regression-ordinary-least-square-using-python-statsmodels-ipynb/v/1&cellId=33&hideInput=true" title="Jovian Viewer" height="544" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>
    


Based on the heatmap, several variables have strong positive or negative correlations. In the simple linear regression analysis, engine-size will be used as the predictor. Then, all the categorical and numerical variables that have been explored before, will be used in the multiple linear regression analysis as well.
<br/>
# Simple Linear Regression

* Scatter plot and histogram (joint plot) of engine-size vs price.


```python
plt.figure(figsize=(12,8))
sns.jointplot(data=df, x='engine-size', y='price', kind='reg').fig.suptitle("Scatter Plot & Histogram of Engine Size vs. Price")
plt.tight_layout()
plt.show()
```
<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/linear-regression-ordinary-least-square-using-python-statsmodels-ipynb/v/1&cellId=37&hideInput=true" title="Jovian Viewer" height="498" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>
    

<br/>
* Duplicate df dataframe and randomly return the sample of the dataframe.
<br/>
Parameter frac=1 means 100% of the dataframe will return randomly and the random_state parameter is for reproducibility (similar to random seed in numpy).


```python
df_slr = df.copy() # slr: simple linear regression
df_slr = df_slr.sample(frac=1, random_state=101).reset_index(drop=True)

df_slr.head()
```



<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/linear-regression-ordinary-least-square-using-python-statsmodels-ipynb/v/1&cellId=39&hideInput=true" title="Jovian Viewer" height="334" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>

<br>
* Assign engine-size as the feature (x) and price as the target (y).

```python
x_slr = df_slr['engine-size']
y_slr = df_slr['price']
```
<br>
* Add constant (represent of intercept) to x variable (where y = a + bx).


```python
x_slr = sm.add_constant(x_slr)
x_slr.head()
```




<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/linear-regression-ordinary-least-square-using-python-statsmodels-ipynb/v/1&cellId=43&hideInput=true" title="Jovian Viewer" height="243" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>

<br>
* Split data into a training set and testing set.
<br/>
Split 75% of the data into a training set and 25% for testing the model. Scikit-learn library has the same function for this kind of task (model_selection.train_test_split).


```python
train_size = int(0.75 * len(x_slr))

x_train_slr = x_slr[:train_size]
y_train_slr = y_slr[:train_size]

x_test_slr = x_slr[train_size:]
y_test_slr = y_slr[train_size:]

x_train_slr.shape, x_test_slr.shape
```




    ((144, 2), (48, 2))


<br>
* Fit linear regression model and view the summary.


```python
lm_slr = sm.OLS(y_train_slr, x_train_slr).fit()

lm_slr.summary()
```



<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/linear-regression-ordinary-least-square-using-python-statsmodels-ipynb/v/1&cellId=47&hideInput=true" title="Jovian Viewer" height="585" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>


<br/>
* Plot the linear regression model.


```python
fig = plt.figure(figsize=(12,8))

fig = sm.graphics.plot_regress_exog(lm_slr, 'engine-size', fig=fig)

plt.show()
```
<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/linear-regression-ordinary-least-square-using-python-statsmodels-ipynb/v/1&cellId=50&hideInput=true" title="Jovian Viewer" height="513" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>
    


The statsmodels' plot_regress_exog function allows for viewing regression results against a single regressor, which in this case is enginesize. Four different plots are generated by this function:

* The upper-left ('Y and Fitted vs. X') plot displays the fitted values of the regression line (in red) versus the actual values of enginesize and price, with vertical lines representing prediction confidence intervals for each fitted value.
* The second plot, showing the residuals of the regression versus the predictor variable (enginesize), can help identify any non-linear patterns. If residuals are evenly spread out around the 0 line, it indicates that the regression model does not have any non-linear patterns.
* The Partial regression plot is used to demonstrate the effect of adding an independent variable to a model that already has one or more independent variables. As this is a single-variable model, the Partial regression plot simply displays a scatter plot of price versus horsepower with a fitted regression line.
* Lastly, the CCPR (Component-Component Plus Residual) plot allows for assessing the impact of one regressor (enginesize) on the response variable (price) while accounting for the effects of other independent variables. In this case, as there are no other independent variables in this regression, the plot simply shows a scatter plot with a linear model fit on the data.

<br/>
* Influence plot of the linear regression model.


```python
sm.graphics.influence_plot(lm_slr)
```

<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/linear-regression-ordinary-least-square-using-python-statsmodels-ipynb/v/1&cellId=53&hideInput=true" title="Jovian Viewer" height="663" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>


The influence_plot can be utilized to gain a deeper understanding of the regression model. This plot enables the identification of records in the dataset that have had a significant influence on the regression analysis. The influential data points can be recognized by their large circles in the plot. For example, the data point with ID 103 has a significant impact on the regression results.
<br/>
* Predict the testing data using the model.


```python
y_pred_slr = lm_slr.predict(x_test_slr)

y_pred_slr.head()
```
<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/linear-regression-ordinary-least-square-using-python-statsmodels-ipynb/v/1&cellId=56&hideInput=true" title="Jovian Viewer" height="185" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>


<br/>
* Plot the the test value and the corresponding predicted value.


```python
fig, ax = plt.subplots(figsize=(12,8))

plt.scatter(x_test_slr['engine-size'], y_test_slr)
plt.plot(x_test_slr['engine-size'], y_pred_slr, color='g')

plt.xlabel('Engine Size')
plt.ylabel('Price')

plt.show()
```


    
<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/linear-regression-ordinary-least-square-using-python-statsmodels-ipynb/v/1&cellId=58&hideInput=true" title="Jovian Viewer" height="473" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>
    


The fitted line of the predicted values can be seen here. Here is a scatter plot of the test data (engine size vs. price) and the matching predicted value for each x value in the test data. It appears to have quite accurately approximated or fitted the test data.


<br/>
# Multiple Linear Regression

* Duplicate the df dataframe and show the columns of the duplicated dataframe.


```python
df_mlr = df.copy() # mlr: multiple linear regression
df_mlr.columns
```
<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/linear-regression-ordinary-least-square-using-python-statsmodels-ipynb/v/1&cellId=64&hideInput=true" title="Jovian Viewer" height="185" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>
<br/>


* Create a list that contains only two unique values and perform the label encoding.
<br/>
Some of the variables in this linear regression analysis are the pre-determined values that are categorical. So to use these variables as predictors, it has to be encoded or converted to numeric values in binary form. It will be using the LabelEncoder from the scikit‑learn library. To every discrete value that these variables take on, the label and quota will assign a unique integral value. For example, gas might be 0, and fuel-type diesel will be 1.


```python
cols = ['fuel-type','aspiration','num-of-doors','engine-location']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for i in cols:
    df_mlr[i] = le.fit_transform(df_mlr[i])

# similar to DataFrame.head() but it returning the data randomly    
df_mlr[cols].sample(5) 
```



<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/linear-regression-ordinary-least-square-using-python-statsmodels-ipynb/v/1&cellId=66&hideInput=true" title="Jovian Viewer" height="243" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>
<br/>


* Perform 'one-hot' encoding to other categorical variables.
<br/>
One‑hot encoding is a converting process to represent categorical variables in numeric form (such as the previous one). One‑hot encoding will replace the original column with a new column, one corresponding to each category value. So there will be a column for convertible, a column for sedan, a column for hatchback, and so on. A value of 1 will indicate that the car belongs to that category. A value of 0 indicates the car does not belong to a category. This one‑hot encoding will be using pd.get_dummies. Then remove the original columns using the DataFrame.drop function.


```python
df_mlr.drop(['make'], axis=1, inplace=True)
cat_columns = ['body-style', 'engine-type', 'drive-wheels', 'num-of-cylinders', 'fuel-system']
for i in cat_columns:
    df_mlr = pd.concat([df_mlr.drop(i, axis=1),
                        pd.get_dummies(df_mlr[i],
                                        prefix = i,
                                        prefix_sep = '_',
                                        drop_first = True)], axis=1)

df_mlr.head()
```



<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/linear-regression-ordinary-least-square-using-python-statsmodels-ipynb/v/1&cellId=68&hideInput=true" title="Jovian Viewer" height="334" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>




```python
# get the number of rows and columns
df_mlr.shape 
```




    (192, 39)

<br/>

* Duplicate the dataframe, assign it into features and target, then add constant.


```python
 # copy the dataframe
df_mlr = df_mlr.sample(frac=1, random_state=101).reset_index(drop=True)

# assign df_mlr into the x variables (exclude price variable)
x_mlr = df_mlr.drop(['price'], axis=1)

# assign df_mlr['price'] as the target
y_mlr = df_mlr['price'] 

# add constant
x_mlr = sm.add_constant(x_mlr) 

x_mlr.head()
```



<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/linear-regression-ordinary-least-square-using-python-statsmodels-ipynb/v/1&cellId=71&hideInput=true" title="Jovian Viewer" height="334" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>

<br/>
* Split the features and the target into train and test set.


```python
train_size = int(0.75 * len(x_mlr))

x_train_mlr = x_mlr[:train_size]
y_train_mlr = y_mlr[:train_size]

x_test_mlr = x_mlr[train_size:]
y_test_mlr = y_mlr[train_size:]
```
<br/>
* Fit the linear model.


```python
lm_mlr = sm.OLS(y_train_mlr, x_train_mlr).fit()

lm_mlr.summary()
```



<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/linear-regression-ordinary-least-square-using-python-statsmodels-ipynb/v/1&cellId=75&hideInput=true" title="Jovian Viewer" height="800" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>




* Predicting the testing data.


```python
y_pred_mlr = lm_mlr.predict(x_test_mlr)

y_pred_mlr.head()
```
<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/linear-regression-ordinary-least-square-using-python-statsmodels-ipynb/v/1&cellId=78&hideInput=true" title="Jovian Viewer" height="263" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>
<br/>


* Create a dataframe to plot the test target values and the predicted values.


```python
data_actual_pred = pd.DataFrame({'Actual Value' : y_test_mlr.ravel(),
                                 'Predicted Value' : y_pred_mlr})

data_actual_pred.head()
```



<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/linear-regression-ordinary-least-square-using-python-statsmodels-ipynb/v/1&cellId=80&hideInput=true" title="Jovian Viewer" height="243" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>
<br/>


* Transform (unpivot) data_actual_pred dataframe into a suitable form for plotting.


```python
melted_data_actual_pred = pd.melt(data_actual_pred.reset_index(),
                                   id_vars=['index'],
                                   value_vars=['Actual Value', 'Predicted Value'])
```


<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/linear-regression-ordinary-least-square-using-python-statsmodels-ipynb/v/1&cellId=82&hideInput=true" title="Jovian Viewer" height="243" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>
<br/>


* Create a comparison lineplot of the actual value vs. the predicted value.

```python
plt.figure(figsize=(12,8))
sns.lineplot(data=melted_data_actual_pred, x='index', y='value', hue='variable')

plt.show()
```

<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/linear-regression-ordinary-least-square-using-python-statsmodels-ipynb/v/1&cellId=84&hideInput=true" title="Jovian Viewer" height="473" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>

The plot shows the actual and the predicted values are close. It indicates this is a good model.




# Conclusions
## Simple Linear Regression
* The model's R-squared is 0.792 or 79.2%, indicating that the linear model is reasonably effective in predicting price, given that it utilizes only one feature. This value signifies that the model captured 79.2% of the variance in the underlying data.
* The F-statistic and corresponding P-value evaluate the validity of the regression analysis in its entirety. A P-value less than 5% indicates the analysis is valid. The null hypothesis for this test is that all regression coefficients equal zero, which is not the case in this scenario. Therefore, the null hypothesis can be rejected, and the alternative hypothesis that regression coefficients are not equal to zero can be accepted.
* For each regression coefficient, a P-statistic and corresponding P-value quantify the precision and validity of the coefficient. The P-values of 0.000 for the intercept and 0.000 for the enginesize coefficient demonstrate their validity. The T-statistic and P-value confirm the statistical significance of the relationship between enginesize and price. In this simple linear regression, the coefficient of engine-size predictor is 181.8057, which means a 1 unit increase in engine-size will be adding 181.8057 to the car price.
## Multiple Linear Regression
* The R‑squared score has increased from 0.792 to 0.956 (95.6%) and the adjusted R-squared score is 0.941. In multiple linear regression, it's necessary to evaluate the adjusted R-squared because not all the predictors are relevant and the adjusted R-squared applies penalty calculations to the irrelevant variables that are included in the regression analysis. The score of R-squared and the adjusted R-squared is slightly different, this indicates there is an irrelevant predictor in this model. Below the adjusted R-squared there are the F-statistics and the corresponding p-value for the analysis. The p‑value is under the significant threshold of 5% indicating that this is a valid regression analysis.
* Each predictor in this model has a coefficient of regression, t-statistic, and p-value that indicates the validity of that regression coefficient. Take a look at the p-value of engine-size coefficient which is 0. The null hypothesis for this statistical test is that this coefficient has no impact or effect on the regression. The alternative hypothesis is that this coefficient has an impact or effect on the regression. With the p‑value of 0, the alternative hypothesis is accepted if the p-value is under the significance threshold (0.05 or 5%). All the predictor's coefficient that has the corresponding p-value above the significance threshold doesn't have effects on this regression, so it's considered irrelevant predictors in this linear regression model.

Lastly, according to Chin (1998), the R-squared score that more than 0.67 is categorized as a substantial. Therefore, the closer the R-squared value is to 1, the better the fit of the model.


# Credits
* Janani Ravi, for providing a clear and insightful explanation of linear regression in the course "Foundations of Statistics and Probability for Machine Learning" on [Pluralsight](https://app.pluralsight.com/library/courses/foundations-statistics-probability-machine-learning/table-of-contents). Janani Ravi's explanation was a valuable resource in the development of this article.
* Chin, W. W. (1998). [The Partial Least Squares Aproach to Structural Equation Modeling. Modern Methods for Business Research](https://www.researchgate.net/publication/311766005_The_Partial_Least_Squares_Approach_to_Structural_Equation_Modeling), 295, 336