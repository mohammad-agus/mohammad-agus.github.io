---
layout: post
title: "Handling Outliers & Missing Values in Python"
info: #
tech: "python, google colab, pandas, numpy, matplotlib, scikit-learn"
type: blog, data preprocessing
---


When analyzing and modeling data, it's common to encounter outliers and missing values, which can have a significant impact on the accuracy and validity of the results. Properly addressing these issues is crucial to ensure that the analysis or modeling is based on dependable data. Outliers can distort statistical models and negatively affect prediction accuracy, while missing values can lead to biased or inaccurate results by reducing the completeness of a dataset. Correctly handling these issues is necessary to enhance the precision and reliability of the analysis or modeling.

Python pandas is a powerful data manipulation library that offers various tools for handling outliers and missing values. Python pandas offers numerous methods for addressing missing values, such as replacing them with mean, median, or mode values, or removing them altogether. Moreover, pandas provides several statistical functions to identify and handle outliers. Additionally, pandas provides several statistical functions for detecting and handling outliers, such as the interquartile range method and the Z-score method.

In this tasks, I'm using [Google Colaboratory (CoLab)](https://colab.research.google.com/) as the coding environment. Google Colaboratory is a free Jupyter notebook interactive development environment provided by Google.

Download the [dataset](https://github.com/mohammad-agus/handling_outliers_and_missing_values_in_python) to follow along.

## Import Library


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# standardize / convert to z-score
from sklearn.preprocessing import StandardScaler

# ignore warning
import warnings
warnings.filterwarnings('ignore')
```

## Handling Outliers

### Connect Google Drive to a Google Colab Notebook


```python
from google.colab import drive
drive.mount('/content/drive')
```
<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/handling-outliers-and-missing-values-in-python/v/1&cellId=5&hideInput=true" title="Jovian Viewer" height="107" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>
    

### Read Data from Google Drive


```python
cust_info = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/dataset_customers/mall_customers_info.csv')
cust_score = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/dataset_customers/mall_customers_score.csv')
customer_data_2 = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/dataset_customers/customers_data_2.csv')
```


```python
cust_info.head()
```
<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/handling-outliers-and-missing-values-in-python/v/1&cellId=8&hideInput=true" title="Jovian Viewer" height="244" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>





```python
cust_score.head()
```
<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/handling-outliers-and-missing-values-in-python/v/1&cellId=9&hideInput=true" title="Jovian Viewer" height="244" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>





```python
customer_data_2.head()
```
<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/handling-outliers-and-missing-values-in-python/v/1&cellId=9&hideInput=true" title="Jovian Viewer" height="244" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>




### Merge & Concat Data


```python
customer_data_1 = pd.merge(cust_info,cust_score[['CustomerID', 'Spending Score (1-100)']],how='inner')
customer_data_1.head()
```
<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/handling-outliers-and-missing-values-in-python/v/1&cellId=12&hideInput=true" title="Jovian Viewer" height="244" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>





```python
customer_data_1.shape
```
<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/handling-outliers-and-missing-values-in-python/v/1&cellId=13&hideInput=true" title="Jovian Viewer" height="80" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>




```python
cust_df = pd.concat([customer_data_1, customer_data_2])
cust_df.shape
```
<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/handling-outliers-and-missing-values-in-python/v/1&cellId=14&hideInput=true" title="Jovian Viewer" height="80" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>



### Outliers Handling with Inter Quartile Range

* Duplicate concatenated dataframe.


```python
df_iqr_outliers = pd.DataFrame.copy(cust_df)
```
<br/>
* Create box plot of df_iqr_outliers dataframe.


```python
df_iqr_outliers.plot(kind='box', rot=45)
```
<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/handling-outliers-and-missing-values-in-python/v/1&cellId=19&hideInput=true" title="Jovian Viewer" height="409" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe><br/>
    


* Calculate inter quartile range to generate the upper limit and the lower limit.


```python
Q1 = df_iqr_outliers['Annual_Income'].quantile(.25)
Q3 = df_iqr_outliers['Annual_Income'].quantile(.75)
iqr = Q3 - Q1
up_l = Q3 + 1.5 * iqr
lw_l = Q1 - 1.5 * iqr
print("upper limit: {} & lower limit: {}".format(up_l, lw_l))
```
<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/handling-outliers-and-missing-values-in-python/v/1&cellId=21&hideInput=true" title="Jovian Viewer" height="83" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe><br/>
    

* Using `np.where` to get index of value of Annual_Income column that greater than upper limit.

Because there is no minus value in the Annual_Income column, filtering process to imputate outliers value (replace oultiers with other value) only using the upper limit  value.


```python
outliers_index = np.where(df_iqr_outliers['Annual_Income']>up_l)
print(outliers_index)
```
<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/handling-outliers-and-missing-values-in-python/v/1&cellId=23&hideInput=true" title="Jovian Viewer" height="83" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe><br/>
    

* Filter dataframe using index from `np.where`.


```python
df_iqr_outliers.iloc[outliers_index]
```
<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/handling-outliers-and-missing-values-in-python/v/1&cellId=25&hideInput=true" title="Jovian Viewer" height="166" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe><br/>




* Replace outliers with mean of Annual_Income column (without filtering outliers).


```python
df_iqr_outliers['Annual_Income'].iloc[outliers_index] = df_iqr_outliers['Annual_Income'].mean()

df_iqr_outliers.iloc[outliers_index]
```
<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/handling-outliers-and-missing-values-in-python/v/1&cellId=27&hideInput=true" title="Jovian Viewer" height="166" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe><br/>




* Create box plot using imputated outliers dataframe.


```python
df_iqr_outliers.plot(kind='box', rot=45)
```
<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/handling-outliers-and-missing-values-in-python/v/1&cellId=29&hideInput=true" title="Jovian Viewer" height="409" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe><br/>
    


### Outliers Handling using Z-Score

* Duplicate concatenated dataframe & create box plot.


```python
df_z_score_outliers = pd.DataFrame.copy(cust_df)

df_z_score_outliers.boxplot(rot=45, grid=False)
```
<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/handling-outliers-and-missing-values-in-python/v/1&cellId=32&hideInput=true" title="Jovian Viewer" height="409" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe><br/>
    


* Standardizing (convert to z-score).


```python
scaled_df_z_score_outliers_annual_income = StandardScaler().fit_transform(df_z_score_outliers['Annual_Income'].values.reshape(-1,1))
```
<br/>
* Add standardized annual income to df_z_score_outliers dataframe.


```python
df_z_score_outliers['scaled_Annual_Income'] = scaled_df_z_score_outliers_annual_income
```


```python
df_z_score_outliers.boxplot(column='scaled_Annual_Income', grid=False)
```
<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/handling-outliers-and-missing-values-in-python/v/1&cellId=37&hideInput=true" title="Jovian Viewer" height="329" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>
    



```python
df_z_score_outliers_outliers_removed = df_z_score_outliers.drop(df_z_score_outliers.index[np.where(df_z_score_outliers['scaled_Annual_Income']>3)])
df_z_score_outliers_outliers_removed = df_z_score_outliers_outliers_removed.drop('scaled_Annual_Income', axis=1)
```


```python
df_z_score_outliers_outliers_removed.boxplot(rot=45, grid=False)
```
<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/handling-outliers-and-missing-values-in-python/v/1&cellId=39&hideInput=true" title="Jovian Viewer" height="409" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe><br/>
    


## Handling Missing Value

#### DataFrame example


```python
data = pd.DataFrame({ "A" : [4, 5, 7, np.nan, np.nan, 5, 8, np.nan, 3],
                       "B" : [100, 150, 130, 140, 180, 115, 155, 120, 105] })

print(data)
```
<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/handling-outliers-and-missing-values-in-python/v/1&cellId=42&hideInput=true" title="Jovian Viewer" height="299" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe><br/>
    

#### Fill missing value using mean (or other specific value)

* Duplicate data DataFrame.


```python
data_v1 = pd.DataFrame.copy(data)
```
<br/>
* Filter null value using isnull() == True.


```python
data_v1[data_v1["A"].isnull() == True]
```
<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/handling-outliers-and-missing-values-in-python/v/1&cellId=47&hideInput=true" title="Jovian Viewer" height="192" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe><br/>




* Calculate mean from existing data in column A that doesn't have null value.


```python
mean = data_v1["A"][data_v1["A"].isnull() == False].mean()

mean
```
<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/handling-outliers-and-missing-values-in-python/v/1&cellId=49&hideInput=true" title="Jovian Viewer" height="80" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe><br/>



* Assign mean as the replacement of missing values in column A.


```python
data_v1["A"][data_v1["A"].isnull() == True] = mean

# or using fillna()
data_v1["A"].fillna(mean, inplace=True)

print(data_v1)
```
<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/handling-outliers-and-missing-values-in-python/v/1&cellId=51&hideInput=true" title="Jovian Viewer" height="299" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>
Other value such as median, mode or a specific value can be assign using this method
<br/>
<br/>
#### Fill missing value using `pandas.fillna` methods

* Duplicate data DataFrame & using numpy.where to return the array index of missing values.


```python
data_v2 = pd.DataFrame.copy(data)

np.where(data_v2['A'].isnull())
```
<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/handling-outliers-and-missing-values-in-python/v/1&cellId=55&hideInput=true" title="Jovian Viewer" height="80" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe><br/>


* Using array index that has been generated from previous step to subset null value.


```python
data_v2.iloc[np.where(data_v2['A'].isnull())]
```
<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/handling-outliers-and-missing-values-in-python/v/1&cellId=57&hideInput=true" title="Jovian Viewer" height="192" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>
Input limit paramater to limit the maximum number of consecutive NaN values to forward/backward fill and inplace=True to fill in-place. Other paramaters can be found on [pandas documentation](https://pandas.pydata.org/docs/index.html).


```python
print(data_v2)
```
<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/handling-outliers-and-missing-values-in-python/v/1&cellId=59&hideInput=true" title="Jovian Viewer" height="299" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe><br/>
    

* Using `fillna` with `ffill` or `pad`.


```python
data_v2.fillna(method="pad")
```
<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/handling-outliers-and-missing-values-in-python/v/1&cellId=61&hideInput=true" title="Jovian Viewer" height="348" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>





```python
data_v2.fillna(method="ffill", limit=1)
```
<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/handling-outliers-and-missing-values-in-python/v/1&cellId=62&hideInput=true" title="Jovian Viewer" height="348" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe><br/>




* Using `fillna` with `backfill` or `bfill`.


```python
data_v2.fillna(method="backfill", limit=1)
```
<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/handling-outliers-and-missing-values-in-python/v/1&cellId=64&hideInput=true" title="Jovian Viewer" height="348" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>


```python
data_v2.fillna(method="bfill")
```
<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/handling-outliers-and-missing-values-in-python/v/1&cellId=65&hideInput=true" title="Jovian Viewer" height="348" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe><br/>




#### Fill missing value using `pandas.interpolate`
Fill NaN values using interpolate method (read [documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html#pandas.DataFrame.interpolate)).


```python
data_v3 = pd.DataFrame.copy(data)

print(data_v3)
```
<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/handling-outliers-and-missing-values-in-python/v/1&cellId=67&hideInput=true" title="Jovian Viewer" height="299" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>   


```python
data_v3.interpolate()
```
<iframe src="https://jovian.com/embed?url=https://jovian.com/mohammadagus1st/handling-outliers-and-missing-values-in-python/v/1&cellId=68&hideInput=true" title="Jovian Viewer" height="348" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe><br/>

Thank you for taking the time to read this post. I hope that the information and insights shared in this post have been valuable to you and have provided some helpful perspectives on the topic at hand.