---
layout: post
title: "Exploratory Data Analysis of 911 Calls Data"
info: #"exploratory data analysis of 911 calls data"
tech: "python, jupyter notebook, pandas, matplotlib, seaborn"
type: blog, EDA
---


This project is a data capstone project which is part of Udemy course: [Python for Data Science and Machine Learning](https://www.udemy.com/course/python-for-data-science-and-machine-learning-bootcamp/) by [Jose Portilla](https://www.udemy.com/user/joseportilla/). This capstone project is about analyzing some 911 call data from [Kaggle](https://www.kaggle.com/mchirico/montcoalert) using python libraries such as pandas, matplotlib and seaborn.

The data contains the following fields:

* lat : String variable, Latitude
* lng: String variable, Longitude
* desc: String variable, Description of the Emergency Call
* zip: String variable, Zipcode
* title: String variable, Title
* timeStamp: String variable, YYYY-MM-DD HH:MM:SS
* twp: String variable, Township
* addr: String variable, Address
* e: String variable, Dummy variable (always 1)


## Data and Setup

____
**Import libraries and set `%matplotlib inline`**<b>
    
Python predefined (magic) function `%matplotlib inline` is used to enable the inline plotting, where the plots/graphs will be displayed just below the cell where plotting commands are written.<br>*Source:* [pythonguides.com](https://pythonguides.com/what-is-matplotlib-inline/)


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


**Read data**<b>
    
Read in the csv file as a dataframe called df.


```python
df = pd.read_csv('911.csv')
```

<br>**df DataFrame Summary**<br>

Check the head of df.


```python
df.head()
```
<iframe src="https://jovian.ai/embed?url=https://jovian.ai/mohammadagus1st/exploratory-analysis-of-911-calls-data/v/1&cellId=8&hideInput=true" title="Jovian Viewer" height="486" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>


Check info() of df dataframe.


```python
df.info()
```
<iframe src="https://jovian.ai/embed?url=https://jovian.ai/mohammadagus1st/exploratory-analysis-of-911-calls-data/v/1&cellId=10&hideInput=true" title="Jovian Viewer" height="539" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>
    

Check number of NA value.<br>
Using `isna()` to generate boolean value (True if NA and False if non-NA) then using `sum()` to sum up all the True value.


```python
df.isna().sum()
```
<iframe src="https://jovian.ai/embed?url=https://jovian.ai/mohammadagus1st/exploratory-analysis-of-911-calls-data/v/1&cellId=12&hideInput=true" title="Jovian Viewer" height="353" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>



There are 3 columns that has NA value (zip, twp, addr).

Value counts of zipcodes for 911 calls.


```python
df['zip'].value_counts().sort_values(ascending=False)
```
<iframe src="https://jovian.ai/embed?url=https://jovian.ai/mohammadagus1st/exploratory-analysis-of-911-calls-data/v/1&cellId=15&hideInput=true" title="Jovian Viewer" height="311" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>



Value counts of townships (twp) for 911 calls.


```python
df['twp'].value_counts().sort_values(ascending=False)
```
<iframe src="https://jovian.ai/embed?url=https://jovian.ai/mohammadagus1st/exploratory-analysis-of-911-calls-data/v/1&cellId=17&hideInput=true" title="Jovian Viewer" height="311" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>



Number of unique value in the title column.


```python
df['title'].nunique()
```
<iframe src="https://jovian.ai/embed?url=https://jovian.ai/mohammadagus1st/exploratory-analysis-of-911-calls-data/v/1&cellId=19&hideInput=true" title="Jovian Viewer" height="80" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>



## Creating new features

In the titles column there are "Reasons/Departments" specified before the title code. These are EMS, Fire, and Traffic. For example, if the title column value is EMS: BACK PAINS/INJURY, the reason column value would be EMS.<br>

Next step is creating a function (first method), to split of each string value in title column by ':' character, append the first index of it into empty list then assign the list as a new column ('reason').


```python
def func_split(a):
    b = []
    for i in a:
        b.append(i.split(':')[0])
    return b

df['reason'] = func_split(df['title'])
```

Second method is using `apply()` with a custom lambda expression.


```python
df['reason'] = df['title'].apply(lambda x:x.split(':')[0])
```

The most common reason for a 911 call based off of the reason column.


```python
df['reason'].value_counts()
```
<iframe src="https://jovian.ai/embed?url=https://jovian.ai/mohammadagus1st/exploratory-analysis-of-911-calls-data/v/1&cellId=26&hideInput=true" title="Jovian Viewer" height="143" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>



Create a countplot of 911 calls by reason. In this plot, set `palette='viridis'` and because of seaborn is a library built on top of matplotlib, matplotlib's colormaps can be use to change color style of seaborn's plot ([more built-in matplotlib colormaps](https://matplotlib.org/stable/tutorials/colors/colormaps.html)).

```python
sns.countplot(x='reason', data=df, palette='viridis')
```
<iframe src="https://jovian.ai/embed?url=https://jovian.ai/mohammadagus1st/exploratory-analysis-of-911-calls-data/v/1&cellId=28&hideInput=true" title="Jovian Viewer" height="380" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>
    


___
**Time information**<br>
The data type of the objects in the timeStamp column.


```python
print(df['timeStamp'].dtypes)
```
<iframe src="https://jovian.ai/embed?url=https://jovian.ai/mohammadagus1st/exploratory-analysis-of-911-calls-data/v/1&cellId=30&hideInput=true" title="Jovian Viewer" height="83" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>
    

The timestamps are still strings (objects). Use [`pd.to_datetime`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_datetime.html) to convert the column from strings to DateTime objects.


```python
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
```

Grab specific attributes from a Datetime object by calling them. For example:

    time = df['timeStamp'].iloc[0]
    time.hour

Jupyter's tab method can be use to explore the various attributes. Now that the timestamp column are actually DateTime objects, use `.apply()` to create 3 new columns called Hour, Month, and Day of Week.

Notice how the Day of Week is an integer 0-6. Use the `.map()` with 'dmap' dictionary to map the actual string names to the day of the week:


```python
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Hour'] = df['timeStamp'].apply(lambda x:x.hour)
df['Day of Week'] = df['timeStamp'].apply(lambda x:x.day_of_week).map(dmap)
df['Month'] = df['timeStamp'].apply(lambda x:x.month)
```


```python
df.head()
```

<iframe src="https://jovian.ai/embed?url=https://jovian.ai/mohammadagus1st/exploratory-analysis-of-911-calls-data/v/1&cellId=36&hideInput=true" title="Jovian Viewer" height="512" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>

**Create a countplot of the Day of Week column with the hue based off of the reason column**


```python
sns.countplot(x='Day of Week', data=df, hue='reason', palette='viridis')
plt.legend(bbox_to_anchor=(1.05, 1),loc=0, borderaxespad=0.)
```
<iframe src="https://jovian.ai/embed?url=https://jovian.ai/mohammadagus1st/exploratory-analysis-of-911-calls-data/v/1&cellId=38&hideInput=true" title="Jovian Viewer" height="342" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>
    


The same for Month.


```python
sns.countplot(x='Month', data=df, hue='reason', palette='viridis')
plt.legend(bbox_to_anchor=(1.25, 1), borderaxespad=0.)
```
<iframe src="https://jovian.ai/embed?url=https://jovian.ai/mohammadagus1st/exploratory-analysis-of-911-calls-data/v/1&cellId=40&hideInput=true" title="Jovian Viewer" height="342" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>
    


_____

The plot was missing some Months, fill this information by plotting the information in another way, possibly a simple line plot that fills in the missing months.

**Create a groupby object called 'byMonth' and it's visualizations**<br>
The DataFrame is grouping by the month column and using the `count()` method for aggregation.


```python
byMonth = df.groupby('Month').count()
byMonth
```
<iframe src="https://jovian.ai/embed?url=https://jovian.ai/mohammadagus1st/exploratory-analysis-of-911-calls-data/v/1&cellId=43&hideInput=true" title="Jovian Viewer" height="393" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>


Create a simple plot of the dataframe indicating the count of calls per month.


```python
byMonth['twp'].plot()
```
<iframe src="https://jovian.ai/embed?url=https://jovian.ai/mohammadagus1st/exploratory-analysis-of-911-calls-data/v/1&cellId=45&hideInput=true" title="Jovian Viewer" height="342" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>
    

Use seaborn's `lmplot()` to create a linear fit on the number of calls per month. In order to create `lmplot()`, the index has to reset to be a new column.


```python
byMonth.reset_index(inplace=True)
sns.lmplot(x='Month', y='twp', data=byMonth)
```
<iframe src="https://jovian.ai/embed?url=https://jovian.ai/mohammadagus1st/exploratory-analysis-of-911-calls-data/v/1&cellId=47&hideInput=true" title="Jovian Viewer" height="432" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>
    


**Create a new column called 'Date' that contains the date from 'timeStamp' column**


```python
df['Date'] = df['timeStamp'].apply(lambda x:x.date())
```

Groupby Date column with the `count()` aggregate and create a plot of counts of 911 calls.


```python
plt.figure(figsize=(10,5))
df.groupby('Date')['twp'].count().plot()
plt.tight_layout()
```

    
<iframe src="https://jovian.ai/embed?url=https://jovian.ai/mohammadagus1st/exploratory-analysis-of-911-calls-data/v/1&cellId=51&hideInput=true" title="Jovian Viewer" height="382" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>
    


Recreate above plot but create 3 separate plots with each plot representing a reason for the 911 call.


```python
plt.figure(figsize=(10,5))
df[df['reason']=='Traffic'].groupby('Date')['twp'].count().plot()
plt.title('Traffic')
plt.tight_layout()
```


    
<iframe src="https://jovian.ai/embed?url=https://jovian.ai/mohammadagus1st/exploratory-analysis-of-911-calls-data/v/1&cellId=53&hideInput=true" title="Jovian Viewer" height="382" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>
    



```python
plt.figure(figsize=(10,5))
df[df['reason']=='Fire'].groupby('Date')['twp'].count().plot()
plt.title('Fire')
plt.tight_layout()
```


    
<iframe src="https://jovian.ai/embed?url=https://jovian.ai/mohammadagus1st/exploratory-analysis-of-911-calls-data/v/1&cellId=54&hideInput=true" title="Jovian Viewer" height="382" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>
    



```python
plt.figure(figsize=(10,5))
df[df['reason']=='EMS'].groupby('Date')['twp'].count().plot()
plt.title('EMS')
plt.tight_layout()
```


    
<iframe src="https://jovian.ai/embed?url=https://jovian.ai/mohammadagus1st/exploratory-analysis-of-911-calls-data/v/1&cellId=55&hideInput=true" title="Jovian Viewer" height="382" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>
    


____
**Create heatmap and clustermap using restructured df dataframe**<br>
First the dataframe need to be restructured, so that the columns become the Hours and the Index becomes the Day of the Week. There are lots of ways to do this, but in this time `pivot_table()` and use `aggfunc='count'` will be use.


```python
dayHour = df.pivot_table(index='Day of Week', columns='Hour', values='twp', aggfunc='count')
dayHour.head()
```



<iframe src="https://jovian.ai/embed?url=https://jovian.ai/mohammadagus1st/exploratory-analysis-of-911-calls-data/v/1&cellId=57&hideInput=true" title="Jovian Viewer" height="354" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>



Create a heatmap using the dayHour dataframe.


```python
plt.figure(figsize=(12,6))
sns.heatmap(data=dayHour, cmap='plasma').tick_params(left=False, bottom=False)
```


    
<iframe src="https://jovian.ai/embed?url=https://jovian.ai/mohammadagus1st/exploratory-analysis-of-911-calls-data/v/1&cellId=59&hideInput=true" title="Jovian Viewer" height="415" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>
    


Create a clustermap using the dayHour dataframe.


```python
sns.clustermap(data=dayHour, cmap='plasma', figsize=(8.5,8)).tick_params(right=False, bottom=False)
```





    
<iframe src="https://jovian.ai/embed?url=https://jovian.ai/mohammadagus1st/exploratory-analysis-of-911-calls-data/v/1&cellId=61&hideInput=true" title="Jovian Viewer" height="648" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>
    


Repeat these same plots and operations, for a dataframe that shows the Month as the column.


```python
dayMonth = df.pivot_table(index='Day of Week', columns='Month', values='twp', aggfunc='count')
dayMonth.head()
```




<iframe src="https://jovian.ai/embed?url=https://jovian.ai/mohammadagus1st/exploratory-analysis-of-911-calls-data/v/1&cellId=63&hideInput=true" title="Jovian Viewer" height="265" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>




```python
plt.figure(figsize=(12,6))
sns.heatmap(data=dayMonth, cmap='plasma').tick_params(left=False, bottom=False)
```


    
<iframe src="https://jovian.ai/embed?url=https://jovian.ai/mohammadagus1st/exploratory-analysis-of-911-calls-data/v/1&cellId=64&hideInput=true" title="Jovian Viewer" height="415" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>
    



```python
sns.clustermap(data=dayMonth, cmap='plasma', figsize=(8.5,8)).tick_params(right=False, bottom=False)
```




<iframe src="https://jovian.ai/embed?url=https://jovian.ai/mohammadagus1st/exploratory-analysis-of-911-calls-data/v/1&cellId=65&hideInput=true" title="Jovian Viewer" height="648" width="100%" style="margin 0 auto; max-width: 800px;" frameborder="0" scrolling="auto"></iframe>