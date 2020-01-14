---
layout: post
title: "Machine Learning Study Material"
tags:
- business analytics 
- business analyst
- data analyst
- python
- machine learning 
- machine learning basis 
- linear algebra 
- regression
- classification 
type: post
published: true
description: Machine Learning Self Study
# Add post description (optional)
img:  # Add image post (optional)
---




# Machine Learning Study 



```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans #사이킷런
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

## Create data points 


```python
df = pd.DataFrame(columns=['x','y'])
```


```python
df.loc[0] = [3,1]
df.loc[1] = [4,1]
df.loc[2] = [3,2]
df.loc[3] = [4,2]
df.loc[4] = [10,5]
df.loc[5] = [10,6]
df.loc[6] = [11,5]
df.loc[7] = [11,6]
df.loc[8] = [15,1]
df.loc[9] = [15,2]
df.loc[10] = [16,1]
df.loc[11] = [16,2]
```


```python
df.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <td>4</td>
      <td>10</td>
      <td>5</td>
    </tr>
    <tr>
      <td>5</td>
      <td>10</td>
      <td>6</td>
    </tr>
    <tr>
      <td>6</td>
      <td>11</td>
      <td>5</td>
    </tr>
    <tr>
      <td>7</td>
      <td>11</td>
      <td>6</td>
    </tr>
    <tr>
      <td>8</td>
      <td>15</td>
      <td>1</td>
    </tr>
    <tr>
      <td>9</td>
      <td>15</td>
      <td>2</td>
    </tr>
    <tr>
      <td>10</td>
      <td>16</td>
      <td>1</td>
    </tr>
    <tr>
      <td>11</td>
      <td>16</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



# Visualize data points in 2D plot


```python
# visualize data point 
sns.lmplot('x','y',data=df,fit_reg=False, scatter_kws={'s':200}) # x-axis, y-axis, data, no line, marker size 

#title 
plt.title('kmean plot')

# x- axis label 
plt.xlabel('x')

# y- axis label 
plt.ylabel('y')
```




    Text(16.299999999999997, 0.5, 'y')



<img src="/assets/img/output_6_1.png" width="200" />


## K mean clustering 


```python
# convert dataframe to numpy array
data_points = df.values
```


```python
kmeans = KMeans(n_clusters=3).fit(data_points)
```


```python
# Cluster id for each data point 
kmeans.labels_
```




    array([1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2], dtype=int32)




```python
# This is final centroids position
kmeans.cluster_centers_
```




    array([[10.5,  5.5],
           [ 3.5,  1.5],
           [15.5,  1.5]])




```python
df['cluster_id'] = kmeans.labels_
```


```python
df.head(12)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
      <th>cluster_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>10</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>10</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <td>6</td>
      <td>11</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <td>7</td>
      <td>11</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <td>8</td>
      <td>15</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <td>9</td>
      <td>15</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <td>10</td>
      <td>16</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <td>11</td>
      <td>16</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.lmplot('x', 'y',data=df, fit_reg=False, # x-axis, y-axis, data, no line
          scatter_kws={"s":150}, # marker size 
          hue = "cluster_id") # color

# title 
plt.title('after kmean clustering')
```




    Text(0.5, 1, 'after kmean clustering')




<img src="/assets/img/output_14_1.png" width="200" />



```python

```


```python

```


```python

```
