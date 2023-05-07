---
layout: wide_default
---


# Predictive Model: Housing Prices

This model utilizes sklearn to train a Lasso regression model to predict the log value of housing sale prices. The model employs a variety of categorical and numerical variables, and is optimized using the alpha and SimpleImputer strategy. To learn more about elements of the following code, visit[sklearn's website.](https://scikit-learn.org/stable/)

### Preprocessing Pipeline

```python
numer_pipe = make_pipeline(SimpleImputer())

cat_pipe = make_pipeline(OneHotEncoder(handle_unknown='ignore',drop='first',sparse=False)) 

preproc_pipe = ColumnTransformer(
     [("num_impute", numer_pipe, num_pipe_features),
     ("cat_trans", cat_pipe, cat_pipe_features)], 
     remainder = 'drop')

lasso_pipe = make_pipeline(preproc_pipe,Lasso(alpha = 0.1))
```

### Optimizing the Model's Parameters

```python
alphas = list(np.linspace(0.00001,0.000012,10))
strats = ['mean','median','most_frequent','constant']
cv = 10

parameters = {
    'lasso__alpha' : alphas,
    'columntransformer__num_impute__simpleimputer__strategy' : strats
}
    
grid_search = GridSearchCV(estimator = lasso_pipe, 
                           param_grid = parameters,
                           cv = cv )

results = grid_search.fit(X_train,y_train)
```

### Fitting and Predicting the Optimal Model

```python
best_lasso = results.best_estimator_ 
best_lasso.fit(X_train,y_train)
r2_score(y_test,best_lasso.predict(X_test,))
```
output: 0.8911620886890559

### Predicting log(sale price) of the Holdout Data

```python
prediction = best_lasso.predict(holdout_X)
df = pd.DataFrame({'parcel':holdout['parcel'],
                   'prediction':prediction})
df
```
output:
	parcel	prediction
0	1_526301100	12.085142
1	988_924100040	12.164829
2	984_923275140	11.684968
3	977_923227080	11.944337
4	803_906203120	12.434177
...	...	...
984	208_903476030	11.891243
985	207_903454060	11.007243
986	187_902401060	11.380163
987	190_902402250	11.648451
988	284_908226130	11.362185
989 rows × 2 columns