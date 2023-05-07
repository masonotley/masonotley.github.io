---
layout: wide_default
---


# Predictive Model: Housing Prices

This model utilizes sklearn to train a Lasso regression model to predict the log value of housing sale prices. The model employs a variety of categorical and numerical variables, and is optimized using the alpha and SimpleImputer strategy. Select input and output of the model are shown below to illustrate how the model was built. To learn more about elements of the following code, visit [sklearn's website.](https://scikit-learn.org/stable/)

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

![/images/Screenshot 2023-05-07 at 6.32.26 PM.png]

The "parcel" variable is a unique identifyer for the housing properties, and the "prediction" variable is the models prediciton of the log value of the property's selling price. On the test data, the model yielded an R-squared score of 0.89116, which I consider fairly accurate given the data provided. This type of model performs well for predicting housing prices, and can be used in a variety of predictive applications. 
