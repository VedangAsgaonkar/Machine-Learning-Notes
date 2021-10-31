### Normalizing
We should always normalize our features and labels. [Why Data Normalization is necessary for Machine Learning models](https://medium.com/@urvashilluniya/why-data-normalization-is-necessary-for-machine-learning-models-681b65a05029).
This can be done by:
```
X_normalized = (X-X.mean())/X.std()
```
where X is a numpy vector. Remember to convert back to original scale in case you are interested in values of particular weights/biases like in linear regression.

### Visualising 
Scatter plots can be used to visualise relation between variables using ```plt.scatter(X,Y)```

### Transforming
If we are looking at an exponential relation, a transformation like ```np.log()``` may help. Similiarly for quadratics, ```np.sqrt()``` is useful

### Datatype
Convert to float
```
X = X.astype(np.float32)
```
