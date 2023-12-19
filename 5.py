import numpy as np
from sklearn.neural_network import MLPRegressor

X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
Y = np.array(([92], [86], [89]), dtype=float)
X /= np.amax(X, axis=0)
Y /= 100
model = MLPRegressor(hidden_layer_sizes=(3,),activation='logistic',max_iter=5000,learning_rate_init=0.1, random_state=42)
model.fit(X, Y.ravel())

predicted_output = model.predict(X)

print("Input:\n", X)
print("Actual Output:\n", Y)
print("Predicted Output:\n", predicted_output)