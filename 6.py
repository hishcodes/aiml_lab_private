import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data = pd.read_csv("data/data6.csv")
print("Dataset:\n", data)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]
print("Features (X):\n", X)
print("Target (y):\n", y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Number of Rows in Training Set: ${len(X_train)} rows")
print(f"Number of Rows in Test Set: ${len(X_test)} rows")

clf = GaussianNB()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nPredictions: {y_pred}")
print(f"Accuracy: {accuracy * 100:.2f}%")