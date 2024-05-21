import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report 

df = pd.read_csv("processed_data.csv")

# indicate label columns
X = df.drop(columns='NObeyesdad')
y = df['NObeyesdad']

# split training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.07, random_state=42)

# initialize and train model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# evaluate model
y_pred = logreg.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification report: {classification_report}")