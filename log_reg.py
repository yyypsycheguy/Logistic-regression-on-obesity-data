import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report 
import matplotlib.pyplot as plt

df = pd.read_csv("processed_data.csv")

# indicate label columns
X = df.drop(columns='NObeyesdad')
y = df['NObeyesdad']

# split training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.069, random_state=42)

# initialize and train model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# get feature importantce
coefficients = logreg.coef_[0]
features = X.columns
df_features = pd.DataFrame({"Features": features, "Coefficients": coefficients})
df_features["Importance"] = df_features["Coefficients"].abs()
df_features = df_features.sort_values(by="Importance", ascending=False)

print(df_features)


# evaluate model
y_pred = logreg.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification report: {classification_report}")


# graph feature importance
values = df_features["Importance"]
categories = df_features["Features"].tolist()
plt.figure(figsize=(20,7))
plt.bar(categories, values)
plt.title("Feature importance")
plt.xticks(rotation=90)
plt.xlabel("Features")
plt.ylabel("importance")

plt.savefig("Feature_importance.png")

plt.show()
