import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("BankNote_Authentication.csv")
print(df.head())

x = df['class']
y = df[['variance','skewness','curtosis','entropy']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)




classifier = LogisticRegression(random_state = 0) 
classifier.fit(x,y)

X_test = np.reshape(x_test, (-1,1))
Y_test = np.reshape(y_test, (-1,1))


x_prediction = classifier.predict(X_test)

predicted_values = []

for i in x_prediction:
    if i == 0:
        predicted_values.append("No")
    else:
        predicted_values.append("Yes")
        
actual_values = []

for i in Y_test.ravel():
    if i == 0:
        actual_values.append("No")
    else:
        actual_values.append("Yes")
        
labels = ["Yes", "No"]
cm = confusion_matrix(actual_values, predicted_values, labels)

ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax)

ax.set_xlabel("Forged")
ax.set_ylabel("Authorized")
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels)
plt.show()





