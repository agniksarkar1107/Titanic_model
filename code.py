import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


data = pd.read_csv("tested.csv")
target = data['Survived']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data,target, test_size = 0.5)
data.head()
y_train = data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
x_train = pd.get_dummies(data[features])
x_test = pd.get_dummies(data[features])

my_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
my_model.fit(x_train, y_train)
survive_predict = my_model.predict(x_test)

output = pd.DataFrame({'PassengerId': data.PassengerId, 'Survived': survive_predict})
output.to_csv('titanic_model_predict.csv', index=False)
print("The file is created")
