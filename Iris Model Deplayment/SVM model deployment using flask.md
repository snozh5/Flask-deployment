# ML Model using SVM and Deploy using Flask
## Flask
[Flask](https://flask.palletsprojects.com/en/2.0.x/) is a python module web framework that helps us develop web applications easily. It is easy to set up and simple to use.

> Since we will be using SVM algorithm, to know more about it click [here](https://github.com/Learn-Write-Repeat/Intern-Work/tree/45ca8245e47186b679bcbb9b4006437d0b29828a/int-ml-4/Support%20Vector%20Machine).  

## Implementation
### Data Loading
```sh
#import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle

# Load the csv file
df = pd.read_csv("iris.csv")
df.head() #print the top 5 rows of the dataset
```
[![head](https://github.com/snozh5/temp/blob/main/Pictures_svm_deploy/head.PNG?raw=true)](https://github.com/snozh5/temp/blob/main/Pictures_svm_deploy/head.PNG)

The above figure shows the top 5 rows of the dataset. It contains 5 columns of which sepal lenght, sepal width, petal length, petal width are independent variables and species is dependent variable. 

### Exploratory Data Analysis (EDA)
```sh
# plotting the count plot of "Species"
sns.countplot(x="Species", data=df)
plt.title("Count plot for Species")
plt.show()
```
[![countplot](https://github.com/snozh5/temp/blob/main/Pictures_svm_deploy/countplot_species.png?raw=true)](https://github.com/snozh5/temp/blob/main/Pictures_svm_deploy/countplot_species.png)

There are total three Species namely "Iris-setosa", "Iris-versicolor" & "Iris-virginica".

```sh
# Pair plot for pairwise relation
sns.pairplot(data=df, hue='Species')
```
[![pairplot](https://github.com/snozh5/temp/blob/main/Pictures_svm_deploy/pairplot_species.png?raw=true)](https://github.com/snozh5/temp/blob/main/Pictures_svm_deploy/pairplot_species.png)

The pairplot graph gives us the pair wair relationship, from the above graph we can see that "Iris-setosa" species differ from the other two species in terms of relationship.

```sh
# Correlation among the attributes plotting using a HeatMap
plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(),annot=True,fmt="f").set_title("Correlation of attributes with Species")
plt.show()
```
[![heatmap](https://github.com/snozh5/temp/blob/main/Pictures_svm_deploy/heatmap.png?raw=true)](https://github.com/snozh5/temp/blob/main/Pictures_svm_deploy/heatmap.png)

The 'PetalWidth' and 'PetalLength' are correlated with each other and 'SepalWidth' and 'SepalLength' are not correlated as shown in the heatmap.

```sh
# Bar plot for Sepal Length and Sepal Width
plt.figure(figsize=(18, 8))
plt.subplot(2,3,1)
sns.barplot(x="Species", y="Sepal_Length", data=df)
plt.subplot(2,3,2)
sns.barplot(x="Species", y="Sepal_Width", data=df)
plt.show()
```
[![sepal](https://github.com/snozh5/temp/blob/main/Pictures_svm_deploy/barplot_sepal_lengthwidth.PNG?raw=true)](https://github.com/snozh5/temp/blob/main/Pictures_svm_deploy/barplot_sepal_lengthwidth.PNG)

For the Sepal length "Iris-virginica" are high while for Sepal width "Iris-setosa" are high on number.

```sh
# Bar plot for Petal Length and Petal Width
plt.figure(figsize=(18, 8))
plt.subplot(2,3,1)
sns.barplot(x="Species", y="Petal_Length", data=df)
plt.subplot(2,3,2)
sns.barplot(x="Species", y="Petal_Width", data=df)
```
[![petal](https://github.com/snozh5/temp/blob/main/Pictures_svm_deploy/barplot_petal_lengthwidth.PNG?raw=true)](https://github.com/snozh5/temp/blob/main/Pictures_svm_deploy/barplot_petal_lengthwidth.PNG)

For both Petal length and Petal width "Iris-virginica" have got the highest number with "Iris-setosa" seems to the lowest on both.

### Model Building
```sh
# linear model
model_linear = SVC(kernel='linear')
model_linear.fit(X_train, y_train)

# predict
y_pred = model_linear.predict(X_test)

# confusion matrix and accuracy
from sklearn import metrics
from sklearn.metrics import confusion_matrix
# accuracy
print("accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")

# cm
print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))
```
[![linear](https://github.com/snozh5/temp/blob/main/Pictures_svm_deploy/linear.PNG?raw=true)](https://github.com/snozh5/temp/blob/main/Pictures_svm_deploy/linear.PNG)

The above figure shows the accuracy and confusion matrix for the linear model.

```sh
# non-linear model
# using rbf kernel, C=1, default value of gamma
non_linear_model = SVC(kernel='rbf')
non_linear_model.fit(X_train, y_train)

# predict
y_pred = non_linear_model.predict(X_test)

# confusion matrix and accuracy
# accuracy
print("accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")

# cm
print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))
```
[![non-linear](https://github.com/snozh5/temp/blob/main/Pictures_svm_deploy/nonlinear.PNG?raw=true)](https://github.com/snozh5/temp/blob/main/Pictures_svm_deploy/nonlinear.PNG)

The above figure shows the accuracy and confusion matrix for the non-linear model.

```sh
# Make pickle file for our model
pickle.dump(non_linear_model, open("model.pkl", "wb"))
```
The pickle file is being created which will be used by the deployment file during routing for prediction.

### Deployment
```sh
# Import necessary liberies
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

# Routing
@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text = "The flower species is {}".format(prediction))

if __name__ == "__main__":
    flask_app.run(debug=True)
```
[![index](https://github.com/snozh5/temp/blob/main/Pictures_svm_deploy/index.PNG?raw=true)](https://github.com/snozh5/temp/blob/main/Pictures_svm_deploy/index.PNG)

The above figure shows the index page which is basically the home page. It is hosted locally at  `http://127.0.0.1:5000`. It is here we give our input data for prediction.  

[![predict](https://github.com/snozh5/temp/blob/main/Pictures_svm_deploy/predicted.PNG?raw=true)](https://github.com/snozh5/temp/blob/main/Pictures_svm_deploy/predicted.PNG)

The above figure shows the predicted output post clicking the predict button. 





