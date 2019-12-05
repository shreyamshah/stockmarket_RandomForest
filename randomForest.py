import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import linear_model
from sklearn import preprocessing
import matplotlib.pyplot as plt


dataset = pd.read_csv('infosys.csv')
close = dataset['Close Price'].values
date = dataset['Date'].values
dataset = dataset.drop(['Close Price','Date'],axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset, close, test_size = 0.25)
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
print("Training Starts")
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(X_train, y_train)
print("Training COmplete")
y_pred = rf.predict(X_test)
# Calculate the absolute errors
errors = abs(y_pred - y_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
#print(y_pred)
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
temp = [700.0,812.2,746.5,810.4429687,179647,4680,142000708,15.7,-5.45,5.5499999999999545,1,99.75969494208216,97.22194954353589,54.59290187891449,787.2099999999999,753.8399999999999]
#temp = [790.0,802.2,786.5,790.4429687,179647,4680,142000708,15.7,5.45,5.5499999999999545,0.007079081632653003,99.75969494208216,97.22194954353589,54.59290187891449,787.2099999999999,753.8399999999999]
#,789.55
#from sklearn.metrics import confusion_matrix
#print(confusion_matrix(y_test,y_pred))
temp = np.array(temp).reshape(1,-1)
print(rf.predict(temp))
y_pred = rf.predict(dataset[1074:])
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_title('Line Plot of Actual Price vs Predicted Price')
ax.plot(date[1074:],close[1074:], color='tab:blue',label="Actual")

ax.plot(date[1074:], y_pred, color='tab:orange',label="Predicted")
ax.axes.get_xaxis().set_visible(False)
#print(y_pred)
#print(X_test)
legend = ax.legend(loc='upper right', shadow=True, fontsize='large')
plt.show()