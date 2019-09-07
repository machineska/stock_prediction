''' this script was convert from fle 
    jupyter notebook 'regline_1_1'
'''

import copy
import datetime
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import style

import numpy as np

import pandas as pd
from pandas import Series, DataFrame
import pandas_datareader.data as web

from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.model_selection import train_test_split



start = datetime.datetime(2000, 1, 1)
end = datetime.datetime(2019, 9, 5)

df = web.DataReader("AAPL", 'yahoo', start, end)

close_px = df['Adj Close']
mavg = close_px.rolling(window=100).mean()

# Adjusting the size of matplotlib
mpl.rc('figure', figsize=(8, 7))
mpl.__version__

# Adjusting the style of matplotlib
style.use('ggplot')

plt.plot(range(3))

plt.figure('History APPL stock')
close_px.plot(label='AAPL')
mavg.plot(label='mavg')
plt.legend()
# plt.show()

dfreg = df.loc[:,['Adj Close','Volume']]
dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

# Drop missing value
dfreg.fillna(value=-99999, inplace=True)

# We want to separate 1 percent of the data to forecast
forecast_out = int(math.ceil(0.01 * len(dfreg)))

# Separating the label here, we want to predict the AdjClose
forecast_col = 'Adj Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
X = np.array(dfreg.drop(['label'], 1))

# Scale the X so that everyone can have the same distribution 
# for linear regression
X = preprocessing.scale(X)

# Finally We want to find Data Series of late X and early X (train) 
# for model generation and evaluation
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

# Separate label and identify it as y
y = np.array(dfreg['label'])
y = y[:-forecast_out]


# Separation of training and testing of model by cross validation train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Linear regression
clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X_train, y_train)

# Quadratic Regression 2
clfpoly_ridge = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly_ridge.fit(X_train, y_train)

# Quadratic Regression 3
clfpoly_lasso = make_pipeline(PolynomialFeatures(3), Lasso())
clfpoly_lasso.fit(X_train, y_train)

confidencereg = clfreg.score(X_test, y_test)
confidence_poly_ridge = clfpoly_ridge.score(X_test,y_test)
confidence_poly_lasso = clfpoly_lasso.score(X_test,y_test)

# Test Model
print("The linear regression confidence is ",confidencereg)
print("The quadratic regression poly 2 confidence with ridge is ",confidence_poly_ridge)
print("The quadratic regression poly 3 confidence with lasso is ",confidence_poly_lasso)

def predict_data(dataframe, clf):
  # Printing the forecast for Linear Regression
  last_date = dataframe.iloc[-1].name
  forecast_set = clf.predict(X_lately)
  dataframe['Forecast'] = np.nanlast_date = dataframe.iloc[-1].name
  last_unix = last_date
  next_unix = last_unix + datetime.timedelta(days=1)

  for i in forecast_set:
      next_date = next_unix
      next_unix += datetime.timedelta(days=1)
      dataframe.loc[next_date] = [
          np.nan for _ in range(len(dataframe.columns)-1)
      ]+[i]
  
  return dataframe
  
def draw_prediction(dataframe, title):
  plt.figure(title)
  dataframe['Adj Close'].tail(500).plot()
  dataframe['Forecast'].tail(50).plot()
  plt.legend(loc=4)
  plt.xlabel('Date')
  plt.ylabel('Price')
  # plt.show()




# draw reguler Regression Linear Prediction
dataframe_reg_regular = predict_data(copy.deepcopy(dfreg), clfreg)
draw_prediction(dataframe_reg_regular, 'Regular Regression Linear Prediction')

dataframe_reg_poly_ridge = predict_data(copy.deepcopy(dfreg), clfpoly_ridge)
draw_prediction(dataframe_reg_poly_ridge, 'Regression Linear Prediction with Ridge')

dataframe_reg_poly_laso = predict_data(copy.deepcopy(dfreg), clfpoly_lasso)
draw_prediction(dataframe_reg_poly_laso, 'Regression Linear Prediction with Laso')

# try to predict for 7th Sept 2019
trading_day = datetime.datetime(2019, 9, 7)
forecast_laso = dataframe_reg_poly_laso['Forecast'].loc[trading_day]
forecast_ridge = dataframe_reg_poly_ridge['Forecast'].loc[trading_day]
forecast_linear = dataframe_reg_regular['Forecast'].loc[trading_day]

print(f'\nPrice Prediction for 7 September 2019 with Laso -> {forecast_laso}')
print(f'Price Prediction for 7 September 2019 with Ridge -> {forecast_ridge}')
print(f'Price Prediction for 7 September 2019 with regular Linear -> {forecast_linear}')

plt.show()
plt.close()
