import cloudscraper
import sklearn
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import time 
import statistics

from time import sleep
scraper = cloudscraper.create_scraper(
        browser={
            'custom': 'ScraperBot/1.0',
        }
    )

data = pd.read_csv('Crash_data.csv')
data1 = pd.read_csv('Older_Crash_Data.csv')

def crashPoint(num):
  info = scraper.get('https://rest-bf.blox.land/games/crash').json()["history"][num]["crashPoint"]
  return info
  
  



try:
  X = np.array([crashPoint(1), crashPoint(2), crashPoint(3), crashPoint(4), crashPoint(5), crashPoint(6), crashPoint(7), crashPoint(8), crashPoint(9), crashPoint(10), crashPoint(11), crashPoint(12), crashPoint(28),crashPoint(29),crashPoint(30) ]).reshape(-1,1)
  y = np.array([crashPoint(13), crashPoint(14), crashPoint(15), crashPoint(16), crashPoint(17), crashPoint(18), crashPoint(19), crashPoint(20), crashPoint(21), crashPoint(22), crashPoint(23), crashPoint(24), crashPoint(25), crashPoint(26),crashPoint(27)]).reshape(-1,1)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#creating a numpy array using the crashpoints then using X_train and the other testing trains
  model = LinearRegression()
  model.fit(X_train, y_train)
#fitting the trains
  score = model.score(X_test, y_test)
  y_pred = model.predict(X_test)
  average = sum(y_pred) / len(y_pred)
  average_one = np.mean(average)
  average_two = average_one + average
  average_three = average_one / average_two * average_one
  average_four = np.mean(average_three) * 0.9
  prediction = (2 / (average_three - average_four) / 2)
  prediction_1 = prediction + 1 * 0.12
  prediction_2 = prediction / prediction * average_four
  prediction_3 = np.mean(prediction_2)
  prediction_4 = np.sum(prediction_3)
  prediction_5 = prediction_1 / prediction_2 / prediction_3 * prediction_4 / 2
  print(prediction_5)




  
except Exception as e:
  print(e)
