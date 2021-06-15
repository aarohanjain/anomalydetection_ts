#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 21:23:44 2021

@author: aarohanjain
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from copy import deepcopy


def arima_anomaly_detection(csv_file, split_factor=0.97, arima_order=(3,1,2), 
                            anomaly_threshold=10, dynamic_threshold=False,
                            verbose=False):
    
    trends = pd.read_csv("searches//" + csv_file).iloc[1:].asfreq('w')
    search_term = csv_file.split('.')[0]
    
    try:
        trends = trends.astype(float)
    except ValueError:
        trends.replace("<1", '0', inplace=True)
        trends = trends.astype(float)
        if verbose:
            print("Data contained uncastable strings like <1. "
                  "Any entry with '<1' is set to be equal to 0.")
    
    # only take past values
    train = trends.iloc[:int(split_factor * len(trends))]
    test = trends.iloc[int(split_factor * len(trends)):]
    
    arima_model = ARIMA(train, order=arima_order)
    results = arima_model.fit()
    if verbose:
        print(results.summary())
    
    start_pred_idx = int(split_factor * len(trends))
    end_pred_idx = len(trends) - 1
    preds = results.predict(start_pred_idx, end_pred_idx, typ="levels")
    
    anomaly_detection = deepcopy(test)
    anomaly_detection["preds"] = preds
    anomaly_detection["diff"] = np.abs(anomaly_detection["Category: All categories"] - anomaly_detection["preds"])
    
    if dynamic_threshold:
        anomaly_threshold = int((train.max() - train.min()) / 3)
        print(anomaly_threshold)
    
    anomalies = anomaly_detection.loc[anomaly_detection["diff"] > anomaly_threshold]["Category: All categories"]
    first_anomaly = anomalies.iloc[0:1]
    try:
        date_at_first_anomaly = str(first_anomaly.index[0])[:10]
        title_str = f"First anomaly detected: {date_at_first_anomaly}"
    except IndexError:
        title_str = "No anomaly detected."
    
    plt.figure(figsize=(8, 6))
    plt.title(f"Google time series data for search term '{search_term.capitalize()}'.\n" + title_str)
    plt.plot(test, label="True Data")
    plt.plot(train, label="Training Data")
    plt.plot(preds, label="ARIMA Predictions")
    plt.plot(first_anomaly, label="Anomaly Detected", marker='x', 
             markersize=10, mew=5, ls='None')
    plt.ylim((0, 100))
    plt.xlabel("Time")
    plt.ylabel("Interest in Search Term")
    plt.legend()
    plt.show()
    
    plt.savefig(f"figures//{search_term}_90.eps", format='eps')
    
    return


if __name__ == "__main__":
        
    csv_list = [c for c in os.listdir(".//searches") if c.endswith(".csv")]
    for csv_file in csv_list:
        arima_anomaly_detection(csv_file,
                                split_factor=0.9,
                                dynamic_threshold=True)
        
