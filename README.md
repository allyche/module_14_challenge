# Module 14 Challenge: Machine Learning Trading Bot

In this Challenge, I was tasked tp improve the existing algorithmic trading systems and maintain the firm’s competitive advantage in the market. To do so, I’ll enhance the existing trading signals with machine learning algorithms that can adapt to new data.

## Key Task
1. Establish a Baseline Performance
2. Tune the Baseline Trading Algorithm
3. Evaluate a New Machine Learning Classifier
4. Create an Evaluation Report

## Installation Guide
Before running the application first install and import the following library & dependencies:

1. import pandas as pd
2. import numpy as np
3. from pathlib import Path
4. import hvplot.pandas
5. import matplotlib.pyplot as plt
6. from sklearn import svm
7. from sklearn.preprocessing import StandardScaler
8. from pandas.tseries.offsets import DateOffset
9. from sklearn.metrics import classification_report

## Contributors
Brought to you by Ally Che (ally.che@gmail.com)

## License
Project Jupyter

## Observations:
1. Base SVC Model: 

With the baseline performance results using SVC model, the classification report from machine learning shows a f1-score at 55%, which is not too high. As for predictions to both 1 (actual return > 0) and -1 (actual return <1), the f1-score gives a low 7% in predicting -1, but a relatively high 71% for 1. 

From the actual returns vs strategy returns plot, we can see the prediction (strategy) still gives a better return from 2019 onwards. However, we should still try adjusting the model parameter to see if we can get better results.

Base model classification report: https://github.com/allyche/module_14_challenge/blob/main/base_report.jpg

Base model actual vs strategy returns plot: https://github.com/allyche/module_14_challenge/blob/main/base_plot.jpg

2. SVC Model - adjusting testing data period

The base model used 3 month worth of testing data, in this adjustment we used a bigger range of test data of 6 months.

From the classification report, the accuracy f1-score stayed about the same at 56%, as for the predictions to both 1 (actual return > 0) and -1 (actual return <1), the f1-score gives a lower 4% in predicting -1, but a same 71% for 1.

From the updated actual returns vs strategy returns plot, we can see there was a perid from 2019 to 2020 the actual return was actually higher than the strategy return. 

Comparing the base model 3 month test data vs the adjusted 6 months data. The prediction didn't do a better job.

Adjusted TestDataPeriod classication report: https://github.com/allyche/module_14_challenge/blob/main/AdjustingTestDataPeriod_Report.jpg

Adjusted TestDataPeriod actual vs strategy returns plot: https://github.com/allyche/module_14_challenge/blob/main/AdjustingTestDataPeriod_plot.jpg

3. SVC Model: Adjusting SMA

The base model used SMA 4 and 10 days. In the adjusted model we updated the SMA to 10 and 200 days. The updated accuracy score is 54% which is similar to the previous ones. The redictions to both 1 (actual return > 0) and -1 (actual return <1), the f1-score gives a higher 22% in predicting -1, but a slightly lower 68% for 1.

Looking at the updated SMA actual vs strategy returns plot, the strategy returns consistently beat the actual returns across the years in the past. This is the best model + parameters so far.

Adjusted SMA classication report: https://github.com/allyche/module_14_challenge/blob/main/AdjustingSMA_Report.jpg

Adjusted SMA actual vs strategy returns plot: https://github.com/allyche/module_14_challenge/blob/main/AdjustingSMA_plot.jpg


4. Linear Regression Model:

Using the Linear Regression model, we got a accuracy  f1-score of 52%, which is slightly lower than the rest of the tests we did. As for the predictions to both 1 (actual return > 0) and -1 (actual return <1), the f1-score gives 38% in predicting -1, but a 61% for 1.

Similar to the base SVC model, we can see the prediction (strategy) return still gives a better performance from the middle of 2019 onwards.  

Linear Regression classification report: https://github.com/allyche/module_14_challenge/blob/main/lr_report.jpg

Linear Regression actual vs strategy returns plot: https://github.com/allyche/module_14_challenge/blob/main/lr_plot.jpg


# Conclusion:

To sum up based on all the reports and plots described above, I would recommend using hte SVC model with adjusted SMA to 20 & 200 days. The classification report gives a higher 22% in predicting -1 across all models, and a stable 68% for predicting 1. According to the plot, the prediction of strategy Returns consistently beat the actual returns across all years in the testing period.
