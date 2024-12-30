# NVIDIA-Stock-Price-Prediction
This beginner-friendly machine learning project uses Long Short-Term Memory (LSTM) networks to predict Nvidia stock prices. The model is trained on historical stock data from 2012 to 2023 and generates predictions for the stock prices between 2021 and 2022, showing a close correlation with actual values. I used the tutorial 'Stock Price Prediction Using Machine Learning' by Simplilearn as a reference for this project. It helped guide the methodology for building the LSTM model, preprocessing the data, and evaluating the predictions. You can find the article here: https://www.simplilearn.com/tutorials/machine-learning-tutorial/stock-price-prediction-using-machine-learning

Steps to run stock prediction:
Download the Python Script:

Download the stock_prediction_script.py script from the repository.

Install Required Libraries:

pip install numpy pandas matplotlib yfinance scikit-learn tensorflow

Run the Program:

After installing the necessary libraries, navigate to the folder where you saved the stock_prediction_script.py script, and run the following command in your terminal:

python stock_prediction.py

The program will download Nvidia stock data, train the model, and display a graph comparing actual vs. predicted stock prices.

View the Results:

Once the program runs, a graph will appear showing the predicted and actual Nvidia stock prices for the test set.
