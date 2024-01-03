import yfinance as yf
from prophet import Prophet
from datetime import date
import matplotlib.pyplot as plt


START = "2019-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

def selected_stock(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data = selected_stock("GME")

data = data.rename(columns={"Date": "ds", "Close": "y"})

model = Prophet()

model.fit(data)

future = model.make_future_dataframe(periods=365)

forecast = model.predict(future)

plt.plot(data['ds'], data['y'], 'b-', label = 'actual')

plt.plot(forecast['ds'], forecast['yhat'], 'r--', label = 'prediction')
plt.legend(); plt.xlabel('Date'); plt.ylabel('Price'); 
plt.title('Prediction vs Actuals')
plt.show()