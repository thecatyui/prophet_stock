import pandas as pd
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import prophet.plot as fp

from datetime import datetime

start = datetime(2020, 7, 21)
end = datetime(2021, 7, 20)

SB = web.DataReader('010950.KS','yahoo',start,end)
SB.head()

SB['Close'].plot(figsize=(12,6), grid=True);
plt.show()

df = pd.DataFrame({'ds':SB.index, 'y':SB['Close']})
df.reset_index(inplace=True)
df["Date"] = SB.index
df["ds"] = pd.to_datetime(df["Date"]).dt.date
df["y"] = np.log(SB["Close"])
df.head()

m = Prophet(daily_seasonality=True)
m.fit(df);

future = m.make_future_dataframe(periods=365)
future.tail()

forecast = m.predict(future)
fig = m.plot(forecast)
fp.add_changepoints_to_plot(fig.gca(), m, forecast);
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()
plt.savefig('./stock1.png')

m.plot(forecast);

m.plot_components(forecast);
plt.savefig('./stock2.png')
plt.show()

