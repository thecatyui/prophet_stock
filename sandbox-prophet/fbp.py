import pandas as pd
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

from datetime import datetime

start = datetime(2016, 7, 20)
end = datetime(2021, 7, 20)

SB = web.DataReader('005930.KS','yahoo',start,end)
SB.head()

SB['Close'].plot(figsize=(12,6));
plt.show()

df = pd.DataFrame({'ds':SB.index, 'y':SB['Close']})
df.reset_index(inplace=True)
del df['Date']
df.head()

m = Prophet(daily_seasonality=True)
m.fit(df);

future = m.make_future_dataframe(periods=365)
future.tail()

forecast = m.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()

m.plot(forecast);

m.plot_components(forecast);
plt.show()