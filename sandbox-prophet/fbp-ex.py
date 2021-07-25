import pandas as pd
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import prophet.plot as fp

from datetime import datetime

start = datetime(2018, 7, 24)
end = datetime(2021, 7, 24)

SB = web.DataReader('051910.KS','yahoo',start,end)
SB.head()

SB['Close'].plot(figsize=(12,6), grid=True);
plt.show()

df = pd.DataFrame({'ds':SB.index, 'y':SB['Close']})
df.reset_index(inplace=True)
del df['Date']
df.head()

playoffs = pd.DataFrame({
  'holiday': 'playoff',
  'ds': pd.to_datetime(['2018-02-16', '2019-02-05', '2020-01-25',
                        '2021-02-12', '2022-02-01', '2023-01-01']),
  'lower_window': -2,
  'upper_window': 2,
})
superbowls = pd.DataFrame({
  'holiday': 'superbowl',
  'ds': pd.to_datetime(['2018-09-24', '2019-09-13', '2020-10-01',
                        '2022-09-10']),
  'lower_window': -2,
  'upper_window': 2,
})
holidays = pd.concat((playoffs, superbowls))

m = Prophet(seasonality_mode='additive', daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False,
            changepoint_prior_scale=3, changepoint_range=0.9, growth='linear',
            )

m.add_seasonality(name='daily', period=1, fourier_order=12)
m.add_seasonality(name='weekly', period=7, fourier_order=20)
m.add_seasonality(name='monthly', period=30.5, fourier_order=15)
m.add_seasonality(name='yearly', period=365, fourier_order=20)
m.add_seasonality(name='quarterly', period=365/4, fourier_order=20, prior_scale=20)
m.fit(df);

future = m.make_future_dataframe(periods=365, freq='d')
future = future[future['ds'].dt.weekday < 5]
future.tail()

forecast = m.predict(future)
for col in ['yhat','yhat_lower','yhat_upper']:
    forecast[col] = forecast[col].clip(lower=0.0)
fig = m.plot(forecast, xlabel='Date', ylabel='Price')

fp.add_changepoints_to_plot(fig.gca(), m, forecast);
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()
plt.savefig('./stock1.png')

m.plot(forecast, xlabel='Date', ylabel='Price');
plt.savefig('./stock2.png')

m.plot_components(forecast);
plt.savefig('./stock3.png')
plt.show()

