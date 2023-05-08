import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
# import statsmodels.api as sm

data = pd.read_csv('freem.csv')
datas = pd.read_csv('BicycleWeather.csv')

data.columns

datas.columns

data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')

datas['Dates'] = pd.to_datetime(datas['DATE'])
datas = datas.set_index('DATE')

daily_traffic = data.groupby('Date')['Fremont Bridge Total'].sum()
daily_traffic_df = pd.DataFrame({'Date':daily_traffic.index,'Fremont Bridge Total':daily_traffic.values})
daily_traffic_df

data['Date'] = pd.to_datetime(data.index.date)
dummies = pd.get_dummies(data['Date'].dt.day_name())
data = pd.concat([data, dummies],axis=1)
daily_bicycle_traffic = data.resample('D').sum()
print(daily_bicycle_traffic.head())

cal = USFederalHolidayCalendar()
holidays = cal.holidays(start=data.index.min(), end=data.index.max())
data['Holiday'] = data.index.isin(holidays).astype(int)

data = data.dropna(subset = ['Fremont Bridge Total','Fremont Bridge East Sidewalk','Fremont Bridge West Sidewalk'])

data.columns

column_names = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday','Holiday']
X= data[column_names]
y= data['Fremont Bridge Total']

model = LinearRegression(fit_intercept=False)
model.fit(X, y)
data['predicted'] = model.predict(X)

data[['Fremont Bridge Total', 'predicted']].plot(alpha=0.5);

params = pd.Series(model.coef_, index=X.columns)
params

np.random.seed(1)
err = np.std([model.fit(*resample(X, y)).coef_
              for i in range(1000)], 0)

print(pd.DataFrame({'effect': params.round(0),
                    'error': err.round(0)}))