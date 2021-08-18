__author__ = 'Li Bai'

"""# get a weekly revenue forecast on a country level for 3 weeks ahead.
using E-commerce dataset by analyzing the daily revenue time series"""


import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
import numpy as np
import string
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from sklearn.metrics import r2_score
import statsmodels.api as sm
import matplotlib
plt.rcParams["figure.figsize"] = (20,12)
font = {#'family' : 'normal',
        # 'weight' : 'normal',
        'size'   : 28}
matplotlib.rc('font', **font)


ALPHA = string.ascii_letters
BETA=string.digits

# pd.set_option('max_colwidth', -1)
df=pd.read_csv("data.csv", encoding='unicode_escape')
df['StockCode']=df['StockCode'].str.upper()
# You can ignore the non encoded UTF-8 by using encoding='unicode_escape'. It works.

# data visualization:
# Index(['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate',
#        'UnitPrice', 'CustomerID', 'Country']
df.InvoiceDate=pd.to_datetime(df.InvoiceDate)
df['date']=df.InvoiceDate.dt.date
df.date=pd.to_datetime(df.date)
df['Revenue']=df.UnitPrice*df.Quantity
# df.loc[df.StockCode.str.startswith(tuple(ALPHA))]['StockCode'].unique()
# ['POST', 'D', 'C2', 'DOT', 'M', 'BANK CHARGES', 'S', 'AMAZONFEE',
#        'DCGS0076', 'DCGS0003', 'GIFT_0001_40', 'DCGS0070', 'GIFT_0001_50',
#        'GIFT_0001_30', 'GIFT_0001_20', 'DCGS0055', 'DCGS0072', 'DCGS0074',
#        'DCGS0069', 'DCGS0057', 'DCGSSBOY', 'DCGSSGIRL', 'GIFT_0001_10',
#        'PADS', 'DCGS0004', 'DCGS0073', 'DCGS0071', 'DCGS0068', 'DCGS0067',
#        'DCGS0066P', 'B', 'CRUK']

# remove invalid items based on stockcode
alpha_stockcode_out=['POST', 'D', 'C2', 'DOT', 'M', 'BANK CHARGES', 'S', 'AMAZONFEE','B', 'CRUK']
# ['POST','C2','DOT', 'M', 'BANK CHARGES', 'S', 'AMAZONFEE', 'B', 'CRUK','D',] # discount is
# considered!!!
df_alpha=df.loc[df.StockCode.isin(alpha_stockcode_out)==False] #538914
# Price
# remove these items that make no contribution to revenue
df_alpha=df_alpha.loc[df_alpha.Revenue!=0]
# generate the continuous daily dates
date0 = df_alpha.date.iloc[0];
date1 = df_alpha.date.iloc[-1]
dates = pd.date_range(date0, date1, freq='D')
country_ids=df_alpha.Country.unique() #38 countries
# price is all positive and the extremely high values are for some products
# quantity can be negative, indicating the returning products

# the missing customers ID and the missing values for description do not matter
#  do country aggregation for revenue; quantity and price; daily dataset; remove the first 3 and make it as 53 weeks
#  in total
df_agg=pd.DataFrame(index=dates[3:].to_list(), columns=country_ids.tolist())
# df_agg_week=pd.DataFrame(index=np.arange((len(dates)-3)//7), columns=country_ids.tolist())
for country in country_ids:
    df_country=df_alpha.loc[df_alpha.Country==country]
    df_country_rev=df_country.groupby(['date']).Revenue.sum()
    df_country_rev.index=pd.to_datetime(df_country_rev.index) # convert to datetime64[ns]

    df_country0=pd.DataFrame(index=dates, columns=['Revenue'])
    df_country0['Revenue']=np.zeros(shape=(len(dates)))
    df_country0.loc[df_country_rev.index,'Revenue']=df_country_rev.to_numpy()


    df_agg.loc[:,country]=df_country0.iloc[3:,:].to_numpy()
    # df_UK0.loc[:,'week']=[k//7 for k in range(df_UK0.shape[0])]
    # df_UK0.plot()
df_agg_qua=pd.DataFrame(index=dates[3:].to_list(), columns=country_ids.tolist())
# df_agg_week=pd.DataFrame(index=np.arange((len(dates)-3)//7), columns=country_ids.tolist())
for country in country_ids:
    df_country=df_alpha.loc[df_alpha.Country==country]
    df_country_rev=df_country.groupby(['date']).Quantity.sum()
    df_country_rev.index=pd.to_datetime(df_country_rev.index) # convert to datetime64[ns]

    df_country0=pd.DataFrame(index=dates, columns=['Quantity'])
    df_country0['Quantity']=np.zeros(shape=(len(dates)))
    df_country0.loc[df_country_rev.index,'Quantity']=df_country_rev.to_numpy()


    df_agg_qua.loc[:,country]=df_country0.iloc[3:,:].to_numpy()
    # df_UK0.loc[:,'week']=[k//7 for k in range(df_UK0.shape[0])]
    # df_UK0.plot()
df_agg_price=pd.DataFrame(index=dates[3:].to_list(), columns=country_ids.tolist())
# df_alpha_price=df_alpha.loc[df_alpha.UnitPrice>0] # df_alpha_price is exactly the same as df_alpha
# cancel behavior!!
for country in country_ids:
    df_country=df_alpha.loc[df_alpha.Country==country]
    df_country_rev=df_country.groupby(['date']).UnitPrice.sum()
    df_country_rev.index=pd.to_datetime(df_country_rev.index) # convert to datetime64[ns]

    df_country0=pd.DataFrame(index=dates, columns=['UnitPrice'])
    df_country0['UnitPrice']=np.zeros(shape=(len(dates)))
    df_country0.loc[df_country_rev.index,'UnitPrice']=df_country_rev.to_numpy()


    df_agg_price.loc[:,country]=df_country0.iloc[3:,:].to_numpy()


# plt.figure()
# df_agg.iloc[:,0].plot(label="Revenue")
# df_agg_qua.iloc[:,0].plot(label="Quantity")
# df_agg_price.iloc[:,0].plot(label="Price")

df_UK_tot=pd.DataFrame(index=df_agg.index, columns=['revenue', 'quantity','price'])
df_UK_tot.iloc[:,:]=pd.concat((df_agg['United Kingdom'],df_agg_qua['United Kingdom'],df_agg_price['United Kingdom']),axis=1).to_numpy()
df_UK_days=pd.DataFrame(index=df_agg.index, columns=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
df_UK_days.iloc[:,:]=np.zeros(shape=(df_agg.shape[0],7))
for num, col in enumerate(df_UK_days.columns.tolist()):
    df_UK_days.loc[df_UK_days.index.weekday==num,col]=1

holidays=['2010-12-24','2010-12-25','2010-12-26','2010-12-27','2010-12-28','2010-12-29','2010-12-30','2010-12-31',
          '2011-01-01','2011-01-02','2011-01-03','2011-04-22','2011-04-23','2011-04-24','2011-04-29','2011-05-02',
          '2011-05-30','2011-08-29','2011-12-24','2011-12-25','2011-12-26','2011-12-27','2011-12-28','2011-12-29',
          '2011-12-30','2011-12-31','2012-01-01','2012-01-02','2011-01-03']

df_agg_days=pd.DataFrame(index=df_agg.index, columns=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
df_agg_days.iloc[:,:]=np.zeros(shape=(df_agg.shape[0],7))
for num, col in enumerate(df_agg_days.columns.tolist()):
    df_agg_days.loc[df_agg_days.index.weekday==num,col]=1
df_agg_days['holiday']=[0]*df_agg_days.shape[0]
df_agg_days.loc[df_agg_days.index.isin(holidays),'holiday']=1



# uk whole dataset
max_norm=df_UK_tot.max();
min_norm=df_UK_tot.min();

df_UK_tot_norm=df_UK_tot.copy()
df_UK_tot_norm=(df_UK_tot_norm-min_norm)/(max_norm-min_norm)


#
#
#
# # uk dataset revenue
# revenue_day_max=df_agg.max();revenue_day_min=df_agg.min()
# df_agg_norm=(df_agg-revenue_day_min)/(revenue_day_max-revenue_day_min)
#

# uk dataset: revenue_day is normalized revenue daily time series
revenue_day=df_UK_tot_norm.revenue.to_numpy()


df_uk=df_UK_tot_norm.copy()
# add week
df_uk.loc[:,'week']=[k//7 for k in range(df_uk.shape[0])]

#  fft analysis
from scipy.fft import fft, fftfreq, ifft
N=364
T=1/364
x = np.linspace(0.0, N*T, N, endpoint=False)
yf = fft(revenue_day[0:364])
yf_copy=yf*0;yf_copy[0::52]=yf[0::52]
iyf_copy_real=np.real(ifft(yf_copy))

# =================add fft
iyf_cycle=iyf_copy_real[0:7].tolist()*53
df_uk['fft']=iyf_cycle

# ==========plot fft
# xf = fftfreq(N, T)[:N//2]
# plt.figure()
# plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
# plt.grid()
# plt.show()




# /* Draw Plot */ analysis for Revenue for UK;
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(16,6), dpi= 80)
plot_acf(df_agg.iloc[:,0], ax=ax1, lags=40)
plot_pacf(df_agg.iloc[:,0], ax=ax2, lags=40)

diff_revenue_day=df_agg.iloc[7:,0].to_numpy()-df_agg.iloc[0:-7,0].to_numpy()
# /* Draw Plot */ analysis for Revenue for UK;
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(16,6), dpi= 80)
plot_acf(diff_revenue_day, ax=ax1, lags=40)
plot_pacf(diff_revenue_day, ax=ax2, lags=40)



# add 3-week revenue, quantity and price as features
index=df_uk.index.to_list()
df_uk.loc[:,'revenue-21']=[0]*df_uk.shape[0]
df_uk.loc[index[21:],'revenue-21']=df_uk.loc[index[0:-21],'revenue'].to_list()

df_uk.loc[:,'quantity-21']=[0]*df_uk.shape[0]
df_uk.loc[index[21:],'quantity-21']=df_uk.loc[index[0:-21],'quantity'].to_list()
#
df_uk.loc[:,'price-21']=[0]*df_uk.shape[0]
df_uk.loc[index[21:],'price-21']=df_uk.loc[index[0:-21],'price'].to_list()


df_uk.pop('price')
df_uk.pop('quantity')

df_uk_1=pd.concat((df_uk,df_agg_days), axis=1)
# 'fft', 'revenue-21', 'quantity-21', 'price-21', 'Mon', 'Tue', 'Wed',
       # 'Thu', 'Fri', 'Sat', 'Sun', 'holiday'



N_tot=53
N_train=35; N_valid=50-N_train


# smoothing exponential average;
u_train=revenue_day[3*7:(N_train+3)*7];u_valid=revenue_day[(N_train+3)*7:]
# df_train=pd.DataFrame(index=df_agg.index[21:43*7], columns=["Real","Persistence","Naive","SmoAvg","ES(N,A)",
#                                                             'ES(A,A)','ES(Ad, A)','SARIMA','SARIMAX','ARX'])
df_valid=pd.DataFrame(index=df_agg.index[(N_train+3)*7:], columns=["Real","Persistence","Naive","SmoAvg","ES(N,A)",
                                                            'ES(A,A)','ES(Ad, A)','SARIMA','SARIMAX','ARX'])


# Real train and valid dataset
# df_train['Real']=revenue_day[21:43*7]
df_valid['Real']=revenue_day[(N_train+3)*7:]

# persistence
# df_train['Persistence']=revenue_day[0:40*7]
df_valid['Persistence']=revenue_day[N_train*7:50*7]

# Naive
# df_train['Naive']=revenue_day[0:40*7].reshape(-1,7).mean(axis=0).tolist()*40
df_valid['Naive']=revenue_day[0:N_train*7].reshape(-1,7).mean(axis=0).tolist()*N_valid


# smoothing average
def smoothavg(revenue_day):
    return [revenue_day[(0*7):((k+1)*7)].reshape(-1,7).mean(axis=0).tolist() for k in range(N_tot-3)]
revenue_day_smo=smoothavg(revenue_day);revenue_day_smo=np.array(revenue_day_smo).reshape(-1,1)
# df_train["SmoAvg"]=revenue_day_smo[0:40*7,0]
df_valid["SmoAvg"]=revenue_day_smo[N_train*7:,0]


def ES_NA(revenue_day):
    # additive seasonal component
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as ES
    trend_pred = revenue_day[0:7].tolist()
    for k in range(1, N_tot-3):
        model=ES(revenue_day[0:(1+k)*7], seasonal='add', seasonal_periods=7).fit()
        pred=model.predict(start=(3+k)*7,end=(3+k+1)*7-1)
        trend_pred=trend_pred+pred.tolist()
    return trend_pred


revenue_day_NA=ES_NA(revenue_day)
# df_train["ES(N,A)"]=revenue_day_NA[0:40*7]#[0:40*7,0]
df_valid["ES(N,A)"]=revenue_day_NA[N_train*7:]#[40*7:,0]

def ES_AA(revenue_day):
    # additive trend and additive seasonal component
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as ES
    trend_pred = revenue_day[0:7].tolist()
    for k in range(1, N_tot-3):
        model=ES(revenue_day[0:(1+k)*7], trend="add", seasonal='add', seasonal_periods=7).fit()
        pred=model.predict(start=(3+k)*7,end=(3+k+1)*7-1)
        trend_pred=trend_pred+pred.tolist()
    return trend_pred

revenue_day_AA=ES_AA(revenue_day);
# df_train["ES(A,A)"]=revenue_day_AA[0:40*7]
df_valid["ES(A,A)"]=revenue_day_AA[N_train*7:]

def ES_AA_damp(revenue_day):
    # additive damped trend and additive seasonal component
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as ES
    trend_pred = revenue_day[0:7].tolist()
    for k in range(1, N_tot-3):
        model=ES(revenue_day[0:(1+k)*7], trend="add", damped_trend=True, seasonal='add', seasonal_periods=7).fit()
        pred=model.predict(start=(3+k)*7,end=(3+k+1)*7-1)
        trend_pred=trend_pred+pred.tolist()
    return trend_pred

revenue_day_AA_d=ES_AA_damp(revenue_day);
# df_train["ES(Ad, A)"]=revenue_day_AA_d[0:40*7]
df_valid["ES(Ad, A)"]=revenue_day_AA_d[N_train*7:]



def arima_forecast(revenue_day):
    from statsmodels.tsa.arima.model import ARIMA
    preds = []#revenue_day[0:7].tolist()+revenue_day[0:7].tolist()
    for k in range(0, N_valid):
        model = ARIMA(revenue_day[0:7*(N_train+k)], order=(2, 0, 0), seasonal_order=(0, 1, 1, 7))
        # model = ARIMA(revenue_day[0:7*(1+k)], order=(2, 0, 0), seasonal_order=(0, 1, 1, 7))
        model_fit = model.fit()
        # pred = model_fit.predict(start=(3 + k) * 7, end=(3 + k + 1) * 7 - 1)
        pred = model_fit.predict(start=7*(N_train+3+k), end=7*(N_train+3+1+k)-1)
        preds=preds+pred.tolist()
    return preds
arima_pred=arima_forecast(revenue_day);
df_valid["SARIMA"]=arima_pred#[40*7:]


# exogenous features
enso=df_uk_1.copy()
enso.pop('revenue-21')
enso.pop('price-21')
enso.pop('quantity-21')

enso=enso.to_numpy()
# the 7th day revenue comes with zero, which calls the singular problem in sovling SARIMAX in statespace models;
# we use the day before holiday (Christmas week with all zeros) to replace the day after that
enso[7*6:7*7,1]=enso[7*5:7*6,1]

def sarimax_forecast(revenue_day):
    # remove the first day of all zeros features to avoid singular problem in solvers
    train_revenue_day=revenue_day[7*3:7*N_train]
    test_revenue_day=revenue_day[7*N_train:]
    # model=sm.tsa.statespace.SARIMAX(train_revenue_day, exog=enso[7*4:7*43,0:2], order=(1, 0, 0), seasonal_order=(0, 1, 1, 7))
    model=sm.tsa.statespace.SARIMAX(train_revenue_day, exog=enso[7*6:7*(3+N_train),:], order=(1, 0, 0), seasonal_order=(0, 1, 1,
                                                                                                             7))
    model_fit=model.fit(disp=False)
    pred=model_fit.predict(start=7*(3+N_train-6), end=7*(3+N_train-6+N_valid)-1, exog=enso[7*(N_train+3):,:])
    return pred

sarimax_pred=sarimax_forecast(revenue_day)
df_valid["SARIMAX"]=sarimax_pred

# OLS framework /;autoregression framework
output=df_uk_1.pop('revenue')
df_uk_1.pop('week')
features=df_uk_1
output_train=output.iloc[3*7:(3+N_train)*7].to_numpy() #280
features_train=features.iloc[3*7:(3+N_train)*7,:].to_numpy() #280*3
beta=np.linalg.inv(features_train.T.dot(features_train)).dot(features_train.T).dot(output_train)

output_test=output.iloc[(3+N_train)*7:].to_numpy()
features_test=features.iloc[(3+N_train)*7:,:].to_numpy()
pred_train=features_train.dot(beta)
pred_test=features_test.dot(beta)
# plt.figure();plt.plot(output_train, label='output');plt.plot(pred_train, label='pred')
# plt.figure();plt.plot(output_test, label='output');plt.plot(pred_test, label='pred')
df_valid["ARX"]=pred_test
# ensure that saturday and holidays are equal to 1
df_valid.loc[df_valid.index.dayofweek==5,:]=0.
df_valid.loc[df_valid.index.isin(holidays),:]=0.
df_valid[df_valid<=0]=0.
# plot validation dataset
df_valid.plot(linewidth=3)


# convert daily forecast time series to weekly forecast time series
df_valid_week=df_valid.copy()
df_valid_week['week']=df_uk.loc[df_valid.index, 'week']#[k//7 for k in range(df_valid_week.shape[0])]
df_valid_week=df_valid_week.groupby('week').sum()
df_valid_week['date']=df_valid.index.tolist()[0::7]
df_valid_week=df_valid_week.set_index('date')

df_metric=pd.DataFrame(index=df_valid_week.columns, columns=['RMSE','MAE','MAPE','R2-score'])
for col in df_metric.index:
    df_metric.loc[col,'RMSE']=mean_squared_error(df_valid_week['Real'], df_valid_week[col], squared=False,
                    multioutput='raw_values')[0]
    df_metric.loc[col,'MAE']=mean_absolute_error( df_valid_week['Real'],df_valid_week[col],
                    multioutput='raw_values')[0]
    df_metric.loc[col,'MAPE']=mean_absolute_percentage_error(df_valid_week['Real'],df_valid_week[col],
                    multioutput='raw_values')[0]

    df_metric.loc[col,'R2-score']=r2_score(df_valid_week['Real'],df_valid_week[col],
                       multioutput='raw_values')[0]

df_metric.plot(kind='bar',rot=15)

# ================save to csv file=================
df_metric.to_csv("day_metric.csv")
df_valid_week.to_csv('day_pred.csv')


