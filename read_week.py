__author__ = 'Li Bai'

"""# get a weekly revenue forecast on a country level for 3 weeks ahead.
using E-commerce dataset """
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
import numpy as np
import string
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import r2_score
plt.rcParams["figure.figsize"] = (20,12)
font = {#'family' : 'normal',
        # 'weight' : 'bold',
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
    df_UK=df_alpha.loc[df_alpha.Country==country]
    df_UK_rev=df_UK.groupby(['date']).Revenue.sum()
    df_UK_rev.index=pd.to_datetime(df_UK_rev.index) # convert to datetime64[ns]

    df_UK0=pd.DataFrame(index=dates, columns=['Revenue'])
    df_UK0['Revenue']=np.zeros(shape=(len(dates)))
    df_UK0.loc[df_UK_rev.index,'Revenue']=df_UK_rev.to_numpy()


    df_agg.loc[:,country]=df_UK0.iloc[3:,:].to_numpy()
    # df_UK0.loc[:,'week']=[k//7 for k in range(df_UK0.shape[0])]
    # df_UK0.plot()
df_agg_qua=pd.DataFrame(index=dates[3:].to_list(), columns=country_ids.tolist())
# df_agg_week=pd.DataFrame(index=np.arange((len(dates)-3)//7), columns=country_ids.tolist())
for country in country_ids:
    df_UK=df_alpha.loc[df_alpha.Country==country]
    df_UK_rev=df_UK.groupby(['date']).Quantity.sum()
    df_UK_rev.index=pd.to_datetime(df_UK_rev.index) # convert to datetime64[ns]

    df_UK0=pd.DataFrame(index=dates, columns=['Quantity'])
    df_UK0['Quantity']=np.zeros(shape=(len(dates)))
    df_UK0.loc[df_UK_rev.index,'Quantity']=df_UK_rev.to_numpy()


    df_agg_qua.loc[:,country]=df_UK0.iloc[3:,:].to_numpy()
    # df_UK0.loc[:,'week']=[k//7 for k in range(df_UK0.shape[0])]
    # df_UK0.plot()
df_agg_price=pd.DataFrame(index=dates[3:].to_list(), columns=country_ids.tolist())
df_alpha_price=df_alpha.loc[df_alpha.UnitPrice>0] # don't consider the zero price or negative price
# cancel behavior!!
for country in country_ids:
    df_UK=df_alpha_price.loc[df_alpha_price.Country==country]
    df_UK_rev=df_UK.groupby(['date']).UnitPrice.sum()
    df_UK_rev.index=pd.to_datetime(df_UK_rev.index) # convert to datetime64[ns]

    df_UK0=pd.DataFrame(index=dates, columns=['UnitPrice'])
    df_UK0['UnitPrice']=np.zeros(shape=(len(dates)))
    df_UK0.loc[df_UK_rev.index,'UnitPrice']=df_UK_rev.to_numpy()


    df_agg_price.loc[:,country]=df_UK0.iloc[3:,:].to_numpy()
    # df_UK0.loc[:,'week']=[k//7 for k in range(df_UK0.shape[0])]
    # df_UK0.plot()

df_agg_week=df_agg.copy()
df_agg_week=(df_agg_week-df_agg.min())/(df_agg.max()-df_agg.min())

# plt.figure()
# df_agg.iloc[:,0].plot(label="Revenue")
# df_agg_qua.iloc[:,0].plot(label="Quantity")
# df_agg_price.iloc[:,0].plot(label="Price")

df_UK_tot=pd.DataFrame(index=df_agg.index, columns=['revenue', 'quantity','price'])
df_UK_tot.iloc[:,:]=pd.concat((df_agg['United Kingdom'],df_agg_qua['United Kingdom'],df_agg_price['United Kingdom']),axis=1).to_numpy()
df_UK_tot_norm=(df_UK_tot-df_UK_tot.min())/(df_UK_tot.max()-df_UK_tot.min())
# df_UK_tot_norm['revenue-France']=df_agg_week['France'].to_numpy()
# df_UK_tot_norm['revenue-Germany']=df_agg_week['Germany'].to_numpy()

# substract daily maximum, minimum and average
df_UK_tot_norm.loc[:,'week']=[k//7 for k in range(df_agg.shape[0])]
df_UK_tot_norm1=df_UK_tot_norm.groupby('week').sum()

df_UK_tot_norm1['revenue max']=df_UK_tot_norm.groupby('week')['revenue'].max().to_list()
df_UK_tot_norm1['revenue mean']=df_UK_tot_norm.groupby('week')['revenue'].mean().to_list()

df_UK_tot_norm1['quantity max']=df_UK_tot_norm.groupby('week')['quantity'].max().to_list()
df_UK_tot_norm1['quantity mean']=df_UK_tot_norm.groupby('week')['quantity'].mean().to_list()


df_UK_tot_norm1['price max']=df_UK_tot_norm.groupby('week')['price'].max().to_list()
df_UK_tot_norm1['price mean']=df_UK_tot_norm.groupby('week')['price'].mean().to_list()

# df_UK_tot_norm1['revenue-France']=df_UK_tot_norm.groupby('week')['revenue-France'].mean().to_list()
# df_UK_tot_norm1['revenue-Germany']=df_UK_tot_norm.groupby('week')['revenue-Germany'].mean().to_list()


# generate one-hot code for each day of week and special holidays!

holidays=['2010-12-24','2010-12-25','2010-12-26','2010-12-27','2010-12-28','2010-12-29','2010-12-30','2010-12-31',
          '2011-01-01','2011-01-02','2011-01-03','2011-04-22','2011-04-23','2011-04-24','2011-04-29','2011-05-02',
          '2011-05-30','2011-08-29','2011-12-24','2011-12-25','2011-12-26','2011-12-27','2011-12-28','2011-12-29',
          '2011-12-30','2011-12-31','2012-01-01','2012-01-02','2011-01-03']
# df_agg with weeks
df_agg_days=pd.DataFrame(index=df_agg.index, columns=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
df_agg_days.iloc[:,:]=np.zeros(shape=(df_agg.shape[0],7))
for num, col in enumerate(df_agg_days.columns.tolist()):
    df_agg_days.loc[df_agg_days.index.weekday==num,col]=1
df_agg_days['holiday']=[0]*df_agg_days.shape[0]
df_agg_days.loc[df_agg_days.index.isin(holidays),'holiday']=1
df_agg_days.loc[:,'week']=[k//7 for k in range(df_agg_days.shape[0])]



df_agg_week_uk_merge=pd.concat((df_UK_tot_norm1, df_agg_days.groupby('week')['holiday'].sum()),axis=1)
df_agg_week_uk_merge['holiday']=df_agg_week_uk_merge['holiday'].to_numpy()+1 # add saturday to each week holiday


# use for ARX regression dataset
df_reg=df_agg_week_uk_merge.copy()
df_reg['date']=df_UK_tot_norm.index.tolist()[0::7]
df_reg=df_reg.set_index('date')

df_reg=df_reg.iloc[3:,:].copy()
df_reg['revenue max']=df_agg_week_uk_merge['revenue max'].to_numpy()[0:-3]
df_reg['revenue mean']=df_agg_week_uk_merge['revenue mean'].to_numpy()[0:-3]
df_reg['quantity max']=df_agg_week_uk_merge['quantity max'].to_numpy()[0:-3]
df_reg['quantity mean']=df_agg_week_uk_merge['quantity mean'].to_numpy()[0:-3]
df_reg['price max']=df_agg_week_uk_merge['price max'].to_numpy()[0:-3]
df_reg['price mean']=df_agg_week_uk_merge['price mean'].to_numpy()[0:-3]
df_reg.pop('quantity')
df_reg.pop('price')
# df_reg['date']=df_UK_tot_norm.index.tolist()[0::7][3:]
# df_reg=df_reg.set_index('date')



# correlation for normalized dataset
# import seaborn as sns
# sns.set_theme(style="white", palette=None)
# sns.set(font_scale=2)
# pp = sns.pairplot(data=df_reg,
#                   y_vars=['revenue'],
#                   x_vars=['revenue','revenue max', 'revenue mean', 'quantity max',
#        'quantity mean', 'price max', 'price mean', 'holiday'])
#




# uk revenue time series to be analyzed!
revenue_week=df_UK_tot_norm1.revenue.to_numpy()

N_tot=53
N_train=35; N_valid=50-N_train
# smoothing exponential average;
u_train=revenue_week[3:3+N_train];u_valid=revenue_week[3+N_train:]
# df_train=pd.DataFrame(index=df_agg_week_uk_merge.index[3:43], columns=["Real","Persistence*","Naive*","SmoAvg*",
#                                                             'ES(A,N)*','ES(Ad, N)*','AR*','ARX*'])

df_valid=pd.DataFrame(index=df_reg.index[N_train:], columns=["Real","Persistence","Naive","SmoAvg",
                                                            'ES(A,N)*','ES(Ad,N)*','AR*','ARX*'])

# Real
# df_train['Real']=revenue_week[3:43]
df_valid['Real']=revenue_week[3+N_train:]

# persistence
# df_train['Persistence']=revenue_week[0:40]
df_valid['Persistence']=revenue_week[N_train:50]

# Naive
# df_train['Naive']=[revenue_week[0:40].mean()]*40
df_valid['Naive']=[revenue_week[0:N_train].mean()]*N_valid

# smoothing average
def smoothavg(revenue_week):
    return [revenue_week[(0):((k+1))].mean(axis=0).tolist() for k in range(N_tot-3)]
revenue_week_smo=smoothavg(revenue_week);revenue_week_smo=np.array(revenue_week_smo).reshape(-1,1)
# df_train["SmoAvg"]=revenue_week_smo[0:40,0]
df_valid["SmoAvg"]=revenue_week_smo[N_train:,0]


def ES_AN(revenue_week):
    # if it keeps increasing, we can add damped elements!
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as ES
    trend_pred=[revenue_week[0]]
    for k in range(1, N_tot-3):
        model=ES(revenue_week[0:(1+k)], trend="add").fit()
        pred=model.predict(start=(3+k),end=(3+k))
        trend_pred=trend_pred+pred.tolist()
    return trend_pred

revenue_week_AN=ES_AN(revenue_week);
# df_train['HWtrend']=revenue_week_AN[0:40]
df_valid["ES(A,N)*"]=revenue_week_AN[N_train:N_train+N_valid]

def ES_AN_damp(revenue_week):
    # if it keeps increasing, we can add damped elements!
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as ES
    trend_pred=[revenue_week[0]]
    for k in range(1, N_tot-3):
        model=ES(revenue_week[0:(1+k)], trend="add", damped_trend=True).fit()
        pred=model.predict(start=(3+k),end=(3+k))
        trend_pred=trend_pred+pred.tolist()
    return trend_pred

revenue_week_AN_damp=ES_AN_damp(revenue_week);
# df_train['HWtrend_damp']=revenue_week_AN_damp[0:40]
df_valid["ES(Ad,N)*"]=revenue_week_AN_damp[N_train:N_train+N_valid]


def arma_forecast(revenue_week):
    from statsmodels.tsa.arima.model import ARIMA
    trend_pred = []
    for k in range(0, N_valid):
        model = ARIMA(revenue_week[0:(N_train + k)], order=(2,0,0)).fit()
        pred = model.predict(start=(N_train+3 + k), end=(N_train+3 + k))
        trend_pred = trend_pred + pred.tolist()
    return trend_pred

arma_pred=arma_forecast(revenue_week)
df_valid["AR*"]=arma_pred

# OLS framework basic regressions!!!
output=df_reg.iloc[:,0]
features=df_reg.iloc[:,1:]

output_train=output.iloc[0:N_train].to_numpy() #280
features_train=features.iloc[0:N_train,:].to_numpy() #280*3


beta=np.linalg.inv(features_train.T.dot(features_train)).dot(features_train.T).dot(output_train)

output_test=output.iloc[N_train:].to_numpy()
features_test=features.iloc[N_train:,:].to_numpy()

pred_train=features_train.dot(beta)
pred_test=features_test.dot(beta)

df_valid['ARX*']=pred_test.tolist()
# plot forecasting results
df_valid.plot(linewidth=3)
plt.ylabel(r'Daily Revenue($\times$105984 )')

# plot acf and pacf for weekly revenue time series
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(16,6), dpi= 80)
plot_acf(revenue_week, ax=ax1, lags=20); ax1.set_xlabel('Lag (week)')
plot_pacf(revenue_week, ax=ax2, lags=20);ax2.set_xlabel('Lag (week)')


df_metric=pd.DataFrame(index=df_valid.columns, columns=['RMSE','MAE','MAPE','R2-score'])
for col in df_metric.index:
    df_metric.loc[col,'RMSE']=mean_squared_error(df_valid['Real'], df_valid[col], squared=False,
                    multioutput='raw_values')[0]
    df_metric.loc[col,'MAE']=mean_absolute_error(df_valid['Real'],df_valid[col],
                    multioutput='raw_values')[0]
    df_metric.loc[col,'MAPE']=mean_absolute_percentage_error(df_valid['Real'],df_valid[col],
                    multioutput='raw_values')[0]
    df_metric.loc[col,'R2-score']=r2_score(df_valid['Real'],df_valid[col],
                    multioutput='raw_values')[0]

# plot metrics
df_metric.plot(kind='bar',width=1, rot=15, alpha=0.3)

df_metric.to_csv("week_metric.csv")
df_valid.to_csv('week_pred.csv')






