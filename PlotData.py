import pandas as pd
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot
from pandas.plotting import bootstrap_plot
import matplotlib.pyplot as plt

DAY=24
TWO_DAYS=24*2
WEEK=24*7
YEAR=24*365

def create_plots(df):
    df_ohlc = df.RT_LMP.resample('D').ohlc()

    pv = pd.pivot_table(df, index=df.index.dayofyear,
            columns=df.index.year, values='RT_LMP')
    pv.plot(title="RT_LMP Year over Year")
    plt.grid(True)

    plt.figure()
    df['RT_LMP'].plot(title="NE-ISO RT_LMP")
    plt.ylabel("RT_LMP ($/MWh")
    plt.grid(True)

    y2013 = df['2012-10-01':'2013-09-30']
    y2014 = df['2013-10-01':'2014-09-30']
    y2013_ohlc = y2013.RT_LMP.resample('D').ohlc()
    y2014_ohlc = y2014.RT_LMP.resample('D').ohlc()

    y2013_ohlc.plot(title="Fiscal Year 2013: Open, High, Low, Close")
    plt.ylabel("Price ($/MWh)")
    plt.grid(True)

    y2014_ohlc.plot(title="Fiscal Year 2014: Open, High, Low, Close")
    plt.ylabel("Price ($/MWh)")
    plt.grid(True)

    y2014.plot(y='DryBulb', title='2014: Temperature', legend=False)
    plt.ylabel('Temperature (F)')
    plt.grid(True)

    # y2014.plot(y=['DryBulb','RT_LMP'], title='2014 Temperature and Price')
    # plt.ylabel('Price ($)/Temperature (F)')
    # plt.grid(True)

    y2014_drybulb = y2014.sort_values('DryBulb')
    y2014_drybulb.plot(x='DryBulb', y='RT_LMP', legend=False,
            title="Fiscal Year 2014: Price vs Temperature")
    plt.xlabel('Temperature (F)')
    plt.ylabel('Price ($)')
    plt.grid(True)

    y2014_demand = y2014.sort_values('DEMAND')
    y2014_demand.plot(x="DEMAND", y='RT_LMP', legend=False,
            title="Fiscal Year 2014: Price vs Demand")
    plt.xlabel('Demand (kW)')
    plt.ylabel('Price ($/MWh)')
    plt.grid(True)

    plt.figure()
    lag_plot(df)
    plt.title("Lag Plot: 1 Hour")
    plt.grid(True)

    plt.figure()
    lag_plot(df, lag=DAY)
    plt.title("Lag Plot: One Day")
    plt.grid(True)

    plt.figure()
    lag_plot(df, lag=TWO_DAYS)
    plt.title("Lag Plot: Two Days")
    plt.grid(True)

    plt.figure()
    lag_plot(df, lag=WEEK)
    plt.title("Lag Plot: One Week")
    plt.grid(True)

    plt.figure()
    lag_plot(df, lag=YEAR)
    plt.title("Lag Plot: One Year")
    plt.grid(True)

    plt.figure()
    for year in df.index.year.unique():
        autocorrelation_plot(df[str(year)].RT_LMP)
    plt.title('Autocorrelation')
    plt.grid(True)

    plt.show()
