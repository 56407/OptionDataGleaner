from pandas_datareader.data import Options
import pandas as pd
import numpy as np
from scipy import stats, optimize
from math import exp,log,sqrt
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import cm




def Option_data(options=None,n=2000):
    '''
    Get option data from Yahoo Finance
    Remove incomplete option data and outliers, provide a concise data description,
    calculate Time to Maturity, and save the clean data as a csv data file
    clean criteria: Last_Trade_Date = 1/1/1970
    clean outliers (implied volatility > 2)
    :param options: enter one name or a ticker name list,
            or randomly select all options of n tickers if no option entered
    :param n: number of tickers, default 2000
    :return: option data review and save as a csv file
    '''

    data = pd.DataFrame()
    if options is None:
        df = pd.read_csv('ticker_list.csv')
        company = df['Symbol']
        i = 0
        selected_list = []
        while i < n:
            if n >len(df):
                print ('\nPlease select fewer tickers, the maximum number of tickers is '+str(len(df)))
                break
            name_list = set(company.sample(n-i).tolist())
            for item in (name_list-set(selected_list)):
                try:
                    name = Options(item, 'yahoo')
                    ticker = name.get_all_data()
                    data = data.append(ticker)
                    i += 1
                except:
                    pass
                continue
            selected_list = data.Underlying.tolist()
    else:
        if type(options) is str: #only select one ticker
            try:
                name = Options(options, 'yahoo')
                data = name.get_all_data()
            except:
                print (str(options)+' is not optionable')
        else:
            for item in list(options): # select a list of tickers
                try:
                    name = Options(item, 'yahoo')
                    ticker = name.get_all_data()
                    data = data.append(ticker)
                except:
                    (str(options) + ' is not optionable')
                continue
    del data['JSON']

    df = data.dropna()
    df['Last_Trade_Date'] = pd.to_datetime(df['Last_Trade_Date'])
    df = df[df['Last_Trade_Date'].dt.year != 1970]
    df = df[df['IV'] <= 2]
    df = df[df['Last'] != 0]
    df['T'] = (pd.to_datetime(df['Expiry']) - pd.to_datetime(df['Quote_Time'])) / np.timedelta64(1, 'Y')
    df.to_csv('option_clean.csv')
    print df.describe()
    return df


def separateEU(df):
    '''
    Separate option data into European options and nonEuropean options and save as csv files
    Set difference 0.005 as benchmark to divide the dataset
    :param df: pandasDataFrame
    :return: European option data and nonEuropean option data
    '''

    df.reset_index(level=df.index.names, inplace=True)
    df['T'] = (pd.to_datetime(df['Expiry'])-pd.to_datetime(df['Quote_Time']))/np.timedelta64(1,'Y')

    def bs_pricing(S, K, T, r, sigma, option_type):
        ### INPUT:
        ## option_type: "call" or "put"
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        if option_type == "call":
            N_d1 = stats.norm.cdf(d1, 0.0, 1.0)
            N_d2 = stats.norm.cdf(d2, 0.0, 1.0)
            call_price = S * N_d1 - K * exp(-r * T) * N_d2
            return call_price
        else:
            N_d1 = stats.norm.cdf(-d1, 0.0, 1.0)
            N_d2 = stats.norm.cdf(-d2, 0.0, 1.0)
            put_price = K * exp(-r * T) * N_d2 - S * N_d1
            return put_price

    for idx, row in df.iterrows():
        S = float(df.loc[idx, 'Underlying_Price'])
        K = float(df.loc[idx, 'Strike'])
        T = float(df.loc[idx, 'T'])
        r = 0.03
        option_type = df.loc[idx, 'Type']
        C_star = float(df.loc[idx, 'Last'])
        imp = float(df.loc[idx, 'IV'])
        f = lambda sigma: bs_pricing(S, K, T, r, sigma, option_type) - C_star
        try:
            df.loc[idx, 'C_imp'] = optimize.brentq(f, -5.0, 10.0)
            df.loc[idx,'diff'] = df.loc[idx,'C_imp'] - imp
        except ValueError:
            df.loc[idx,'C_imp'] = 9999
            df.loc[idx, 'diff'] = df.loc[idx, 'C_imp'] - imp
            continue

    df_eu = df[df['diff'].abs() < 0.005]
    df_eu.index = range(len(df_eu))
    df_noneu = df[df['diff'].abs() >= 0.005]
    df_eu.to_csv('EuropeanOption.csv')
    df_noneu.to_csv('NonEuropeOption.csv')
    return df_eu,df_noneu


def plot3d(df,ticker_list,option_type):
    '''
    3d plot of time to maturity, strike price, and implied volatility for certain ticker
    :param df: pandasDataFrame
    :param ticker_list: ticker name list
    :param option_type: 'put' or 'call'
    :return: 3D plot
    '''
    for ticker in ticker_list:
        df_plot = df[df.loc[:, "Root"] == ticker.upper()]
        df_plot = df_plot[df_plot["Type"] == option_type.lower()]
        df_plot['T'] = (pd.to_datetime(df_plot['Expiry']) - pd.to_datetime(df_plot['Quote_Time'])) / np.timedelta64(1, 'Y')

        df_plot1_IV= df_plot['IV']
        df_plot1_T= df_plot['T']
        df_plot1_p= df_plot['Strike']

        fig = plt.figure(figsize=(10, 8))
        ax = Axes3D(fig)
        ax.plot_trisurf(df_plot1_T, df_plot1_p, df_plot1_IV, cmap=cm.coolwarm, linewidth=0.5, antialiased=False)
        ax.set_xlabel('TimeToMaturity')
        ax.set_ylabel('Strike Price')
        ax.set_zlabel('Implied Volatility')
        ax.set_title("Visualization {} {} options 3D plot".format(ticker,option_type))

        plt.show()
    return
    #fig.savefig("{}3D.eps".format(ticker), format='eps')



def iv_hist(df,ticker):
    '''
    Histogram of implied volatility
    :param df: pandasDataFrame
    :param ticker: ticker name
    :return: histogram
    '''
    df = df[df['Root']==ticker.upper()]
    df_call = df[df["Type"] == "call"]
    df_put = df[df["Type"] == "put"]

    fig = plt.figure(figsize=(10, 8))
    fig.suptitle("Implied Volatility of {} option".format(ticker))
    # visualization of call price
    plt.subplot(2, 1, 1)
    plt.hist(df_call["IV"], range=[0.0,2.0],histtype="barstacked", label="call", color="xkcd:sky blue", align="left",rwidth=1.5)
    plt.ylabel("Frequency")
    plt.xlabel("Implied Volatility")
    plt.grid('on')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.hist(df_put["IV"],range=[0.0,2.0], histtype="barstacked", label="put", align="left", rwidth=1.5)
    plt.ylabel("Frequency")
    plt.xlabel("Implied Volatility")
    plt.grid('on')
    plt.legend()
    plt.show()
    return


def any_two_(df,ticker,fac1,fac2):
    '''
    Plot any two factors
    choose the two factors from "Strike,Last,Big,Ask,Chg,PctChg,Vol,Open_Int,IV,Underlying_Price,T"
    :param df: pandasDataFrame
    :param ticker: ticker name
    :param fac1: the first factor
    :param fac2: the second factor
    :return: plot
    '''
    df = df[df['Root']==ticker.upper()]
    df_call = df[df['Type']=='call']
    df_put = df[df['Type']=='put']

    fig = plt.figure(figsize=(10, 8))
    fig.suptitle("Correlation between {} and {} of {}".format(fac1,fac2,ticker))

    plt.subplot(2, 1, 1)
    plt.plot(df_call[fac1],df_call[fac2], label='call',color='xkcd:sky blue')
    plt.ylabel(fac2)
    plt.xlabel(fac1)
    plt.legend(loc='lower right')
    plt.grid("on")

    plt.subplot(2, 1, 2)
    plt.plot(df_put[fac1], df_put[fac2], label= 'put')
    plt.ylabel(fac2)
    plt.xlabel(fac1)
    plt.legend(loc='lower right')
    plt.grid("on")
    plt.show()
    return


def any_three_3D(df,ticker,fac1,fac2,fac3):
    '''
    Provide 3D Visualization of any 3 factors
    choose the three factors from "Strike,Last,Big,Ask,Chg,PctChg,Vol,Open_Int,IV,Underlying_Price,T"
    :param df: pandasDataFrame
    :param ticker: ticker name
    :param fac1: the x label
    :param fac2: the y label
    :param fac3: the z label
    :return: 3D plot
    '''
    df = df[df['Root']==ticker.upper()]
    df_call = df[df['Type']=='call']
    df_put = df[df['Type']=='put']

    fig = plt.figure(figsize=(10, 8))
    fig.suptitle("Correlation between {} and {} of {}".format(fac1,fac2,ticker))

    ax = fig.add_subplot(2,1,1,projection='3d')
    ax.plot_trisurf(df_call[fac1], df_call[fac2], df_call[fac3], cmap=cm.coolwarm, linewidth=0.5, antialiased=False)
    ax.set_xlabel(fac1)
    ax.set_ylabel(fac2)
    ax.set_zlabel(fac3)
    ax.set_title("{} call options 3D plot".format(ticker))

    ax = fig.add_subplot(2, 1, 2, projection='3d')
    ax.plot_trisurf(df_put[fac1], df_put[fac2], df_put[fac3], cmap=cm.coolwarm, linewidth=0.5, antialiased=False)
    ax.set_xlabel(fac1)
    ax.set_ylabel(fac2)
    ax.set_zlabel(fac3)
    ax.set_title("{} put options 3D plot".format(ticker))
    plt.show()
    return





