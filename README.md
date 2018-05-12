# OptionDataGleaner

  OptionDataGleaner is designed to get option information from Yahoo Finance, separate the dataset into European option data and nonEuropean option data and provide data visualization for users.

  Given a ticker name or a tickers list this program attempts to crawl from web by Option_data function. This function returns clean dataset that removes incomplete data and outliers (implied volatility > 2) and save it to csv file.
  The function separateEU(df) serves to partition option data into European options and nonEuropean options by BS model and save as csv files.
  The histgram of implied volatility can be viewed by iv_hist(df,ticker). You can also visualize the correlation between any two factors by the function any_two(df,ticker,fac1,fac2). What's more, any_three_3D(df,ticker, fac1,fac2,fac3) can provide 3D visualization for any three factors.
  
  Graphics interface is coming soon.
