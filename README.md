# OptionDataGleaner

OptionDataGleaner is designed to get option information from Yahoo Finance, separate the dataset into European option data and nonEuropean option data and provide data visualization for users.

Main functions are shown as below:

	1. Option_data(options,n)
Given a ticker name or a ticker list this program attempts to crawl the option information from web. If you don't have a target, it will randomly select n (default 2000) tickers to download. This function returns clean dataset that incomplete data and outliers (implied volatility > 2) have already been removed and you can get a csv file.
  
	2. separateEU(df) 
Serves to partition option data into European options and nonEuropean options by BS model and save as csv files.
  

For data visualization, the histogram of implied volatility can be provided by

	iv_hist(df,ticker)
![hist example](https://github.com/zzhou59/OptionDataGleaner/blob/master/screenshot/hist%20Example.png)
However, you can visualize any two or three factors freely by
	
	any_two(df,ticker,fac1,fac2)
and even 3D visualization

	any_three_3D(df,ticker, fac1,fac2,fac3)
 
![3DPlot Example](https://github.com/zzhou59/OptionDataGleaner/raw/master/screenshot/3Dplot%20Example.png)	

Graphics interface is coming soon.
