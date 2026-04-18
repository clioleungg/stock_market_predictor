"""
Stock Price Prediction Using Historical Data & Machine Learning
DS2500 Final Project — Team: Justin, Lam, Simon (Haoming), Clio
Section: 1:35–3:15 PM Tuesday & Friday, Dehan Yu

Analyzes ALL S&P 500 companies as a whole:
  1. Historical OHLCV data for S&P 500 tickers (Jan 2020 – Jan 2026)
     collected via the yfinance Python library from Yahoo Finance
  2. Index-wide EDA: aggregate return distributions, sector comparisons,
     cross-stock correlation structure, volume patterns
  3. Feature engineering: 17 technical indicators applied to every ticker
  4. ML models trained & evaluated on the full pooled dataset
  5. Per-sector and per-stock accuracy breakdowns

Data source: Yahoo Finance (https://finance.yahoo.com)
Retrieved via: yfinance (https://pypi.org/project/yfinance/)
Install: pip install yfinance
"""

# Import core libraries for numerical operations, data handling, and plotting
import numpy as np
import pandas as pd
import matplotlib
# Use a non-interactive backend so plots can be saved without opening a GUI window
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
# Import machine learning models for both classification and regression tasks
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
# Import scaler for normalizing features
from sklearn.preprocessing import MinMaxScaler
# Import evaluation metrics for classification and regression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score)
# Import system utilities for file handling and warnings management
import os, warnings
# Suppress warning messages to keep output cleaner
warnings.filterwarnings('ignore')
# Create an output folder where generated figures will be saved
OUT = 'figures'
os.makedirs(OUT, exist_ok=True)



# 1. DATA COLLECTION — Full S&P 500 via yfinance
# We retrieve historical daily OHLCV data for all current S&P 500
# constituents using the yfinance library, which pulls publicly
# available market data from Yahoo Finance.
# This section downloads historical stock market data for the
# current S&P 500 companies using the yfinance package.
# The data includes:
# - Open price
# - High price
# - Low price
# - Close price
# - Adjusted close price
# - Volume
# The goal is to build a dataset that can later be used for
# analysis, visualization, or machine learning.

import yfinance as yf 
# List of S&P 500 ticker symbols to download
SP500_TICKERS = [
    "AAPL","ABBV","ABT","ACN","ADBE","ADI","ADM","ADP","ADSK","AEE",
    "AEP","AES","AFL","AIG","AIZ","AJG","AKAM","ALB","ALGN","ALK",
    "ALL","ALLE","AMAT","AMCR","AMD","AME","AMGN","AMP","AMT","AMZN",
    "ANET","ANSS","AON","AOS","APA","APD","APH","APTV","ARE","ATO",
    "AVGO","AVY","AWK","AXP","AZO","BA","BAC","BAX","BBY","BDX",
    "BEN","BIO","BIIB","BK","BKNG","BKR","BLK","BMY","BR","BRK-B",
    "BRO","BSX","BWA","BXP","C","CAG","CAH","CARR","CAT","CB",
    "CBOE","CBRE","CCI","CCL","CDNS","CDW","CE","CEG","CF","CFG",
    "CHD","CHRW","CHTR","CI","CINF","CL","CLX","CMA","CMCSA","CME",
    "CMG","CMI","CMS","CNC","CNP","COF","COO","COP","COST","CPAY",
    "CPB","CPRT","CPT","CRL","CRM","CSCO","CSGP","CSX","CTAS","CTLT",
    "CTRA","CTSH","CTVA","CVS","CVX","CZR","D","DAL","DAY","DD",
    "DE","DECK","DFS","DG","DGX","DHI","DHR","DIS","DLTR","DOV",
    "DOW","DPZ","DRI","DTE","DUK","DVA","DVN","DXCM","EA","EBAY",
    "ECL","ED","EFX","EIX","EL","EMN","EMR","ENPH","EOG","EPAM",
    "EQIX","EQR","EQT","ES","ESS","ETN","ETR","EVRG","EW","EXC",
    "EXPD","EXPE","EXR","F","FANG","FAST","FBHS","FCX","FDS","FDX",
    "FE","FFIV","FI","FICO","FIS","FISV","FITB","FLT","FMC","FOX",
    "FOXA","FRT","FSLR","FTNT","FTV","GD","GDDY","GE","GEHC","GEN",
    "GEV","GILD","GIS","GL","GLW","GM","GNRC","GOOG","GOOGL","GPC",
    "GPN","GRMN","GS","GWW","HAL","HAS","HBAN","HCA","HD","HOLX",
    "HON","HPE","HPQ","HRL","HSIC","HST","HSY","HUBB","HUM","HWM",
    "IBM","ICE","IDXX","IEX","IFF","ILMN","INCY","INTC","INTU","INVH",
    "IP","IPG","IQV","IR","IRM","ISRG","IT","ITW","IVZ","J",
    "JBHT","JCI","JKHY","JNJ","JNPR","JPM","K","KDP","KEY","KEYS",
    "KHC","KIM","KLAC","KMB","KMI","KMX","KO","KR","KVUE","L",
    "LDOS","LEN","LH","LHX","LIN","LKQ","LLY","LMT","LNT","LOW",
    "LRCX","LULU","LUV","LVS","LW","LYB","LYV","MA","MAA","MAR",
    "MAS","MCD","MCHP","MCK","MCO","MDLZ","MDT","MET","META","MGM",
    "MHK","MKC","MKTX","MLM","MMC","MMM","MNST","MO","MOH","MOS",
    "MPC","MPWR","MRK","MRNA","MRO","MS","MSCI","MSFT","MSI","MTB",
    "MTCH","MTD","MU","NCLH","NDAQ","NDSN","NEE","NEM","NFLX","NI",
    "NKE","NOC","NOW","NRG","NSC","NTAP","NTRS","NUE","NVDA","NVR",
    "NWL","NWS","NWSA","NXPI","O","ODFL","OGN","OKE","OMC","ON",
    "ORCL","ORLY","OTIS","OXY","PARA","PAYC","PAYX","PCAR","PCG","PEAK",
    "PEG","PEP","PFE","PFG","PG","PGR","PH","PHM","PKG","PLD",
    "PM","PNC","PNR","PNW","POOL","PPG","PPL","PRU","PSA","PSX",
    "PTC","PVH","PWR","PXD","PYPL","QCOM","QRVO","RCL","REG","REGN",
    "RF","RHI","RJF","RL","RMD","ROK","ROL","ROP","ROST","RSG",
    "RTX","RVTY","SBAC","SBUX","SCHW","SEE","SHW","SJM","SLB","SMCI",
    "SNA","SNPS","SO","SOLV","SPG","SPGI","SRE","STE","STLD","STT",
    "STX","STZ","SWK","SWKS","SYF","SYK","SYY","T","TAP","TDG",
    "TDY","TECH","TEL","TER","TFC","TFX","TGT","TJX","TMO","TMUS",
    "TPR","TRGP","TRMB","TROW","TRV","TSCO","TSLA","TSN","TT","TTWO",
    "TXN","TXT","TYL","UAL","UBER","UDR","UHS","ULTA","UNH","UNP",
    "UPS","URI","USB","V","VICI","VLO","VLTO","VMC","VRSK","VRSN",
    "VRTX","VST","VTR","VTRS","VZ","WAB","WAT","WBA","WBD","WDC",
    "WEC","WELL","WFC","WHR","WM","WMB","WMT","WRB","WRK","WST",
    "WTW","WY","WYNN","XEL","XOM","XRAY","XYL","YUM","ZBH","ZBRA",
    "ZION","ZTS"
]

# GICS sector mapping
# Dictionary mapping sectors to the tickers that belong to them
# This helps categorize each stock by its industry sector later
SECTOR_MAP = {
    'Technology': ['AAPL','ADBE','ADI','ADSK','AMAT','AMD','ANET','ANSS','AVGO',
                   'CDNS','CRM','CSCO','CTSH','ENPH','EPAM','FFIV','FICO','FTNT','GEN',
                   'GOOGL','GOOG','HPE','HPQ','IBM','INTC','INTU','IT','KEYS','KLAC',
                   'LRCX','MCHP','MPWR','MSFT','MSI','MU','NOW','NXPI','ON','ORCL',
                   'PAYC','PTC','QCOM','QRVO','SMCI','SNPS','STX','SWKS','TDY','TECH',
                   'TEL','TER','TRMB','TXN','TYL','VRSN','VRTX','WDC','ZBRA','ACN',
                   'CDW','CSGP','FSLR','GDDY','META','NFLX','AKAM'],
    'Healthcare': ['ABBV','ABT','ALGN','AMGN','BAX','BDX','BIO','BIIB','BMY',
                   'BSX','CAH','CI','CNC','CRL','CTLT','CVS','DXCM','DHR','DVA','EW',
                   'GEHC','GILD','HCA','HOLX','HSIC','HUM','IDXX','ILMN','INCY','IQV',
                   'ISRG','JNJ','LH','LLY','MCK','MDT','MOH','MRK','MRNA','MTD','OGN',
                   'PFE','REGN','RMD','RVTY','STE','SYK','TMO','UHS','UNH','VTRS',
                   'WAT','WST','ZBH','ZTS','KVUE','SOLV','DGX'],
    'Financials': ['AIG','AIZ','AJG','ALL','AMP','AON','AXP','BAC','BEN','BK',
                   'BKNG','BLK','BR','BRK-B','BRO','C','CB','CBOE','CFG','CINF','CMA',
                   'CME','COF','CPAY','DFS','FDS','FI','FIS','FISV','FITB','FLT','GL',
                   'GPN','GS','HBAN','ICE','IVZ','JKHY','JPM','KEY','L','LDOS','MA',
                   'MCO','MKTX','MMC','MS','MSCI','MTB','NDAQ','NTRS','PFG','PGR','PNC',
                   'PRU','PYPL','RF','RJF','SCHW','SPGI','STT','SYF','TROW','TRV','USB',
                   'V','WFC','WRB','WTW','TFC'],
    'Cons. Disc.': ['AMZN','APTV','AZO','BBY','BWA','CCL','CMG','CZR','DAY',
                    'DECK','DG','DHI','DIS','DLTR','DPZ','DRI','EBAY','EXPE','F','FBHS',
                    'GM','GNRC','GPC','GRMN','HAS','HD','KMX','LEN','LOW','LULU','LVS',
                    'LYV','MAR','MAS','MCD','MGM','MHK','NCLH','NKE','NVR','ORLY','PARA',
                    'PHM','POOL','PVH','RCL','RL','ROST','SBUX','SWK','TAP','TGT','TJX',
                    'TPR','TSCO','TSLA','TTWO','TXT','UBER','ULTA','WHR','WYNN','YUM'],
    'Industrials': ['AOS','BA','CAT','CHRW','CMI','CPRT','CSX','CTAS','DAL','DE',
                    'DOV','EMR','ETN','EXPD','FAST','FDX','FTV','GD','GE','GEV','GWW',
                    'HWM','IEX','IR','ITW','J','JBHT','JCI','LHX','LMT','MMM','NDSN',
                    'NOC','NSC','ODFL','OTIS','PCAR','PH','PNR','PWR','ROK','ROL','ROP',
                    'RSG','RTX','SNA','TDG','TT','UAL','UNP','UPS','URI','VRSK','WAB',
                    'WM','XYL','CARR','ALLE','AME','RHI','JNPR'],
    'Comm. Svc.': ['CHTR','CMCSA','EA','FOX','FOXA','IPG','MTCH','NWS','NWSA',
                   'OMC','T','TMUS','VZ','WBD'],
    'Energy': ['APA','BKR','COP','CTRA','CVX','DVN','EOG','EQT','FANG','HAL',
               'KMI','MPC','MRO','OKE','OXY','PSX','PXD','SLB','TRGP','VLO','WMB','XOM'],
    'Utilities': ['AEE','AEP','AES','ATO','AWK','CEG','CMS','CNP','D','DTE',
                  'DUK','ED','EIX','ES','ETR','EVRG','EXC','FE','LNT','NEE','NI','NRG',
                  'PCG','PEG','PNW','PPL','SO','SRE','VST','WEC','XEL'],
    'Real Estate': ['AMT','ARE','BXP','CCI','CBRE','CPT','EQIX','EQR','ESS',
                    'FRT','INVH','IRM','KIM','MAA','O','PEAK','PLD','PSA','REG','SBAC',
                    'SPG','UDR','VICI','VTR','WELL','EXR','HST'],
    'Materials': ['ALB','AMCR','APD','AVY','CE','CF','DD','DOW','ECL','EMN',
                  'FCX','FMC','IFF','IP','LIN','LYB','MLM','MOS','NEM','NUE','PKG',
                  'PPG','SEE','SHW','STLD','VMC','WRK','WY'],
    'Cons. Staples': ['ADM','CAG','CHD','CL','CLX','COST','CPB','EL','GIS',
                      'HRL','HSY','K','KDP','KHC','KMB','KO','KR','LW','MDLZ','MKC',
                      'MNST','MO','PEP','PG','PM','SJM','STZ','SYY','TSN','WBA','WMT','MET'],
}

# Build ticker -> sector lookup
# Create a reverse lookup dictionary: given a ticker, find its sector
# This makes it easier to match each company with its sector
_ticker_to_sector = {}
for sector, tickers in SECTOR_MAP.items():
    for t in tickers:
        _ticker_to_sector[t] = sector

# Build a DataFrame containing each ticker and its assigned sector
# If a ticker is missing from the sector map, label it as "Other"
sectors = pd.DataFrame([
    {'Ticker': tk, 'Sector': _ticker_to_sector.get(tk, 'Other')}
    for tk in SP500_TICKERS
])

# --- Download data from Yahoo Finance via yfinance ---
print(f"Downloading data for {len(SP500_TICKERS)} S&P 500 tickers...")
print("Date range: 2020-01-01 to 2026-01-01\n")

# Download historical market data for all tickers
# group_by='ticker' organizes the result so each ticker has its own column group
# auto_adjust=False keeps the original price series unchanged
# threads=True speeds up downloading using parallel requests
raw = yf.download(SP500_TICKERS, start="2020-01-01", end="2026-01-01",
                  group_by='ticker', auto_adjust=False, threads=True)

# Reshape from multi-level columns to long format
frames = []
for tk in SP500_TICKERS:
    try:
        # Extract this ticker's data from the downloaded dataset
        stk = raw[tk].copy()
        stk = stk[['Open','High','Low','Close','Adj Close','Volume']].dropna()
        # Skip tickers with too little data
        # This helps avoid including incomplete or unreliable series
        if len(stk) < 100:
            print(f"  Skipping {tk}: only {len(stk)} rows")
            continue
        stk['Ticker'] = tk
        frames.append(stk)
    except (KeyError, TypeError):
        print(f"  Skipping {tk}: no data returned")
# Combine all ticker DataFrames into one large DataFrame
data = pd.concat(frames)
data.index.name = 'Date'

# Attach sector labels
data = data.reset_index().merge(sectors, on='Ticker', how='left').set_index('Date')

print(f"\nDataset: {len(data):,} rows, {data['Ticker'].nunique()} tickers, "
      f"{data.index.min().date()} – {data.index.max().date()}")
# Print how many tickers belong to each sector
print(f"\nSector counts:\n{sectors['Sector'].value_counts().to_string()}")


# 2. DATA CLEANING
# This section checks for missing values, fills gaps in the data,
# computes daily returns, and creates summary statistics for each stock.
# These cleaned values are later used for analysis and modeling.

# Print the total number of missing values in the dataset
print(f"Missing values: {data.isnull().sum().sum()}")

# Forward-fill missing values using the most recent previous value,
# then remove any remaining missing rows
data = data.ffill().dropna()

# Compute daily percentage return for each stock separately
# pct_change() calculates (today_close - yesterday_close) / yesterday_close
data['Daily_Return'] = data.groupby('Ticker')['Close'].pct_change()


# Aggregate return statistics for each ticker
# For each stock, calculate:
# - mean daily return
# - standard deviation of daily return
ret_stats = (
    data.groupby('Ticker')['Daily_Return']
        .agg(['mean', 'std'])
        .dropna()
        .rename(columns={'mean': 'mu', 'std': 'sigma'})
)

# Convert daily return and volatility into annualized values
# 252 is the approximate number of trading days in a year
ret_stats['Ann_Ret'] = ret_stats['mu'] * 252
ret_stats['Ann_Vol'] = ret_stats['sigma'] * np.sqrt(252)

# Compute Sharpe ratio as return per unit of risk
# This version assumes a risk-free rate of 0
ret_stats['Sharpe'] = ret_stats['Ann_Ret'] / ret_stats['Ann_Vol']

# Add sector labels back to the statistics table
ret_stats = ret_stats.merge(sectors, left_index=True, right_on='Ticker')

# Print overall market-wide daily return behavior
print(
    f"\nIndex-wide daily return: mean={data['Daily_Return'].mean():.5f}, "
    f"std={data['Daily_Return'].std():.5f}"
)

# Print average annualized return, volatility, and Sharpe ratio by sector
print(f"\nSector summary (annualized):")
print(
    ret_stats.groupby('Sector')[['Ann_Ret', 'Ann_Vol', 'Sharpe']]
             .mean()
             .round(3)
             .to_string()
)


# 3. EXPLORATORY DATA ANALYSIS — Index-Wide

# This section creates charts to explore:
# - return distributions
# - sector-level differences
# - risk-return relationships
# - market-wide trends over time

# Use a clean plotting style for all figures
plt.style.use('seaborn-v0_8-whitegrid')


# Fig 1: Aggregate S&P 500 return distribution

# This figure contains:
# 1. pooled daily return distribution across all stocks
# 2. annualized return distribution across individual stocks
# 3. annualized volatility distribution across individual stocks

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Collect all daily returns into one series and remove missing values
all_ret = data['Daily_Return'].dropna()

# Plot histogram of all pooled daily returns
axes[0].hist(
    all_ret,
    bins=120,
    density=True,
    color='steelblue',
    alpha=0.7,
    edgecolor='none'
)

# Fit a normal distribution using the sample mean and standard deviation
mu_all, sig_all = all_ret.mean(), all_ret.std()
x = np.linspace(mu_all - 4 * sig_all, mu_all + 4 * sig_all, 200)

# Plot the fitted normal curve on top of the histogram
axes[0].plot(x, stats.norm.pdf(x, mu_all, sig_all), 'r-', lw=2, label='Normal fit')
axes[0].set_title(
    'S&P 500 Daily Return Distribution\n(All S&P 500 Stocks Pooled)',
    fontsize=12,
    fontweight='bold'
)
axes[0].set_xlabel('Daily Return')
axes[0].set_ylabel('Density')
axes[0].legend()


# Histogram of annualized returns across stocks
axes[1].hist(
    ret_stats['Ann_Ret'],
    bins=40,
    color='seagreen',
    alpha=0.7,
    edgecolor='black',
    lw=0.3
)

# Add a vertical line for the median annualized return
axes[1].axvline(
    ret_stats['Ann_Ret'].median(),
    color='red',
    ls='--',
    lw=1.5,
    label=f"Median: {ret_stats['Ann_Ret'].median():.1%}"
)
axes[1].set_title(
    'Distribution of Annualized Returns\nAcross S&P 500 Stocks',
    fontsize=12,
    fontweight='bold'
)
axes[1].set_xlabel('Annualized Return')
axes[1].legend()


# Histogram of annualized volatility across stocks

axes[2].hist(
    ret_stats['Ann_Vol'],
    bins=40,
    color='coral',
    alpha=0.7,
    edgecolor='black',
    lw=0.3
)

# Add a vertical line for the median annualized volatility
axes[2].axvline(
    ret_stats['Ann_Vol'].median(),
    color='red',
    ls='--',
    lw=1.5,
    label=f"Median: {ret_stats['Ann_Vol'].median():.1%}"
)
axes[2].set_title(
    'Distribution of Annualized Volatility\nAcross S&P 500 Stocks',
    fontsize=12,
    fontweight='bold'
)
axes[2].set_xlabel('Annualized Volatility')
axes[2].legend()

plt.tight_layout()
plt.savefig(f'{OUT}/01_index_return_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 01")


# Fig 2: Sector comparison

# This compares sectors using average:
# - annualized return
# - annualized volatility
# - Sharpe ratio

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Compute average sector-level values
sec_agg = ret_stats.groupby('Sector')[['Ann_Ret', 'Ann_Vol', 'Sharpe']].mean().sort_values('Ann_Ret')

# Bar chart: average annualized return by sector
c_ret = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(sec_agg)))
sec_agg['Ann_Ret'].plot(kind='barh', ax=axes[0], color=c_ret, edgecolor='black', lw=0.4)
axes[0].set_title('Avg Annualized Return by Sector', fontweight='bold')
axes[0].set_xlabel('Return')

# Bar chart: average annualized volatility by sector
sec_agg.sort_values('Ann_Vol')['Ann_Vol'].plot(
    kind='barh',
    ax=axes[1],
    color=plt.cm.OrRd(np.linspace(0.2, 0.8, len(sec_agg))),
    edgecolor='black',
    lw=0.4
)
axes[1].set_title('Avg Annualized Volatility by Sector', fontweight='bold')
axes[1].set_xlabel('Volatility')

# Bar chart: average Sharpe ratio by sector
sec_agg.sort_values('Sharpe')['Sharpe'].plot(
    kind='barh',
    ax=axes[2],
    color=plt.cm.PuBuGn(np.linspace(0.2, 0.85, len(sec_agg))),
    edgecolor='black',
    lw=0.4
)
axes[2].set_title('Avg Sharpe Ratio by Sector', fontweight='bold')
axes[2].set_xlabel('Sharpe Ratio')

plt.tight_layout()
plt.savefig(f'{OUT}/02_sector_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 02")


# Fig 3: Risk-return scatter plot
# Each point represents one stock.
# X-axis = annualized volatility
# Y-axis = annualized return
# Color = sector

fig, ax = plt.subplots(figsize=(12, 8))

unique_sectors = sorted(ret_stats['Sector'].unique())
palette = sns.color_palette('husl', len(unique_sectors))

for sec, color in zip(unique_sectors, palette):
    sub = ret_stats[ret_stats['Sector'] == sec]
    ax.scatter(sub['Ann_Vol'], sub['Ann_Ret'], s=25, alpha=0.65, label=sec, color=color)

ax.set_xlabel('Annualized Volatility', fontsize=12)
ax.set_ylabel('Annualized Return', fontsize=12)
ax.set_title('Risk vs Return — All S&P 500 Stocks (Color = Sector)', fontsize=14, fontweight='bold')

# Draw a horizontal line at 0 return to separate gains from losses
ax.axhline(0, color='black', lw=0.5, ls='--')

ax.legend(fontsize=8, ncol=3, loc='upper left')
plt.tight_layout()
plt.savefig(f'{OUT}/03_risk_return_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 03")


# Fig 4: Market-wide time series
# This figure shows:
# 1. an equal-weight index built from all stocks
# 2. aggregate market trading volume over time

fig, axes = plt.subplots(2, 1, figsize=(14, 9))

# Create a pivot table where each column is a stock and each row is a date
pivot_close = data.pivot_table(index='Date', columns='Ticker', values='Close')

# Normalize each stock's price series so all begin at 1
# This makes them comparable despite different price levels
norm_df = pivot_close.div(pivot_close.iloc[0])

# Equal-weight index = average normalized stock price across all tickers
eq_idx = norm_df.mean(axis=1) * 100

# Plot the equal-weight index
axes[0].plot(eq_idx.index, eq_idx.values, color='steelblue', lw=1.5)

# Shade green when above the starting base level and red when below
eq_vals = eq_idx.values.flatten()
axes[0].fill_between(eq_idx.index, eq_vals, 100, where=eq_vals >= 100, alpha=0.1, color='green')
axes[0].fill_between(eq_idx.index, eq_vals, 100, where=eq_vals < 100, alpha=0.1, color='red')

# Add a horizontal baseline at 100
axes[0].axhline(100, color='gray', ls='--', lw=0.8)

axes[0].set_title('S&P 500 Equal-Weight Index (Normalized, Base=100)', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Index Level')
axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Highlight the COVID crash period
axes[0].axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-04-15'), alpha=0.12, color='red')
axes[0].annotate(
    'COVID',
    xy=(pd.Timestamp('2020-03-15'), eq_idx.min() * 1.02),
    fontsize=9,
    color='red',
    fontstyle='italic'
)


# Aggregate volume chart
# Sum trading volume across all stocks for each day
agg_vol = data.groupby(data.index)['Volume'].sum()

# Plot daily total volume and 20-day rolling average
axes[1].fill_between(agg_vol.index, agg_vol.values, alpha=0.4, color='steelblue')
axes[1].plot(
    agg_vol.rolling(20).mean().index,
    agg_vol.rolling(20).mean().values,
    color='red',
    lw=1.5,
    label='20-Day Avg'
)

axes[1].set_title('Aggregate Daily Volume Across S&P 500', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Total Volume')
axes[1].legend()
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig(f'{OUT}/04_index_timeseries.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 04")



# 4. FEATURE ENGINEERING — All Stocks
# This function creates technical indicators and target variables
# for machine learning.
#
# Feature engineering transforms raw price/volume data into
# signals that may help predict next-day movement or price.

def engineer_features(df):
    # Work on a copy so the original DataFrame is not modified directly
    df = df.copy()

    # Daily percent return from one day to the next
    df['Daily_Return'] = df['Close'].pct_change()

    # Log return, often used in finance because it is additive over time
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))


    # Simple moving averages

    # These smooth price data and help identify trend direction
    for w in [5, 10, 20, 50]:
        df[f'SMA_{w}'] = df['Close'].rolling(w).mean()

    # Indicator for whether the short-term average is above the long-term average
    # Often interpreted as a bullish trend signal
    df['SMA_20_50_Cross'] = (df['SMA_20'] > df['SMA_50']).astype(int)


    # Exponential moving averages and MACD

    # EMA gives more weight to recent prices than SMA
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()

    # MACD measures momentum by comparing short-term and long-term EMA
    df['MACD'] = df['EMA_12'] - df['EMA_26']

    # Signal line is a smoothed version of MACD, often used for trade signals
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()

    
    # Rolling volatility
    # Standard deviation of returns over recent windows
    # Higher values suggest more unstable price movement
    df['Volatility_10'] = df['Daily_Return'].rolling(10).std()
    df['Volatility_20'] = df['Daily_Return'].rolling(20).std()


    # Relative Strength Index (RSI)
    # RSI is a momentum oscillator ranging from 0 to 100.
    # Higher RSI can indicate overbought conditions,
    # while lower RSI can indicate oversold conditions.
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - 100 / (1 + gain / (loss + 1e-10))

    
    # Bollinger Band width
    # Uses the 20-day moving average and standard deviation to estimate
    # how wide the price bands are, which reflects recent volatility
    bb_m = df['Close'].rolling(20).mean()
    bb_s = df['Close'].rolling(20).std()
    df['BB_Width'] = (4 * bb_s) / (bb_m + 1e-10)

    
    # Volume-based features
    # These measure how trading activity is changing over time
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Volume_SMA_20'] = df['Volume'].rolling(20).mean()

    # Ratio above 1 means today's volume is above its recent average
    df['Volume_Ratio'] = df['Volume'] / (df['Volume_SMA_20'] + 1e-10)


    # Lagged returns
    # These provide recent past return information to the model
    for lag in [1, 2, 3, 5]:
        df[f'Return_Lag_{lag}'] = df['Daily_Return'].shift(lag)


    # Momentum features
    # Percentage price change over different lookback periods
    for w in [5, 10, 20]:
        df[f'Momentum_{w}'] = df['Close'] / df['Close'].shift(w) - 1

    # Target variables
    # Classification target:
    # 1 if tomorrow's close is higher than today's close, else 0
    df['Target_Direction'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # Regression target:
    # actual next day's closing price
    df['Target_Price'] = df['Close'].shift(-1)

    return df

# Display progress so the user knows features are being created
print(f"Engineering features for all {data['Ticker'].nunique()} stocks...")

feat_frames = []

# Group data by ticker so features are engineered stock by stock
# Sector is removed temporarily because the function only needs numeric market data
grouped = data.drop(columns=['Sector']).groupby('Ticker')

actual_tickers = data["Ticker"].unique()

for i, (tk, stk) in enumerate(grouped):
    feat_frames.append(engineer_features(stk))

    # Print progress every 100 stocks
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(SP500_TICKERS)}")

# Combine engineered features for all stocks
data_feat = pd.concat(feat_frames)

# Re-attach sector information after feature creation
data_feat = data_feat.reset_index().merge(sectors, on='Ticker').set_index('Date')

print(f"Featured dataset: {len(data_feat):,} rows, {data_feat.shape[1]} columns")

# Fig 5: Correlation heatmap
# This examines the relationships between selected engineered features

feat_cols = [
    'Daily_Return', 'Volatility_20', 'RSI', 'MACD', 'Volume_Ratio',
    'Momentum_5', 'Momentum_20', 'SMA_20_50_Cross', 'BB_Width',
    'Return_Lag_1', 'Return_Lag_2', 'Return_Lag_3', 'Target_Direction'
]

# Sample 100,000 rows to speed up computation while still keeping a large sample
corr_sample = data_feat[feat_cols].dropna().sample(100000, random_state=42)
corr_mat = corr_sample.corr()

fig, ax = plt.subplots(figsize=(11, 9))

# Mask the upper triangle so the heatmap is easier to read
mask = np.triu(np.ones_like(corr_mat, dtype=bool))

sns.heatmap(
    corr_mat,
    mask=mask,
    annot=True,
    fmt='.2f',
    cmap='RdBu_r',
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={'shrink': 0.8},
    ax=ax
)

ax.set_title('Feature Correlation (Pooled Across All S&P 500 Stocks)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/05_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 05")


# 5. MACHINE LEARNING — Full Index
# This section trains models on all stocks pooled together.
#
# Two prediction tasks are used:
# 1. Classification: predict whether tomorrow's price goes up or down
# 2. Regression: predict tomorrow's actual closing price

# Feature list for classification models
CLS_FEAT = [
    'Daily_Return', 'Volatility_10', 'Volatility_20', 'RSI', 'MACD',
    'MACD_Signal', 'Volume_Ratio', 'Volume_Change',
    'Momentum_5', 'Momentum_10', 'Momentum_20',
    'SMA_20_50_Cross', 'BB_Width',
    'Return_Lag_1', 'Return_Lag_2', 'Return_Lag_3', 'Return_Lag_5'
]

# Regression uses all classification features plus price-level moving average features
REG_FEAT = CLS_FEAT + ['Close', 'SMA_5', 'SMA_10', 'SMA_20']

# Keep only rows with all required features and targets present
df_all = data_feat[REG_FEAT + ['Target_Direction', 'Target_Price', 'Ticker', 'Sector']].dropna()

print(f"Pooled model-ready rows: {len(df_all):,}")
print(
    f"Target split: Up {df_all['Target_Direction'].mean():.1%} / "
    f"Down {1 - df_all['Target_Direction'].mean():.1%}"
)

# Chronological train/test split
# This prevents future information from leaking into training
SPLIT = pd.Timestamp('2024-06-01')
train_all = df_all[df_all.index <= SPLIT]
test_all = df_all[df_all.index > SPLIT]

print(f"Train: {len(train_all):,} (≤{SPLIT.date()}) | Test: {len(test_all):,} (>{SPLIT.date()})")

# Subsample training set for speed if it is too large
MAX_TRAIN = 80000
if len(train_all) > MAX_TRAIN:
    train_sub = train_all.sample(MAX_TRAIN, random_state=42)
    print(f"Subsampled train to {MAX_TRAIN:,}")
else:
    train_sub = train_all

# Prepare classification input and target
X_tr = train_sub[CLS_FEAT]
y_tr = train_sub['Target_Direction']
X_te = test_all[CLS_FEAT]
y_te = test_all['Target_Direction']

# Scale features to the 0–1 range
# This is especially helpful for models like logistic regression and KNN
scaler = MinMaxScaler()
X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=CLS_FEAT, index=X_tr.index)
X_te_s = pd.DataFrame(scaler.transform(X_te), columns=CLS_FEAT, index=X_te.index)


# Classification models

print("\n  CLASSIFICATION (Next-Day Direction, All S&P 500)")

classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=20,
        random_state=42,
        n_jobs=-1
    ),
}

cls_results = {}

for name, mdl in classifiers.items():
    # Fit the model on training data
    mdl.fit(X_tr_s, y_tr)

    # Predict on test data
    yp = mdl.predict(X_te_s)

    # Store performance metrics
    cls_results[name] = {
        'Accuracy': accuracy_score(y_te, yp),
        'Precision': precision_score(y_te, yp, zero_division=0),
        'Recall': recall_score(y_te, yp, zero_division=0),
        'F1 Score': f1_score(y_te, yp, zero_division=0),
        'preds': yp
    }

    r = cls_results[name]
    print(
        f"    {name:25s} | Acc {r['Accuracy']:.3f} | Prec {r['Precision']:.3f} | "
        f"Rec {r['Recall']:.3f} | F1 {r['F1 Score']:.3f}"
    )

# KNN classification on a smaller training subset

# KNN is more computationally expensive, so a smaller subset is used
print("    Training KNN (50k subsample)...")

knn_idx = np.random.choice(len(X_tr_s), min(50000, len(X_tr_s)), replace=False)
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_tr_s.iloc[knn_idx], y_tr.iloc[knn_idx])

yp_knn = knn.predict(X_te_s)

cls_results['KNN (k=10)'] = {
    'Accuracy': accuracy_score(y_te, yp_knn),
    'Precision': precision_score(y_te, yp_knn, zero_division=0),
    'Recall': recall_score(y_te, yp_knn, zero_division=0),
    'F1 Score': f1_score(y_te, yp_knn, zero_division=0),
    'preds': yp_knn
}

r = cls_results['KNN (k=10)']
print(
    f"    {'KNN (k=10)':25s} | Acc {r['Accuracy']:.3f} | Prec {r['Precision']:.3f} | "
    f"Rec {r['Recall']:.3f} | F1 {r['F1 Score']:.3f}"
)

# Choose the best classifier based on highest accuracy
best_cls = max(cls_results, key=lambda x: cls_results[x]['Accuracy'])

# Regression models
print("\n  REGRESSION (Next-Day Price, All S&P 500)")

X_tr_reg = train_sub[REG_FEAT]
y_tr_reg = train_sub['Target_Price']
X_te_reg = test_all[REG_FEAT]
y_te_reg = test_all['Target_Price']

# Scale regression features
sc_reg = MinMaxScaler()
X_tr_reg_s = pd.DataFrame(sc_reg.fit_transform(X_tr_reg), columns=REG_FEAT, index=X_tr_reg.index)
X_te_reg_s = pd.DataFrame(sc_reg.transform(X_te_reg), columns=REG_FEAT, index=X_te_reg.index)

regressors = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regressor': RandomForestRegressor(
        n_estimators=80,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    ),
}

reg_results = {}

for name, mdl in regressors.items():
    mdl.fit(X_tr_reg_s, y_tr_reg)
    yp = mdl.predict(X_te_reg_s)

    reg_results[name] = {
        'MAE': mean_absolute_error(y_te_reg, yp),
        'RMSE': np.sqrt(mean_squared_error(y_te_reg, yp)),
        'R2': r2_score(y_te_reg, yp),
        'preds': yp
    }

    r = reg_results[name]
    print(f"    {name:25s} | MAE ${r['MAE']:.2f} | RMSE ${r['RMSE']:.2f} | R² {r['R2']:.4f}")

# Choose best regression model by highest R²
best_reg = max(reg_results, key=lambda x: reg_results[x]['R2'])


# Per-sector accuracy breakdown
# Evaluate how the Random Forest classifier performs in each sector
print("\n  PER-SECTOR ACCURACY (Random Forest)")

rf_preds = cls_results['Random Forest']['preds']
test_all_eval = test_all.copy()
test_all_eval['Pred'] = rf_preds

sec_acc = test_all_eval.groupby('Sector').apply(
    lambda g: accuracy_score(g['Target_Direction'], g['Pred']),
    include_groups=False
)
sec_acc = sec_acc.sort_values()

print(sec_acc.to_string())

# Per-stock accuracy breakdown
# Compute prediction accuracy for each ticker individually
# Stocks with very few test rows are ignored
stock_acc = test_all_eval.groupby('Ticker').apply(
    lambda g: accuracy_score(g['Target_Direction'], g['Pred']) if len(g) > 10 else np.nan,
    include_groups=False
).dropna()

print(f"\n  Per-stock accuracy stats:")
print(f"    Mean:   {stock_acc.mean():.3f}")
print(f"    Median: {stock_acc.median():.3f}")
print(f"    Std:    {stock_acc.std():.3f}")
print(f"    Min:    {stock_acc.min():.3f} ({stock_acc.idxmin()})")
print(f"    Max:    {stock_acc.max():.3f} ({stock_acc.idxmax()})")




# 6. RESULTS VISUALIZATION
# This section visualizes model performance using bar charts,
# confusion matrices, feature importance, and predicted-vs-actual plots.


# Fig 6: Classification model comparison

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Convert classification results into a DataFrame, excluding raw predictions
mdf = pd.DataFrame({
    n: {k: v for k, v in r.items() if k != 'preds'}
    for n, r in cls_results.items()
}).T

# Plot model performance metrics side by side
mdf.plot(kind='bar', ax=axes[0], colormap='Set2', edgecolor='black', lw=0.5)

axes[0].set_title('Classification Model Comparison\n(All S&P 500 Pooled)', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Score')
axes[0].set_ylim(0.35, 0.72)

# Add a 50% baseline to represent random guessing in a binary task
axes[0].axhline(0.5, color='red', ls='--', alpha=0.7, label='Random Baseline')
axes[0].legend(fontsize=7)
axes[0].tick_params(axis='x', rotation=20)

# Plot confusion matrix for the best classifier
cm = confusion_matrix(y_te, cls_results[best_cls]['preds'])
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=axes[1],
    xticklabels=['Down', 'Up'],
    yticklabels=['Down', 'Up']
)

axes[1].set_title(f'Confusion Matrix — {best_cls}\n(All S&P 500)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig(f'{OUT}/06_classification_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 06")


# Fig 7: Per-sector accuracy

fig, ax = plt.subplots(figsize=(10, 6))

colors_sec = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(sec_acc)))
sec_acc.plot(kind='barh', ax=ax, color=colors_sec, edgecolor='black', lw=0.4)

# Add random baseline for comparison
ax.axvline(0.5, color='red', ls='--', lw=1.5, label='Random Baseline')

ax.set_title('Random Forest Accuracy by Sector', fontsize=14, fontweight='bold')
ax.set_xlabel('Accuracy')

# Label each bar with its accuracy value
for i, (sec, val) in enumerate(sec_acc.items()):
    ax.text(val + 0.003, i, f'{val:.1%}', va='center', fontsize=9)

ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT}/07_sector_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 07")


# Fig 8: Per-stock accuracy distribution

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of stock-level prediction accuracy
axes[0].hist(stock_acc, bins=30, color='steelblue', edgecolor='black', lw=0.3, alpha=0.7)
axes[0].axvline(0.5, color='red', ls='--', lw=1.5, label='Random (50%)')
axes[0].axvline(stock_acc.mean(), color='green', ls='--', lw=1.5, label=f'Mean ({stock_acc.mean():.1%})')

axes[0].set_title('Distribution of Per-Stock Accuracy\n(Random Forest, All S&P 500)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Accuracy')
axes[0].set_ylabel('Number of Stocks')
axes[0].legend()

# Top 10 and bottom 10 stocks by prediction accuracy
top10 = stock_acc.nlargest(10)
bot10 = stock_acc.nsmallest(10)
combined = pd.concat([bot10, top10])

# Red for bottom 10, green for top 10
colors_tb = ['#EF4444'] * 10 + ['#10B981'] * 10
combined.plot(kind='barh', ax=axes[1], color=colors_tb, edgecolor='black', lw=0.3)

axes[1].axvline(0.5, color='gray', ls='--')
axes[1].set_title('Top 10 & Bottom 10 Stocks by Accuracy', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Accuracy')

plt.tight_layout()
plt.savefig(f'{OUT}/08_perstock_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 08")


# Fig 9: Feature importance from Random Forest
# This shows which input features contributed the most to the model's decisions
rf_model = classifiers['Random Forest']
feat_imp = pd.Series(rf_model.feature_importances_, index=CLS_FEAT).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
feat_imp.plot(kind='barh', ax=ax, color='steelblue', edgecolor='black', lw=0.4)

ax.set_title('Feature Importance — Random Forest\n(Trained on All S&P 500)', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance')

plt.tight_layout()
plt.savefig(f'{OUT}/09_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 09")


# Fig 10: Regression actual vs predicted for selected stocks

# This compares true next-day prices to predicted values for sample tickers

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

sample_tickers = ['AAPL', 'MSFT', 'JPM', 'XOM', 'UNH', 'TSLA']
best_reg_model = regressors[best_reg]

for ax, tk in zip(axes.flat, sample_tickers):
    stk_test = test_all[test_all['Ticker'] == tk]

    # Skip stocks with too little test data
    if len(stk_test) < 10:
        continue

    # Scale this stock's test features using the regression scaler
    X_stk = pd.DataFrame(
        sc_reg.transform(stk_test[REG_FEAT]),
        columns=REG_FEAT,
        index=stk_test.index
    )

    # Predict next-day prices
    yp_stk = best_reg_model.predict(X_stk)

    # Plot actual vs predicted values over time
    ax.plot(stk_test.index, stk_test['Target_Price'], label='Actual', lw=1)
    ax.plot(stk_test.index, yp_stk, label='Predicted', lw=1, alpha=0.8)

    # Compute ticker-specific R² score
    r2_stk = r2_score(stk_test['Target_Price'], yp_stk)
    ax.set_title(f'{tk} (R²={r2_stk:.3f})', fontweight='bold')
    ax.legend(fontsize=8)
    ax.tick_params(axis='x', rotation=30)

plt.suptitle(
    f'Actual vs Predicted Price — {best_reg} (Trained on Full S&P 500)',
    fontsize=14,
    fontweight='bold',
    y=1.01
)

plt.tight_layout()
plt.savefig(f'{OUT}/10_regression_sample_stocks.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 10")




# 7. SUMMARY
# This final section prints a clean summary of the full project results
# and saves important outputs to CSV files for later review.
#
# It reports:
# - dataset size
# - number of features used
# - classification performance
# - regression performance
# - sector and stock-level accuracy
# - most important model features
# - key takeaways from the analysis

print("FINAL RESULTS SUMMARY")

# Basic dataset summary
# Count the number of unique stocks in the dataset
n_tickers = data['Ticker'].nunique()

# Estimate the average number of trading days per stock
# using total row count divided by number of tickers
n_days_avg = len(data) // n_tickers

# Print overall dataset dimensions
print(f"\nDataset: {n_tickers} S&P 500 tickers × ~{n_days_avg} trading days = {len(data):,} rows")

# Print the number of features used for each machine learning task
print(f"Features: {len(CLS_FEAT)} classification / {len(REG_FEAT)} regression")

# Classification results summary
# Show the accuracy and F1 score for each classification model
print(f"\n--- Classification (All S&P 500 Pooled) ---")

for n, r in cls_results.items():
    print(f"  {n:25s} | Acc {r['Accuracy']:.3f} | F1 {r['F1 Score']:.3f}")

# Print a random baseline for comparison
# Since this is a binary up/down prediction problem,
# random guessing would achieve about 50% accuracy
print(f"  {'Random Baseline':25s} | Acc 0.500")

# Print the best-performing classifier
print(f"  Best: {best_cls} ({cls_results[best_cls]['Accuracy']:.1%})")

# Regression results summary
# Show the prediction error and fit quality for each regression model
print(f"\n--- Regression (All S&P 500 Pooled) ---")

for n, r in reg_results.items():
    print(
        f"  {n:25s} | MAE ${r['MAE']:.2f} | "
        f"RMSE ${r['RMSE']:.2f} | R² {r['R2']:.3f}"
    )

# Sector-level Random Forest accuracy
# Show how prediction accuracy varies across sectors
print(f"\n--- Per-Sector Accuracy (RF) ---")

for sec, acc in sec_acc.items():
    print(f"  {sec:25s}: {acc:.1%}")

# Stock-level Random Forest accuracy distribution
# Summarize how model accuracy varies across individual stocks
print(f"\n--- Per-Stock Accuracy Distribution (RF) ---")

print(
    f"  Mean {stock_acc.mean():.3f} | Median {stock_acc.median():.3f} | "
    f"Std {stock_acc.std():.3f} | Range [{stock_acc.min():.3f}, {stock_acc.max():.3f}]"
)

# Top 5 most important features
# feat_imp is sorted in ascending order, so tail(5) gives the 5 most important features
print(f"\n--- Top 5 Features ---")

for f, v in feat_imp.tail(5).items():
    print(f"  {f:25s}: {v:.4f}")

# Key findings
# Print the main takeaways from the analysis in sentence form
print(f"\n--- Key Findings ---")

print(f"  1. Analyzed {len(SP500_TICKERS)} stocks with {len(data):,} total observations.")

print(
    f"  2. Index-wide classification accuracy: {cls_results[best_cls]['Accuracy']:.1%} ({best_cls}),"
)
print(f"     modestly above the 50% random baseline.")

print(
    f"  3. Accuracy varies across sectors ({sec_acc.min():.1%}–{sec_acc.max():.1%}),"
)
print(f"     suggesting some sectors have slightly more predictable short-term behavior.")

print(
    f"  4. Per-stock accuracy ranges from {stock_acc.min():.1%} to {stock_acc.max():.1%},"
)
print(f"     with most stocks clustered near 50% — consistent with market efficiency.")

print(
    f"  5. {best_reg} achieves R²={reg_results[best_reg]['R2']:.3f} for next-day price,"
)
print(f"     largely due to price autocorrelation (today's price ≈ tomorrow's).")

print(f"  6. The strongest features are lagged returns, momentum, and volume ratio.")

# Save result tables as CSV files
# Save classification results without raw prediction arrays
pd.DataFrame({
    n: {k: v for k, v in r.items() if k != 'preds'}
    for n, r in cls_results.items()
}).T.to_csv(f'{OUT}/classification_results.csv')

# Save regression results without raw prediction arrays
pd.DataFrame({
    n: {k: v for k, v in r.items() if k != 'preds'}
    for n, r in reg_results.items()
}).T.to_csv(f'{OUT}/regression_results.csv')

# Save sector-level accuracy results
sec_acc.to_csv(f'{OUT}/sector_accuracy.csv')

# Save stock-level accuracy results
stock_acc.to_csv(f'{OUT}/per_stock_accuracy.csv')

# Save stock return statistics
ret_stats.to_csv(f'{OUT}/sp500_return_stats.csv', index=False)

# Confirm that all figures and tables were saved successfully
print(f"\nAll 10 figures + 5 CSVs saved to {OUT}/")