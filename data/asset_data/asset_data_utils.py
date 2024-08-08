import requests
import pandas as pd
import time
import os
from api_keys import eodhd_api_key

import yfinance as yf 

def get_eodhd_prices_for_ticker(ticker, exchange_symbol, api_key, start_date=None, end_date=None):
    # build url
    url = f"https://eodhd.com/api/eod/{ticker}.{exchange_symbol}?"
    if start_date:
        url += f"from={start_date}&"
    if end_date:
        url += f"to={end_date}&"
    url += f"period=d&api_token={api_key}&fmt=json"
    # get data
    data = requests.get(url).json()
    return data

def get_US_trading_dates(start_date, end_date):
    # AAPL is confirmed to have data for all trading days (and none for non-trading days)
    data = get_eodhd_prices_for_ticker("AAPL", "US", eodhd_api_key, start_date, end_date)
    # return as single-column df
    return pd.DataFrame(data)[["date"]]


def full_download(tickers, exchange_symbol, start_date, end_date, save_path, progress_path, min_time_between_requests=0.1, return_exceptions_list=False, data_source_list=None):
    # check if .csv already exists
    if os.path.exists(save_path):
        raise FileExistsError(f"File {save_path} already exists. Delete or move it.")

    print(f"Starting download of prices for {len(tickers)} tickers from {start_date} to {end_date}.")
    prices_df = pd.DataFrame(columns=["date"])
    exceptions_list = []
    for i, ticker in enumerate(tickers):
        start_time = time.time()
        # get data
        try: 
            # eodhd (default source)
            if data_source_list is None or data_source_list[i] == "eodhd":
                new_json = get_eodhd_prices_for_ticker(ticker=ticker if not exchange_symbol == "CC" else f"{ticker}-USD", # cryptos need suffix
                                                        exchange_symbol=exchange_symbol, 
                                                        start_date=start_date, 
                                                        end_date=end_date, 
                                                        api_key=eodhd_api_key)
                # convert to df, keep only date and adjusted_close columns, rename column to ticker
                new_df = pd.DataFrame(new_json)[["date", "adjusted_close"]].rename(columns={"adjusted_close": ticker})
            elif data_source_list[i] == "yahoo":
                # careful: yahoo end_date is exclusive while eodhd is inclusive!
                new_df = yf.download(ticker, start=start_date, end=end_date, progress=False).reset_index()
                # adjust column names and date format to match eodhd
                new_df = new_df.reset_index()[["Date", "Adj Close"]].rename(columns={'Date': 'date', 'Adj Close': ticker})
                new_df['date'] = new_df['date'].dt.strftime('%Y-%m-%d')

            # outer merge with existing df on date column
            prices_df = pd.merge(prices_df, new_df, on="date", how="outer")
        except Exception as e: # exceptions should only be due to non-available data in the requested time frame
            #print(f"Error for ticker {ticker}: {e} -> skipping.")
            exceptions_list.append((ticker, e))
        # avoid minute rate limit (max 1k requests / minute -> max 1 per 0.06 seconds)
        time_since_last_request = time.time() - start_time
        if time_since_last_request < min_time_between_requests: # 0.1 to be sure
            time.sleep(min_time_between_requests - time_since_last_request)

        # print progress every 100 tickers
        if (i+1) % 100 == 0:
            print(f"-- Progress: {i+1}/{len(tickers)} completed. (total exceptions so far: {len(exceptions_list)})")

        # save intermediate results every 1k tickers (overwrite previous progress)
        if (i+1) % 1000 == 0:
            # read in progress if exists (might be from previous aborted runs)
            if os.path.exists(progress_path):
                old_progress = pd.read_csv(progress_path, sep=";")

            prices_df.to_csv(progress_path, sep=";", index=False)
            print(f"Saved intermediate progress after {i+1} tickers.")

    # save final results (and delete progress save(s))
    prices_df.sort_values(by="date", ascending=True).to_csv(save_path, sep=";", index=False)
    print(f"{'-'*30}\nCompleted all downloads and saved final results to {save_path}.")
    if os.path.exists(progress_path):
        os.remove(progress_path)
        print("Deleted intermediate progress file.")
    
    # show/return exceptions
    #print(f"{'-'*30}\nList of exceptions which occured: {exceptions_list}")
    if return_exceptions_list:
        return exceptions_list
    
def get_returns_from_prices(prices_df, drop_full_na_cols=True):
    # prices_df needs an index or a column "date" and at least one column with prices
    # dfs passed to this function should be pre-filtered to only include price series with not too many missing valuess

    ### prep
    # set "date" column as index (if not already set)
    if "date" not in prices_df.columns and prices_df.index.name != "date":
        raise ValueError("get_returns_from_prices(): prices_df needs a column 'date' or an index 'date'.")
    elif "date" in prices_df.columns and prices_df.index.name != "date":
        prices_df = prices_df.set_index("date")

    # ensure df is sorted by date ascending
    prices_df = prices_df.sort_index(ascending=True)

    # fill missing values with preceding value, but ONLY IN BETWEEN FIRST AND LAST AVAILABLE PRICE!
    # (we would like to use df.ffill(limit_area="inside") but it is not available in our pandas version (2.1.4))
    # instead, we use a workaround with a mask df
    mask = prices_df.bfill().isna()
    prices_df = prices_df.ffill().mask(mask)

    # compute returns (using pct_change())
    returns_df = prices_df.pct_change(fill_method=None) # fill_method=None to avoid forward filling NaN values (which would lead to wrong returns

    # drop any columns with all NaN values (e.g. due to only one price value)
    if drop_full_na_cols:
        returns_df = returns_df.dropna(axis=1, how="all")
        dropped_cols = prices_df.columns.difference(returns_df.columns)
        if len(dropped_cols) > 0:
            print(f"get_returns_from_prices(): Dropped {len(dropped_cols)} return columns with all NaN values: {dropped_cols}")

    # check for potential issues (which should have been avoided by pre-processing the input already)
    # inf values (due to zero prices)
    if returns_df.isin([float("inf"), float("-inf")]).sum().sum() > 0:
        inf_colnames = returns_df.columns[returns_df.isin([float("inf"), float("-inf")]).any()].tolist()
        print(f"get_returns_from_prices(): WARNING: Found infinite values in returns_df in the following columns: {inf_colnames}")
    # NaN values inbetween valid values (should not be possible when forward filling prices without limit)
    for t in returns_df.columns:
        # get earliest and last price
        first_idx = returns_df[t].first_valid_index()
        last_idx = returns_df[t].last_valid_index()
        # check if there are NaN values inbetween
        if returns_df.loc[first_idx:last_idx, t].isna().sum() > 0:
            print(f"get_returns_from_prices(): WARNING: Found NaN values inbetween valid prices for ticker {t}.")

    return returns_df

def detect_errors_in_returns(returns_df, asset_type_name, drop_cols_with_likely_errors=False, print_info=False):
    if "date" in returns_df.columns:
        raise ValueError("detect_errors_in_returns(): returns_df should not have a 'date' column, set as index instead!")
    
    if asset_type_name not in ["cryptos", "stocks", "commodities", "etfs"]:
        raise ValueError("detect_errors_in_returns(): asset_type_name must be one of ['cryptos', 'stocks', 'commodities', 'etfs']")
    
    # define anomalous return thresholds for the different asset classes
    pos_return_thresholds = {"cryptos": 8, # most difficult to choose for cryptos as some can actually have extreme returns, but almost all ~500%+ cases are either extremely low volume or faulty data
                             "stocks": 5, # only a few penny stocks might have extreme returns
                             "commodities": 5, # checked, should all be in normal ranges and error-free
                             "etfs": 4 # some leveraged ETFs can have extreme returns
                             }
    # negative return thresholds are not really appropriate. 
    # in most cases of data errors, huge negative returns tend to be preceded by huge positive returns (and thus already filtered out)
    # especially for small-cap cryptos, extreme one-day drops are known to happen occasionally and should not be discarded
    neg_return_thresholds = {"cryptos": -1, 
                             "stocks": -1, 
                             "commodities": -1, # checked, should all be in normal ranges and error-free
                             "etfs": -1 # there are some leveraged ETFs which can go to zero
                             }

    # set thresholds
    pos_return_threshold = pos_return_thresholds[asset_type_name]
    neg_return_threshold = neg_return_thresholds[asset_type_name]

    # iterate over columns and check for anomalies
    likely_error_cols = []
    for t in returns_df.columns:
        r = returns_df[t]

        # error condition(s)

        # extreme return thresholds
        if r.max() > pos_return_threshold or r.min() < neg_return_threshold:
            likely_error_cols.append(t)
            if print_info:
                print(f"detect_errors_in_returns(): Ticker {t} has likely errors  (max return: {r.max()}, min return: {r.min()}")

        # z-scores?

    if drop_cols_with_likely_errors:
        # drop columns with likely errors (e.g. due to faulty data)
        returns_df = returns_df.drop(columns=likely_error_cols)
        return returns_df, likely_error_cols 
    else:
        return likely_error_cols



# while checking price availability for the top 50 tickers for each asset class we found some wrong data entries which need manual fixes
manual_price_fixes_dict = {
    "cryptos":      [{"ticker": "SAFEMOON", "date": "2018-08-07", "new_val": pd.NA}, # faulty price entry (years before token existed)
                     {"ticker": "AXS", "date": "2022-02-10", "new_val": pd.NA}, # single near-zero price error
                     {"ticker": "LUNA", "date": "2022-05-11", "new_val": pd.NA}, # single extreme value error
                     {"ticker": "GALA", "date": "2018-01-31", "new_val": pd.NA}, # faulty price entry 
                     {"ticker": "ICP", "date": "2021-05-06", "new_val": pd.NA}, # faulty price entry
                    ],
    "stocks":       [],
    "commodities":  [],
    "etfs":         [],
}

def get_daily_3m_tbill_returns(start_date, end_date):
    # download ^IRX (3 month us treasury bill rates) from yahoo finance and convert to EOD-to-EOD returns for US trading days

    # download annualized returns
    irx = yf.download("^IRX", start=start_date, end=end_date)
    irx = irx.reset_index().rename(columns={"Date": "date", "Adj Close": "annualized_pct_return"})[["date", "annualized_pct_return"]]
    irx["annualized_return"] = irx["annualized_pct_return"] / 100 # 5% -> 0.05
    irx["date"] = irx["date"].dt.strftime('%Y-%m-%d')
    # extend to full daterange (incl. weekends, holidays) and fill up missing days with previous day's value 
    df = pd.DataFrame(pd.date_range(start=start_date, end=end_date).strftime('%Y-%m-%d'), columns=["date"])
    df = pd.merge(df, irx, on="date", how="left")
    df["annualized_return"] = df["annualized_return"].ffill()
    # get daily returns from annualized values
    df["daily_return"] = df["annualized_return"].apply(lambda x: (1+x)**(1/365)-1)
    # compute total return index (= theoretical price of the risk-free asset)
    df["total_return_index"] = (1+df["daily_return"]).cumprod()
    # now only keep trading day rows
    df = pd.merge(get_US_trading_dates(start_date, end_date), df, on="date", how="left")
    # use the index to compute trading day returns (accounting for weekends, holidays etc.)
    df["3m_tbills"] = df["total_return_index"].pct_change()
    df = df[["date", "3m_tbills"]]
    return df

