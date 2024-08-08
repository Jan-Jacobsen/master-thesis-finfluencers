"""Implements all necessary classes and functions for the channel portfolio computation and performance analysis."""

import pandas as pd
from scipy.stats import ttest_1samp
import numpy as np
import json
import time

### performance metrics (standalone functions, so we can also compute them for benchmarks etc.)
def beta(pf_returns, bm_returns):
    assert pf_returns.index.equals(bm_returns.index), "Portfolio and benchmark returns indices do not match."
    cov_matrix = np.cov(pf_returns, bm_returns)
    return cov_matrix[0, 1] / cov_matrix[1, 1]

def total_return(pf_returns):
    return (1 + pf_returns).prod() - 1

def total_excess_return(pf_returns, bm_returns):
    assert pf_returns.index.equals(bm_returns.index), "Portfolio and benchmark returns indices do not match."
    return total_return(pf_returns) - total_return(bm_returns)

def sharpe_ratio(pf_returns, bm_returns):
    assert pf_returns.index.equals(bm_returns.index), "Portfolio and benchmark returns indices do not match."

    excess_returns = pf_returns - bm_returns
    sharpe_ratio = excess_returns.mean() / excess_returns.std()
    return sharpe_ratio

def sortino_ratio(pf_returns, bm_returns, daily_target_return=0):
        assert pf_returns.index.equals(bm_returns.index), "Portfolio and benchmark returns indices do not match."
        if isinstance(daily_target_return, pd.Series):
            assert pf_returns.index.equals(daily_target_return.index), "Portfolio and daily_target_return indices do not match."

        excess_returns = pf_returns - bm_returns
        # downside returns: set all returns above daily_target_return to zero
        downside_returns = pf_returns - daily_target_return # daily_target_return can be a scalar or a series with matching index
        downside_returns[downside_returns > 0] = 0
        sortino_ratio = excess_returns.mean() / downside_returns.std()
        return sortino_ratio

def value_at_risk(pf_returns, alpha):
    return -np.quantile(pf_returns, alpha) # var is usually defined as a positive number and we expect negative returns -> invert sign

def max_drawdown(pf_values):
    cum_max = pf_values.cummax()
    drawdown = (cum_max - pf_values) / cum_max
    max_drawdown = drawdown.max()
    return max_drawdown


class Portfolio:
    """
    The Portfolio class holds all relevant info about a portfolio and provides performance metrics, etc.
    """
    def __init__(self, 
                 pos_df_bt, # positions eod BEFORE trading
                 pos_df_at, # positions eod AFTER trading
                 trade_logs_df,
                 compute_settings, channel_id=None):
        self.channel_id = channel_id
        if pos_df_bt.empty or pos_df_bt.empty:
            raise ValueError("One of the positions dataframes is empty.")
        
        self.trade_logs_df = trade_logs_df
        self.pos_df_bt = pos_df_bt # positions eod BEFORE trading
        self.pos_df_at = pos_df_at # positions eod AFTER trading
        self.compute_settings = compute_settings

        ### compute info from portfolio data
        self.has_trades = not trade_logs_df[trade_logs_df["executed"]].empty
        if not self.has_trades:
            #print(f"Portfolio for channel_id {self.channel_id}: No trades executed.")
            pass
        else:
            self.pf_values = self.pos_df_at.sum(axis=1).rename("pf_val")
            self.pf_returns = self.pf_values.pct_change()
            # total period
            self.start_date = self.pos_df_at.index[0]
            self.end_date = self.pos_df_at.index[-1]
            self.n_days_total_period = self.pos_df_at.shape[0]
            # active days (all days with non-zero positions, not necessarily contiguous)
            self.active_holding_days = self.pos_df_bt.index[(self.pos_df_bt.iloc[:, :-1] != 0).any(axis=1)]
            self.n_days_active_holdings = len(self.active_holding_days)
            # active period (with all gaps included -> contiguous series)
            self.first_holding_date = self.active_holding_days[0] # since we trade EOD we consider the first day that with non-zero positions from market open
            self.last_holding_date = self.active_holding_days[-1] # last day with non-zero positions
            self.active_period_days = self.pos_df_at.loc[self.first_holding_date:self.last_holding_date].index
            self.n_days_active_period = len(self.active_period_days)
            
            self.neutral_asset_ticker = self.pos_df_at.columns[-1] # last column of positions df is always neutral asset
            
            # portfolio allocation info
            self.n_buys = len(self.trade_logs_df[(self.trade_logs_df["executed"]) & (self.trade_logs_df["sentiment"] == "buy")])
            self.n_sells = len(self.trade_logs_df[self.trade_logs_df["executed"] & (self.trade_logs_df["sentiment"] == "sell")])
            self.unique_tickers = self.pos_df_at.columns[:-1].tolist()
            self.n_unique_positions = len(self.unique_tickers) # excluding neutral asset
            self.n_buys_stocks = ((self.trade_logs_df["executed"]) & (self.trade_logs_df["sentiment"] == "buy") & (self.trade_logs_df["ticker"].str.startswith("stock"))).sum()
            all_buys = self.trade_logs_df[(self.trade_logs_df["executed"]) & (self.trade_logs_df["sentiment"] == "buy")]
            self.n_buys_stocks = ((all_buys["ticker"].str.startswith("stock"))).sum()
            self.n_buys_cryptos = ((all_buys["ticker"].str.startswith("crypto"))).sum()
            self.n_buys_etfs = ((all_buys["ticker"].str.startswith("etf"))).sum()
            self.n_buys_commodities = ((all_buys["ticker"].str.startswith("commodity"))).sum()

    # various performance metrics (determine correct series subsets to call standalone functions with)
    def get_sharpe_ratio(self, bm_returns, period="full"):
        idx = self._get_idx(period)
        return sharpe_ratio(self.pf_returns.loc[idx], bm_returns.loc[idx])

    def get_sortino_ratio(self, bm_returns, daily_target_return=0, period="full"):
        idx = self._get_idx(period)
        return sortino_ratio(pf_returns=self.pf_returns.loc[idx], 
                             bm_returns=bm_returns.loc[idx], 
                             # daily target return can be a scalar or a series with matching index
                             daily_target_return=(daily_target_return.loc[idx] if isinstance(daily_target_return, pd.Series) else daily_target_return))

    def get_value_at_risk(self, alpha, period="full"):
        idx = self._get_idx(period)
        return value_at_risk(self.pf_returns.loc[idx], alpha)
    
    def get_max_drawdown(self, period="full"):
        idx = self._get_idx(period)
        pf_values = (1 + self.pf_returns.loc[idx]).cumprod() # can't use self.pf_values if period is "active_days" due to gaps
        return max_drawdown(pf_values)

        
    def get_total_return(self, period="full"):
        idx = self._get_idx(period)
        return total_return(self.pf_returns.loc[idx])
    
    def get_total_excess_return(self, bm_returns, period="full"):
        idx = self._get_idx(period)
        return total_excess_return(self.pf_returns.loc[idx], bm_returns.loc[idx])

    def get_beta(self, bm_returns, period="full"):
        idx = self._get_idx(period)
        return beta(self.pf_returns.loc[idx], bm_returns.loc[idx])
    
    # helper
    def _get_idx(self, period_arg):
        if period_arg == "active_days":
            idx = self.active_holding_days
        elif period_arg == "active_period":
            idx = self.active_period_days
        elif period_arg == "full":
            idx = self.pf_returns.index
        # if index contains very first day ("2016-01-04"), drop it (no returns available yet)
        if idx[0] == "2016-01-04":
            idx = idx[1:]
        return idx

class PortfolioBuilder:
    """
    The PortfolioBuilder class is used to build portfolios from extractions and returns data. Holds full set of returns data but selects appropriate subset in preparation for portfolio computation. 
    This allows for more efficient computation of a big number of portfolios.
    """
    def __init__(self, extractions_df, returns_df, settings, ticker_sep="+"):

        ### base data
        self.ext = extractions_df[["video_id", "upload_date", "channel_id", "trade_info"]] # "trade_info" should contain list of json objects
        self.returns = returns_df # column names should be f"{asset_type}{ticker_sep}{ticker}" (e.g. "stock+AAPL", "crypto+BTC", etc.), index should be "YYYY-MM-DD" date strings.
        self.asset_types = ["stock", "crypto", "etf", "commodity", 
                            "benchmark" # benchmark assets such as cash, risk-free rate, SPY, etc.
        ]
        self.ticker_sep = ticker_sep # separator to be used between asset type and ticker in column names

        # settings
        if settings is not None:
            self.settings = settings
        else:
            # default settings
            self.settings = {"pf_start_date": "2016-01-04", 
                             "pf_end_date": "2023-12-29",
                             "pf_initial_value": 1, # arbitrary, 1 is simplest to interpret
                             "portfolio_type": "equal_weight", # with each trade, readjust positions to equal weight
                             "max_positions": 10000, # max number of non-cash positions in a portfolio (sell oldest positions if exceeded)
                             "max_holding_period": 252, # max number of trading days to hold a position (sell if exceeded)
                             "neutral_asset": "cash", # asset to hold in place of cash (must be available as column in benchmark returns data)
                             "min_days_wait_after_upload": 1 # min number of days (not trading days!) after video upload date to start trading -> should be at least 1 to avoid lookahead bias (we trade EOD but video could have been uploaded after market close with new information)
                             }
        self.neutral_asset_ticker = f"benchmark{self.ticker_sep}{self.settings['neutral_asset']}"
        
        print(f"PortfolioBuilder: Initialized with {len(self.ext)} videos from {len(self.ext['channel_id'].unique())} unique channels.")

        # get trading days from returns data (as datetimes)
        self.trading_days = pd.to_datetime(returns_df.index)

        ### a few preprocessing steps for the extraction data
        # 1. remove rows with empty trade_info
        self.ext = self.ext[self.ext["trade_info"] != "[]"]
        # 2. compute trade_date col (next trading day after upload_date)
        self.ext["trade_date"] = self.ext["upload_date"].apply(lambda x: self._get_next_trading_day(x, min_days_wait=self.settings["min_days_wait_after_upload"]))
        # 3. remove rows with trade_date outside of start_date and end_date (if given) (note: also removes rows with missing trade_date)
        if self.settings["pf_start_date"] is not None:
            self.ext = self.ext[self.ext["trade_date"] >= self.settings["pf_start_date"]]
        if self.settings["pf_end_date"] is not None:
            self.ext = self.ext[self.ext["trade_date"] <= self.settings["pf_end_date"]]

        print(f"PortfolioBuilder: Only considering videos with non-empty trade_info: {len(self.ext)} videos from {len(self.ext['channel_id'].unique())} unique channels.")
        
    def update_settings(self, new_settings):
        if "pf_start_date" in new_settings or "pf_end_date" in new_settings:
            raise ValueError("Updating start or end date requires re-instantiation of the PortfolioBuilder object.")
        self.settings.update(new_settings)
        self.neutral_asset_ticker = f"benchmark{self.ticker_sep}{self.settings['neutral_asset']}"
        print("PortfolioBuilder: Updated settings.")

    def compute_portfolio(self, channel_id, return_before_and_after_trading_positions=True, return_portfolio_object=False, debug=False):
        
        # get initial trades_df
        trades_df = self._get_raw_trades_from_recs(channel_id, remove_sells_without_preceding_buys=True, remove_buys_with_same_day_sells=True, debug=debug)

        # get unique tickers for this portfolio
        unique_assets = trades_df["ticker"].unique().tolist()

        # get required subsets of return data and return availability for this portfolio
        ret_df, ret_av = self._prep_pf_returns_data(unique_assets)

        # perform portfolio computation with desired strategy
        if self.settings["portfolio_type"] == "equal_weight":
            pos_df_bt, pos_df_at, trade_logs_df = self._compute_equal_weight_portfolio(trades_df, 
                                                                                       ret_df, 
                                                                                       ret_av, 
                                                                                       return_before_and_after_trading_positions=return_before_and_after_trading_positions, 
                                                                                       debug=debug)
        else:
            raise ValueError("Invalid portfolio_type option!")
        
        if return_portfolio_object:
            return Portfolio(trade_logs_df, pos_df_at, pos_df_bt, self.settings)
        else:
            return pos_df_bt, pos_df_at, trade_logs_df
        
            
            
    def _get_raw_trades_from_recs(self, channel_id, remove_sells_without_preceding_buys=False, remove_buys_with_same_day_sells=False, debug=False):
        
        ### get raw trades df (not yet taking into account returns data availability, needing to buy before being able to sell, etc.)
        # get relevant rows from extractions_df
        df = self.ext[self.ext["channel_id"] == channel_id].copy()
        
        # expand trade info json lists into df (-> columns asset_type, ticker, sentiment)
        df.loc[:, "trade_info"] = df["trade_info"].apply(json.loads)
        df = df.explode('trade_info').reset_index(drop=True)
        trades_df = pd.json_normalize(df["trade_info"])
        # re-add date cols from (now exploded!) extractions df
        trades_df["trade_date"] = df["trade_date"]
        trades_df["upload_date"] = df["upload_date"] # only kept for debugging purposes

        # add asset_type prefix to ticker col
        trades_df["ticker"] = trades_df["asset_type"] + "+" + trades_df["ticker"]

        if remove_buys_with_same_day_sells:
            # create helper column
            trades_df['has_sameday_sell'] = trades_df.groupby(['ticker', 'trade_date'])['sentiment'].transform(lambda x: 'sell' in x.values)
            # remove buy trades with same-day sells
            trades_df = trades_df[~((trades_df['sentiment'] == 'buy') & (trades_df['has_sameday_sell']))]
            # remove helper col
            trades_df = trades_df.drop(columns=['has_sameday_sell']).reset_index(drop=True)

        if remove_sells_without_preceding_buys:
            # add boolean helper column preceding_buy_exists
            trades_df['sentiment_order'] = trades_df['sentiment'].map({'sell': 0, 'buy': 1})
            trades_df = trades_df.sort_values(by=['ticker', 'trade_date', 'sentiment_order'])
            trades_df = trades_df.groupby('ticker', group_keys=False).apply(
                lambda group: group.assign(preceding_buy_exists=(group['sentiment'] == 'buy').cumsum().shift().fillna(0) > 0))
            drop_cond = (trades_df['sentiment'] == 'sell') & (~trades_df['preceding_buy_exists'])
            # collect info about rows to be dropped
            num_dropped = drop_cond.sum()
            dropped_tickers = trades_df[drop_cond]['ticker'].unique()
            
            # remove sells without preceding buys
            trades_df = trades_df[~drop_cond]
            # remove helper cols
            trades_df = trades_df.drop(columns=['preceding_buy_exists', 'sentiment_order']).reset_index(drop=True)
            if num_dropped > 0 and debug:
                print(f"Removed {num_dropped} sell trades without preceding buy trades for the following tickers: {dropped_tickers}")

        return trades_df
    

    def _get_next_trading_day(self, date, min_days_wait):
        # given a date + waiting period, return the next trading day 
        date = pd.to_datetime(date)
        target_date = date + pd.Timedelta(days=min_days_wait)
        next_trading_day = self.trading_days[self.trading_days >= target_date].min()

        return next_trading_day # none if no trading day after target_date

    
    def _prep_pf_returns_data(self, unique_assets):
        # get required subsets of return data for this portfolio, and dict with data availability info
        # note: neutral asset MUST NOT be in unique_assets
        
        ret_df = self.returns[[t for t in unique_assets if t in self.returns.columns]].copy() # avoid SettingWithCopyWarning
        # add neutral asset as very last column (position is important!)
        ret_df[self.neutral_asset_ticker] = self.returns[self.neutral_asset_ticker]
        # get shifted returns data (shifted early by 1 trading day)
        ret_df_shifted = ret_df.shift(-1)
        # get available first and last trading days for each asset in dict to make lookup possible: ret_av[ticker][...] -> date
        # note: neutral asset is not included in ret_av, we assume full returns availability for it
        ret_av = {}
        for t in unique_assets:
            ret_av[t] = {}
            ret_av[t]["has_returns"] = t in ret_df.columns
            if ret_av[t]["has_returns"]:
                ret_av[t]["first_possible_buy"] = ret_df_shifted[t].first_valid_index() # day before first return is available
                ret_av[t]["last_possible_buy"] = ret_df_shifted[t].last_valid_index() # day before last available return
                ret_av[t]["last_possible_sell"] = ret_df[t].last_valid_index() # day of last available return
        return ret_df, ret_av
    
    def _compute_equal_weight_portfolio(self, trades_df, returns_df, ret_av, return_before_and_after_trading_positions=True, debug=False):
        # the equal weight portfolio is computed by rebalancing all non-neutral positions to equal weight each trading day.
        return_factors_df = returns_df + 1
        tickers = returns_df.columns[:-1].tolist() # tickers of all considered assets for this portfolio (except neutral asset!) in correct order
        holding_days_tracker = {t: 0 for t in tickers} # dict to track holding days for each ticker
        # initialize positions list 
        positions_list_eod_before_trading = [] 
        positions_list_eod_after_trading = [] # (will contain the positions lists for each trading day EOD (-> AFTER trading))

        # helper function(s)
        def add_to_trade_logs(trades_list, date, tickers, sentiment, executed, reason):
            for t in tickers:
                trades_list.append({"trade_date": date, "ticker": t, "sentiment": sentiment, "executed": executed, "reason": reason})

        trade_logs = [] # will be used to build executed_trades_df, containing 1 row for every actually executed trade

        pf_running_days = self.trading_days[(self.trading_days >= self.settings["pf_start_date"]) & 
                                            (self.trading_days <= self.settings["pf_end_date"])]

        for date in pf_running_days.strftime("%Y-%m-%d"):
            ### get current positions (from previous day) or initialize on first day
            if len(positions_list_eod_after_trading) == 0: # first iteration
                pos = np.array([0.] * len(tickers) + [self.settings["pf_initial_value"]]) # starting with 100% neutral asset
            else:
                pos = positions_list_eod_after_trading[-1]

            ### apply returns for the day
            # element-wise multiplication of pos with return factors for the day. replace missing return factors with 1 
            # note: we ensure that missing return factors are only mutliplied with zero-positions through the availability checking in the trading part
            if debug:
                # DEBUG: check if any NA return factors have non-zero pos values at the same index
                if np.any(np.isnan(return_factors_df.loc[date]) & (pos != 0)) and date != self.settings["pf_start_date"]:
                    problematic_tickers = [(t, pos[i], return_factors_df.loc[date][i]) for i, t in enumerate(tickers) if np.isnan(return_factors_df.loc[date][i]) and pos[i] != 0]
                    print("ERROR: Missing return factors for non-zero positions!")
                    print(f"Date: {date}, problematic cases (ticker, pos, return_factor): {[(t, p, r) for t, p, r in problematic_tickers]}")
            pos = pos * return_factors_df.loc[date].fillna(1).to_numpy()
            # save to pre-trading list
            positions_list_eod_before_trading.append(pos)
            # get tickers for current holdings
            current_tickers = set([t for i, t in enumerate(tickers) if pos[i] > 0]) # doesn't include neutral asset
            # update holding days tracker
            holding_days_tracker = {t: holding_days_tracker[t] + 1 if t in current_tickers else 0 for t in tickers}

            ### process trades for the day
            # for equal weight portfolio we simply need to get the set of tickers we are still holding after all buys and sells
            # 1. BUYS
            potential_buys = set(trades_df[(trades_df["trade_date"] == date) & (trades_df["sentiment"] == "buy")]["ticker"])
            already_holding = potential_buys & current_tickers
            add_to_trade_logs(trade_logs, date, already_holding, "buy", False, "ALREADY HOLDING") # not executed!
            potential_buys = potential_buys - already_holding
            no_returns = {t for t in potential_buys if not (ret_av[t]["has_returns"] and date >= ret_av[t]["first_possible_buy"] and date <= ret_av[t]["last_possible_buy"])}
            add_to_trade_logs(trade_logs, date, no_returns, "buy", False, "NO RETURNS") # not executed!
            buys = potential_buys - no_returns
            current_tickers = current_tickers | buys
            add_to_trade_logs(trade_logs, date, buys, "buy", True, "NORMAL") # actually executed


            # 2. SELLS
            potential_sells = set(trades_df[(trades_df["trade_date"] == date) & (trades_df["sentiment"] == "sell")]["ticker"])
            not_holding = potential_sells - current_tickers
            add_to_trade_logs(trade_logs, date, not_holding, "sell", False, "NOT HOLDING") # not executed!
            sells = potential_sells - not_holding 
            current_tickers = current_tickers - sells
            add_to_trade_logs(trade_logs, date, sells, "sell", True, "NORMAL") # actually executed
            
            # check for forced sells 
            # - due to returns data no longer available
            no_more_data = {t for t in current_tickers if ret_av[t]["has_returns"] and date == ret_av[t]["last_possible_sell"]}
            add_to_trade_logs(trade_logs, date, no_more_data, "sell", True, "DATA AVAILABILITY")
            current_tickers = current_tickers - no_more_data
            # - due to max holding period reached
            mhp_reached = {t for t in current_tickers if holding_days_tracker[t] >= self.settings["max_holding_period"]}
            add_to_trade_logs(trade_logs, date, mhp_reached, "sell", True, "MHP REACHED")
            current_tickers = current_tickers - mhp_reached
            # - due to max positions number reached
            if len(current_tickers) > self.settings["max_positions"]:
                # determine tickers to sell (oldest ones, i.e. with highest holding days)
                n_to_sell = len(current_tickers) - self.settings["max_positions"]
                sorted_oldest_first = sorted(current_tickers, key=lambda x: holding_days_tracker[x], reverse=True)
                to_sell = sorted_oldest_first[:n_to_sell]
                add_to_trade_logs(trade_logs, date, to_sell, "sell", True, "MAX POSITIONS")
                current_tickers = current_tickers - set(to_sell)
            
            ### adjust positions (in this case: rebalance to equal weight)
            pf_val = np.sum(pos)
            if len(current_tickers) == 0: # case 1) no holdings left after trading -> hold only neutral asset
                pos = np.array([0.] * len(tickers) + [pf_val])
            else: # case 2) at least 1 holding -> balance to equal weight
                pos_val = pf_val / len(current_tickers)
                pos = np.array([pos_val if t in current_tickers else 0. for t in tickers] + [0.])
            
            # save to post-trading list
            positions_list_eod_after_trading.append(pos)


        ### post-processing
        # build positions_df from positions list
        # note: we drop any columns (except for the neutral asset column) which are all zeros (-> e.g. due to returns data availability, or any other reason which only became apparent during the portfolio computation)
        # this way we can take the column names of the returned positions df as unique assets actually held in the portfolio at some point
        df = pd.DataFrame(positions_list_eod_after_trading, columns=tickers + [self.neutral_asset_ticker], index=pf_running_days)
        positions_df_eod_after_trading = pd.concat([df.iloc[:, :-1].loc[:, (df.iloc[:, :-1] != 0).any(axis=0)], # cols without neutral asset col
                                                    df.iloc[:, -1]], axis=1) # neutral asset col
        if debug:
            print(f"DEBUG: positions_df_eod_after_trading shapes before and after dropping zero cols: {df.shape}, {positions_df_eod_after_trading.shape}")
        df = pd.DataFrame(positions_list_eod_before_trading, columns=tickers + [self.neutral_asset_ticker], index=pf_running_days)
        positions_df_eod_before_trading = pd.concat([df.iloc[:, :-1].loc[:, (df.iloc[:, :-1] != 0).any(axis=0)], # cols without neutral asset col
                                                    df.iloc[:, -1]], axis=1)
        if debug:
            print(f"DEBUG: positions_df_eod_before_trading shapes before and after dropping zero cols: {df.shape}, {positions_df_eod_before_trading.shape}")
        # build trades_log_df from trade_logs
        trade_logs_df = pd.DataFrame(trade_logs, columns=["trade_date", "ticker", "sentiment", "executed", "reason"])

        if return_before_and_after_trading_positions:
            return positions_df_eod_before_trading, positions_df_eod_after_trading, trade_logs_df
        else:
            return positions_df_eod_after_trading, trade_logs_df
        
                        

