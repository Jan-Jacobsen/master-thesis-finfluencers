def deduplicate_asset_list(asset_list, retain_unmatched=True):
    # input list contains objects of form {"asset_name": "Apple", "asset_type": "stock", "sentiment": "buy", "match_info": {"matched_ticker": "AAPL", ...}}
    # matched_ticker is the unique identifier for each asset!
    
    # separate assets which are not part of the deduplication process
    # -> any unmatched assets (which includes all of type "other")
    matched_assets, unmatched_assets = [], []
    for a in asset_list:
        if ("match_info" in a) and (a["match_info"]["matched_ticker"] is not None):
            matched_assets.append(a)
        else:
            unmatched_assets.append(a)

    # add prefix to all crypto tickers to avoid conflicts with other asset types 
    crypto_prefix = "C+" # use character which is certainly not part of any ticker
    for a in matched_assets:
        if a["asset_type"] == "crypto":
            a["match_info"]["matched_ticker"] = f"{crypto_prefix}{a['match_info']['matched_ticker']}"

    # note: we don't use set() here to preserve order of assets
    unique_tickers = dict.fromkeys([a["match_info"]["matched_ticker"] for a in matched_assets]).keys()

    # iterate over tickers and check for multiple occurences
    deduped_assets = []
    for ticker in unique_tickers:
        ticker_assets = [a for a in matched_assets if a["match_info"]["matched_ticker"] == ticker]
        if len(ticker_assets) == 1: # no deduplication required
            selection = ticker_assets[0]
        else: # multiple occurences -> apply defined rules to decide which one to keep

            # 1. sentiment disagreements
            unique_sentiments = set([a["sentiment"] for a in ticker_assets])
            if len(unique_sentiments) > 1:
                # all three sentiments (buy/neutral/sell) are present -> keep only first neutral one
                if set(["buy", "neutral", "sell"]).issubset(unique_sentiments):
                    selection = next(a for a in ticker_assets if a["sentiment"] == "neutral")
                # only buy and sell are present -> transform to neutral
                elif set(["buy", "sell"]).issubset(unique_sentiments):
                    selection = ticker_assets[0] # deepcopy not necessary before updating (no further use of ticker_assets in the iteration)
                    selection.update({"sentiment": "neutral"})
                else: # neutral and 1 non-neutral sentiment -> keep first non-neutral
                    selection = next(a for a in ticker_assets if a["sentiment"] != "neutral")
            else:
                # 2. asset type disagreements (commodity and stock/etf)
                unique_asset_types = set([a["asset_type"] for a in ticker_assets])
                if len(unique_asset_types) > 1:
                    # keep first commodity
                    selection = next(a for a in ticker_assets if a["asset_type"] == "commodity")
                else:
                    # 3. asset name disagreements or simply duplicate extractions -> keep first
                    selection = ticker_assets[0]

        # append asset object(s) which should be kept for the current ticker
        if selection is list: # not possible in current implementation
            deduped_assets.extend(selection)
        else:
            deduped_assets.append(selection)

    # remove crypto prefix from tickers
    for a in deduped_assets:
        if a["asset_type"] == "crypto":
            a["match_info"]["matched_ticker"] = a["match_info"]["matched_ticker"][len(crypto_prefix):]

    # add back unmatched assets if desired
    if retain_unmatched:
        deduped_assets.extend(unmatched_assets)
    
    return deduped_assets