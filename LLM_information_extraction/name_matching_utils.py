from cleanco.clean import custom_basename, prepare_default_terms
import rapidfuzz as rf
import re
import pandas as pd

### PRE-PROCESSING ###

def could_be_ticker(name):
    # check if the name could be a ticker (e.g. short, all uppercase or lowercase, all letters (true for US only! otherwise use .isalnum()))
    return (name.isupper() or name.islower()) and len(name) <= 5 and name.isalpha()

def get_stock_basename_or_ticker(full_name):

    # skip any cleaning if the name looks like a ticker
    if could_be_ticker(full_name):
        return full_name
    
    # remove legal terms (using cleanco)
    name_without_legal_terms = custom_basename(full_name, 
                                terms=prepare_default_terms(), 
                                prefix=False, # we don't expect legal terms as first word in our dataset
                                middle=True, 
                                suffix=True)
    # in some cases custom_basename returns an empty string (e.g. for "LTD Inc."), in this case don't apply the legal term removal
    if name_without_legal_terms != "": 
        name = name_without_legal_terms
    else:
        name = full_name

    # further data-specific removals (mostly related to share types)
    remove_exact = [
        r" Class A", # we only remove this for "A" class to match these easier than other classes!
        r" ADR",
        r" ADS",
    ]
    remove_case_insensitive = [
        r" American Depositary Shares",
        r" Common Stock",
        r" Common Shares",
        r" Ordinary Shares",
        r" ADR$", # remove case-insensitive only at the end
        r" ADS$", # remove case-insensitive only at the end
        # other
        r"\.com",
        r"^The ", # remove only if it's the first word
    ] # note: we don't remove terms such as "Warrants", "Units", "Rights" etc. to make matching these more difficult, usually there is a normal version of the stock which should be matched instead

    for pattern in remove_exact:
        name = re.sub(pattern, "", name)
    for pattern in remove_case_insensitive:
        name = re.sub(pattern, "", name, flags=re.IGNORECASE)
    
    # lowercase
    name = name.lower()
    return name

def get_etf_basename_or_ticker(full_name):

    # skip any cleaning if the name looks like a ticker
    if could_be_ticker(full_name):
        return full_name
    else:
        name = full_name

    # removals
    remove_case_insensitive = [
        r" ETF",
        r" Trust",
        r" Fund",
        r" Shares",
        r" Index",
    ]
    for pattern in remove_case_insensitive:
        name = re.sub(pattern, "", name, flags=re.IGNORECASE)
    # replace hyphens with spaces
    name = name.replace("-", " ")
    # remove extra spaces
    name = re.sub(r"\s+", " ", name)
    # lowercase
    name = name.lower()
    return name

def get_crypto_basename_or_ticker(full_name):
    # for cryptos we don't do much cleaning
    name = full_name
    # removals
    remove_case_insensitive = [
        r"^The ", # remove only if it's the first word
    ]
    for pattern in remove_case_insensitive:
        name = re.sub(pattern, "", name, flags=re.IGNORECASE)

    name = name.lower()

    return name

def get_commodity_basename_or_ticker(full_name):
    name = full_name.lower()
    return name

### MATCHING UTILITIES ###

def match_stock(query, candidates_dict_all):
    # query: stock name string, to be matched to one of candidates
    # candidates_dict: dict with tickers as keys and preprocessed (!) names as values


    # IMPORTANT: 
    # the candidates dict should be ALREADY SORTED as follows: SP500 constituents first, then all currently listed stocks, then delisted stocks
    # this will yield better results in cases of equal match scores


    match_info = {"query": query, 
                  "query_basename": None, # only not None if preprocessing was applied
                  "matched_ticker": None, 
                  "match_score": None, # fuzzy match score, if used
                  "match_type": None, # which matching round was successful?
            }
    

    # 1. Try exact ticker match first
    #     note: this should only be implemented without extra checks when the model outputs for company names tend to capitalize the first letter while tickers are either all uppercase or all lowercase
    if could_be_ticker(query):
        if query.upper() in candidates_dict_all.keys():
            match_info["matched_ticker"] = query.upper()
            match_info["match_type"] = "round 1 (exact ticker match)"
            return match_info
    
    # -- preprocess
    query_basename = get_stock_basename_or_ticker(query)
    match_info["query_basename"] = query_basename

    # 2. Try manual matches
    if query_basename in manual_stock_match_dict:
        match_info["matched_ticker"] = manual_stock_match_dict[query_basename]
        match_info["match_type"] = "round 2 (manual match dict)"
        return match_info

    # 3. Try fuzzy matching using Jaro-Winkler distance
    result = rf.process.extractOne(query=query_basename, # preprocessed query!
                                  choices=candidates_dict_all, # 
                                  scorer=rf.distance.JaroWinkler.normalized_similarity,
                                  processor=None,
                                  score_cutoff=0.94, # 0.94 has proven to be a good threshold for stocks
                                  score_hint=None, 
                                  scorer_kwargs={"prefix_weight": 0.2} # high weight of 0.2 (max is 0.25) to prefer matches with same (4-letter-)prefix strongly
                                  )
    if result is not None: # tuple of (matched basename, score, matched ticker)
        match_info["matched_ticker"] = result[2]
        match_info["match_score"] = result[1]
        match_info["match_type"] = "round 3 (Jaro-Winkler matching)"
        return match_info
    
    # another round with Levensthein here?
    
    # 4. Try ticker matching again for short names with first-letter capitalization (e.g. "Mara" -> "MARA" referring to Marathon Digital Holdings)
    if could_be_ticker(query.upper()):
        if query.upper() in candidates_dict_all.keys():
            match_info["matched_ticker"] = query.upper()
            match_info["match_type"] = "round 4 (exact ticker match, miscapitalized)"
            return match_info
    
    # no match found: leave match_info fields as None
    return match_info


def match_etf(query, candidates_dict_all, candidates_dict_listed=None):
    # if candidates_dict_listed is provided we perform a multi-step matching process (only listed etfs first, then also include delisted)

    match_info = {"query": query, 
                  "query_basename": None, # only not None if preprocessing was applied
                  "matched_ticker": None, 
                  "match_score": None, # fuzzy match score, if used
                  "match_type": None, # which matching round was successful?
            }
    

    # 1. Try exact ticker match first (for all short names)
    if could_be_ticker(query.upper()):
        if query.upper() in candidates_dict_all.keys():
            match_info["matched_ticker"] = query.upper()
            match_info["match_type"] = "round 1 (exact ticker match)"
            return match_info
        
    # -- preprocess
    query_basename = get_etf_basename_or_ticker(query)
    match_info["query_basename"] = query_basename
    # 2. Try manual matches
    # special adjustment since variants of sp500 are super common
    sp500_strings = ["sp500", "s&p500", "s&p 500", "sp 500", "s&p", "s and p", "s and p 500"]
    avoid_strings = ["short", "bear", "inverse", "2x", "3x"]
    if any(s in query_basename for s in sp500_strings) and (not any(s in query_basename for s in avoid_strings)):
        query_basename = "s&p500" # matches SPY via manual matching dict
    # manual matching
    if query_basename in manual_etf_match_dict:
        match_info["matched_ticker"] = manual_etf_match_dict[query_basename]
        match_info["match_type"] = "round 2 (manual match dict)"
        return match_info

    # 3. Try fuzzy matching using WRatio (only listed etfs)
    if candidates_dict_listed is not None:
        result = rf.process.extractOne(query=query_basename,
                                    choices=candidates_dict_listed,
                                    scorer=rf.fuzz.WRatio, 
                                    processor=None, 
                                    score_cutoff=89, # WRatio returns values in [0, 100]
                                    score_hint=None, 
                                    scorer_kwargs=None,
                                    )
        if result is not None: # tuple of (matched basename, score, matched ticker)
            match_info["matched_ticker"] = result[2]
            match_info["match_score"] = result[1]
            match_info["match_type"] = "round 3 (strict WRatio matching, listed only)"
            return match_info
    
    # 4. Try fuzzy matching using WRatio (incl delisted and less strict)
    result = rf.process.extractOne(query=query_basename,
                                choices=candidates_dict_all,
                                scorer=rf.fuzz.WRatio, 
                                processor=None, 
                                score_cutoff=86, # note: tests showed that at 85.5 a lot of bad matches start happening
                                score_hint=None, 
                                scorer_kwargs=None,
                                )
    if result is not None: # tuple of (matched basename, score, matched ticker)
        match_info["matched_ticker"] = result[2]
        match_info["match_score"] = result[1]
        match_info["match_type"] = "round 4 (less strict WRatio matching, incl delisted)"
        return match_info
    
    # no match found: leave match_info fields as None
    return match_info


def match_crypto(query, candidates_dict_all, candidates_dict_top200=None, candidates_dict_listed=None):
    # if candidates_dict_listed is provided we perform a multi-step matching process (only listed cryptos first, then also include delisted)

    match_info = {"query": query, 
                  "query_basename": None, # only not None if preprocessing was applied
                  "matched_ticker": None, 
                  "match_score": None, # fuzzy match score, if used
                  "match_type": None, # which matching round was successful?
            }

    # 1. Try exact ticker match first (for all short names)
    if len(query) <= 4:
        if query.upper() in candidates_dict_all.keys():
            match_info["matched_ticker"] = query.upper()
            match_info["match_type"] = "round 1 (exact ticker match)"
            return match_info
        
    # -- preprocess
    query_basename = get_crypto_basename_or_ticker(query)
    match_info["query_basename"] = query_basename

    # 2. Try manual matches
    if query_basename in manual_crypto_match_dict:
        match_info["matched_ticker"] = manual_crypto_match_dict[query_basename]
        match_info["match_type"] = "round 2 (manual match dict)"
        return match_info
    
    # 3. Try fuzzy matching using Jaro-Winkler (top_200 only)
    if candidates_dict_top200 is not None:
        result = rf.process.extractOne(query=query_basename,
                                    choices=candidates_dict_top200,
                                    scorer=rf.distance.JaroWinkler.normalized_similarity,
                                    processor=None,
                                    score_cutoff=0.95,
                                    score_hint=None, 
                                    scorer_kwargs={"prefix_weight": 0.2} # high weight of 0.2 (max is 0.25) to prefer matches with same (4-letter-)prefix strongly
                                    )
        if result is not None:
            match_info["matched_ticker"] = result[2]
            match_info["match_score"] = result[1]
            match_info["match_type"] = "round 3 (Jaro-Winkler matching, top200 only)"
            return match_info
    
    # 4. Try fuzzy matching using Jaro-Winkler (listed)
    if candidates_dict_listed is not None:
        result = rf.process.extractOne(query=query_basename,
                                    choices=candidates_dict_listed,
                                    scorer=rf.distance.JaroWinkler.normalized_similarity,
                                    processor=None,
                                    score_cutoff=0.95,
                                    score_hint=None, 
                                    scorer_kwargs={"prefix_weight": 0.2} # high weight of 0.2 (max is 0.25) to prefer matches with same (4-letter-)prefix strongly
                                    )
        if result is not None:
            match_info["matched_ticker"] = result[2]
            match_info["match_score"] = result[1]
            match_info["match_type"] = "round 4 (Jaro-Winkler matching, listed only)"
            return match_info
    
    # 5. Try fuzzy matching using Jaro-Winkler (all)
    result = rf.process.extractOne(query=query_basename,
                                choices=candidates_dict_all,
                                scorer=rf.distance.JaroWinkler.normalized_similarity,
                                processor=None,
                                score_cutoff=0.94, # slightly lower threshold
                                score_hint=None, 
                                scorer_kwargs={"prefix_weight": 0.2} # high weight of 0.2 (max is 0.25) to prefer matches with same (4-letter-)prefix strongly
                                )
    if result is not None:
        match_info["matched_ticker"] = result[2]
        match_info["match_score"] = result[1]
        match_info["match_type"] = "round 5 (Jaro-Winkler matching, incl delisted)"
        return match_info
        
    # 6. Try fuzzy matching using standard Levenshtein (all)
    result = rf.process.extractOne(query=query_basename,
                                choices=candidates_dict_all,
                                scorer=rf.distance.Levenshtein.normalized_similarity,
                                processor=None, 
                                score_cutoff=0.9,
                                score_hint=None, 
                                scorer_kwargs=None,
                                )
    if result is not None:
        match_info["matched_ticker"] = result[2]
        match_info["match_score"] = result[1]
        match_info["match_type"] = "round 6 (Levenshtein matching, all)"
        return match_info
    
    # 7. Try exact ticker match again (ignoring length, crypto tickers can be longer)
    if query.upper() in candidates_dict_all.keys():
        match_info["matched_ticker"] = query.upper()
        match_info["match_type"] = "round 7 (exact ticker match, long name)"
        return match_info

    # no match found: leave match_info fields as None
    return match_info


def match_commodity(query, candidates_dict_all):

    match_info = {"query": query, 
                    "query_basename": None, # only not None if preprocessing was applied
                    "matched_ticker": None, 
                    "match_score": None, # fuzzy match score, if used
                    "match_type": None, # which matching round was successful?
                }
    # preprocess
    query_basename = query.lower()
    match_info["query_basename"] = query_basename

    # 1. Try manual matches
    if query_basename in manual_commodity_match_dict:
        match_info["matched_ticker"] = manual_commodity_match_dict[query_basename]
        match_info["match_type"] = "round 1 (manual match dict)"
        return match_info

    # 2. Try Jaro-Winkler (high threshold)
    result = rf.process.extractOne(query=query_basename,
                                choices=candidates_dict_all,
                                scorer=rf.distance.JaroWinkler.normalized_similarity,
                                processor=None,
                                score_cutoff=0.95,
                                score_hint=None, 
                                scorer_kwargs={"prefix_weight": 0.2} # high weight of 0.2 (max is 0.25) to prefer matches with same (4-letter-)prefix strongly
                                )
    if result is not None:
        match_info["matched_ticker"] = result[2]
        match_info["match_score"] = result[1]
        match_info["match_type"] = "round 2 (Jaro-Winkler matching)"
        return match_info

    # 3. Try Levenshtein (lower threshold)
    result = rf.process.extractOne(query=query_basename,
                                choices=candidates_dict_all,
                                scorer=rf.distance.Levenshtein.normalized_similarity,
                                processor=None, 
                                score_cutoff=0.85,
                                score_hint=None, 
                                scorer_kwargs=None,
                                )
    if result is not None:
        match_info["matched_ticker"] = result[2]
        match_info["match_score"] = result[1]
        match_info["match_type"] = "round 3 (Levenshtein matching)"
        return match_info
    
    # 4. Try WRatio (candidate names are always short here, queries sometimes have several words)
    result = rf.process.extractOne(query=query_basename,
                                choices=candidates_dict_all,
                                scorer=rf.fuzz.WRatio, 
                                processor=None, 
                                score_cutoff=89, # WRatio returns values in [0, 100]
                                score_hint=None, 
                                scorer_kwargs=None,
                                )
    if result is not None:
        match_info["matched_ticker"] = result[2]
        match_info["match_score"] = result[1]
        match_info["match_type"] = "round 4 (WRatio matching)"
        return match_info
    
    # no match found: leave match_info fields as None
    return match_info



### PREP & APPLICATION ###

def load_candidate_dicts():
    # load names and tickers data and create candidate dicts required by the matching functions
    #   - with appropriate sorting
    #   - with preprocessing applied

    # load names and tickers data
    path = "../data/asset_data/names_and_tickers"
    stocks = pd.read_csv(f"{path}/eodhd_stocks.csv", sep=";")
    etfs = pd.read_csv(f"{path}/eodhd_etfs.csv", sep=";")
    cryptos = pd.read_csv(f"{path}/eodhd_cryptos.csv", sep=";")
    commodities = pd.read_csv(f"{path}/yahoo_eodhd_commodities.csv", sep=";")

    # drop some old/incorrect identified tickers which are present in the eodhd data and might otherwise get matched instead of the correct ones
    stocks = stocks[~stocks["Code"].isin(["PYPL1", # old/unused PayPal ticker
                                          "AEN", ])] # old AMC ticker]
    
    cryptos = cryptos[~cryptos["Code_clean"].isin(["LUNA"])] # 

    # stocks
    stocks = stocks.sort_values(by=["in_sp500_as_of_may_2024", "delisted_as_of_may_2024"], ascending=[False, True])
    stocks_candidates = dict(zip(stocks["Code"], stocks["Name"].apply(get_stock_basename_or_ticker)))

    # etfs
    etfs = etfs.sort_values(by=["delisted_as_of_may_2024"], ascending=[True]) # listed etfs first
    etfs_candidates_all = dict(zip(etfs["Code"], etfs["Name"].apply(get_etf_basename_or_ticker)))
    etfs_candidates_listed = {t:n for t, n in etfs_candidates_all.items() if t in etfs[~etfs["delisted_as_of_may_2024"]]["Code"].values}

    # cryptos
    cryptos = cryptos.sort_values(["in_top200_as_of_dec_2022", "delisted_as_of_may_2024"], ascending=[False, True])
    cryptos_candidates_all = dict(zip(cryptos["Code_clean"], cryptos["Name"].apply(get_crypto_basename_or_ticker)))
    cryptos_candidates_top200 = {t:n for t, n in cryptos_candidates_all.items() if t in cryptos[cryptos["in_top200_as_of_dec_2022"]]["Code_clean"].values}
    cryptos_candidates_listed = {t:n for t, n in cryptos_candidates_all.items() if t not in cryptos[cryptos["delisted_as_of_may_2024"]]["Code_clean"].values}
    
    # commodities
    # no sorting needed
    commodities_candidates_all = dict(zip(commodities["Code"], commodities["Name"].apply(get_commodity_basename_or_ticker)))
    
    # return in a dict
    return_dict = {"stocks": {"candidates_dict_all": stocks_candidates},
                    "etfs": {"candidates_dict_all": etfs_candidates_all, "candidates_dict_listed": etfs_candidates_listed},
                    "cryptos": {"candidates_dict_all": cryptos_candidates_all, "candidates_dict_top200": cryptos_candidates_top200, "candidates_dict_listed": cryptos_candidates_listed},
                    "commodities": {"candidates_dict_all": commodities_candidates_all},
                    }                
    return return_dict

"""
def get_matches(output_jsons, names_and_tickers_dfs):
    
    # prepare candidate dicts for each asset type
    stock_candidates_dict = 

    return outputs_with_matches
"""

#############################################################################################################################################

# manual matches for some common hard-to-match cases (bad youtube transcriptions, company name changes etc.)
# key: query name (cleaned), value: matched ticker
# "->" in comments indicates rename
manual_stock_match_dict = {
    "facebook": "META", # -> Meta Platforms
    "FB": "META", # -> Meta Platforms
    "fb": "META", # -> Meta Platforms
    "meta": "META", # Platforms
    "google": "GOOG", # Alphabet
    "square": "SQ", # -> Block
    "carnival cruise line": "CCL", # -> Carnival Corp
    "philip morris": "MO", # -> Altria (but Philip Morris International is still PMI)
    "coinbase": "COIN", # Global
    "disney": "DIS", # Walt Disney
    "rivian": "RIVN", # Automotive
    "united technologies": "UTX", # -> Raytheon Technologies
    "neo": "NIO", # mistranscription
    "teledoc": "TDOC", # Teladoc
    "general electric": "GE", # -> GE Aerospace
    "costco": "COST", # Wholesale
    "zoom": "ZM", # Video Communications
    "paypal": "PYPL", # Holdings
    "verizon": "VZ", # Communications
}

manual_etf_match_dict = {
    "s&p500": "SPY", # -> SPDR S&P 500 ETF Trust
    "russell": "IWM", # -> iShares Russell 2000 ETF
    "russell2000": "IWM", # -> iShares Russell 2000 ETF
    "nasdaq": "QQQ", # -> Invesco QQQ Trust
    "nasdaq100": "QQQ", # -> Invesco QQQ Trust

    "bitcoin": "GBTC", # Grayscale Bitcoin Trust

    "china": "MCHI", # iShares MSCI China ETF
    "us": "VTI", # Vanguard Total Stock Market ETF
    "europe": "VGK", # Vanguard FTSE Europe ETF
    "asia": "AAXJ", # iShares MSCI All Country Asia ex Japan ETF
    "japan": "EWJ", # iShares MSCI Japan ETF

}

manual_crypto_match_dict = {
    "ripple": "XRP",
    "ripple xrp": "XRP",
    "safemoon": "SFM", # 
    "v-chain": "VET", # VeChain
    "v chain": "VET", # VeChain
    "aether": "ETH", # Ethereum
    "aetherium": "ETH", # Ethereum
    "tazos": "XTZ", # Tezos
    "wren": "REN", # Ren
    "silica": "ZIL", # Zilliqa
    "luna": "LUNC", # Terra Luna (original, renamed from LUNA to LUNC after split but eodhd has new name)
    "luna coin": "LUNC", # Terra Luna
    "terra": "LUNC", # Terra Luna
    "terra luna": "LUNC", # Terra Luna
}

manual_commodity_match_dict = {
    "wti": "CL=F", # Crude Oil
    "crude": "CL=F", # Crude Oil
    "oil": "CL=F", # Crude Oil
    "west texas intermediate": "CL=F", # Crude Oil

    "brent": "BZ=F", # Brent Crude Oil
    "brent oil": "BZ=F", # Brent Crude Oil

    "pork": "HE=F", # Hogs
    "lean hogs": "HE=F", # Hogs
}

