"""
Contains various helpful functions for the data gathering process,
mostly related to cleaning and filtering csv and infojson files from the yt-dlp calls. 

# note: some functions manipulate files and/or directories -> consider potential directory issues when importing this module.

"""

import os
import pandas as pd
import json



def rename_infojsons(infojsons_dir):
    """
    Fixes format of .info.json filenames in the given directory by removing the .{ext}_info. part. 
    """
    for filename in os.listdir(infojsons_dir):
        if filename.endswith(".info.json"):
            if ".webm_info." in filename:
                os.rename(f"{infojsons_dir}/{filename}", f"{infojsons_dir}/{'.'.join(filename.split('.webm_info.'))}")
            elif ".mkv_info." in filename:
                os.rename(f"{infojsons_dir}/{filename}", f"{infojsons_dir}/{'.'.join(filename.split('.mkv_info.'))}")
            elif ".mp4_info." in filename:
                os.rename(f"{infojsons_dir}/{filename}", f"{infojsons_dir}/{'.'.join(filename.split('.mp4_info.'))}")
            else:
                print(f"could not rename file: {filename}")

def load_channel_search_results(channel_search_dir, query_types_to_include, return_infojsons=False):
    # some lines contain the separator in tags -> fix by passing passing custom function to read_csv (only applied to 'bad' lines)
    def line_fix_seps_in_tags(line):
        # reconstruct line
        line = ";".join(line)
        # try tag fix (remove semicolons from tags list)
        try:
            a, b = line.split(f";[")
            b, c = b.split(f"];")
            b = b.replace(";", "")
            line = f"{a};[{b}];{c}"
            if len(line.split(";")) == 9: # number of expected fields
                print(f"fixed line (semicolons in tags): {line}")
                return line.split(";")
            else:
                raise Exception
        except:
            print(f"could not fix line, skipping: {line}")
            return None # to make pandas skip this line

    # read in searchresults csvs, create query_type col and join into one df
    dfs = []
    for query_type in query_types_to_include:
        df = pd.read_csv(f"{channel_search_dir}/searchresults_{query_type}.csv", 
                            sep=";", 
                            header=0, 
                            on_bad_lines=line_fix_seps_in_tags, 
                            engine="python")
        df['query_type'] = query_type
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    # read in info jsons and add some fields to the df
    info_jsons = {}

    # new fields (note: should have the exact same name as in the info jsons)
    df['view_count'] = None
    df['channel_follower_count'] = None


    for idx, row in df.iterrows():
        if row['video_id'] not in info_jsons.keys(): # duplicate video ids might still be present at this point
            # read in info json
            filepath = f"{channel_search_dir}/info_jsons/{row['uploader_id']}_{row['video_id']}.info.json"
            if not os.path.exists(filepath):
                print(f"Could not find file: {filepath}")
            else:
                with open(filepath, 'r', encoding='utf-8') as f: # encoding='utf-8' to avoid UnicodeDecodeError
                    info_jsons[row['video_id']] = json.load(f)
            
                # add new field values (if present in info json)
                for field in ['view_count', 'channel_follower_count']:
                    if field in info_jsons[row['video_id']].keys():
                        df.loc[idx, field] = info_jsons[row['video_id']][field]
                    else:
                        print(f"{field} missing in: {filepath}")
    
    # dtype conversions
    df['upload_date'] = pd.to_datetime(df['upload_date'], format="%Y%m%d")
    # convert to int while allowing na values
    df['duration'] = pd.to_numeric(df['duration'])
    df['view_count'] = pd.to_numeric(df['view_count'])


    print("-"*60)
    print(f"Read in {len(df)} videos (query categories: {df['query_type'].unique()})")
    print(f"Missing value counts:\n{df.isna().sum()}")

    return df if not return_infojsons else (df, info_jsons)

def filter_search_results(df, 
                          query_types_to_include, 
                          langs_to_include, 
                          max_duration, 
                          min_view_count, 
                          min_channel_follower_count, 
                          return_intermediate_df=False):
    """
    Filters the given df (containing search results) according to given criteria, prints intermediate filtering results and finally returns 
    df with unique channels for whom at least one video passed all filters.
    Optionally returns intermediate df (after filtering, before collapsing to unique channels).
    """
    print("-"*60)
    print("Applying filters to search results...")
    print("-"*60)
    print(f"{len(df)} initial videos")

    df = df.drop_duplicates(subset=["video_id"])
    print(f"{len(df)} unique videos (before any filters)")
    print(f"{df['uploader_id'].nunique()} unique channels (before any filters)\n")

    df = df[df['language'].isin(langs_to_include)]
    print(f"{len(df)} unique videos after language filter")
    print(f"{df['uploader_id'].nunique()} unique channels after language filter\n")

    df = df[df['duration'] <= max_duration]
    print(f"{len(df)} unique videos after duration filter")
    print(f"{df['uploader_id'].nunique()} unique channels after duration filter\n")

    df = df[df['view_count'] >= min_view_count]
    print(f"{len(df)} unique videos after view count filter")
    print(f"{df['uploader_id'].nunique()} unique channels after view count filter\n")

    df = df[df.channel_follower_count >= min_channel_follower_count]
    print(f"{len(df)} unique videos after follower count filter")
    print(f"{df['uploader_id'].nunique()} unique channels after follower count filter\n")

    ## step 3: obtain unique channels from data (while keeping query and query category info, as rough channel topic indicators)

    # add count column (to represent number of times a channel appeared in the searches)
    df['n_results'] = df.groupby('uploader_id')['uploader_id'].transform('count')
    # collapse query and query_types columns
    df[['query', 'query_type']] = df.groupby('uploader_id')[['query', 'query_type']].transform(lambda x: ','.join(set(x)))

    channels = df.drop_duplicates(subset=['uploader_id'])[['uploader_id', 'n_results', 'query', 'query_type']]
    channels = channels.sort_values(by='n_results', ascending=False)
    channels = channels.rename(columns={'query': 'queries', 'query_type': 'query_types'}).reset_index(drop=True)
    print("-"*60)
    print(f"Final number of unique channels after filtering: {len(channels)}")

    return channels if not return_intermediate_df else (channels, df)

def fix_channel_playlist_csvs(old_dir, new_dir, return_faulty_lines=False):
    """
    There are two potential issues with lines in the csvs:
        1. some lines have too many fields because the title (last field) contains the separator, i.e. a semicolon
            -> fix by replacing semicolons in the title with commas

        2. some lines have too few fields because the title (last field) of the previous line contains a newline character (or other weirdly coded characters)
            -> fix by simply eliminating the faulty line (only a handful of occurences, and title is not that important anyway)
    """
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    faulty_lines = []
    files_skipped = []

    for filename in os.listdir(old_dir):
        if filename.startswith("video_list_") and filename.endswith(".csv"): # just for safety, but there shouldn't be other files in this folder
            if os.path.isfile(f"{new_dir}/{filename}"): # skip file if it already exists in new folder
                files_skipped.append(filename)
            else:
                # read in lines from old file
                with open(f"{old_dir}/{filename}", "r", encoding="utf-8") as f:
                    lines = f.readlines()

                # write lines to new file (with fixes if necessary)
                with open(f"{new_dir}/{filename}", "w", encoding="utf-8") as f:
                    for line_idx, line in enumerate(lines):
                        
                        if len(line.split(";")) < 7: # too few fields -> skip line
                            pass 
                            faulty_lines.append(['skipped', filename, line_idx, line])
                        elif len(line.split(";")) > 7: # too many fields -> title fix (replace semicolons with commas)
                            fields = line.split(";", 6)
                            fixed_title = fields[6].replace(";", ",")
                            fixed_line = ";".join(fields[:6] + [fixed_title])
                            f.write(fixed_line)
                            faulty_lines.append(['title fix', filename, line_idx, line, fixed_line])
                        else: # correct number of fields -> no changes
                            f.write(line)

    print("-"*60)
    print(f"number of files skipped because they already exist in new folder: {len(files_skipped)}")
    print(f"number of faulty lines: {len(faulty_lines)}")
    print(f"number of lines skipped: {len([faulty_line for faulty_line in faulty_lines if faulty_line[0] == 'skipped'])}")
    print(f"number of titles fixed: {len([faulty_line for faulty_line in faulty_lines if faulty_line[0] == 'title fix'])}")

    if return_faulty_lines:
        return faulty_lines

def load_channel_playlist_csvs(channel_playlist_dir):
    """
    Load channel playlist csvs in the given dir into a single pandas df.
    """
    # read in all files in new dir and concatenate them into one df
    # note: this already serves as a check for correct csv structure, pandas will throw an error otherwise
    dfs = []
    for filename in os.listdir(channel_playlist_dir):
        if filename.startswith("video_list_") and filename.endswith(".csv"):
            df = pd.read_csv(f"{channel_playlist_dir}/{filename}", 
                                    sep=";", 
                                    header=0, 
                                    index_col=False, 
                                    quoting=3) # to deal with titles containing quotes
            # check if empty
            if len(df) == 0:
                print(f"empty file: {filename}")
            # check for rows which are all na in df
            else: 
                dfs.append(df)
            # check for na rows
            if len(df.dropna(how="all")) != len(df):
                print(f"{len(df) - len(df.dropna(how='all'))} full na row(s) present in: {filename}")
            
    df = pd.concat(dfs, axis=0, ignore_index=True)
    print("-"*60)
    print(f"Read in {len(df)} videos from {df['channel_id'].nunique()} channels.")

    # dtype conversions
    df['approx_upload_date'] = pd.to_datetime(df['approx_upload_date'], format="%Y%m%d")
    df['duration'] = pd.to_numeric(df['duration'])
    df['view_count'] = pd.to_numeric(df['view_count'])

    return df

def check_channel_playlist_df(df):
    # check dtypes
    print("-"*60)
    print(f"data types:\n{df.dtypes}")
    # check na counts
    print("-"*60)
    print(f"n videos with na values: {df[df.yt_video_type == 'video'].isna().any(axis=1).sum()}/{len(df[df.yt_video_type == 'video'])}")
    print(f"n shorts with na values: {df[df.yt_video_type == 'short'].isna().any(axis=1).sum()}/{len(df[df.yt_video_type == 'short'])} (duration and upload date aren't available for shorts)")

def filter_channel_playlist_df( df, 
                                max_duration, # (sec), to filter out super long-form podcasts etc.
                                min_view_count,
                                min_upload_date,
                                max_upload_date):
    """
    Filters the given df (1 row = 1 video) according to given criteria, prints intermediate filtering results and returns filtered df.
    """

    print("-"*60)
    print(f"number of videos before any filters: {len(df)}")
    df = df[(df.yt_video_type == "short") | (df.duration <= max_duration)]
    print(f"number of videos after duration filter: {len(df)}")

    df = df[df.view_count >= min_view_count] # note: shorts should have view count available
    print(f"number of videos after view count filter: {len(df)}")

    # upload timeframe (only apply to videos, not shorts) (note: dates are only approximate! But accurate to a month up until one year ago.)
    df = df[(df.yt_video_type == "short") | ((df.approx_upload_date >= min_upload_date) & (df.approx_upload_date <= max_upload_date))]
    print(f"number of videos after upload date filter: {len(df)}")
    print(f"shorts/normal videos: {len(df[df.yt_video_type == 'short'])}/{len(df[df.yt_video_type == 'video'])}")
    
    return df

def add_infojson_fields(df, fields, infojsons_dir, tags_as_string=False, print_missing_fields=True):
    """
    Add given fields from info jsons to df, if they exist.
    df must have a column 'video_id' and a column 'uploader_id'.
    """
    # check if any of the requested fields are already present in the df
    for field in fields:
        if field in df.columns:
            raise ValueError(f"Field '{field}' already present in df.")
    
    # ensure rangeindex
    if not (isinstance(df.index, pd.RangeIndex) and df.index.start == 0 and df.index.step == 1):
        print("Resetting df index.")
        df = df.reset_index(drop=True)
    
    # start message
    print("-"*60)
    print(f"Adding fields from info jsons to df (nrows: {len(df)}, fields: {fields})...")

    # intermediate lists
    res = {}
    for field in fields:
        res[field] = []

    # iterate over df and add field values from info jsons and store in intermediate lists
    for idx in range(len(df)):
        filepath = f"{infojsons_dir}/{df.at[idx, 'uploader_id']}_{df.at[idx, 'video_id']}.info.json"
        if not os.path.exists(filepath):
            print(f"Could not find file: {filepath}")
            for field in fields:
                res[field].append(None)
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                info_json = json.load(f)
            for field in fields:
                if field in info_json.keys():
                    # store in intermediate lists
                    res[field].append(info_json[field])
                else: 
                    res[field].append(None)
                    if print_missing_fields:
                        print(f"field '{field}' missing in: {filepath}")

        # print progress message every x rows
        if (idx+1) % 2500 == 0:
            print(f"{idx+1}/{len(df)} rows processed")

    print("adding intermediate lists to df...")
    # add intermediate lists to df
    for field in fields:
        df[field] = res[field]
    del res

    print("postprocessing...")
    if "description" in fields:
        # clean/remove special characters
        df["description"] = df["description"].apply(lambda x: None if not x else x.replace("\n", "\\n").replace("\r", "").replace(";", "").replace(",", ""))

    if "tags" in fields:
        if tags_as_string:
            # convert list to string and clean tags
            df["tags"] = df["tags"].apply(lambda x: None if not x else ",".join([tag.replace(";", "").replace(",", "") for tag in x]))
        else:
            # just clean tag list
            df["tags"] = df["tags"].apply(lambda x: None if not x else [tag.replace(";", "").replace(",", "") for tag in x])

    if "categories" in fields:
        # clean category list
        df["categories"] = df["categories"].apply(lambda x: None if not x else [cat.replace(";", "").replace(",", "") for cat in x])

    if "upload_date" in fields:
        # correct date format
        df["upload_date"] = pd.to_datetime(df["upload_date"], format="%Y%m%d")
    
    if "chapters" in fields: 
        # clean chapter titles
        df["chapters"] = df["chapters"].apply(lambda x: None if not x else json.dumps(x).replace(";", ""))


    return df
    