"""This module contains various functions for cleanning and transforming transcript data."""

import csv
import pandas as pd
import re
import os


def ttml_extract_lines(ttml_str, fix_youtube_timestamps=False):
    """
    Extracts lines of text and timestamps from a raw ttml document string to a list to be further processed.
    The output list contains tuples of the form (start, end, text). All values are strings.
    """

    # pattern for identifying lines with transcript text (i.e. begin="00:00:00.000" end="00:00:00.000")
    pattern = re.compile(r'begin="\d{2}:\d{2}:\d{2}.\d{3}" end="\d{2}:\d{2}:\d{2}.\d{3}"')
    # get lines which contain transcript text
    lines = [line for line in ttml_str.split('\n') if pattern.search(line)]

    # extract timestamps and text from transcript lines
    transcript = []
    for line in lines:
        # extract timestamps
        begin = line.split('"')[1]
        end = line.split('"')[3]
        # extract text)
        text = line.split('>')[1].split('<')[0]

        # replace unicode characters in text (present in some older transcripts)
        text = text.replace('&amp;', '&')
        text = text.replace('&quot;', '"')
        text = text.replace('&#39;', "'")

        # replace separator token (to avoid csv problems later)
        text = text.replace(';', ',')

        # append to transcript array
        transcript.append([begin, end, text])


    if fix_youtube_timestamps:
        # true 'end' time is 'start' time of next line
        for i in range(len(transcript)-1):
            transcript[i][1] = transcript[i+1][0]

    return transcript

def vtt_extract_lines(vtt_str):
    """
    Extracts lines of text and timestamps from a raw vtt document string to a list to be further processed.
    The output list contains tuples of the form (start, end, text). All values are strings.

    Note: This function can't deal with youtube .vtt transcripts, as they contain duplicated lines. For youtube, use ttml format instead.
    Edit: It seems that most Youtube .vtts are actually normal, i.e. no duplicated lines. The function is thus applicable for YT .vtts as well.
    """

    # get lines which contain transcript text

    # re pattern for identifying lines with timestamps (e.g. '00:00:00.000 --> 00:00:00.000')
    pattern = re.compile(r'\d{2}:\d{2}:\d{2}.\d{3} --> \d{2}:\d{2}:\d{2}.\d{3}')

    lines = [line for line in vtt_str.split('\n')]

    # extract timestamps and text from transcript lines
    transcript = []

    in_block = False # flag to indicate if we are currently in a block of text (transcripts lines in vtt can be one or more per timestamp)
    block_lines = [] # list to store lines of text in current block
    for idx, line in enumerate(lines):
        # check if timestamp line
        if not in_block and pattern.match(line):
            in_block = True # indicate 
            # extract timestamps
            begin = line.split(' --> ')[0]
            end = line.split(' --> ')[1]
            
        elif in_block:
            if line == '': # q: does every vtt file have a blank line at the very end? if not, adjustment needed here!
                in_block = False
                transcript.append((begin, end, ' '.join(block_lines)))
                block_lines = []
            else:
                # line cleaning
                # replace unicode characters in text (present in some older transcripts)
                line = line.replace('&amp;', '&')
                line = line.replace('&quot;', '"')
                line = line.replace('&#39;', "'")

                # replace separator token (to avoid csv problems later)
                line = line.replace(';', ',')

                block_lines.append(line)

    return transcript

def lines_to_csv(lines, csv_path):
    """
    Saves a list of lines to a csv file.
    File headers: start, end, text
    """
    # if file already exists, abort
    if os.path.exists(csv_path):
        raise FileExistsError(f"File already exists: ' + csv_path)")

    # create new csv file
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        # header row
        writer.writerow(['start', 'end', 'text'])
        # transcript lines
        for line in lines:
            writer.writerow(line)

def csv_to_lines(csv_path):
    """
    Reads a csv file and returns a list of transcript lines (line = tuple of strings: (start, end, text)).
    """
    with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        # skip header row
        next(reader)
        # read transcript lines
        lines = [tuple(line) for line in reader]
    return lines

def lines_to_df(lines):
    """
    Converts a list of transcript lines to a pandas dataframe.
    """

    df = pd.DataFrame(lines, columns=['start', 'end', 'text'])

    # fix column types
    # note: We use datetime to allow for time calculations. Year, month, and day are meaningless here (but required by datetime).
    df['start'] = pd.to_datetime(df['start'], format='%H:%M:%S.%f')
    df['end'] = pd.to_datetime(df['end'], format='%H:%M:%S.%f')
    df['text'] = df['text'].astype(str)

    return df

def extract_entire_dir(old_dir, new_dir, clean_filenames=False):
    """Extracts all transcript files (.vtt or .ttml) from a directory and saves them as csv files in a new directory."""
    # create new directory
    if os.path.exists(new_dir):
        raise FileExistsError(f"Directory already exists: ' + new_dir)")
    else:
        os.makedirs(new_dir)

    # get filenames of transcript files to process
    filenames = [filename for filename in os.listdir(old_dir) if filename.endswith('.vtt') or filename.endswith('.ttml')]
    print(f"Beginning extraction for {len(filenames)} transcript files from {old_dir} to {new_dir}...")

    # iterate over files in old directory
    n_processed = 0
    for filename in filenames:
        file_ending = filename.split('.')[-1]
        if file_ending == 'vtt':
            # read vtt file
            with open(os.path.join(old_dir, filename), 'r', encoding='utf-8') as file:
                vtt_str = file.read()
            # extract lines
            lines = vtt_extract_lines(vtt_str)
        elif file_ending == 'ttml':
            # read ttml file
            with open(os.path.join(old_dir, filename), 'r', encoding='utf-8') as file:
                ttml_str = file.read()
            # extract lines
            lines = ttml_extract_lines(ttml_str)

        # new filename
        if clean_filenames:
            new_filename = filename.replace(f"_subs.en.{file_ending}", ".csv")
        else:
            new_filename = filename[:-len(file_ending)] + ".csv"

        # save to new file
        lines_to_csv(lines, os.path.join(new_dir, new_filename))
        n_processed += 1

        # print progress (every 10%)
        if n_processed % (len(filenames)//10) == 0:
            print(f"{n_processed}/{len(filenames)} files processed.")
        
    print("-"*60)
    print(f"Extraction complete. {len(os.listdir(new_dir))} files saved to {new_dir}.")