"""
Various utility functions for preparing the LLM finetuning and inference datasets. 
"""

import re


def clean_text(text):
    """
    Cleans transcript (or any other) text to prepare it for tokenization/LLM input. 
    """
    # Note: Transcripts should already be mostly free from unwanted characters because the downloaded transcripts were processed when saving them to csv files.

    # remove unwanted characters 
    text = text.replace("\n", " ") # replace newline characters
    text = text.replace(";", ",") # replace separator (in case we want to save to csv later)
    text = text.replace('"', "") # remove double quotes

    # remove youtube-specific special characters
    text = text.replace("[\xa0__\xa0]", "") # used by youtube to censor inappropriate language

    # replace unicode characters in text (present in some older transcripts)
    text = text.replace('&amp;', '&')
    text = text.replace('&quot;', '"')
    text = text.replace('&#39;', "'")

    # remove any unusual characters (emojis, etc.) which might cause problems for the tokenizer
    # NOT SURE IF NECESSARY -> TEST!
    #text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # remove multiple whitespaces (might be introduced by removals above)
    text = re.sub(r'\s+', ' ', text)

    return text

def get_text_chunks(text, tokenizer, max_tokens_per_chunk=2048, overlap=30, return_token_ids=False):
    """
    Splits a transcript text into overlapping chunks of a maximum token (!) number. 
    The last chunk may be smaller.
    """
    # check input
    if len(text) == 0:
        print("Warning: get_text_chunks() called with empty text input.")
    if overlap >= max_tokens_per_chunk:
        raise ValueError("overlap must be smaller than max_tokens_per_chunk.")

    # tokenize transcript (without adding special tokens, such as bos, eos, etc.)
    tokenized = tokenizer(text, add_special_tokens=False)
    n_tokens = len(tokenized["input_ids"])

    # get chunks
    chunks = []
    start_idx = 0
    while True:
        if start_idx + max_tokens_per_chunk < n_tokens: # not last chunk
            end_idx = start_idx + max_tokens_per_chunk 
            chunks.append(tokenized["input_ids"][start_idx:end_idx])
            # update start index for next chunk, offset by overlap value
            start_idx = end_idx - overlap
        else: # last chunk 
            chunks.append(tokenized["input_ids"][start_idx:])
            break

    if return_token_ids:
        return chunks
    else:
        # decode chunks back to text (skipping special tokens, such as bos, eos, etc.)
        text_chunks = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]
        return text_chunks
    

