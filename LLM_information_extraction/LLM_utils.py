import re

# all-in-one function: supports different model formats and can produce finetuning as well as inference prompts
# important: this function defines the entire prompt around the provided transcript chunk in the feature_row of the dataset. 
def format_prompt(feature_row, prompt_format, include_answer_tease=True, include_label=False, include_bos=True, include_eos=True):

    # reject invalid parameters
    if include_label and 'label' not in feature_row:
        raise ValueError("Cannot include label in prompt if label is not present in feature_row.")
    if prompt_format not in ["mistral", "llama3", "plain"]:
        raise ValueError("Invalid prompt format. Please choose one of 'mistral', 'llama3', 'plain'.")
    
    ### define prompt parts 
    # tags are only mentioned if they are available
    tags_part = "" if not feature_row['first_three_tags'] else f"The top tags for the video are: '{feature_row['first_three_tags']}'. "
    pre_chunk_instruction = f"The triple-quoted text below is part of a youtube video transcript by channel {feature_row['uploader_id']} with the title '{feature_row['title']}'. {tags_part}Read the transcript carefully in order to perform the asset name extraction task specified below the transcript."
    json_format_part = '[{"asset_name": "name of asset 1", "asset_type": "stock/etf/crypto/commodity/other", "sentiment": "buy/sell/hold"}, "asset_name": "name of asset 2", "asset_type": "...", "sentiment": "..."}, ...]'
    post_chunk_instruction = f"""Your task is to extract trade recommendations from the text, if it contains any. Use the transcript content and the information provided before the transcript to determine whether the video is actually about financial topics and contains concrete trading/investment recommendations. If it does not, simply return an empty list. Please return a json list with an object for each mentioned asset, including the following fields:\n\n- asset name: name of the mentioned asset (e.g. company name). For etfs (=any fund or index) provide either a ticker, index name, sector name or country/region name. Some names might be mistranscribed in the text, in which case you should infer the correct name from the context.\n- asset type: either "stock", "crypto", "etf", "commodity" or "other" if none of the previous apply\n- sentiment: corresponding recommendation, according to the speaker's sentiment in the transcript (positive -> "buy", neutral -> "neutral", negative -> "sell").\n\nThe output MUST follow this exact format: {json_format_part}"""
    
    # assemble main prompt message
    prompt_msg = f"{pre_chunk_instruction}\n\n\"\"\"{feature_row['chunk_text']}\"\"\"\n\n{post_chunk_instruction}"

    # optional answer tease, to be inserted at the beginning of the model response part!
    answer_tease = "Sure, here's the json list with extracted trade recommendations:"

    ### build prompt with model-specific format
    if prompt_format == "mistral":
        # mistral format: "<s>[INST] Instruction [/INST] Answer</s>"
        # note: no system prompt!
        prompt = f"""[INST] {prompt_msg} [/INST] """ # trailing space on purpose
        if include_bos:
            prompt = f"<s>{prompt}"
        if include_answer_tease:
            prompt = f"{prompt}{answer_tease}\n\n"
        if include_label:
            prompt = f"{prompt}{feature_row['label']}"
        if include_eos:
            prompt = f"{prompt}</s>"

    elif prompt_format == "llama3":
        # llama3 instruct format: see https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/
        # note: allows system prompt (and it is recommended to use it)
        system_prompt = "You are a smart and efficient assistant specialized at extracting relevant information from text and replying in json format. You always follow the user's instructions carefully."
        prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        if include_bos:
            prompt = f"<|begin_of_text|>{prompt}"
        if include_answer_tease:
            prompt = f"{prompt}{answer_tease}\n\n" # whether we have a whitespace, newline, etc. here is important for finetuning! Also handled differently by different tokenizers...
        if include_label:
            prompt = f"{prompt}{feature_row['label']}"
        if include_eos:
            prompt = f"{prompt}<|eot_id|>" # note: the eos token always appears after system and user prompts, we only make it optional at the end of the assistant prompt!
    
    elif prompt_format == "plain":
        # no special tokens added
        prompt = prompt_msg
        if include_answer_tease:
            prompt = f"{prompt}\n\n{answer_tease}\n\n "
        if include_label:
            prompt = f"{prompt}{feature_row['label']}"

    feature_row['prompt'] = prompt
    return feature_row

# function to extract the json part from the model response (raising error if none to be found)
def extract_json_from_output(text): 
    pattern = r"\[.*?\]" # should match the first occurrence of square brackets and everything in between
    matches = re.findall(pattern, text, re.DOTALL) # use re.DOTALL to match newlines as well
    if matches:
        return matches[0]
    else:
        raise ValueError(f"Could not find json output in model response: '{text}'")



# json schema definition for constrained output generation (using e.g. outlines)
output_json_schema_string = '''
{
  "type": "array",
  "items": {
    "type": "object",
    "required": ["asset_name", "asset_type", "sentiment"],
    "properties": {
      "asset_name": {"type": "string", "minLength": 1, "maxLength": 100},
      "asset_type": {"type": "string", "enum": ["crypto", "stock", "etf", "commodity", "other"]},
      "sentiment": {"type": "string", "enum": ["buy", "sell", "neutral"]}
    }
  }
}'''