Note: This repository does **not contain any data files** (including scraping/asset price/intermediate results/outputs). A version with data included is available on request.

### Overview

#### ``scraping``

- Finding relevant YouTube channels through automated searches
- Downloading video metadata including transcripts
- Filtering and cleaning the data

All steps carried out in ``scraping_and_processing.ipynb``. Functions in ``scraping_utils.py`` and ``transcript_utils.py``.

#### ``LLM_information_extraction``

- Data preparation for LLM inference/finetuning: ``dataset_creation.ipynb``, ``transcript_chunking.ipynb`` with functions in  ``chunks_to_video_utils.py``, ``data_prep_utils.py``, ``LLM_utils.py``
- Finetuning, inference, and quantization notebooks: ``finetuning_and_inference_colab``
- Results postprocessing, including asset name matching and chunk recombination: ``results_processing_inf.ipynb``, ``name_matching_utils.py``
- Validation: ``results_processing_val_runs.ipynb``

#### ``analysis``

- Portfolio computation: ``portfolio_building.ipynb`` with classes/functions in  ``portfolio_utils.py``
- Computing portfolio statistics: ``portfolio_analysis.ipynb``
- Aggregate recommendation analysis: ``rec_analysis.ipynb``

#### ``data/asset_data``

- Downloading and post-processing asset price and name/ticker data: ``names_and_tickers_data_gathering_and_processing.ipynb``, ``price_data_gathering_and_processing.ipynb`` with functions in ``asset_data_utils.py``

#### ``viz``

- Creating plots and tables for thesis: ``plots_gen.ipynb``, ``tables_gen.ipynb``
