# Overview of SemDisc
This repository contains the code for `Qualitative Join Discovery in Data Lakes using Examples (SIGMOD 2026)`.

SemDisc is an end-to-end join discovery system that supports a QbE (Query-by-Example) interface for join discovery. SemDisc is divided into two phases. In the offline processing stage- SemDisc builds the indexes including the join graph and the Inverted Join Path Index. In the online phase, SemDisc takes a query table as input and outputs a joined table by searching for the optimal join path from the data lake.

We have tested the code on a CHPC machine with a linux OS, 400 GB RAM, 40 2.75 GHz AMD EPYC 9454 CPU cores, and an NVIDIA A800 GPU. We will update the code to run on MacOS and Windows soon. The code ships with data lake and queries of DrugCentral for testing initial setups. You will need to create an openai API key in order to run the code.

# Install environment
* Open a bash terminal and make sure that `semdisc-code` is the active directory after cloning. Make sure `conda` command is available system-wide.
* `conda create --prefix ./semdiscenv python=3.9`
* `conda activate ./semdiscenv`
* `cat requirements.txt | xargs -n 1 pip install`

# Run the Offline Stage
* Activate conda environment `conda activate ./semdiscenv`
* Create an openai API key and put the following line in `./custom_lib/dotenv/.env`:

`OPENAI_API_KEY=<your_openai_key>`

* Run `python -m semdisc.stages.offline_stage --startstep 3 --endstep 12 --datalake drugcentral`

# Run the Online Stage
* Run `python -m semdisc.stages.online_stage --datalake drugcentral` 

The following output should appear for the default query table given with the code:

```
joined table shape:  (685, 9)
Precision@5 for finding all examples: 1.0
```

# Handling New Data Lakes
* Create new directory `./data/all_datalakes/<new_datalake_name>/uploaded_datasets` and put all data lake CSV files in that directory.
* Create new file `./data/all_datalakes/<new_datalake_name>/query_tables.json` and create as many query tables as you need with the similar structure of `./query_tables.json` file in root folder.
* Set hyperparameters in `./semdisc/stages/datalake_config.py`. Adjust hyperparameters if needed.
```
    "<new_datalake_name>":{
        'simhash_size': 18,
        'join_edge_threshold': 0.9,
        'semantic_type_similarity_threshold': 0.9,
        'number_of_hashes_for_minhash': 128,
        'diversity_multiplier_threshold': 0.4,
        'simple_path_max_length': 5,
        'normalize_embeddings': True,
        'diversity_enabled': True
    }
```
* Run `python -m semdisc.stages.offline_stage --startstep 2 --endstep 12 --datalake <new_datalake_name>`
* Run `python -m semdisc.stages.online_stage --datalake <new_datalake_name>` to find join paths for the query tables you specified.

