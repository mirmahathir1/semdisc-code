# Overview of SemDisc
SemDisc is an end-to-end join discovery system that supports a QbE (Query-by-Example) interface for join discovery. SemDisc is divided into two phases. In the offline processing stage- SemDisc builds the indexes including the join graph and the Inverted Join Path Index. In the online phase, SemDisc takes a query table as input and outputs a joined table by searching for the optimal join path from the data lake.

# Data Lake Download Link
https://drive.google.com/drive/folders/1h_0oKvPW_IksQgJ9tu8CsV2wMh4ikJy7?usp=drive_link

# Install environment
* Open a bash terminal.
* `conda create --prefix ./semdiscenv python=3.9`
* `conda activate ./semdiscenv`
* `cat requirements.txt | xargs -n 1 pip install`

# Run the Offline Stage
* Create directory `./data/all_datalakes/drugcentral/uploaded_datasets` and paste all .csv files of the DrugCentral data lake.
* Activate conda environment `conda activate ./semdiscenv`
* Create an openai API key and put the following line in `./custom_lib/dotenv/.env`:

`OPENAI_API_KEY=<your_openai_key>`

* Run `python -m semdisc.stages.offline_stage --startstep 3 --endstep 12 --datalake drugcentral`

# Run the Online Stage
* Copy `./query_tables.json` to folder `./data/all_datalakes/drugcentral/`
* Run `python -m semdisc.stages.online_stage --datalake drugcentral` 
* For any custom query table, clear the `query_tables.json` and add a query table as follows with your own semantic types and examples:
```
[
    {
        "query": [
            {
                "semantic_type": "Gene Symbol",
                "examples": [
                    "IL23A",
                    "SCN5A",
                    "GABRA1|GABRG2|GABRB2",
                    "GART",
                    "FCGR1A"
                ]
            },
            {
                "semantic_type": "Activity Measurement Value",
                "examples": [
                    "11.0",
                    "5.17",
                    "5.7",
                    "6.42",
                    "8.15"
                ]
            }
        ],
        "query_id": "0"
    }
]
```

The following output should appear for the default query table given with the code:

```
joined table shape:  (685, 9)
Precision@5 for finding all examples: 1.0
```