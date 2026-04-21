## Installation
Create conda env using environment.yml



### Dataset:
The dataset is in the experiments/babyLFM2b1k/input folder. 
The dataset.inter file is the interactions. 
The demographics.tsv file is the user demographics.
The tracks.tsv file is the tracks statistics.
The files have _raw version, which is all the dataset statistics. The other version (without the _raw ending) is the data processed like the model needs it. 


### Runing the model
You can run the model by running:

`python run_loop.py -n 5 --dataset babyLFM2b1k --model BPR --choice-model consume_all --config recbole_config_default.yaml` --train-from-checkpoint

| Option                  | Description                                                                                                                                                                     |
|-------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| -n                      | Amount of loops to run. Put n=1 to run the training without simulation                                                                                                          |
| --dataset               | Experiment name (subfolder name) with dataset to evaluate                                                                                                                       |
| --model                 | `Recbole`-model to train and generate recommendations from. So far I only tested BPR                                                                                            |
| --choice-model          | User choice model to simulate acceptance with. consume_all means that the users consume all recommended items and is the best option for a low number of loops(I think)         |
| --config                | Recbole config file to be used. In that config file you can change some training details.                                                                                       |
| --train-from-checkpoint | Initialize training from the embeddings of the last iteration                                                                                                                   |
| --starting-iteration    | If you want to rerun the last run starting from a specific iteration (you can e.g. give a new choice model from that point)                                                     |


### Output
After finishing the full experiment, you'll be left with the following folder structure:
```
experiments/
    <experiment name>/
        datasets/
            iteration_1.inter     # input data at interation 1
            iteration_2.inter
            ...
            iteration_<n>.inter   # input data at interation n
            iteration_<n+1>.inter # final dataset with all new artificial interactions added
        input/
            dataset.inter         # Interactions, unmodified
            demographics.tsv      # User demographic information, unmodified
            tracks.tsv            # Music Track information, unmodified
        log/
            ...
        log_tensorboard/
            ...
        output/
            iteration_1_accepted_songs.tsv      # Accepted songs of iteration 1
            iteration_1_top_k.tsv               # Suggested recommendations of iteration 1
            iteration_1_item_embedding.pt       # item embeddings after training iteration 1
            iteration_1_user_embedding.pt       # user embeddings after training 1 iteration
            iteration_1_ndcg_per_user           # ndcg per user: main accuracy metric!!
            ...
            iteration_<n>_accepted_songs.tsv
            iteration_<n>_top_k.tsv
        params.json              # Internal file with saved simulation parameters (model, choice model, config)
```
The ndcg is the main accuracy metric used in recommendation. It will be only calculated for the first iteration and will not reapeare in the user_based_metric.csv file!!!!!

#### Datasets
The dataset folder contains the data that was used to train Recbole in a given iteration. 
**This also means that iteration_1.inter is always identical to input/dataset.inter**.
The last file in this folder was not used for training, but instead contains the resulting dataset after the
experiment finished.

#### Output
For each iteration, there are two resulting output files

**iteration_\<n\>_top_k.tsv**: Tab-Separated Tab-Separated Values file (TSV) with the following columns (with header row) detailing the best k (default k=10) recommendations per user

| Column Order | Column name | Description                                                                                     |
|--------------|-------------|-------------------------------------------------------------------------------------------------|
| 1            | user_id     | ID (row number, starting at 0) of user in demographics table                                  |
| 2            | track_id    | ID (row number, starting at 0) of track in tracks table                                         |
| 3            | rank        | rank of the recommendation. Best recommendation is always in rank 1, second best in rank 2, etc. |
| 4            | score       | Score given to the recommendation by Recbole                                                    |

This table records what the model trained by recbole has determined to be the best k items to recommend to each user.
Note that repeated recommendation is not allowed and such recommendations are filtered out beforehand.

**iteration_\<n\>_accepted.tsv**: Tab-Separated Tab-Separated Values file (TSV) with identical file structure as `dataset.inter` (see above).

This table records the subset of the top_k table that was selected and "consumed" by the user choice model. Its content
is appended to the dataset to be used in the next simulation step.

### Evaluation
To generate the metrics used in the research paper, set your experiment(s) in the `compute_all_metrics.py` script 
and run it as a result, 3 new files will appear in the experiment folder

#### baselines.csv: Comma-separated values file containing the following columns
| Column Order | Column name | Description                                                                                             |
|--------------|-------------|---------------------------------------------------------------------------------------------------------|
| 1            | country     | Country Code of users                                                                                   |
| 2            | us          | Percentage of interactions of users from `country` with music tracks originating from the US            |
| 3            | local       | Percentage of interactions of users from `country` with music tracks originating from their own country |
| 4            | other       | Percentage of interactions of users from `country` neither coming from the US nor their own country     |

#### user_based_metrics.csv Comma-separated values file containing the following columns
This table contains information about the recommendations of each user at each simulation step

| Column name                  | Description                                                                               |
|------------------------------|-------------------------------------------------------------------------------------------|
| user_id                      | ID (row number, starting at 0) of user in demographics table                              |
| model                        | Recbole model used to generate recommendations                                            |
| choice_model                 | Choice model used to simulate user behavior                                               |
| iteration                    | Simulation iteration                                                                      |
| country                      | Country Code of user                                                                      |
| user_count                   | Amount of users originating from this country                                             |
| jsd                          | Artist country JSD divergence - original input file vs. recommendations at this iteration |
| interaction_jsd              | Artist country JSD divergence - original input file vs. input dataset at this iteration   |
| jsd_sumamrized               | like `jsd`, but countries are grouped into US/Local/Other                                 |
| interaction_jsd_sumamrized   | like `interaction_jsd`, but countries are grouped into US/Local/Other                     |
| bin_jsd                      | Popularity bin JSD divergence - original input file vs. recommendations at this iteration |
| interaction_bin_jsd          | Popularity bin JSD divergence - original input file vs. input dataset at this iteration   |
| us_proportion                | Percentage at this iteration of recommendations for tracks originating in US              |
| us_interaction_proportion    | Percentage of input data at this iteration for tracks originating in US                   |
| local_proportion             | Percentage of recommendations for tracks originating in user country                      |
| local_interaction_proportion | Percentage of input data at this iteration for tracks originating in user country         |
| male_proportion              | Percentage of recommendations for tracks by male artists                                  |
| male_interaction_proportion  | Percentage of input data at this iteration for tracks by male artists                     |

#### metrics.csv
The metrics file contains the same information as `user_based_metrics`, but grouped by user country and iteration.
It is useful to compare the RecSys behaviour between different demographic groups.


