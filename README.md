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

`python run_loop.py -n 5 --dataset babyLFM2b1k --model BPR --choice-model consume_all --config recbole_config_default.yaml`

| Option         | Description                                                                                                                                                                     |
|----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| -n             | Amount of loops to run                                                                                                                                                          |
| --dataset      | Experiment name (subfolder name) with dataset to evaluate                                                                                                                       |
| --model        | `Recbole`-model to train and generate recommendations from. So far I only tested BPR                                                                                            |
| --choice-model | User choice model to simulate acceptance with. consume_all means that the users consume all recommended items and is the best option for a low number of loops(I think)         |
| --config       | Recbole config file to be used. In that config file you can change some training details.                          




