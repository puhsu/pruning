# Pruning recurrent neural network

This is a simple implementation of pruning method from https://arxiv.org/abs/1704.05119

## Running instructions
First of all setup the environment by creating virtualenv and installing all the
requirements.
```bash
# this is done via virtualenvwrapper
mkvirtuaenv pruning
pip install -r requirements.txt
# extra requirement for spacy
python -m spacy download en_core_web_sm
```
Then run `python main.py --help` to see options

## Train with pruning
In order to get q parameter, run training with `--collectq` flag, then create `your_config.yaml` file like [this](configs/base.yaml) one, and run training again with `--prune --config path/to/your/config.yaml` flags.

## Results
You can see the results [here](reports.org)
