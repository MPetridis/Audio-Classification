# Audio Classification

Download [FSDKaggle2018](https://zenodo.org/records/2552860/files/FSDKaggle2018.audio_train.zip?download=1) audio files and [Csv File](https://zenodo.org/records/2552860/files/FSDKaggle2018.meta.zip?download=1) ` train_post_competition.csv`

and replace:
in `main.py`

- `dataset_csv` : path to csv file
- `root_dir at` : path to audio dir

if `./partitions` exist and has no csv files in it delete the dir and run `main.py` after

### Requires Python 3.11
