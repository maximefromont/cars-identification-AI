![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

# cars-identification-AI

## How to install

1. Clone this repository
2. Install requirements

```bash
pip install -r requirements.txt
```

## How get the data and sorted the data

To get the dataset necessary to trained the model you can use the `initialize.py`

Or un separately the scripts : 

```
get-dataset-from-kaggle.py
```

```
sort_dataset_into_brand.py
```

```
sort_dataset_into_folder.py
```

/!\ You need to get the dataset before to run the sort scripts.

## How use the brand model classifier

### main utilisation

If you just want to build a trained model, you can use the script : `build-specific-model.py`

The script will ask wich model use and the number of necessary epochs

That will build the model in the model file, you will also find in the plot folder a resume of the model data , and a resume of this data in the model-data folder in acsv file.

### detailled script

#### brand_model script

This script contain different function, whose permit to build the trained model.

#### brand-model-comparator script

This script comparate different trained model.

#### sort_dataset_into_brand script

This script sort the initial dataset to a new one to trained the brans models.

#### test-brand-model.py 

This script is used to test a model with different picture from the test-dataset.
