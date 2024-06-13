# Suncorp Take Home Assignment

## Problem Statement
Our claims handlers spend a lot of time going through claim characteristics and inputting cost estimates, which are crucial for managing our reserves but are currently inaccurate and inconsistent, requiring frequent adjustments. Additionally, identifying complex claims and triaging them to the appropriate handlers is challenging, so early detection and proper management of these claims would greatly improve efficiency.

## Objective
1. Develop a model to predict the claim amount for each claim.
2. Develop a method to detect claims that are potentially complex and should be triaged to the appropriate team.

## Project Structure
```bash
.
├── README.md
├── data
│   ├── data.csv
|── src
|   ├── data_processing.py
|   ├── model.py
|   ├── main.ipynb
|   ├── utils.py
|── models
|   ├── best_xgb_model.pkl
|── requirements.txt
|── poetry.lock
|── poetry.toml
```

- data_processing.py: Contains the code to preprocess the data.

- model.py: Contains the code to train the model.

- main.ipynb: Jupyter notebook containing the code to preprocess the data, train the model and evaluate the model.

## Reproducing the Results
Python version: 3.11.6

If you have poetry installed, you can create a virtual environment and install the dependencies using the following commands:
```bash
poetry install
```
You can run spin up a jupyter notebook by running the following command:
```bash
poetry run jupyter notebook
```
Or you can run the code from VSCode or any other IDE.

Instruction to install poetry can be found [here](https://python-poetry.org/docs/). (poetry 1.7.1)

if you don't have poetry ready, you can install the dependencies using pip:
```bash
pip install -r requirements.txt
```

