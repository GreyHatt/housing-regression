# housing-regression

This is the final Major Assignment for MLOps

#### Report By
Shivam Bhardwaj (G24AI1088)

## Steps For Completing the Code

#### Step 1:
- Create git repo with readme and git ignore files.
- Create `requirement.txt` and commited.

#### Step 2:
- Create `config.py` to load the paths of file and other env variables.
- Created `utils.py` file with below functionalities.
    - load_datasets
    - train_and_split
    - save_model
    - evaluate_model
- Created `train.py` and test it.
- To test it created conda env locally.
    - Command
        ```bash
        conda create env --name mlops python=3.11
        conda activate mlops
        pip install -r requirements.txt
        python src/train.py
        ```
    - After testing out locally below are the logs
        ```json
            {'mean_absolute_error': 0.5332001304956976, 'mean_squared_error': 0.5558915986952426, 'r2_score': 0.5757877060324521}
        ```
        ```text
        Model saved to data/trained_model.pkl
        ```