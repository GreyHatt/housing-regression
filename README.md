# housing-regression

This is the final Major Assignment for MLOps

#### Report By
Shivam Bhardwaj (G24AI1088)

## Steps For Completing the Code

#### Step 1:
Commit ID: [ec1ae75](https://github.com/GreyHatt/housing-regression/commit/ec1ae75f44acdf652bc82aab8ba69f2bbc01e200)
- Create git repo with readme and git ignore files.
- Create `requirement.txt` and commited.

#### Step 2:
Commit ID: [36cd468](https://github.com/GreyHatt/housing-regression/commit/36cd468b356a4373b6f40d729b2914aec80d0f9e)
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
#### Step 3:
Commit ID: 
- Added commit ids and link to the previous steps.
- Created `tests/test_train.py` file.
- Changed a bit in `src/train.py` file to fetch the metrics.
- Run the test locally.
    - Command and O/P:
```text
(mlops) logan@Bhardwajs-MacBook-Air housing-regression % pytest tests/test_train.py  
==================================== test session starts ================================================================
platform darwin -- Python 3.11.13, pytest-8.4.1, pluggy-1.6.0
rootdir: /Users/logan/Documents/Learning & Documentation/Learning/IITJ/Trimester 3-2025/MLOps/Major Assignment/housing-regression
plugins: cov-6.2.1
collected 4 items                                                                                                                       
teststest_trainpy ....                  [100%]                                                                                          

===================== 4 passed in 4.11s ================================================================
(mlops) logan@Bhardwajs-MacBook-Air housing-regression % 
```