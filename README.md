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
        python -m src.train.py
        ```
    - After testing out locally below are the logs
        ```text
            {
                'mean_absolute_error': 0.5332001304956976, 
                'mean_squared_error': 0.5558915986952426, 
                'r2_score': 0.5757877060324521
            }
        ```
        ```text
        Model saved to data/trained_model.pkl
        ```

#### Step 3:
Commit ID: [bfe7fdd](https://github.com/GreyHatt/housing-regression/commit/bfe7fdda5f6bf14c35b6bf5eb6874052573d20a2)
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

#### Step 4:
Commit ID: [25bef92](https://github.com/GreyHatt/housing-regression/commit/25bef9299a512f9c549a41c4f08158aa09d6eefa)
- Created `src/quantize.py` file.
- Add `load_model` method in `src/utils.py` file.
- In `quantize.py` file
    - I have created a generic script to save params in `src/utils.py`
    - Commands to test
        ```bash
        python -m src.train
        python -m src.quantize
        ```
    - Output
        ```text
        Quantization metrics: 
            {
                'mean_absolute_error': 1.016262825917015, 
                'mean_squared_error': 1.546130459757296, 
                'r2_score': -0.17988390298792845
            }
        ```
    - However, if used in16
        ```text
        Quantization metrics: 
            {
                'mean_absolute_error': 0.5344431924510482, 
                'mean_squared_error': 0.5559100099242476, 
                'r2_score': 0.5757736560455315
            }
        ```

#### Step 5:
Commit ID: [bcaae43](https://github.com/GreyHatt/housing-regression/commit/bcaae435ae5b60b54cfb6a753fb39a99b16da798)
- Created `Dockerfile` file.
- Created `src/predict.py` file.
    - Printed sample predictions
    - Added metrics for evaluation
- Command to test it
    ```bash
    docker build -t housing-regression .
    docker run --rm housing-regression
    ```
- Output when test it locally
```text
(mlops) logan@Bhardwajs-MacBook-Air housing-regression % docker run  housing-model      
{'mean_absolute_error': 0.5332001304956985, 'mean_squared_error': 0.5558915986952427, 'r2_score': 0.575787706032452}
Model saved to data/trained_model.joblib
Model loaded from data/trained_model.joblib
Sample Outputs
Predicted: 0.7191, Actual: 0.4770
Predicted: 1.7640, Actual: 0.4580
Predicted: 2.7097, Actual: 5.0000
Predicted: 2.8389, Actual: 2.1860
Predicted: 2.6047, Actual: 2.7800
Evaluation Metrics: {'mean_absolute_error': 0.5332001304956985, 'mean_squared_error': 0.5558915986952427, 'r2_score': 0.575787706032452}
(mlops) logan@Bhardwajs-MacBook-Air housing-regression % 
```

#### Step 6:
- Created `CI/CD` workflow file.