# Online Retail, Customer Churn Prediction

This repo predicts customer churn (90-day inactivity) from the UCI Online Retail dataset using time aware feature/label windows. It includes RFM, lifecycle features, leakage-safe splits, and three models (LogReg, RandomForest, XGBoost). Note: The data folder is not included in repo, create a data folder and add subfolders (cleaned, processed, raw) and adjust paths in files if needed.

## Repository Structure:
1- data folder: contains cleaned and raw data for training models and predicting churn.
2- models folder: conatins the rf_best.pkl file, created when we run train_model.py
3- pipelines folder: contains .py scripts for cleaning and transforming data for training and prediction.
4- reports folder: stores all outputs, results, and visuals generated after analysis or model training.
5- src folder: contains the all modular code used across the project.
5.5- app: contains py code for UI based app for this project.
6- .venv: the environemnt created.
7- predict.py: file that runs, inputs data and provides output churn predicted excel or csv file.
8- requiremnts.txt: contains all dependencies required for running scripts.
9- train_model.py: file that trains our model on the online retail sales data.

## How to create and train the model:
0- open the churn prediction model folder in any IDE like VS code

1- Create an environment, run following command in terminal :

python -m venv .venv
.venv\Scripts\Activate

2- Install dependencies, run: pip install -r requirements.txt

2.5- Select the python environemnt we jsut created as interpreter if not automatically selected.

3- prepare data for training, put the raw online retail sales file in data\raw folder and run the pipelines\model_data_pipeline.py

4- run the train_model.py file:
   python train_model.py

   After running, three models will be trained random forest, xgboost and Logistic Regression, the best one will be chosen on AUC score.

   this will generate the models/best_model.pkl file, your trained ml model !!!

## How to predict:

1- For prediction, download a test data set (already in data\raw\test_sales_data.csv). Run the training_data_pipeline.py file, run the pipelines/test_data_pipeline.py

this will generate a cleaned and transformed csv file for predcition.

2- Finally run this command:
   python predcit.py
   
   this will output an excel or csv file in report\tables\churn_predictions.csv containing all customers and their predicted churn probability with other metrics.

3- You can also achieve all this, after training model, using our mini UI app, run the following command in terminal:

streamlit run app/app.py

and it will locally run our app where you can upload test data in predict section, and it will give you predictions. You can also go to explainability section, upload there the file data/processed/main_ds.csv, its the processed file while training the model, upload there, select a customer, and press explain button, wait it ll take some time for processing, and it will give you nunced insights, waterfall charts for that specific customer.
 


