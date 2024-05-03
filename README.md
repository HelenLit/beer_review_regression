# Project: Regression Analysis on "Beer Profile and Ratings Data Set"  
   
## Introduction  
   
This project is based on the dataset "Beer Profile and Ratings Data Set" obtained from the open source Kaggle by user ruthgn. The dataset contains information about different types of beer, their characteristics, and user ratings. The dataset consists of 9,386 entries, each representing a unique type of beer.  
   
## Data Description  
   
The dataset includes the following fields:  
   
1. "Name": Beer name.  
2. "Style": Beer style.  
3. "Brewery": Brewery producing the beer.  
4. "Beer Name (Full)": Full beer name.  
5. "Description": Beer description.  
6. "ABV": Alcohol by volume.  
7. "Min IBU" and "Max IBU": Minimum and maximum International Bitterness Units.  
8. "Astringency", "Body", "Alcohol", "Bitter", "Sweet", "Sour", "Salty", "Fruits", "Hoppy", "Spices", "Malty": Flavor characteristics of the beer.  
9. "review_aroma", "review_appearance", "review_palate", "review_taste": User ratings on different aspects of the beer.  
10. "review_overall": Overall user rating.  
11. "number_of_reviews": Number of reviews.  
   
## Data Preparation  
   
The target variable for prediction in this project is 'review_overall'. The features used for prediction include 'Style', 'ABV', 'Astringency', 'Sweet', 'Sour', 'Salty', 'Hoppy', 'Spices', 'Malty'.  
   
## Model Selection  
   
A range of models were used, including linear regression, gradient boosting, stochastic gradient descent, random forest, elastic net, Bayesian ridge, SVR, decision tree, ridge, lasso, K-nearest neighbors, XGBoost, and CatBoost. The best result was achieved using the CatBoostRegressor model, with an RMSE of 0.07676. This model uses gradient boosting on decision trees, allowing it to effectively handle various types of data and complex structures. The model's performance on new data was evaluated with an RMSE of 0.082.   
  
## How to use   
  
Scripts for training the model and using it to make predictions on new data have been created. These scripts are essential for reproducibility and real-world application of the model.  
   
The training script can be run independently and is used to train the model on any dataset. The script saves the trained model for future use. This script is stored in the GitHub repository for ease of use and collaboration.  
   
To use the model to make predictions on new data, a separate script has been created. This script loads the trained model, accepts new data, and uses the model to make predictions.  
   
New data were created using the custom function "save_fraction_csv(DATASET_PATH, "../data", "new_input.csv", 0.1)", which separated 10% of the initial dataset and saved it in a separate file new_input.csv. This file with new data can now be used to make predictions using the trained model.  
