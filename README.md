# Random-Forest-Regression

This project demonstrates the use of Random Forest Regression for predictive modeling. The training and test data are loaded and prepared using NumPy, and the baseline regression model is computed by taking the mean of the target variable, Y_train. The Mean Absolute Error (MAE) is then calculated to establish a baseline accuracy for reference.  

In section (b), the Random Forest Regressor from the scikit-learn library is utilized. The model is trained on the training data (X_train and Y_train), and predictions are made on the test data (X_test). The MAE is computed by comparing the actual target values (Y_test) with the predicted values (Y_pred). This evaluation allows us to measure the accuracy of the Random Forest Regressor compared to the baseline model.  

Furthermore, section (c) explores the impact of varying the maximum depth of the Random Forest Regressor. Multiple regressors with different depths are trained and evaluated to observe how the MAE changes with increasing depth. The results are plotted using Matplotlib, showcasing the relationship between depth and MAE. This analysis helps in selecting the appropriate depth for the model, as too shallow or too deep trees can lead to overfitting or underfitting.  

This project provides valuable insights into the performance of the Random Forest Regression model and its sensitivity to the maximum depth parameter. By visualizing the MAE against different depths, data scientists and analysts can make informed decisions to optimize the model's accuracy for the specific dataset.  
