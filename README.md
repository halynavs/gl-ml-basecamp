# gl-ml-basecamp

## Folder HW-1 contains:
1. realizations of Gradient Descent, Stochastic Gradient Descent from scratch and sklearn Linear Regression for data with:
- normilized normal distribution
- normal distribution with 3 types of abnormal data
2. image Compression Using SVD realization
3. conclusions about implementation 1. 2. puncts

## Folder HW-2 contains:
1. In HW-2-cross-validation.ipynb:
   - realized linear regression model from scratch with mape loss and opt method Powell
   - used pipeline(polynomial feature engineering + custom linear regression model)
   - polynom degree are chosen by sklearn cross validation 
2. In HW-2-L2-regulization.ipynb:
   - l2 regulization implemented for custom linear regression. Weight of regulization estimated by cross validation
   - RidgeCV used for data without polynomial features
3. In HW-2-L1-regulization.ipynb:
   - l1 regulization implemented for custom linear regression. Weight of regulization estimated by cross validation
   - LassoCV used for data without polynomial features. Alpha selected by yellowbrick.regressor.AlphaSelection
4. In HW-2-Elastic-net-regulization.ipynb:
   - Elastic net regulization implemented for custom linear regression.Drawback - weights for regulization not estimated in pair
5. In HW-2-weight-linear-regression.ipynb:
   - custom weight-linear-regression model compared with optimal linear with polynomial feature engineering(degree=10)
   - Drawback -  Evaluation lowess in progress
6. In HW-2-diff-loss-func-and-opt-methods.ipynb
   - impleminted different loss functions as MAPE, MSE, RMSE, MAE and MSLE for regression task
   - used 'BFGS', 'Nelder-Mead', 'Powell', 'CG', 'TNC' numerical optimization funcs for minimization mentioned loss funcs(in pairs loss-opt method)
7. In HW-2-classification.ipynb:
   - realized logistic regression from scratch, log-loss used for learning
   - model estimated by cross validation with log-loss estimator, confusion matrix and accuracy, precision, recall, F1 score metrics
 

