# gl-ml-basecamp

## Folder HW-1 contains:
1. **realizations of Gradient Descent, Stochastic Gradient Descent from scratch and sklearn Linear Regression** for data with:
- normilized normal distribution
- normal distribution with 3 types of abnormal data
2. **image Compression Using SVD realization**
3. conclusions about implementation 1. 2. puncts

## Folder HW-2 contains:
1. In HW-2-cross-validation.ipynb:
   - realized **linear regression model from scratch with mape loss and opt method Powell**
   - used **pipeline(polynomial feature engineering + custom linear regression model)**
   - **polynom degree are chosen by sklearn cross validation **
2. In HW-2-L2-regulization.ipynb:
   - **l2 regulization implemented for custom linear regression**. Weight of regulization estimated by cross validation
   - **RidgeCV used** for data without polynomial features
3. In HW-2-L1-regulization.ipynb:
   - **l1 regulization implemented for custom linear regression**. Weight of regulization estimated by cross validation
   - **LassoCV used** for data without polynomial features. Alpha selected by yellowbrick.regressor.AlphaSelection
4. In HW-2-Elastic-net-regulization.ipynb:
   - **Elastic net regulization implemented** for custom linear regression.Drawback - weights for regulization not estimated in pair
5. In HW-2-weight-linear-regression.ipynb:
   - **custom weight-linear-regression model** compared with optimal linear with polynomial feature engineering(degree=10)
   - Drawback -  Evaluation lowess in progress
6. In HW-2-diff-loss-func-and-opt-methods.ipynb
   - impleminted **different loss functions as MAPE, MSE, RMSE, MAE and MSLE for regression task**
   - **used 'BFGS', 'Nelder-Mead', 'Powell', 'CG', 'TNC' numerical optimization funcs** for minimization mentioned loss funcs(in pairs loss-opt method)
7. In HW-2-classification.ipynb:
   - realized **logistic regression from scratch**, log-loss used for learning
   - model **estimated by cross validation with log-loss estimator**, confusion matrix and accuracy, precision, recall, F1 score metrics

## Folder HW-3 contains:
1. In HW-3-spam-classifier-custom-sklearn-BOW-Multi-vercoriz.ipynb:
   - classilied spam messages using **custom classifier** and **sklearn GaussianNB** from [SpamAssassin Public Corpus](https://spamassassin.apache.org/old/publiccorpus/).
   - estemated **2 types of text vectorization**(BOW and multinomial(count of words))
   - **cross validation used for better model estimation**
2. In Descriptive_statistics_usage.ipynb:
      for random and real data
   - ploted **Probability Density Function(pdf)** and **Cumulative distribution function(cdf)**
   - canculated(**mean, median, mode, quantiles, interquartile range, sample varience, standard deviation**)
3. In Custom-LDA-QDA-impl-tested-on-artificial-data.ipynb:
   - Custom **Quadratic Discriminant Analysis** and **Custom Linear Discriminant Analysis** are used for classification and compared to sklearn models
4. In Gaussian-process-for-regression.ipynb: 
   - **Gaussian process** impl for regression with exponentiated quadratic kernel

## Folder NN contains:
1. In Multilayer_perceptron_for_classification.ipynb and Multilayer_perceptron_for_regression.ipynb:
   - custom **multilayer perceptron** with backpropagation learning for classification and regression problem
   - compared with **keras Sequential model**
   - **TensorBoard** used for visualization workflow
2. In  Baseline-algorithms.ipynb:
   - **Random and zero rule baseline algorithms** implemented for regression and classification problems

## Folder HW-4 contains:
1. In [LogReg_with_NN_logic_for_image_classif.ipynb](https://github.com/halynavs/gl-ml-basecamp/blob/hw-4/HW-4/LogReg_with_NN_logic_for_image_classif.ipynb):
   - impl **image classification by Logistic Regression**
2. In [HW-4-Testing-activ-func-on-specific-data.ipynb](https://github.com/halynavs/gl-ml-basecamp/blob/hw-4/HW-4/HW-4-Testing-activ-func-on-specific-data.ipynb):
   - Implemented a 2-class classification neural network with a single hidden layer
   - Used units with a non-linear activation function, ReLu, Sigmoid, Tanh
   - Computed the cross entropy loss
   - Implementd forward and backward propagation
3. In [HW-4-ReLU-flower-data-classif.ipynb][1], [HW-4-Sigmoid-flower-data-classif.ipynb][2], [HW-4-Tanh-flower-data-classif.ipynb][3]:
   - **ReLu, Sigmoid, Tanh activation funcs tested on different data: noisy_circles, noisy_moon, blobs, gaussian_quantiles**
  ![alt text](https://github.com/halynavs/gl-ml-basecamp/blob/hw-4/HW-4/images/act_funcs.png)
4. In [HW-4-Building-funcs-for-DNN.ipynb](https://github.com/halynavs/gl-ml-basecamp/blob/hw-4/HW-4/HW-4-Building-funcs-for-DNN.ipynb):
   - impl funcs for creating deep NN
5. In [HW-4-Deep-Neural-Network-Application.ipynb](https://github.com/halynavs/gl-ml-basecamp/blob/hw-4/HW-4/HW-4-Deep-Neural-Network-Application.ipynb):
   - created a two-layer neural network
   - created an **L-layer neural network**
   This models are used to classify cat vs non-cat images
6. In [Logic-Of-Decision-Tree-Classifier.ipynb](https://github.com/halynavs/gl-ml-basecamp/blob/hw-4/HW-4/Logic-Of-Decision-Tree-Classifier.ipynb):
   - impl logic of Decision Tree Classifier from scratch
7. In [HW-4-Comparison-LR-LRpoly-DT-DTpoly-OptDT.ipynb]():
   - created comparision between 5 models for classification on artificial data.
      Model are used:
        * Linear Regression
        * Linear Regression with 5 degree polynomial feature engineering
        * Decision Tree
        * Decision Tree with 5 degree polynomial feature engineering
        * Optimal Decision Tree with choosen by GridSearchCV best criterion and depth of tree
  ![alt text](https://github.com/halynavs/gl-ml-basecamp/blob/hw-4/HW-4/images/decision_tree_vs_logreg.png)
 
[1]: https://github.com/halynavs/gl-ml-basecamp/blob/hw-4/HW-4/HW-4-ReLU-flower-data-classif.ipynb
[2]: https://github.com/halynavs/gl-ml-basecamp/blob/hw-4/HW-4/HW-4-Sigmoid-flower-data-classif.ipynb
[3]: https://github.com/halynavs/gl-ml-basecamp/blob/hw-4/HW-4/HW-4-Tanh-flower-data-classif.ipynb
