# Customer Churn Prediction with Neural Networks

This project focuses on predicting customer churn using an Artificial Neural Network (ANN) built with TensorFlow/Keras. The dataset is sourced from a bank's customer records, and the goal is to identify customers who are likely to leave the service.



##  Dataset

**Source:** [Churn_Modelling.csv](https://raw.githubusercontent.com/rasushi/customer_churn_prediction/refs/heads/main/Churn_Modelling.csv)

**Key Columns:**
- CreditScore
- Geography
- Gender
- Age
- Tenure
- Balance
- NumOfProducts
- HasCrCard
- IsActiveMember
- EstimatedSalary
- Exited (Target)


##  Workflow

1. **Data Preprocessing**
   - Removed non-informative columns (`RowNumber`, `CustomerId`, `Surname`)
   - One-hot encoded categorical variables (`Gender`, `Geography`)
   - Scaled numerical features using `StandardScaler`

2. **Model Building**
   - Used `Sequential` API from Keras
   - Architecture: 2 hidden layers (10 neurons each, ReLU), 1 output layer (Sigmoid)
   - Compiled with `binary_crossentropy` loss and `Adam` optimizer

3. **Model Training**
   - Trained for 100 epochs with 20% validation split
   - Tracked training and validation metrics

4. **Evaluation**
   - Accuracy Score
   - Confusion Matrix
   - Classification Report (Precision, Recall, F1-score)
   - Plots for loss and accuracy over epochs


## Results

- Final **test accuracy**: _8.54_% (fill this based on your run)
- Confusion matrix and metric report provided
- Minimal overfitting observed (training vs validation accuracy differed by ~0.2%)


## Libraries Used

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `sklearn` for preprocessing and evaluation
- `tensorflow` and `keras` for deep learning


