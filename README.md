# Credit Card Approval Prediction

## 📌 Project Overview
This project focuses on analyzing and predicting credit card approvals using machine learning techniques. The dataset includes numerical and categorical features related to customers, such as income, age, occupation, and credit history, which influence the approval decision. The goal is to explore the dataset, clean it, and develop classification models to predict whether a credit card application will be approved or rejected.

## 📂 Dataset Information
- **Number of Rows:** 690
- **Number of Columns:** 16
- **Target Variable:** `Class` (0 = Rejected, 1 = Approved)
- **Feature Overview:**
  - `CustomerID`: Unique identifier for each customer
  - `A1` - `A14`: Customer characteristics (numerical and categorical)
  - `Class`: Credit card approval outcome

## 📊 Exploratory Data Analysis (EDA)
We perform **EDA on numerical columns** to identify patterns and distributions:
1. Summary statistics & missing values
2. Distribution plots (Histograms, Boxplots, KDE plots)
3. Correlation heatmap
4. Pair plots & feature relationships
5. Outlier detection and removal

## 🔧 Data Preprocessing
1. Handling missing values (if any)
2. Encoding categorical variables
3. Scaling numerical features
4. Outlier removal using IQR method
5. Feature selection based on importance

## 🤖 Machine Learning Models
We train and compare **Top 10 Classification Models**:
1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**
4. **Support Vector Machine (SVM)**
5. **K-Nearest Neighbors (KNN)**
6. **Gradient Boosting Classifier**
7. **XGBoost Classifier**
8. **LightGBM Classifier**
9. **CatBoost Classifier**
10. **Artificial Neural Network (ANN)**

## 📈 Model Evaluation & Comparison
Each model is evaluated using:
- **Accuracy, Precision, Recall, F1-score**
- **Confusion Matrix**
- **ROC-AUC Curve & Score**
- **Feature Importance Plots**
- **Comparison of Model Performance**

## 🔥 Key Findings & Insights
- Identified key factors affecting credit approval
- Outlier handling improved model performance
- Random Forest & XGBoost provided the best results

## 🛠 Installation & Usage
### Install Required Libraries:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost
```


## 📌 Future Work
- Improve feature engineering
- Apply hyperparameter tuning for best models
- Experiment with deep learning techniques

## 📜 License
This project is open-source under the MIT License.

## 💡 Contributing
Feel free to contribute by submitting pull requests or suggesting improvements!

## 📞 Contact
For questions or collaboration, reach out via GitHub Issues.

## 📎 Links
- **Kaggle Notebook:** [https://www.kaggle.com/code/arifmia/customer-segmentation-risk-prediction-using-ml/notebook](#)
- **LinkedIn Profile:** [www.linkedin.com/in/arif-miah-8751bb217](#)

---
### ⭐ If you find this project useful, consider giving it a star!

