# 🔄 Customer Churn Prediction

A Machine Learning project to predict whether a telecom customer will churn (leave) 
using their usage data and demographics.

## 📌 Problem Statement
Customer churn is a major business problem. Retaining an existing customer is 5x 
cheaper than acquiring a new one. This model helps identify at-risk customers early.

## 📊 Dataset
- Source: IBM Telco Customer Churn (via Kaggle)
- 7,043 customers | 21 features

## 🧠 Models Used
| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~80% |
| Random Forest | ~79% |
| Gradient Boosting | ~81% |

## 🔑 Key Insights
- Month-to-month contract customers churn the most
- Higher monthly charges increase churn probability
- Customers with longer tenure are more loyal

## 🚀 How to Run
```bash
pip install -r requirements.txt
python main.py
```

## 📁 Project Structure
```
customer-churn-prediction/
├── data/               ← dataset
├── src/                ← source code modules
├── outputs/            ← saved plots and model
├── main.py             ← run this to execute pipeline
└── requirements.txt    ← dependencies
```