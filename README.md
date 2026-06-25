# Online Payment Fraud Detection using Machine Learning

As online payment methods have grown, so has the incidence of digital payment fraud. This project builds a Machine Learning classifier in Python to detect fraudulent transactions, helping to secure digital payments and minimize financial losses.

## 🚀 Features
* **Data Preprocessing & Cleaning:** Handles missing values, balances class distributions, and encodes categorical transaction types.
* **Feature Engineering:** Extracts key insights from transaction amounts, balances, and origins.
* **Exploratory Data Analysis (EDA):** Visualizes transaction trends, fraud distribution, and correlation matrices (saved in the `images/` directory).
* **Predictive Modeling:** Implements a Machine Learning model (such as a Decision Tree, Random Forest, or XGBoost classifier) to accurately flag fraudulent transactions.

---

## 🛠️ Project Structure

The repository is structured simply and cleanly as follows:
```text
├── images/                # Saved EDA plots, confusion matrices, and ROC curves
├── model.py               # Main Python script for data processing, training, and evaluation
├── .gitignore             # Specific files and folders ignored by Git (like venv and data)
├── README.md              # Project documentation
└── requirements.txt       # List of Python dependencies
