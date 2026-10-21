# fraudguard-credit-card-fraud-detection
Machine learning model that detects fraudulent credit card transactions, built with CRISP-DM.
en the notebook in Jupyter.

FraudGuard: Credit Card Fraud Detection

A machine learning project that flags likely fraudulent credit card transactions, built following the CRISP-DM process. The main lesson: on data where fraud is 0.17% of transactions, accuracy is a trap, so every model is judged on precision, recall and PR-AUC instead.

The problem

Banks and payment providers need to block or review fraud before money is lost, without constantly blocking genuine customers. Fraud is very rare, so a model that always says "not fraud" scores 99.8% accuracy while catching nothing. The challenge is building and judging a model on the rare class.

Data
Source: ULB Credit Card Fraud Detection on Kaggle
Size: 284,807 anonymised card transactions from two days in September 2013, with 492 confirmed frauds (0.17%)
Columns: Time, Amount, V1 to V28 (anonymised PCA features) and Class (0 = normal, 1 = fraud)
No missing values. The data file is not included in this repo because of its size.
Method (CRISP-DM)
Business understanding: catch as much fraud as possible while keeping false alarms low enough for a fraud team to act on.
Data understanding: confirmed the extreme class imbalance, and looked at amount and time-of-day patterns for fraud versus normal transactions.
Data preparation: scaled Amount and Time, then split 80/20 with stratification so both sets keep the same fraud rate. Two ways of handling imbalance were compared on the training set only: class weighting and SMOTE.
Modelling: a "never fraud" baseline, Logistic Regression (plain, balanced and with SMOTE), Random Forest, and XGBoost.
Evaluation: precision, recall, ROC-AUC, PR-AUC and a confusion matrix on a held-out test set of 56,962 transactions (98 of them fraud).
Results
Model	Fraud precision	Fraud recall	PR-AUC	ROC-AUC
Never predict fraud (baseline)	0.00	0.00	n/a	n/a
Logistic Regression (plain)	0.83	0.64	0.744	0.957
Logistic Regression (balanced)	0.06	0.92	0.719	0.972
Logistic Regression (SMOTE)	0.06	0.92	0.725	0.970
Random Forest (balanced)	0.96	0.76	0.865	0.958
XGBoost (scale_pos_weight)	0.89	0.83	0.875	0.970

Key result: XGBoost gave the best balance. It caught about 83% of the fraud cases in the test set, and about 89% of its fraud alerts were real. The baseline reached 99.8% accuracy while catching none.

What the trade-off means: the balanced and SMOTE logistic models catch more fraud (92%) but raise many false alarms (6% precision). Random Forest raises very few false alarms but misses more fraud. A real fraud team would choose a point on this trade-off based on what a false alarm costs compared with a missed fraud.

Limitations
The test set holds only 98 fraud cases from a single split, so the numbers could move with a different split.
The data is from 2013 and from European cardholders, and the features are anonymised, so the model is a learning project and not a production fraud system.
The decision threshold was left at the default. Tuning it is the next step.
Tools

Python, pandas, NumPy, scikit-learn, imbalanced-learn, XGBoost, Matplotlib, Seaborn

How to run
Download creditcard.csv from the Kaggle page.
Put it in the same folder as fraud_detection_project.ipynb.
Install the libraries: pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn
Open the notebook in Jupyter, VS Code or Google Colab and run all cells.
Next steps
Tune the decision threshold to show precision and recall trading off.
Wrap the XGBoost model in a small API or Streamlit app.
Author

Victor Waweru, data science graduate, Moringa School. GitHub
