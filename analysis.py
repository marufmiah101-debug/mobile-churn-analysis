# Mobile Customer Churn Analysis

Python-based exploratory data analysis and churn prediction on telecom customer data, built to complement an interactive Tableau dashboard developed at Northumbria University.

## What this project does

- Cleans and analyses a real-world telecom dataset (7,000+ customers)
- Identifies the key drivers of churn: contract type, monthly charges, tenure
- Visualises churn patterns across customer segments
- Builds a logistic regression model to predict which customers are likely to leave

## Key findings

- Overall churn rate: 26.5%
- Month-to-month contract customers churn at 3x the rate of annual contract customers
- Churned customers pay on average £10/month more than retained customers
- Tenure is the strongest single predictor — customers who stay past 12 months rarely leave

## Tech stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core analysis |
| pandas | Data cleaning and manipulation |
| matplotlib / seaborn | Visualisation |
| scikit-learn | Logistic regression model |
| Tableau | Interactive dashboard (linked below) |

## How to run

```bash
git clone https://github.com/marufmiah101-debug/mobile-churn-analysis
cd mobile-churn-analysis
pip install -r requirements.txt
python analysis.py
```

Dataset: [IBM Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Note: download the CSV from Kaggle and place it in a `data/` folder before running.

## Related work

This analysis extends my Tableau dashboard built for the Digital Business module at Northumbria University. [View dashboard on Tableau Public](https://public.tableau.com/app/profile/maruf.miah8527/viz/groupproject1_17709818503140/Genderandaveragemonthlycharges)

---

*Built by Maruf Miah — Marketing and Business student at Northumbria University | Founder @ AIgnite | [LinkedIn](https://www.linkedin.com/in/maruf-miah-775795227/)*
