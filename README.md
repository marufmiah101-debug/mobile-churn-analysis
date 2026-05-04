# Mobile Customer Churn Analysis

Python-based exploratory data analysis and churn prediction model, built to extend
an interactive multi-dashboard Tableau workbook developed for the Digital Business
module at Northumbria University.

## Dataset

IBM Telco Customer Churn dataset — 6,687 customers across contract types,
payment methods, international plans, and demographics.

## Key findings (from Tableau + Python analysis)

- **Overall churn rate: 26.86%** — approximately 1 in 4 customers leave
- **Top churn driver: competition** — 805 churned customers left for a competitor,
  more than attitude (287), dissatisfaction (286), and price (200) combined
- **Contract type is the strongest lever** — month-to-month customers churn
  at ~55%+; one-year and two-year contract customers churn far less
- **International plan inactivity is a major risk** — customers with an
  international plan who are NOT actively using it churn at 71.19%,
  compared to just 7.59% for active users
- **Churn spikes after age 60** — customers aged 60+ show churn rates
  approaching 50%, a clear at-risk segment
- **Group plans protect retention** — solo customers churn most;
  churn rate falls consistently as group size increases
- **Geographic variation is wide** — state-level churn ranges from 19.44% to 63.24%

## Analysis structure

| File | What it does |
|------|-------------|
| `analysis.py` | Full EDA + logistic regression model |
| `requirements.txt` | Python dependencies |
| `outputs/` | Auto-generated charts |

## Charts produced

- Churn rate by contract type
- Churn rate by customer tenure band
- Churn rate: international plan vs no plan
- Monthly charges distribution (churned vs retained)
- Top predictors of churn (logistic regression)

## How to run

```bash
git clone https://github.com/marufmiah101-debug/mobile-churn-analysis
cd mobile-churn-analysis
pip install -r requirements.txt
mkdir outputs
python analysis.py
```

Download the dataset from [Kaggle — IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
and save it as `data/telco_churn.csv` before running.

## Related Tableau work

This Python analysis directly extends a multi-tab Tableau dashboard covering:
customer churn overview, age and group analysis, contract and behaviour analysis,
data and international usage, and a customer churn story.

[View the full interactive Tableau dashboard](https://public.tableau.com/app/profile/maruf.miah8527/viz/groupproject1_17709818503140/Genderandaveragemonthlycharges)

---

*Maruf Miah — Marketing and Business student, Northumbria University |
Founder @ AIgnite | [LinkedIn](https://www.linkedin.com/in/maruf-miah-775795227/)*
