import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/telco_churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

print("=== Dataset Overview ===")
print(f"Total customers: {len(df)}")
print(f"Overall churn rate: {df['Churn'].mean()*100:.2f}%")
print(f"Churned customers: {df['Churn'].sum()}")
print(f"Retained customers: {(df['Churn'] == 0).sum()}")

# --- Chart 1: Churn by contract type ---
contract_churn = df.groupby('Contract')['Churn'].mean() * 100
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(contract_churn.index, contract_churn.values, color=['#E24B4A', '#EF9F27', '#1D9E75'])
ax.set_title('Churn rate by contract type', fontsize=14)
ax.set_ylabel('Churn rate (%)')
ax.set_xlabel('Contract type')
for bar, val in zip(bars, contract_churn.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/churn_by_contract.png', dpi=150)
plt.close()
print("\nChart 1 saved: churn_by_contract.png")
print(contract_churn.to_string())

# --- Chart 2: Churn rate by age group ---
df['AgeGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72],
                         labels=['0-12 months', '13-24 months', '25-48 months', '49-72 months'])
age_churn = df.groupby('AgeGroup', observed=True)['Churn'].mean() * 100
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(age_churn.index.astype(str), age_churn.values, marker='o',
        color='#185FA5', linewidth=2, markersize=8)
ax.fill_between(range(len(age_churn)), age_churn.values, alpha=0.1, color='#185FA5')
ax.set_title('Churn rate by customer tenure', fontsize=14)
ax.set_ylabel('Churn rate (%)')
ax.set_xlabel('Tenure band')
for i, val in enumerate(age_churn.values):
    ax.text(i, val + 0.8, f'{val:.1f}%', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('outputs/churn_by_tenure.png', dpi=150)
plt.close()
print("\nChart 2 saved: churn_by_tenure.png")

# --- Chart 3: International plan vs churn ---
if 'International plan' in df.columns:
    intl_churn = df.groupby('International plan')['Churn'].mean() * 100
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(intl_churn.index, intl_churn.values, color=['#1D9E75', '#E24B4A'])
    ax.set_title('Churn rate: international plan vs no plan', fontsize=13)
    ax.set_ylabel('Churn rate (%)')
    for bar, val in zip(bars, intl_churn.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/churn_intl_plan.png', dpi=150)
    plt.close()
    print("\nChart 3 saved: churn_intl_plan.png")

# --- Chart 4: Monthly charges distribution ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
churned = df[df['Churn'] == 1]['MonthlyCharges']
retained = df[df['Churn'] == 0]['MonthlyCharges']
axes[0].hist(churned, bins=30, color='#E24B4A', alpha=0.7, label=f'Churned (mean: £{churned.mean():.0f})')
axes[0].hist(retained, bins=30, color='#185FA5', alpha=0.7, label=f'Retained (mean: £{retained.mean():.0f})')
axes[0].set_title('Monthly charges: churned vs retained')
axes[0].set_xlabel('Monthly charges (£)')
axes[0].set_ylabel('Number of customers')
axes[0].legend()
df.boxplot(column='MonthlyCharges', by='Churn', ax=axes[1])
axes[1].set_title('Monthly charges distribution by churn')
axes[1].set_xlabel('Churn (0 = retained, 1 = churned)')
plt.suptitle('')
plt.tight_layout()
plt.savefig('outputs/monthly_charges.png', dpi=150)
plt.close()
print("Chart 4 saved: monthly_charges.png")

# --- Predictive model ---
cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'InternetService',
            'Contract', 'PaperlessBilling', 'PaymentMethod']
df_model = df.copy()
for col in cat_cols:
    if col in df_model.columns:
        df_model[col] = df_model[col].astype('category').cat.codes

feature_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
feature_cols += [c for c in cat_cols if c in df_model.columns]
X = df_model[feature_cols]
y = df_model['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n=== Predictive Model — Logistic Regression ===")
print(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))

importance = pd.Series(abs(model.coef_[0]), index=feature_cols).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
importance.head(8).plot(kind='bar', ax=ax, color='#185FA5')
ax.set_title('Top predictors of churn (logistic regression coefficients)', fontsize=13)
ax.set_ylabel('Absolute coefficient value')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('outputs/feature_importance.png', dpi=150)
plt.close()
print("Chart 5 saved: feature_importance.png")
print("\nAll outputs saved to outputs/ folder.")
