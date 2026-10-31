"""
Generates a synthetic-but-realistic 'Campus Placement' dataset.
Columns match the well-known Kaggle 'Campus Recruitment' dataset structure:
sl_no, gender, ssc_p, ssc_b, hsc_p, hsc_b, hsc_s, degree_p, degree_t,
workex, etest_p, specialisation, mba_p, status, salary

Note: If you have downloaded the real dataset from Kaggle
(search: "Campus Recruitment - benroshan"), just replace this CSV with
that one — the column names are identical, so nothing else needs to change.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 300

gender = np.random.choice(['M', 'F'], n, p=[0.55, 0.45])
ssc_b = np.random.choice(['Central', 'Others'], n)
hsc_b = np.random.choice(['Central', 'Others'], n)
hsc_s = np.random.choice(['Science', 'Commerce', 'Arts'], n, p=[0.4, 0.45, 0.15])
degree_t = np.random.choice(['Sci&Tech', 'Comm&Mgmt', 'Others'], n, p=[0.45, 0.45, 0.10])
workex = np.random.choice(['Yes', 'No'], n, p=[0.35, 0.65])
specialisation = np.random.choice(['Mkt&HR', 'Mkt&Fin'], n)

ssc_p = np.clip(np.random.normal(67, 10, n), 40, 98).round(2)
hsc_p = np.clip(np.random.normal(66, 11, n), 40, 98).round(2)
degree_p = np.clip(np.random.normal(66, 8, n), 45, 95).round(2)
etest_p = np.clip(np.random.normal(72, 13, n), 40, 99).round(2)
mba_p = np.clip(np.random.normal(62, 6, n), 50, 78).round(2)

# Placement probability driven by academics + workex (like real-world trends)
score = (
    0.28 * (ssc_p - 50) +
    0.22 * (hsc_p - 50) +
    0.30 * (degree_p - 50) +
    0.10 * (etest_p - 50) +
    0.05 * (mba_p - 50) +
    15 * (workex == 'Yes')
)
prob = 1 / (1 + np.exp(-(score - 25) / 6))
status = np.where(np.random.rand(n) < prob, 'Placed', 'Not Placed')

salary = np.where(
    status == 'Placed',
    np.clip(np.random.normal(280000, 70000, n) + (degree_p - 60) * 2000, 180000, 900000).round(-3),
    np.nan
)

df = pd.DataFrame({
    'sl_no': np.arange(1, n + 1),
    'gender': gender,
    'ssc_p': ssc_p,
    'ssc_b': ssc_b,
    'hsc_p': hsc_p,
    'hsc_b': hsc_b,
    'hsc_s': hsc_s,
    'degree_p': degree_p,
    'degree_t': degree_t,
    'workex': workex,
    'etest_p': etest_p,
    'specialisation': specialisation,
    'mba_p': mba_p,
    'status': status,
    'salary': salary,
})

df.to_csv('/home/claude/placement_prediction/dataset/Placement_Data_Full_Class.csv', index=False)
print(df.head())
print("\nShape:", df.shape)
print("\nPlacement counts:\n", df['status'].value_counts())
