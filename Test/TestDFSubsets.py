import numpy as np
import pandas as pd
import datetime

# Create dataset
np.random.seed(123)
n = 100000

date = pd.to_datetime({
    'year': np.random.choice(range(2017, 2020), size=n),
    'month': np.random.choice(range(1, 13), size=n),
    'day': np.random.choice(range(1, 28), size=n),
    'hour': np.random.choice(range(0, 25), size=n),
    'minute': np.random.choice(range(0, 61), size=n),
    'second': np.random.choice(range(0, 61), size=n)
})

random_num = np.random.choice(
    range(0, 1000),
    size=n)

df = pd.DataFrame({'date': date, 'random_num': random_num})
# Sort by date in ascending order:
df = df.sort_values(by="date")

for year in df['date'].dt.year.unique():
    year_df_subset = df[df['date'].dt.year == year]
    # Iterate by week over all the existing data in the year:
    for week in year_df_subset['date'].dt.week.unique():
        week_df_subset = year_df_subset[year_df_subset['date'].dt.week == week]
        pass

print('break')