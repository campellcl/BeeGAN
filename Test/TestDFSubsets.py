import numpy as np
import pandas as pd
import datetime
from typing import List

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
df['year'] = df['date'].dt.year
df['week'] = df['date'].dt.week
df['day_of_year'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek

df['yr_week_grp_idx'] = df['date'].apply(
    lambda x: '%s-%s' % (x.year, '{:02d}'.format(x.week)))


def perform_weekly_train_val_test_split(week_data: pd.Series):
    # Randomly select from day-of-week indices [0-6] inclusive, 4 days for train, 2 for val, 1 for test
    day_of_week_indices = np.arange(0, 7)
    day_of_week_indices = np.random.permutation(day_of_week_indices)
    train_days: np.ndarray = day_of_week_indices[0: 4]
    val_days = day_of_week_indices[4: 6]
    test_day = day_of_week_indices[-1]
    week_train_meta_data_series: pd.Series = week_data.query('day_of_week in @train_days')
    week_val_meta_data_series: pd.Series = week_data.query('day_of_week in @val_days')
    week_test_meta_data_series: pd.Series = week_data.query('day_of_week == @test_day')
    # TODO: Append each series to their respective train/val/test dataframes:

    # Return separate train, test, and val pd.Series for audio files belonging to the specified days of the week:
    return week_train_meta_data_series, week_val_meta_data_series, week_test_meta_data_series

df.groupby('yr_week_grp_idx').apply(perform_weekly_train_val_test_split)

# for year in df['date'].dt.year.unique():
#     year_df_subset = df[df['date'].dt.year == year]
#     # Iterate by week over all the existing data in the year:
#     for week in year_df_subset['date'].dt.week.unique():
#         week_df_subset = year_df_subset[year_df_subset['date'].dt.week == week]
#         pass

print('break')