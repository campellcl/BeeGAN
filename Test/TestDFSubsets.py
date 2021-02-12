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
# Add convenience columns:
df['year'] = df['date'].dt.year
df['week'] = df['date'].dt.isocalendar().week
df['day_of_year'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek
# Add a column indicating if the sample belongs to the 'train', 'val', or 'test' dataset:
df['split'] = ""

df['yr_week_grp_idx'] = df['date'].apply(
    lambda x: '%s-%s' % (x.year, '{:02d}'.format(x.week)))

# Create placeholder dataframes for each of the train-val-test splits:
train_df: pd.DataFrame = pd.DataFrame(data=None, index=None, columns=['date', 'random_num', 'year', 'week', 'day_of_year', 'day_of_week'])
val_df: pd.DataFrame = pd.DataFrame(data=None, index=None, columns=['date', 'random_num', 'year', 'week', 'day_of_year', 'day_of_week'])
test_df: pd.DataFrame = pd.DataFrame(data=None, index=None, columns=['date', 'random_num', 'year', 'week', 'day_of_year', 'day_of_week'])

def perform_weekly_train_val_test_split(week_data: pd.Series, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    # Randomly select from day-of-week indices [0-6] inclusive, 4 days for train, 2 for val, 1 for test
    day_of_week_indices = np.arange(0, 7)
    day_of_week_indices = np.random.permutation(day_of_week_indices)
    train_days: np.ndarray = day_of_week_indices[0: 4]
    val_days = day_of_week_indices[4: 6]
    test_day = day_of_week_indices[-1]
    week_train_meta_data_series: pd.Series = week_data.query('day_of_week in @train_days')
    week_val_meta_data_series: pd.Series = week_data.query('day_of_week in @val_days')
    week_test_meta_data_series: pd.Series = week_data.query('day_of_week == @test_day')
    # Append each series to their respective train/val/test dataframes:
    train_df = train_df.append(week_train_meta_data_series)
    val_df = val_df.append(week_val_meta_data_series)
    test_df = test_df.append(week_test_meta_data_series)
    # Return the updated provided dataframes:
    return train_df, val_df, test_df

# Group the dataframe and then interate over the grouped object:
df_grouped_by_week_and_year = df.groupby('yr_week_grp_idx')

# Iterate through the unique weeks in the dataframe:
for year_and_week, week_data_subset in df_grouped_by_week_and_year:
    train_df, val_df, test_df = perform_weekly_train_val_test_split(week_data=week_data_subset, train_df=train_df, val_df=val_df, test_df=test_df)

assert (train_df.shape[0] + val_df.shape[0] + test_df.shape[0]) == df.shape[0]