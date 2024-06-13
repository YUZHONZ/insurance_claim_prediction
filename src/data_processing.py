import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

class DataProcessing:

    def __init__(self, path, save_to_file=False):
        self.path = path
        self.drop_unnecessary_columns = None
        self.save_to_file = save_to_file

    def load_data(self, path=None):

        if self.path is None:
            self.path = path
        
        LOGGER.info(f"Loading data from {self.path}")
        self.df = pd.read_csv(self.path)
        return self.df

    def clean_data(self):

        LOGGER.info("Transforming datetime data")
        self.df = self.transform_datetime_data(self.df)

        LOGGER.info("Cleaning missing data")
        self.df = self.clean_missing_data(self.df)

        LOGGER.info("Transforming employment data")
        self.df = self.transform_employment_data(self.df)

        LOGGER.info("Transforming claim cost data")
        self.df = self.transform_claim_cost_data(self.df)

        LOGGER.info("Transforming categorical data")
        self.df = self.transfoer_categorical_data(self.df)

        df = df.drop(columns=['ClaimNumber', 'Gender', 'MaritalStatus', 'PartTimeFullTime', 'WageCatgory'])

        if self.save_to_file:
            save_path = '../data'
            self.df.to_csv(f'{save_path}/processed_data.csv', index=False)

    def transform_datetime_data(self, df):

        LOGGER.info("Transforming datetime data")
        df['DateTimeOfAccident'] = pd.to_datetime(df['DateTimeOfAccident'])
        df['DateReported'] = pd.to_datetime(df['DateReported'])

        LOGGER.info("Creating accident time related features")
        # Accident time related features
        df['WeekDayOfAccident'] = df['DateTimeOfAccident'].dt.dayofweek
        df['MonthOfAccident'] = df['DateTimeOfAccident'].dt.month
        df['YearOfAccident'] = df['DateTimeOfAccident'].dt.year
        df['HourOfAccident'] = df['DateTimeOfAccident'].dt.hour

        LOGGER.info("Creating report time related features")
        # Report time related features
        df['MonthOfReport'] = df['DateReported'].dt.dayofweek
        df['YearOfReport'] = df['DateReported'].dt.year

        LOGGER.info("Creating interval of date between accident and report date time")
        # get interval date between accident and report
        df['IntervalOfTime'] = df['DateReported'] - df['DateTimeOfAccident']
        # interval day
        df['IntervalofDay'] = df['IntervalOfTime'].dt.days
        df.loc[df['IntervalofDay'] < 0, 'IntervalofDay'] = 0
        # interval month
        df['IntervalofYear'] = df['YearOfReport']  - df['YearOfAccident']
        df.loc[df['IntervalofYear'] < 0, 'IntervalofYear'] = 0

        # drop the original time related features
        df = df.drop(['DateTimeOfAccident', 'DateReported', 'IntervalOfTime'], axis=1)

        LOGGER.info("Create new features: WeekDayOfAccident, MonthOfAccident, YearOfAccident, HourOfAccident, MonthOfReport, YearOfReport, IntervalofDay, IntervalofYear")
        LOGGER.info("Drop the original time related features: DateTimeOfAccident, DateReported, IntervalOfTime")

        return df

    def clean_missing_data(self, df):
        df['MaritalStatus'].fillna('U', inplace=True)
        return df
    
    def transform_employment_data(self, df):

        # convert the employment data to hourly and daily wages
        df['HourlyWages'] = df.apply(lambda x: x['WeeklyWages'] / x['HoursWorkedPerWeek'] if 0 < x['HoursWorkedPerWeek'] < 100 else 0, axis=1)
        df['DailyWages'] = df.apply(lambda x: x['WeeklyWages'] / x['DaysWorkedPerWeek'] if x['DaysWorkedPerWeek'] != 0 else 0, axis=1)        
        
        # wagecategory data
        bins = [0, 500, 1000, float('inf')]
        labels = ['Low', 'Medium', 'High']
        df['WageCatgory'] = pd.cut(df['WeeklyWages'], bins=bins, labels=labels)

        # label overtime if overtime hours is greater than 0
        df['Overtime'] = df['HoursWorkedPerWeek'].apply(lambda x: 1 if x - 40 > 0 else 0)
        return df

    def transform_claim_cost_data(self, df):
        df['InitialIncurredClaimsCost_log'] = np.log1p(df['InitialIncurredClaimsCost'])
        df['UltimateIncurredClaimCost'] = np.log1p(df['UltimateIncurredClaimCost'])
        return df

    def transfoer_categorical_data(self, df):

        LOGGER.info("Transforming categorical data for: Gender, MaritalStatus, PartTimeFullTime, WageCatgory")
        # gender
        gender_dummies = pd.get_dummies(df['Gender'], prefix='Gender')

        # MaritalStatus
        marital_dummies = pd.get_dummies(df['MaritalStatus'], prefix='MaritalStatus')

        # part time full time
        part_time_full_time_dummies = pd.get_dummies(df['PartTimeFullTime'], prefix='PartTimeFullTime')

        # wage category
        wage_category_dummies = pd.get_dummies(df['WageCatgory'], prefix='WageCatgory')

        df = pd.concat([df, gender_dummies, marital_dummies, part_time_full_time_dummies, wage_category_dummies], axis=1)

        LOGGER.info("Drop columns: Gender, MaritalStatus, PartTimeFullTime, WageCatgory")
        df = df.drop(columns=['Gender', 'MaritalStatus', 'PartTimeFullTime', 'WageCatgory'])

        return df