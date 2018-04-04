import pandas as pd
import numpy as np

# Reading in the team data frame from previous scripts
team_df = pd.read_csv('../Data/modified_team_df.csv', index_col=0 )

# There are a bunch of columns that are not from data that was gathered
# before the 15 minute mark so we are going to go ahead and drop those
cols_to_drop = ['teamdragkills', 'oppdragkills', 'oppelementals',
               'firedrakes', 'waterdrakes', 'earthdrakes', 'airdrakes',
               'elders', 'oppelders', 'firstmidouter', 'firsttothreetowers',
               'teamtowerkills', 'opptowerkills', 'fbaron', 'teambaronkills',
               'oppbaronkills', 'fbarontime', 'dmgtochamps', 'wards',
               'wardshare', 'wardkills', 'totalgold', 'goldspent',
               'minionkills', 'monsterkills', 'elementals','monsterkills',
               'monsterkillsownjungle', 'monsterkillsenemyjungle', 'k', 'd', 'a',
               'teamkills', 'teamdeaths', 'gspd', 'gamelength', 'visionwards',
               'visionwardbuys', 'earnedgpm', 'cspm', 'kpm', 'okpm', 'ckpm',
               'dmgtochampsperminute', 'wpm' , 'wcpm', 'visiblewardclearrate',
               'invisiblewardclearrate']

dropped_team_df = team_df.drop(cols_to_drop, axis=1)

# This is going to have the per minute data as well as data from before
# the 15 minute mark. The idea here is that if the data is normalized to
# a per minute basis maybe we can try and get some important information
# as to features we should look at by the 15 minute mark as good indicators
# of future wins
per_min_cols_drop = ['teamdragkills', 'oppdragkills', 'oppelementals',
               'firedrakes', 'waterdrakes', 'earthdrakes', 'airdrakes',
               'elders', 'oppelders', 'firstmidouter', 'firsttothreetowers',
               'teamtowerkills', 'opptowerkills', 'fbaron', 'teambaronkills',
               'oppbaronkills', 'fbarontime', 'dmgtochamps', 'wards',
               'wardshare', 'wardkills', 'totalgold', 'goldspent',
               'minionkills', 'elementals', 'k', 'd', 'a',
               'teamkills', 'teamdeaths', 'gspd']

# These are columns that we can make into per min columns by dividing the value
# by the game time
per_min_cols = ['monsterkills', 'monsterkillsenemyjungle', 'monsterkillsownjungle',
               'visionwardbuys','visionwards']


def per_min_creation(df, cols):
    '''
    This function calculates the per minute stats by dividing the columns
    passed to it by the gamelength column

    : param df - The data frame to be analyzed needs to have a 'gamelength'
                column as well as the columns in the cols param
    : param cols - A list of columns to be divided by the 'gamelength' col
    : RETURNS a new dataframe with the cols values/gamelength
    '''

    # Loop to do the division for reach col in cols
    for col in cols:
        df[col] = df[col].divide(df['gamelength'])

    return df

# Dropping unnecessary columns and creating the per minute columns
# also asigning to a new dataframe
per_min_df = team_df.drop(per_min_cols_drop, axis=1)
per_min_df = per_min_creation(per_min_df, per_min_cols)
per_min_df.drop('gamelength', 1, inplace=True)

def above_time(df, time_list):
    '''
    This function replaces features that have a time value to them with
    -9999 if they are above a certain threshold in this case 15.0

    : param df - The data frame to be manipulated
    : param time_list - The list of features to be checked
    : RETURNS - A new dataframe with -9999 if the time_list cols is above
    the threshold of 15.0
    '''

    # Loop to check for > 15.0 and replace with -9999
    for col in time_list:
        df.loc[(df[col] > 15.0), col] = -9999

    return df

# Creating the time list.
time_list = ['fdtime', 'fbtime', 'fttime', 'heraldtime']

# Replacing time_list with -9999
per_min_df = above_time(per_min_df, time_list)
dropped_team_df = above_time(dropped_team_df, time_list)

# These features depend on a feature in the time_list so we are replaceing
# Them manually with -9999
fb_l = ['fb', 'fbassist', 'fbvictim']
per_min_df.loc[per_min_df['fbtime'] == -9999, fb_l] = -9999
per_min_df.loc[per_min_df['fttime'] == -9999, 'ft'] = -9999
per_min_df.loc[per_min_df['fdtime'] == -9999, 'fd'] = -9999
per_min_df.loc[per_min_df['heraldtime'] == -9999, 'herald'] = -9999

dropped_team_df.loc[dropped_team_df['fbtime'] == -9999, fb_l] = -9999
dropped_team_df.loc[dropped_team_df['fttime'] == -9999, 'ft'] = -9999
dropped_team_df.loc[dropped_team_df['fdtime'] == -9999, 'fd'] = -9999
dropped_team_df.loc[dropped_team_df['heraldtime'] == -9999, 'herald'] = -9999

# Getting dummies for the side (red/blue)
per_min_df = pd.get_dummies(per_min_df)
dropped_team_df = pd.get_dummies(dropped_team_df)

# Saving to a new file
# Commented out as they are provided in the github download
# per_min_df.to_csv('../Data/per_min_teamdf.csv')
# dropped_team_df.to_csv('../Data/dropped_teamdf.csv')

