# This data comes from http://oracleselixir.com/
# Support the work that he does!

import pandas as pd
import numpy as np

# Reading in the raw file and then looking at some of the data. Most of this
# work was actually done in juptyr so some of the things I do don't make a
# lot of sense just looking at the script.
raw_df = pd.read_csv('../Data/2018-spring-match-data-OraclesElixir-2018-03-27.csv')
print(raw_df['playerid'].unique())

# I noticed that those with a playerid of 100 or 200 are the teams so I am
# going to create two new dfs and then save those for later
team_df = raw_df.loc[(raw_df['playerid'] == 100) | (raw_df['playerid'] == 200)]
player_df = raw_df.loc[(raw_df['playerid'] != 100) | (raw_df['playerid'] != 200)]

# Commented out as they are included in the github repo
# team_df.to_csv('./Data/team_data.csv')
# player_df.to_csv('./Data/player_data.csv')

# A list of columns to drop as they shouldn't be important to prediction
# of winners which is the goal of this set of scripts
drop_list = ['split', 'date', 'week', 'patchno', 'playerid',
            'position', 'player', 'champion', 'doubles', 'triples',
            'quadras', 'pentas', 'dmgshare', 'earnedgoldshare',
             'ban1', 'ban2', 'ban3', 'ban4', 'ban5', 'team']

team_df.drop(drop_list, axis=1, inplace=True)

# Getting rid of the LPL league as a lof of potentially relevent information
# is missing
team_df = team_df[team_df['league'] != 'LPL']
print(team_df['league'].unique())

# Getting rid of some more solumns
team_df.drop(['league', 'game'], 1, inplace=True)

# Taking a look at some of the columns that have null data in them
print(team_df[team_df['fbtime'].isnull()]['fbtime'])

# Filling with -9999 as the columns that are null actually mean that the
# objective was not taken
team_df.fillna(-9999, inplace=True)
print(team_df.columns[team_df.isnull().any()])

# Converting everything to numeric if it isn't already except side
cols = team_df.columns.drop('side')
team_df[cols] = team_df[cols].apply(pd.to_numeric, errors='coerce', axis=1)

# Commented out as is already included in repo
#team_df.to_csv('./Data/modified_team_df.csv')

