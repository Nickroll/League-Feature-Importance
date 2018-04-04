Readme
-----------
This repo contains my scripts to analyze feature importance in League of Legends games.
Analysis was done on two different types of files. The Dropped type and the per type.
Most of the scripts files were actually .ipynb files created in juptyr notebook and
were since converted to .py files. I have gone through and tried to make them
as useful as possible and clean them up but there is no gaurentee that they will work.

The original data was downloaded from: http://oracleselixir.com/

**WARNING**

The Model Training file will take forever to run if not done on a computer with
a large number of cores. This is because it implements scikit-learn's
GridSearchCv and cross_val_score with a large set of paramater dictonaries.
It was run overnight on AWS EC2 c5.2xlarge compute optimized with 8 vCPU's.

Dropped
_______________
The dropped type of file is one whcih only has those features that that could be
known at 15 minutes into the game. THis means that any feautre that is not explicitly
stated to be before 15 minutes is dropped from the data frame.

Per
_______________
The per type includes data that is per minute. In this case the data could be known
by 15 mintues, if it was being kept track of, but since it is for an entire game the
data is likely skewed as many fights, kills, and csing happens after 15 minutes. However,
the feature importance ranking that is present may be useful if data can be
collected that has those stats at the 15 minute mark.

