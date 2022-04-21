import numpy as np
import csv
import seaborn as sns
import pandas as pd
from itertools import chain

data1 = {"rewards":[], "Episodes x 10":[], "type":[]}

for i in range(1):
	games_df = pd.read_csv(f'/lab/ssontakk/cs699_dynamics_of_representation_learning/loss_landscape/data/halfcheetah-expert-v1/Vanilla/{i}/monitor.csv', skiprows=2, header = None)
	# sumRew = []
	for j in range(600):
		# print(i, j)
		data1["rewards"].append(games_df[0].iloc[j])
		data1["Episodes x 10"].append(j)
		data1["type"].append("No Pretraining")


for i in range(1):
	games_df = pd.read_csv(f'/lab/ssontakk/cs699_dynamics_of_representation_learning/loss_landscape/data/halfcheetah-expert-v1/BCBackboneRL/{i}/monitor.csv', skiprows=2, header = None)
	# sumRew = []
	for j in range(600):
		# print(i, j)
		data1["rewards"].append(games_df[0].iloc[j])
		data1["Episodes x 10"].append(j)
		data1["type"].append("BC Backbone")


# for i in range(9):
# 	games_df = pd.read_csv(f'/lab/ssontakk/ShELL/src/data/pong-mixed-v4/LB20Episode-Split-exp0.1/{i}_tmpYMult/monitor.csv', skiprows=2, header = None)
# 	# sumRew = []
# 	for j in range(750):
# 		# print(i, j)
# 		data1["rewards"].append(games_df[0].iloc[j])
# 		data1["Episodes x 10"].append(j)
# 		data1["type"].append("Pretrain 0.1")
# for i in range(10):
# 	games_df = pd.read_csv(f'/lab/ssontakk/ShELL/src/data/2000PointsPretrainLBMixed/{i}_tmpY1Mult/monitor.csv', skiprows=2, header = None)
# 	# sumRew = []
# 	for j in range(2355):
# 		print(i, j)
# 		data1["rewards"].append(games_df[0].iloc[j])
# 		data1["Episodes x 10"].append(j)
# 		data1["type"].append("2000 Points Pretrain LB Mixed")

# for i in range(10):
# 	games_df = pd.read_csv(f'/lab/ssontakk/ShELL/src/data/20pointsPretrain/{i}_tmpY1Mult/monitor.csv', skiprows=2, header = None)
# 	# sumRew = []
# 	for j in range(2500):
# 		print(i, j)
# 		data1["rewards"].append(games_df[0].iloc[j])
# 		data1["Episodes x 10"].append(j)
# 		data1["type"].append("20 Points Pretrained")

sns.set_theme(style="darkgrid")
sns_plot = sns.lineplot(data=data1,x="Episodes x 10",y="rewards",hue="type")
# sns_plot.set(xscale="log")
# sns_plot = sns.lineplot(data=data,x="t",y="rewards")

sns_plot.figure.savefig("HCBC.png")