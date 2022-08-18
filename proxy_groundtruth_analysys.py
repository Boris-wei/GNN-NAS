import pandas as pd
import matplotlib.pyplot as plt

path1 = 'results/proxy_grid_personal_graph_mod/agg/test.csv'
csvData1 = pd.read_csv(path1)
csvData1.sort_values(["loss"], axis=0, ascending=[True], inplace=True)
x = csvData1.serial.values
x = pd.Series(x)
path2 = 'results/personal_graph_grid_personal_graph_mod/agg/test.csv'
csvData2 = pd.read_csv(path2)
csvData2.sort_values(["auc"], axis=0, ascending=[False], inplace=True)
y = csvData2.serial.values
y = pd.Series(y)

rho = x.corr(y, method="spearman")  # compute spearman's rho
tau = x.corr(y, method="kendall")  # compute kendall's tau
print('spearman\'s rho = %f'% rho)
print('kendall\'s tau = %f'% tau)
fig, ax = plt.subplots()
plt.scatter(x, y)
plt.show()