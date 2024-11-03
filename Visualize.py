import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import OrderedDict
# Data to visualize
data ={'CMG': {'ORG': 60.80792079207921, 'SENSEI': 93.31353135313532, 'DIST': 82.32343234323433, 'PGD': 82.33663366336634, 'BIM': 100.0, 'CMG': 91.34323432343234}, 'PGD': {'ORG': 55.10495049504951, 'SENSEI': 96.81518151815182, 'DIST': 84.81518151815182, 'PGD': 100.0, 'BIM': 100.0, 'CMG': 46.73927392739274}, 'SENSEI': {'ORG': 56.13465346534653, 'SENSEI': 99.996699669967, 'DIST': 85.25742574257426, 'PGD': 87.20132013201321, 'BIM': 100.0, 'CMG': 47.76237623762376}, 'BIM': {'ORG': 55.912871287128716, 'SENSEI': 96.55445544554455, 'DIST': 84.21782178217822, 'PGD': 86.82838283828383, 'BIM': 100.0, 'CMG': 47.584158415841586}, 'DIST': {'ORG': 56.32079207920792, 'SENSEI': 99.48514851485149, 'DIST': 97.6996699669967, 'PGD': 94.93399339933994, 'BIM': 100.0, 'CMG': 47.587458745874585}}
pth = 'FOOD101-FEW-OVERALL'
save_pth = pth + '.png'


new_order = ['ORG', 'DIST', 'BIM', 'SENSEI', 'PGD','CMG']
reordered_data = {}
for key, sub_dict in data.items():
    reordered_data[key] = OrderedDict((k, sub_dict[k]) for k in new_order if k in sub_dict)

data=reordered_data

df = pd.DataFrame(data)

#ORG 
df.loc['ORG', 'ORG'] = 44.59

save_pth = pth + '_ncb.png'


# df.loc[df.index != 'ORG', 'ORG'] = '/'

# 计算前五列的均值并添加为新列
mean_row = df.iloc[:, :5].mean()
mean_row.name = 'Mean'
df = pd.concat([df, mean_row.to_frame().T])


# 绘制热力图
plt.figure(figsize=(10, 8))
heatmap=sns.heatmap(df, annot=True, cmap='YlGnBu', fmt='.2f',annot_kws={'size': 25}, cbar=False)

# colorbar = heatmap.collections[0].colorbar
# colorbar.ax.tick_params(labelsize=14)

plt.tight_layout()
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.subplots_adjust(top=0.95,left=0.05)
plt.title(pth,fontsize=25)
plt.savefig("FigRes/" + save_pth)
plt.show()
