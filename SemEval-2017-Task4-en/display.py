import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('score.csv')
fig, ax = plt.subplots()
index = list(df['model'])
for i in range(len(index)):
    ax.bar(index[i], df["AvgF1"].values[i], label=index[i])
ax.set_xlabel('SemEval 2017-4a')
ax.set_ylabel('Average F1 score')
# ax.set_xticklabels((''))
ax.set_ylim(top=1)
# ax.legend()
plt.title('Evaluation of embedding methods with SemEval 2017 task 4-a')
plt.show()
