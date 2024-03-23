import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('MIA_Rank_ordered_number_all_epoch_all_layer.csv')
print(df.columns)
df_avg = df['0']
plt.plot(df_avg)
plt.show()