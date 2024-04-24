import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def create_plot(df_acc, df_mia, process_name):
    # df_f1 = pd.read_csv('Result_rank_random_variable_epoch_all_layer.csv')
    # df_mia = pd.read_csv('MIA_Rank_ordered_number_all_epoch_all_layer.csv')
    df_mia[0] = df_mia[0].round(3)

    x = [i + 1 for i in range(df_acc.shape[0])]
    y = [i for i in df_acc[0]]
    n = df_mia[0].to_list()

    plt.figure(figsize=(8, 6))

    # fig, ax = plt.subplots()

    plt.plot(x, y, color='red', marker='o', ms=3, mec='k', mfc='k')

    for i, txt in enumerate(n):
        plt.annotate(txt, (x[i], y[i]), fontsize=6)

    plt.xlabel('Epochs', fontsize='10', horizontalalignment='center')
    plt.ylabel('F1 score', fontsize='10', horizontalalignment='center')

    blue_circle = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=3, label='MIA Score')
    plt.legend(handles=[blue_circle], loc='lower right')
    plt.savefig(f'mnist/result/{process_name}.png')