import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler

palette = ['#3c3744', '#048BA8', '#EE6352', '#E1BB80', '#78BC61']
grey_palette = ['#8e8e93', '#636366', '#48484a', '#3a3a3c', '#2c2c2e', '#1c1c27']
bg_color = '#F6F5F5'
white_color = '#d1d1d6'

custom_params = {
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.spines.left': False,
    'grid.alpha':0.2,
    'figure.figsize': (16, 6),
    'axes.titlesize': 'large',
    'axes.labelsize': 'large',
    'ytick.labelsize': 'medium',
    'xtick.labelsize': 'medium',
    'legend.fontsize': 'large',
    'lines.linewidth': 1,
    'axes.prop_cycle': cycler('color',palette),
    'figure.facecolor': bg_color,
    'figure.edgecolor': bg_color,
    'axes.facecolor': bg_color,
    'text.color':grey_palette[1],
    'axes.labelcolor':grey_palette[1],
    'axes.edgecolor':grey_palette[1],
    'xtick.color':grey_palette[1],
    'ytick.color':grey_palette[1],
    'figure.dpi':150,
}
sns.set_theme(
    style='whitegrid',
    palette=sns.color_palette(palette),
    rc=custom_params)


def get_fi(model, X, CFG, model_name):
    feature_importance =  [model[x].feature_importances_ for x in range(CFG.NFOLDS*CFG.REPEATS)]
    feature_importance = np.average(feature_importance,axis=0)
    feature_df = pd.DataFrame(feature_importance, index=X.columns)

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.barplot(x=feature_df.values.squeeze(), y=feature_df.index,
                color=palette[-3], linestyle="-", width=0.5, errorbar='sd',
                linewidth=0.5, edgecolor="black", ax=ax)
    ax.set_title(f'Feature Importance for {model_name}', fontdict={'fontweight': 'bold'})
    ax.set(xlabel=None)
