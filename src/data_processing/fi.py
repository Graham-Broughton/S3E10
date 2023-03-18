import pandas as pd
from pandas.io.formats.style import Styler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# import os
# import sys
import datetime

rc = {
    "axes.facecolor": "#FFFEF8",
    "figure.facecolor": "#FFFEF8",
    "axes.edgecolor": "#000000",
    "grid.color": "#EBEBE7" + "30",
    "font.family": "serif",
    "axes.labelcolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "grid.alpha": 0.4
}
sns.set(rc=rc)
palette = ['#302c36', '#037d97', '#E4591E', '#C09741',
           '#EC5B6D', '#90A6B1', '#6ca957', '#D8E3E2']


def get_fi(model, X, CFG):
    feature_importance =  [model[x].feature_importances_ for x in range(CFG.NFOLDS*CFG.REPEATS)]
    feature_importance = np.average(feature_importance,axis=0)
    feature_df = pd.DataFrame(feature_importance, index=X.columns)

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.barplot(x=feature_df.values.squeeze(), y=feature_df.index,
                color=palette[-3], linestyle="-", width=0.5, errorbar='sd',
                linewidth=0.5, edgecolor="black", ax=ax)
    ax.set_title('Feature Importance', fontdict={'fontweight': 'bold'})
    ax.set(xlabel=None)

    for s in ['right', 'top']:
        ax.spines[s].set_visible(False)