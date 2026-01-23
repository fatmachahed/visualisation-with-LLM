import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_bar(df: pd.DataFrame, x: str, y: str, title: str = "", save_path: str = None):
    """
    Génère un graphique en barres simple.
    """
    plt.figure(figsize=(8,5))
    sns.barplot(data=df, x=x, y=y)
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_scatter(df: pd.DataFrame, x: str, y: str, hue: str = None, title: str = "", save_path: str = None):
    """
    Génère un scatter plot.
    """
    plt.figure(figsize=(8,5))
    sns.scatterplot(data=df, x=x, y=y, hue=hue)
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_line(df: pd.DataFrame, x: str, y: str, title: str = "", save_path: str = None):
    """
    Génère un graphique linéaire.
    """
    plt.figure(figsize=(8,5))
    sns.lineplot(data=df, x=x, y=y)
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def save_figure(fig, filename: str):
    """
    Sauvegarde une figure matplotlib au format PNG.
    """
    fig.savefig(filename, dpi=300)
    print(f"Figure sauvegardée : {filename}")
