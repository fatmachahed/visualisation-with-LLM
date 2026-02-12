# src/visualisation_with_llm/viz_utils.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# ==========================================================
# THEME & GLOBAL CONFIG
# ==========================================================

def apply_theme():
    """
    Applique un thème propre et académique.
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 100


# ==========================================================
# VALIDATION UTILITIES
# ==========================================================

def validate_columns(df: pd.DataFrame, columns: list):
    """
    Vérifie que les colonnes existent dans le DataFrame.
    """
    for col in columns:
        if col and col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataset.")


def get_numeric_columns(df: pd.DataFrame):
    return df.select_dtypes(include="number").columns.tolist()


def get_categorical_columns(df: pd.DataFrame):
    return df.select_dtypes(exclude="number").columns.tolist()


# ==========================================================
# PLOT FUNCTIONS
# ==========================================================

def plot_bar(df, x, y, title="", palette="deep"):
    validate_columns(df, [x, y])
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x=x, y=y, palette=palette, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_scatter(df, x, y, hue=None, title="", palette="deep"):
    validate_columns(df, [x, y])
    if hue:
        validate_columns(df, [hue])

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        palette=palette if hue else None,
        ax=ax
    )
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_line(df, x, y, title="", palette="deep"):
    validate_columns(df, [x, y])
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=df, x=x, y=y, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_boxplot(df, x, y, title="", palette="deep"):
    validate_columns(df, [x, y])
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x=x, y=y, palette=palette, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_histogram(df, column, title="", palette="deep"):
    validate_columns(df, [column])
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[column], kde=True, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_heatmap(df, title="Correlation Heatmap", cmap="coolwarm"):
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        raise ValueError("Heatmap requires at least two numeric columns.")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap=cmap,
        ax=ax
    )
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ==========================================================
# EXPORT FUNCTION
# ==========================================================

def save_figure(fig, filename: str):
    """
    Sauvegarde une figure au format PNG.
    """
    fig.savefig(filename, dpi=300)
    return filename


# ==========================================================
# DISPATCHER (LLM SPEC -> GRAPH)
# ==========================================================

def generate_plot(df: pd.DataFrame, spec: dict):
    """
    Génère un graphique à partir d'une spécification fournie par le LLM.

    spec attendu :
    {
        "type": "scatter",
        "x": "surface",
        "y": "price",
        "hue": "district",
        "title": "Surface vs Price",
        "palette": "colorblind"
    }
    """

    apply_theme()

    plot_type = spec.get("type")
    x = spec.get("x")
    y = spec.get("y")
    hue = spec.get("hue")
    title = spec.get("title", "")
    palette = spec.get("palette", "deep")

    if plot_type == "bar":
        return plot_bar(df, x, y, title, palette)

    elif plot_type == "scatter":
        return plot_scatter(df, x, y, hue, title, palette)

    elif plot_type == "line":
        return plot_line(df, x, y, title, palette)

    elif plot_type == "boxplot":
        return plot_boxplot(df, x, y, title, palette)

    elif plot_type == "histogram":
        return plot_histogram(df, x, title, palette)

    elif plot_type == "heatmap":
        return plot_heatmap(df, title)

    else:
        raise ValueError(f"Unsupported plot type: {plot_type}")
