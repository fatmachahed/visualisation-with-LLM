import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from io import BytesIO
import base64

# =========================================================
# THEME
# =========================================================
def apply_theme():
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12

# =========================================================
# DATA PREPROCESSING
# =========================================================
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().replace(["", "nan", "None", "NULL"], pd.NA)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df

# =========================================================
# HELPER
# =========================================================
def empty_plot(message="Aucune donnée valide"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.axis("off")
    return fig

def auto_layout_labels(ax, axis='x'):
    """Adapte la rotation et layout selon le nombre de ticks"""
    if axis == 'x':
        labels = ax.get_xticklabels()
    else:
        labels = ax.get_yticklabels()
    n = len(labels)
    if n > 10:
        rotation = 90 if axis == 'x' else 0
        ha = "center" if axis == 'x' else "right"
    elif n > 5:
        rotation = 45 if axis == 'x' else 0
        ha = "right"
    else:
        rotation = 0
        ha = "center"
    if axis == 'x':
        plt.setp(labels, rotation=rotation, ha=ha)
    else:
        plt.setp(labels, rotation=rotation, ha=ha)

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"

# =========================================================
# GENERAL PLOT FUNCTION
# =========================================================
def plot(df, spec, palette='deep', color='#4F8BF9'):
    apply_theme()
    df = preprocess_dataframe(df)

    plot_type = spec.get("type", "").lower()
    x = spec.get("x")
    y = spec.get("y")
    hue = spec.get("hue") if spec.get("hue") in df.columns else None
    title = spec.get("title", "")

    try:
        fig, ax = plt.subplots(figsize=(max(6, len(df[x].unique())/2), 5) if x else (8,5))

        if plot_type == "bar":
            if y:
                sns.barplot(data=df, x=x, y=y, palette=palette, ax=ax)
            else:
                sns.countplot(data=df, x=x, palette=palette, ax=ax)
            auto_layout_labels(ax, 'x')

        elif plot_type == "count":
            sns.countplot(data=df, x=x, palette=palette, ax=ax)
            auto_layout_labels(ax, 'x')

        elif plot_type == "scatter":
            sns.scatterplot(data=df, x=x, y=y, hue=hue, palette=palette if hue else None, color=None if hue else color, ax=ax)

        elif plot_type == "line":
            sns.lineplot(data=df, x=x, y=y, ax=ax)

        elif plot_type == "boxplot":
            sns.boxplot(data=df, x=x, y=y, palette=palette, ax=ax)
            auto_layout_labels(ax, 'x')

        elif plot_type == "histogram":
            data = df[x].dropna()
            sns.histplot(data, kde=True, bins=spec.get("bins", 20), color=color, ax=ax)

        elif plot_type == "heatmap":
            numeric_df = df.select_dtypes(include="number")
            if numeric_df.shape[1] < 2:
                return empty_plot("Pas assez de colonnes numériques")
            corr = numeric_df.corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            auto_layout_labels(ax, 'x')
            auto_layout_labels(ax, 'y')

        elif plot_type == "pairplot":
            numeric_cols = df.select_dtypes(include="number").columns
            if len(numeric_cols) < 2:
                return empty_plot("Pas assez de colonnes numériques")
            g = sns.pairplot(df[numeric_cols], diag_kind='kde', corner=True)
            g.fig.tight_layout()
            return g.fig

        else:
            return empty_plot("Type de plot inconnu")

        ax.set_title(title)
        fig.tight_layout()
        return fig

    except Exception as e:
        return empty_plot(f"Erreur: {e}")
