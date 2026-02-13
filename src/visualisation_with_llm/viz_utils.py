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
    df = df.copy()

    # Drop fully empty rows/columns
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")

    # Clean strings
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace(["", "nan", "None", "NULL"], pd.NA)

    # Smart numeric conversion
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    return df


# =========================================================
# VALIDATION
# =========================================================

def column_is_valid(df: pd.DataFrame, col: str) -> bool:
    if not col or col not in df.columns:
        return False

    series = df[col].dropna()

    if series.empty:
        return False

    if series.nunique() == 0:
        return False

    return True


def is_numeric(df, col):
    return col in df.select_dtypes(include="number").columns


def is_categorical(df, col):
    return col in df.select_dtypes(exclude="number").columns


# =========================================================
# AUTO CORRECTION (ANTI-LLM ERRORS)
# =========================================================

def auto_correct_spec(df, spec):

    plot_type = spec.get("type", "").lower().strip()
    x = spec.get("x")
    y = spec.get("y")

    # Histogram sur colonne catégorielle → count
    if plot_type == "histogram" and is_categorical(df, x):
        spec["type"] = "count"

    # Scatter → x et y doivent être numériques
    if plot_type == "scatter":
        if not (is_numeric(df, x) and is_numeric(df, y)):
            spec["type"] = "invalid"

    # Boxplot → y doit être numérique
    if plot_type == "boxplot":
        if not is_numeric(df, y):
            spec["type"] = "invalid"

    # Bar avec y → y doit être numérique
    if plot_type == "bar" and y:
        if not is_numeric(df, y):
            spec["type"] = "invalid"

    return spec


# =========================================================
# SAFE PLOT FUNCTIONS
# =========================================================

def plot_bar(df, x, y=None, title="", palette="deep"):
    fig, ax = plt.subplots(figsize=(8, 5))

    if y:
        data = df.dropna(subset=[x, y])
        if data.empty:
            return empty_plot("Aucune donnée valide")
        sns.barplot(data=data, x=x, y=y, palette=palette, ax=ax)
    else:
        value_counts = df[x].dropna().value_counts()
        if value_counts.empty:
            return empty_plot("Aucune donnée valide")

        sns.barplot(
            x=value_counts.index,
            y=value_counts.values,
            palette=palette,
            ax=ax
        )
        ax.set_ylabel("Count")

    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_count(df, x, title="", palette="deep"):
    data = df.dropna(subset=[x])
    if data.empty:
        return empty_plot("Aucune donnée valide")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=data, x=x, palette=palette, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_scatter(df, x, y, hue=None, title="", palette="deep"):
    data = df.dropna(subset=[x, y])
    if data.empty:
        return empty_plot("Aucune donnée valide")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=data,
        x=x,
        y=y,
        hue=hue if hue and column_is_valid(df, hue) else None,
        palette=palette if hue else None,
        ax=ax
    )
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_line(df, x, y, title=""):
    data = df.dropna(subset=[x, y])
    if data.empty:
        return empty_plot("Aucune donnée valide")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=data, x=x, y=y, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_boxplot(df, x, y, title="", palette="deep"):
    data = df.dropna(subset=[x, y])
    if data.empty:
        return empty_plot("Aucune donnée valide")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=data, x=x, y=y, palette=palette, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_histogram(df, column, title="", bins=20):
    data = df[column].dropna()
    if data.empty:
        return empty_plot("Aucune donnée valide")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data, kde=True, bins=bins, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_heatmap(df, title="Correlation Heatmap", cmap="coolwarm"):
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        return empty_plot("Pas assez de colonnes numériques")

    corr_matrix = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap=cmap, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_pairplot(df, hue=None):
    numeric_cols = df.select_dtypes(include="number").columns

    if len(numeric_cols) < 2:
        return empty_plot("Pairplot nécessite 2 colonnes numériques")

    df_clean = df[numeric_cols].dropna().copy()

    if hue and hue in df.columns:
        df_clean[hue] = df[hue]

    g = sns.pairplot(df_clean, hue=hue if hue else None)
    return g.fig


def empty_plot(message):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.axis("off")
    return fig


# =========================================================
# EXPORT
# =========================================================

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"


# =========================================================
# MAIN DISPATCHER (FINAL SAFE VERSION)
# =========================================================

def generate_plot_base64(df: pd.DataFrame, spec: dict):

    apply_theme()

    df = preprocess_dataframe(df)
    spec = auto_correct_spec(df, spec)

    plot_type = spec.get("type", "").lower().strip()
    x = spec.get("x")
    y = spec.get("y")
    hue = spec.get("hue")
    title = spec.get("title", "")

    # Validate columns
    for col in [x, y, hue]:
        if col and not column_is_valid(df, col):
            return fig_to_base64(empty_plot(f"Colonne '{col}' invalide"))

    if plot_type == "invalid":
        return fig_to_base64(empty_plot("Spécification invalide"))

    try:
        if plot_type == "bar":
            fig = plot_bar(df, x, y, title)
        elif plot_type == "count":
            fig = plot_count(df, x, title)
        elif plot_type == "scatter":
            fig = plot_scatter(df, x, y, hue, title)
        elif plot_type == "line":
            fig = plot_line(df, x, y, title)
        elif plot_type == "boxplot":
            fig = plot_boxplot(df, x, y, title)
        elif plot_type == "histogram":
            fig = plot_histogram(df, x, title)
        elif plot_type == "heatmap":
            fig = plot_heatmap(df, title)
        elif plot_type == "pairplot":
            fig = plot_pairplot(df, hue)
        else:
            return fig_to_base64(empty_plot("Type de plot inconnu"))

        return fig_to_base64(fig)

    except Exception as e:
        return fig_to_base64(empty_plot(f"Erreur: {e}"))
