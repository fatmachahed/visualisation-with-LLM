import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
from io import BytesIO

# =========================================================
# CONFIG PAGE
# =========================================================
st.set_page_config(
    page_title="DataViz AI",
    layout="wide",
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
.main { background-color: #F7F9FC; }
.block-container { padding-top: 2rem; }
div[data-testid="stMetric"] {
    background-color: white; padding: 15px; border-radius: 12px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
h1, h2, h3 { color: #1F2937; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("🎨 Personnalisation")
palette = st.sidebar.selectbox(
    "Palette Seaborn",
    ["deep", "muted", "bright", "pastel", "dark", "colorblind"]
)
custom_color = st.sidebar.color_picker("Couleur principale", "#4F8BF9")
show_grid = st.sidebar.checkbox("Afficher la grille", True)
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Infos dataset")

# =========================================================
# HEADER
# =========================================================
st.title("📊 DataViz AI Dashboard")
problem = st.text_area(
    "Problématique",
    placeholder="Ex: analyser les relations entre variables..."
)
uploaded_file = st.file_uploader("Uploader un fichier CSV", type=["csv"])

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def apply_palette_or_color(plot_type, kwargs):
    if plot_type in ["scatter", "line", "histogram"]:
        kwargs["color"] = custom_color
    else:
        kwargs["palette"] = palette
    return kwargs

def auto_layout(ax):
    labels = ax.get_xticklabels()
    if len(labels) > 10:
        plt.setp(labels, rotation=90, ha="center")
    elif len(labels) > 5:
        plt.setp(labels, rotation=45, ha="right")
    else:
        plt.setp(labels, rotation=0, ha="center")

def fig_to_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf

def plot_graph(df, plot_type, x=None, y=None, hue=None, title=""):
    fig, ax = plt.subplots(figsize=(max(6, len(df[x].unique())/2 if x else 6), 5))
    kwargs = apply_palette_or_color(plot_type, {})

    try:
        if plot_type == "scatter":
            sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax, **kwargs)
        elif plot_type == "line":
            sns.lineplot(data=df, x=x, y=y, hue=hue, ax=ax, **kwargs)
        elif plot_type == "boxplot":
            sns.boxplot(data=df, x=x, y=y, hue=hue, ax=ax, **kwargs)
        elif plot_type == "bar":
            sns.barplot(data=df, x=x, y=y, hue=hue, ax=ax, **kwargs)
        elif plot_type == "histogram":
            sns.histplot(df[x], kde=True, ax=ax, **kwargs)
        elif plot_type == "heatmap":
            numeric_cols = df.select_dtypes(include="number").columns
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        else:
            ax.text(0.5, 0.5, f"Type '{plot_type}' non supporté", ha="center", va="center")

        if plot_type not in ["heatmap"]:
            auto_layout(ax)
        ax.set_title(title)
        if show_grid:
            ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # Bouton de téléchargement
        buf = fig_to_bytes(fig)
        st.download_button(
            label="📥 Télécharger ce graphique",
            data=buf,
            file_name=f"{title.replace(' ','_')}.png",
            mime="image/png"
        )

    except Exception as e:
        st.error(f"Erreur génération graphique: {e}")

# =========================================================
# MAIN
# =========================================================
if uploaded_file:
    df = pd.read_csv(uploaded_file).dropna(how="all").dropna(axis=1, how="all")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    st.sidebar.metric("Lignes", df.shape[0])
    st.sidebar.metric("Colonnes", df.shape[1])

    st.markdown("## 📊 Visualisations Automatiques")

    n_graphs = st.sidebar.number_input("Nombre de graphes", min_value=1, max_value=6, value=3, step=1)
    regenerate = st.sidebar.button("Regénérer les graphes")
    plot_types = ["scatter", "boxplot", "bar", "histogram", "line", "heatmap"]

    for i in range(n_graphs):
        plot_type = random.choice(plot_types)
        title = f"{plot_type.capitalize()} automatique #{i+1}"
        x = random.choice(numeric_cols) if numeric_cols else None
        y = random.choice([c for c in numeric_cols if c != x]) if numeric_cols else None
        hue = random.choice(categorical_cols) if categorical_cols else None
        plot_graph(df, plot_type, x=x, y=y, hue=hue, title=title)

else:
    st.info("Veuillez importer un fichier CSV pour commencer.")
