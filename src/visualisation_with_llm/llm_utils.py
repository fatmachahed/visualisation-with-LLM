import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =========================================================
# CONFIG PAGE
# =========================================================
st.set_page_config(
    page_title="DataViz AI",
    layout="wide",
)

# =========================================================
# CUSTOM CSS (design clair + cartes)
# =========================================================
st.markdown("""
<style>
.main {
    background-color: #F7F9FC;
}

.block-container {
    padding-top: 2rem;
}

div[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}

h1, h2, h3 {
    color: #1F2937;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("🎨 Personnalisation")

color_mode = st.sidebar.radio("Mode de couleur", ["Palette Seaborn", "Couleur Unique"])
palette = None
custom_color = "#4F8BF9"

if color_mode == "Palette Seaborn":
    palette = st.sidebar.selectbox(
        "Palette Seaborn",
        ["deep", "muted", "bright", "pastel", "dark", "colorblind"]
    )
else:
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
# HELPER FUNCTION
# =========================================================
def plot_graph(df, spec):
    """Génère un graphique avec bonnes pratiques selon le type."""
    plot_type = spec["type"]
    x = spec["x"]
    y = spec["y"]
    hue = spec["hue"]
    bins = spec.get("bins", 20)
    orientation = spec.get("orientation", "v")

    fig, ax = plt.subplots(figsize=(10,6))
    
    # Définir kwargs couleurs/palette
    kwargs = {}
    if color_mode == "Palette Seaborn" and palette:
        if plot_type in ["boxplot", "violin", "bar", "stacked_bar"]:
            kwargs["palette"] = palette
    else:
        kwargs["color"] = custom_color

    # Scatter
    if plot_type == "scatter":
        sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax, **kwargs)
    # Line
    elif plot_type == "line":
        sns.lineplot(data=df, x=x, y=y, hue=hue, ax=ax, **kwargs)
    # Boxplot
    elif plot_type == "boxplot":
        sns.boxplot(data=df, x=x, y=y, hue=hue, ax=ax, **kwargs)
    # Violin
    elif plot_type == "violin":
        sns.violinplot(data=df, x=x, y=y, hue=hue, ax=ax, **kwargs)
    # Bar
    elif plot_type == "bar":
        sns.barplot(data=df, x=x, y=y, hue=hue, ax=ax, **kwargs)
    # Stacked bar
    elif plot_type == "stacked_bar":
        # empilement simulé via pivot
        pivot = df.pivot_table(index=x, columns=hue, values=y, aggfunc="sum").fillna(0)
        pivot.plot(kind="bar", stacked=True, ax=ax, color=sns.color_palette(palette) if palette else None)
    # Histogram
    elif plot_type == "histogram":
        sns.histplot(df[x], bins=bins, kde=True, ax=ax, **kwargs)
    # Heatmap
    elif plot_type == "heatmap":
        numeric_cols = df.select_dtypes(include="number").columns
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    # Count plot
    elif plot_type == "count":
        sns.countplot(data=df, x=x, hue=hue, ax=ax, **kwargs)
    # Pairplot
    elif plot_type == "pairplot":
        sns.pairplot(df.select_dtypes(include="number"), palette=palette if palette else None)
        st.pyplot(plt.gcf())
        return

    # Bonnes pratiques : labels lisibles
    plt.xticks(rotation=45, ha="center")
    plt.title(spec.get("title", "Graphique"))
    if show_grid:
        ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# =========================================================
# MAIN
# =========================================================
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = df.dropna(how="all").dropna(axis=1, how="all")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    # Sidebar stats
    st.sidebar.metric("Lignes", df.shape[0])
    st.sidebar.metric("Colonnes", df.shape[1])

    st.markdown("## 📊 Visualisations Automatiques")

    # Générer 3 graphiques automatiquement
    specs = [
        {"type":"scatter","x":numeric_cols[0] if numeric_cols else None,
         "y":numeric_cols[1] if len(numeric_cols)>1 else None,
         "hue":None,"title":"Scatter Plot","bins":20,"orientation":"v","palette":"deep","justification":""},
        {"type":"boxplot","x":categorical_cols[0] if categorical_cols else None,
         "y":numeric_cols[0] if numeric_cols else None,
         "hue":None,"title":"Boxplot","bins":20,"orientation":"v","palette":"deep","justification":""},
        {"type":"heatmap","x":None,"y":None,"hue":None,"title":"Matrice de corrélation","bins":20,"orientation":"v","palette":"deep","justification":""}
    ]

    for spec in specs:
        plot_graph(df, spec)

else:
    st.info("Veuillez importer un fichier CSV pour commencer.")
