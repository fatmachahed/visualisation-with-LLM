import streamlit as st
import pandas as pd
import base64
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Nettoyage du cache
for module_name in ['src', 'src.visualisation_with_llm', 'src.visualisation_with_llm.llm_utils', 'src.visualisation_with_llm.viz_utils']:
    if module_name in sys.modules:
        del sys.modules[module_name]

from src.visualisation_with_llm.llm_utils import init_llm, generate_visualization_proposals
from src.visualisation_with_llm.viz_utils import plot, fig_to_base64

# =========================================================
# CONFIG PAGE
# =========================================================
st.set_page_config(
    page_title="📊 DataViz AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
    .main { background-color: #F7F9FC; }
    .block-container { padding-top: 2rem; }
    div[data-testid="stMetric"] {
        background-color: white; 
        padding: 15px; 
        border-radius: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    h1 { 
        color: #1F2937;
        font-size: 3rem;
        font-weight: 700;
    }
    h2, h3 { color: #1F2937; }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        border: none;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    .stButton>button:hover {
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
        transform: translateY(-2px);
    }
    .justification {
        background: #f8fafc;
        padding: 0.75rem;
        border-left: 3px solid #667eea;
        border-radius: 4px;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }
    .tech-details {
        background: #f0f4ff;
        padding: 0.5rem;
        border-radius: 6px;
        font-size: 0.85rem;
        margin: 0.5rem 0;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("🎨 Personnalisation")

color_mode = st.sidebar.radio("Mode de couleur", ["Palette Seaborn", "Couleur Unique"])
palette = "deep"
custom_color = "#4F8BF9"

if color_mode == "Palette Seaborn":
    palette = st.sidebar.selectbox("Palette", ["deep", "muted", "bright", "pastel", "colorblind"])
else:
    custom_color = st.sidebar.color_picker("Couleur", "#4F8BF9")

show_details = st.sidebar.checkbox("Détails techniques", False)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Configuration")

num_proposals = st.sidebar.slider(
    "Nombre de propositions",
    min_value=3,
    max_value=10,
    value=3,
    step=1,
    help="Le LLM générera ce nombre de visualisations"
)

graph_mode = st.sidebar.radio("Mode", ["🤖 Automatique", "🎛 Filtrage"])

graph_options = {
    "📈 Scatter": "scatter",
    "📉 Line": "line",
    "📊 Bar": "bar",
    "📦 Boxplot": "boxplot",
    "📊 Histogram": "histogram",
    "🔥 Heatmap": "heatmap",
    "🔢 Count": "count",
}

selected_types = []
if graph_mode == "🎛 Filtrage":
    selected_types = st.sidebar.multiselect(
        "Types autorisés",
        list(graph_options.keys()),
        help="Le LLM utilisera UNIQUEMENT ces types"
    )

st.sidebar.caption("💡 Une problématique claire = meilleures visualisations")

# =========================================================
# HEADER
# =========================================================
st.title("📊 DataViz AI Dashboard")
st.markdown("Génération automatique par Intelligence Artificielle")

# =========================================================
# INPUTS
# =========================================================
col1, col2 = st.columns([2, 1])

with col1:
    problem = st.text_area(
        "💡 Problématique",
        placeholder="Ex: Analyser les relations, identifier les tendances...",
        height=100
    )

with col2:
    uploaded_file = st.file_uploader("📁 CSV", type=["csv"])

if not uploaded_file:
    st.info("👆 Uploadez un fichier CSV")
    st.stop()

# Charger
try:
    df = pd.read_csv(uploaded_file)
    df = df.dropna(how="all").dropna(axis=1, how="all")
    
    if df.empty:
        st.error("❌ Dataset vide")
        st.stop()
    
    st.success(f"✅ {len(df)} lignes × {len(df.columns)} colonnes")
    
except Exception as e:
    st.error(f"❌ Erreur : {e}")
    st.stop()

# Aperçu
with st.expander("👁️ Aperçu", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Lignes", f"{len(df):,}")
    with col2:
        st.metric("📋 Colonnes", len(df.columns))
    with col3:
        numeric = df.select_dtypes(include='number').columns
        st.metric("🔢 Numériques", len(numeric))
    with col4:
        st.metric("❓ Manquantes", f"{df.isnull().sum().sum():,}")
    
    st.dataframe(df.head(10), use_container_width=True)

if not problem:
    st.warning("⚠️ Saisissez une problématique")
    st.stop()

# Boutons
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    gen_btn = st.button("🤖 Générer les propositions", use_container_width=True)

with col2:
    if "specs" in st.session_state:
        regen_btn = st.button("🔄 Régénérer", use_container_width=True)
    else:
        regen_btn = False

with col3:
    if "specs" in st.session_state:
        if st.button("🗑️ Effacer", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# =========================================================
# DATASET SUMMARIZATION
# =========================================================
def summarize_dataset(df: pd.DataFrame) -> str:
    summary = []
    summary.append(f"Nombre de lignes : {len(df)}")
    summary.append(f"Nombre de colonnes : {len(df.columns)}")
    summary.append("\nColonnes :")
    for col in df.columns:
        dtype = str(df[col].dtype)
        if pd.api.types.is_numeric_dtype(df[col]):
            summary.append(
                f"- {col} ({dtype}) | min={df[col].min()} | max={df[col].max()} | moyenne={df[col].mean():.2f}"
            )
        else:
            uniques = df[col].nunique()
            summary.append(f"- {col} ({dtype}) | valeurs uniques={uniques}")
    return "\n".join(summary)

# =========================================================
# GÉNÉRATION
# =========================================================
if gen_btn or regen_btn:
    with st.spinner("🔄 Génération des propositions..."):
        try:
            llm = init_llm()
            dataset_summary = summarize_dataset(df)
            
            if show_details:
                with st.expander("📄 Résumé LLM"):
                    st.code(dataset_summary)
            
            preferred_types = None
            allow_duplicates = False
            
            if graph_mode == "🎛 Filtrage" and selected_types:
                preferred_types = [graph_options[t] for t in selected_types]
                if len(preferred_types) < num_proposals:
                    allow_duplicates = True
                    st.info(f"🎯 Types autorisés : {', '.join(preferred_types)} (doublons autorisés)")
                else:
                    st.info(f"🎯 Types autorisés : {', '.join(preferred_types)}")
            else:
                st.info(f"🤖 Mode automatique - {num_proposals} visualisation(s)")
            
            specs = generate_visualization_proposals(
                llm,
                problem,
                dataset_summary,
                preferred_types=preferred_types,
                num_proposals=num_proposals,
                allow_duplicates=allow_duplicates
            )
            
            if not specs:
                st.error("❌ Aucune visualisation générée")
                st.stop()
            
            st.session_state["specs"] = specs
            st.session_state["df"] = df
            st.session_state["palette"] = palette
            st.session_state["color"] = custom_color
            st.session_state["selected_viz"] = None
            
            st.success(f"✅ {len(specs)} proposition(s) générée(s)")
            
        except Exception as e:
            st.error(f"❌ Erreur : {e}")
            if show_details:
                import traceback
                st.code(traceback.format_exc())
            st.stop()

# =========================================================
# AFFICHAGE DES PROPOSITIONS
# =========================================================
if "specs" in st.session_state and st.session_state["specs"]:
    specs = st.session_state["specs"]
    df = st.session_state["df"]
    palette = st.session_state.get("palette", "deep")
    color = st.session_state.get("color", "#4F8BF9")
    
    st.divider()
    st.header("📋 Propositions de visualisations")
    st.markdown("**Sélectionnez une visualisation pour la générer**")
    
    # Grille adaptative
    for row_start in range(0, len(specs), 3):
        cols = st.columns(3)
        row_specs = specs[row_start:row_start + 3]
        
        for idx_in_row, (col, spec) in enumerate(zip(cols, row_specs)):
            actual_idx = row_start + idx_in_row
            
            with col:
                st.subheader(f"📊 Proposition {actual_idx + 1}")
                st.markdown(f"**{spec.get('title', 'Sans titre')}**")
                
                st.markdown(f"**Type :** {spec.get('type', 'N/A').upper()}")
                st.markdown(f"**Variables :** {spec.get('x', 'N/A')} {('vs ' + spec.get('y')) if spec.get('y') else ''}")
                
                # JUSTIFICATION
                if spec.get('justification'):
                    st.markdown(f"""
                    <div class="justification">
                        <strong>💡 Justification :</strong><br>
                        {spec.get('justification')}
                    </div>
                    """, unsafe_allow_html=True)
                
                # DÉTAILS TECHNIQUES (TOUJOURS VISIBLES)
                with st.expander("🔧 Détails techniques", expanded=False):
                    st.markdown(f"""
                    <div class="tech-details">
                    📊 <strong>Type :</strong> {spec.get('type', 'N/A')}<br>
                    📈 <strong>Axe X :</strong> {spec.get('x', 'null')}<br>
                    📈 <strong>Axe Y :</strong> {spec.get('y', 'null')}<br>
                    📝 <strong>Titre :</strong> {spec.get('title', 'Sans titre')}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Bouton de sélection
                is_selected = st.session_state.get("selected_viz") == actual_idx
                
                if st.button(
                    "✓ Sélectionné" if is_selected else "Sélectionner",
                    key=f"select_{actual_idx}",
                    use_container_width=True,
                    disabled=is_selected,
                    type="primary" if is_selected else "secondary"
                ):
                    st.session_state["selected_viz"] = actual_idx
                    # Initialiser les paramètres de personnalisation
                    st.session_state["custom_title"] = spec.get('title', '')
                    st.session_state["filter_enabled"] = False
                    st.rerun()
    
    # =========================================================
    # VISUALISATION FINALE AVEC PERSONNALISATION
    # =========================================================
    if st.session_state.get("selected_viz") is not None:
        st.divider()
        st.header("📊 Visualisation finale")
        
        selected_idx = st.session_state["selected_viz"]
        selected_spec = specs[selected_idx].copy()
        
        st.info(f"**Visualisation sélectionnée :** {selected_spec.get('title', 'Sans titre')}")
        
        # =========================================================
        # PERSONNALISATION (NOUVEAU)
        # =========================================================
        with st.expander("⚙️ Personnaliser la visualisation", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                # Titre personnalisé
                custom_title = st.text_input(
                    "📝 Titre personnalisé",
                    value=st.session_state.get("custom_title", selected_spec.get('title', '')),
                    help="Modifiez le titre de la visualisation"
                )
                if custom_title:
                    selected_spec['title'] = custom_title
                    st.session_state["custom_title"] = custom_title
                
                # Options selon le type
                if selected_spec.get('type') == 'histogram':
                    bins = st.slider("📊 Nombre de bins", 5, 100, 20, help="Nombre de barres dans l'histogramme")
                    selected_spec['bins'] = bins
            
            with col2:
                # Filtrage des données
                filter_enabled = st.checkbox(
                    "🔍 Filtrer les données",
                    value=st.session_state.get("filter_enabled", False),
                    help="Afficher seulement certaines valeurs"
                )
                st.session_state["filter_enabled"] = filter_enabled
                
                if filter_enabled:
                    x_col = selected_spec.get('x')
                    y_col = selected_spec.get('y')
                    
                    # Filtrage selon les colonnes disponibles
                    if x_col and x_col in df.columns:
                        if pd.api.types.is_numeric_dtype(df[x_col]):
                            min_val = float(df[x_col].min())
                            max_val = float(df[x_col].max())
                            filter_range_x = st.slider(
                                f"Plage de {x_col}",
                                min_val,
                                max_val,
                                (min_val, max_val),
                                help=f"Filtrer les valeurs de {x_col}"
                            )
                            # Appliquer le filtre
                            df = df[(df[x_col] >= filter_range_x[0]) & (df[x_col] <= filter_range_x[1])]
                    
                    if y_col and y_col in df.columns and pd.api.types.is_numeric_dtype(df[y_col]):
                        min_val = float(df[y_col].min())
                        max_val = float(df[y_col].max())
                        filter_range_y = st.slider(
                            f"Plage de {y_col}",
                            min_val,
                            max_val,
                            (min_val, max_val),
                            help=f"Filtrer les valeurs de {y_col}"
                        )
                        # Appliquer le filtre
                        df = df[(df[y_col] >= filter_range_y[0]) & (df[y_col] <= filter_range_y[1])]
                    
                    st.caption(f"📊 Données filtrées : {len(df)} lignes")
        
        # Générer et afficher
        try:
            fig = plot(df, selected_spec, palette=palette, color=color)
            img_base64 = fig_to_base64(fig)
            
            st.markdown(
                f'<img src="{img_base64}" style="width:100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">',
                unsafe_allow_html=True
            )
            
            # Export PNG
            img_bytes = base64.b64decode(img_base64.split(',')[1])
            st.download_button(
                label="💾 Télécharger PNG",
                data=img_bytes,
                file_name=f"visualization_{selected_spec.get('type', 'chart')}.png",
                mime="image/png",
                use_container_width=True
            )
            
            if show_details:
                with st.expander("ℹ️ Détails de la visualisation"):
                    st.json(selected_spec)
            
        except Exception as e:
            st.error(f"❌ Erreur graphique")
            st.exception(e)
        
        # Options supplémentaires
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📋 Export JSON", use_container_width=True):
                import json
                specs_json = json.dumps(specs, indent=2, ensure_ascii=False)
                st.download_button(
                    "💾 JSON",
                    specs_json,
                    "specs.json",
                    "application/json",
                    use_container_width=True
                )
        
        with col2:
            if st.button("📊 Export CSV", use_container_width=True):
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "💾 CSV",
                    csv,
                    "data.csv",
                    "text/csv",
                    use_container_width=True
                )

else:
    st.info("👆 Cliquez sur 'Générer les propositions' pour commencer")

st.divider()
st.caption("🤖 Propulsé par Google Gemini 2.0 • DataViz AI")