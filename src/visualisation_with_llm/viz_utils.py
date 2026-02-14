import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from io import BytesIO
import base64
import numpy as np

# =========================================================
# CONFIGURATION GLOBALE
# =========================================================

def apply_theme():
    """Applique un thème professionnel et lisible"""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "figure.dpi": 120,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "axes.labelweight": "bold",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.frameon": True,
        "legend.shadow": True,
        "legend.fancybox": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "grid.alpha": 0.3,
        "grid.linestyle": "--"
    })


# =========================================================
# PREPROCESSING
# =========================================================

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie le DataFrame :
    - Supprime lignes/colonnes entièrement vides
    - Nettoie les strings
    - Convertit en numérique si possible
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Supprimer colonnes et lignes complètement vides
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")
    
    # Nettoyer les colonnes string
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace(["", "nan", "None", "NULL", "NaN"], pd.NA)
    
    # Convertir en numérique si possible
    for col in df.columns:
        try:
            # Essayer de convertir en numérique
            converted = pd.to_numeric(df[col], errors='coerce')
            # Si au moins 50% des valeurs sont converties avec succès, garder
            if converted.notna().sum() / len(df) > 0.5:
                df[col] = converted
        except Exception:
            pass
    
    return df


# =========================================================
# HELPERS
# =========================================================

def empty_plot(message="Aucune donnée valide"):
    """Crée un graphique vide avec un message"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, message, 
            ha="center", va="center", 
            fontsize=14, color="#666",
            bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
    ax.axis("off")
    fig.tight_layout()
    return fig


def auto_layout_labels(ax, axis='x', max_labels=15):
    """
    Ajuste automatiquement la rotation et l'alignement des labels
    selon leur nombre et longueur
    """
    if axis == 'x':
        labels = [label.get_text() for label in ax.get_xticklabels()]
    else:
        labels = [label.get_text() for label in ax.get_yticklabels()]
    
    n_labels = len(labels)
    max_length = max([len(str(l)) for l in labels]) if labels else 0
    
    # Trop de labels : réduire le nombre
    if n_labels > max_labels:
        if axis == 'x':
            step = n_labels // max_labels + 1
            ticks = ax.get_xticks()[::step]
            ax.set_xticks(ticks)
            labels = [labels[i] for i in range(0, len(labels), step)]
    
    # Déterminer la rotation selon la longueur et le nombre
    if axis == 'x':
        if max_length > 15 or n_labels > 10:
            rotation = 90
            ha = "center"
        elif max_length > 8 or n_labels > 5:
            rotation = 45
            ha = "right"
        else:
            rotation = 0
            ha = "center"
        
        plt.setp(ax.get_xticklabels(), rotation=rotation, ha=ha)
    else:
        if max_length > 20:
            rotation = 0
            ha = "right"
        else:
            rotation = 0
            ha = "right"
        
        plt.setp(ax.get_yticklabels(), rotation=rotation, ha=ha)


def fig_to_base64(fig):
    """Convertit une figure matplotlib en base64 pour affichage web"""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"


def validate_columns(df, columns):
    """Vérifie que les colonnes existent dans le DataFrame"""
    missing = [col for col in columns if col and col not in df.columns]
    if missing:
        available = ", ".join(df.columns.tolist())
        raise ValueError(f"Colonnes manquantes: {', '.join(missing)}\nDisponibles: {available}")


# =========================================================
# FONCTION PRINCIPALE DE PLOTTING
# =========================================================

def plot(df, spec, palette='deep', color='#4F8BF9'):
    """
    Génère un graphique à partir d'une spec et d'un DataFrame
    
    Args:
        df: DataFrame pandas avec les données
        spec: dict avec clés 'type', 'x', 'y', 'hue', 'title', 'bins'
        palette: Palette seaborn (deep, muted, pastel, colorblind)
        color: Couleur par défaut si pas de hue
    
    Returns:
        Figure matplotlib
    """
    apply_theme()
    
    # Prétraiter le DataFrame
    df = preprocess_dataframe(df)
    
    if df.empty:
        return empty_plot("Le dataset est vide après nettoyage")
    
    # ===== EXTRACTION DES PARAMÈTRES =====
    # Convertir spec en dict si c'est autre chose
    if not isinstance(spec, dict):
        return empty_plot("Spec invalide : doit être un dictionnaire")
    
    plot_type = str(spec.get("type", "")).lower().strip()
    x = spec.get("x")
    y = spec.get("y")
    hue = spec.get("hue")
    title = str(spec.get("title", "Visualisation")).strip()
    bins = spec.get("bins", 20)
    
    # Convertir None/null strings en None
    if isinstance(x, str) and x.lower() in ["none", "null", ""]:
        x = None
    if isinstance(y, str) and y.lower() in ["none", "null", ""]:
        y = None
    if isinstance(hue, str) and hue.lower() in ["none", "null", ""]:
        hue = None
    
    # Vérifier que hue existe
    if hue and hue not in df.columns:
        print(f"⚠️ Colonne hue '{hue}' introuvable, ignorée")
        hue = None
    
    print(f"📊 Génération {plot_type}: x={x}, y={y}, hue={hue}")
    
    try:
        # ===== BAR CHART =====
        if plot_type == "bar":
            if not x:
                return empty_plot("Bar chart nécessite une colonne x")
            
            validate_columns(df, [x] + ([y] if y else []))
            
            # Déterminer la largeur de la figure selon le nombre de catégories
            n_categories = df[x].nunique()
            fig_width = max(10, min(20, n_categories * 0.8))
            fig, ax = plt.subplots(figsize=(fig_width, 6))
            
            if y:
                # Bar avec agrégation
                data_clean = df[[x, y]].dropna()
                if data_clean.empty:
                    return empty_plot("Aucune donnée valide pour ce bar chart")
                
                sns.barplot(
                    data=data_clean,
                    x=x,
                    y=y,
                    palette=palette,
                    ax=ax,
                    errorbar=None,
                    edgecolor='black',
                    linewidth=1.2
                )
                ax.set_ylabel(y, fontweight='bold')
            else:
                # Count plot
                data_clean = df[[x]].dropna()
                if data_clean.empty:
                    return empty_plot("Aucune donnée valide pour ce count plot")
                
                sns.countplot(
                    data=data_clean,
                    x=x,
                    palette=palette,
                    ax=ax,
                    edgecolor='black',
                    linewidth=1.2
                )
                ax.set_ylabel("Nombre d'occurrences", fontweight='bold')
            
            ax.set_xlabel(x, fontweight='bold')
            auto_layout_labels(ax, 'x')
        
        # ===== COUNT PLOT =====
        elif plot_type == "count":
            if not x:
                return empty_plot("Count plot nécessite une colonne x")
            
            validate_columns(df, [x])
            
            n_categories = df[x].nunique()
            fig_width = max(10, min(20, n_categories * 0.8))
            fig, ax = plt.subplots(figsize=(fig_width, 6))
            
            data_clean = df[[x]].dropna()
            if data_clean.empty:
                return empty_plot("Aucune donnée valide")
            
            sns.countplot(
                data=data_clean,
                x=x,
                palette=palette,
                ax=ax,
                edgecolor='black',
                linewidth=1.2
            )
            ax.set_ylabel("Nombre d'occurrences", fontweight='bold')
            ax.set_xlabel(x, fontweight='bold')
            auto_layout_labels(ax, 'x')
        
        # ===== SCATTER PLOT =====
        elif plot_type == "scatter":
            if not x or not y:
                return empty_plot("Scatter plot nécessite x et y")
            
            validate_columns(df, [x, y] + ([hue] if hue else []))
            
            cols_to_check = [x, y] + ([hue] if hue else [])
            data_clean = df[cols_to_check].dropna()
            
            if data_clean.empty:
                return empty_plot("Aucune donnée valide")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            sns.scatterplot(
                data=data_clean,
                x=x,
                y=y,
                hue=hue,
                palette=palette if hue else None,
                color=None if hue else color,
                ax=ax,
                s=80,
                alpha=0.7,
                edgecolor='white',
                linewidth=0.5
            )
            
            ax.set_xlabel(x, fontweight='bold')
            ax.set_ylabel(y, fontweight='bold')
            
            if hue:
                ax.legend(title=hue, loc='best', frameon=True, shadow=True)
        
        # ===== LINE PLOT =====
        elif plot_type == "line":
            if not x or not y:
                return empty_plot("Line plot nécessite x et y")
            
            validate_columns(df, [x, y])
            
            data_clean = df[[x, y]].dropna()
            if data_clean.empty:
                return empty_plot("Aucune donnée valide")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            sns.lineplot(
                data=data_clean,
                x=x,
                y=y,
                ax=ax,
                linewidth=2.5,
                marker='o',
                markersize=6,
                color=color
            )
            
            ax.set_xlabel(x, fontweight='bold')
            ax.set_ylabel(y, fontweight='bold')
            auto_layout_labels(ax, 'x')
        
        # ===== BOX PLOT =====
        elif plot_type == "boxplot":
            if not x or not y:
                return empty_plot("Boxplot nécessite x et y")
            
            validate_columns(df, [x, y])
            
            data_clean = df[[x, y]].dropna()
            if data_clean.empty:
                return empty_plot("Aucune donnée valide")
            
            n_categories = data_clean[x].nunique()
            fig_width = max(10, min(20, n_categories * 1.2))
            fig, ax = plt.subplots(figsize=(fig_width, 6))
            
            sns.boxplot(
                data=data_clean,
                x=x,
                y=y,
                palette=palette,
                ax=ax,
                linewidth=1.5,
                fliersize=5
            )
            
            ax.set_xlabel(x, fontweight='bold')
            ax.set_ylabel(y, fontweight='bold')
            auto_layout_labels(ax, 'x')
        
        # ===== HISTOGRAM =====
        elif plot_type == "histogram":
            if not x:
                return empty_plot("Histogram nécessite une colonne x")
            
            validate_columns(df, [x])
            
            data_clean = df[x].dropna()
            if data_clean.empty:
                return empty_plot("Aucune donnée valide")
            
            # S'assurer que c'est numérique
            try:
                data_clean = pd.to_numeric(data_clean, errors='coerce').dropna()
            except:
                return empty_plot(f"La colonne {x} n'est pas numérique")
            
            if data_clean.empty:
                return empty_plot("Aucune valeur numérique valide")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            sns.histplot(
                data_clean,
                kde=True,
                bins=bins,
                color=color,
                ax=ax,
                edgecolor='black',
                linewidth=1.2,
                alpha=0.7
            )
            
            # Ajouter stats
            mean_val = data_clean.mean()
            median_val = data_clean.median()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Moyenne: {mean_val:.2f}')
            ax.axvline(median_val, color='blue', linestyle='--', linewidth=2, label=f'Médiane: {median_val:.2f}')
            ax.legend(loc='best', frameon=True, shadow=True)
            
            ax.set_xlabel(x, fontweight='bold')
            ax.set_ylabel("Fréquence", fontweight='bold')
        
        # ===== HEATMAP =====
        elif plot_type == "heatmap":
            numeric_df = df.select_dtypes(include="number")
            
            if numeric_df.shape[1] < 2:
                return empty_plot("Heatmap nécessite au moins 2 colonnes numériques")
            
            corr = numeric_df.corr()
            
            n_cols = len(corr.columns)
            fig_size = max(8, min(16, n_cols * 0.8))
            fig, ax = plt.subplots(figsize=(fig_size, fig_size))
            
            sns.heatmap(
                corr,
                annot=True,
                fmt=".2f",
                cmap='RdBu_r',
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                linewidths=1,
                linecolor='white',
                cbar_kws={"shrink": 0.8, "label": "Corrélation"},
                ax=ax
            )
            
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            auto_layout_labels(ax, 'x')
            auto_layout_labels(ax, 'y')
            
            fig.tight_layout()
            return fig
        
        # ===== PAIRPLOT =====
        elif plot_type == "pairplot":
            numeric_cols = df.select_dtypes(include="number").columns
            
            if len(numeric_cols) < 2:
                return empty_plot("Pairplot nécessite au moins 2 colonnes numériques")
            
            # Limiter à 5 colonnes max pour la lisibilité
            if len(numeric_cols) > 5:
                numeric_cols = numeric_cols[:5]
                print(f"⚠️ Pairplot limité aux 5 premières colonnes numériques")
            
            data_clean = df[numeric_cols].dropna()
            
            if data_clean.empty:
                return empty_plot("Aucune donnée valide")
            
            g = sns.pairplot(
                data_clean,
                diag_kind='kde',
                plot_kws={'alpha': 0.6, 's': 30},
                diag_kws={'linewidth': 2}
            )
            
            g.fig.suptitle(title, y=1.02, fontsize=16, fontweight='bold')
            g.fig.tight_layout()
            return g.fig
        
        else:
            return empty_plot(f"Type de plot '{plot_type}' non reconnu")
        
        # Ajouter le titre (sauf pour pairplot et heatmap qui l'ont déjà)
        if plot_type not in ["pairplot", "heatmap"]:
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        fig.tight_layout()
        return fig
    
    except Exception as e:
        print(f"❌ Erreur génération graphique: {e}")
        import traceback
        traceback.print_exc()
        return empty_plot(f"Erreur: {str(e)}")