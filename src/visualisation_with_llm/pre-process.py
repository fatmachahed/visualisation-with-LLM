import pandas as pd

def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prépare le dataset avant de le passer au LLM :
    - Nettoie les noms de colonnes
    - Convertit les colonnes numériques
    - Supprime les colonnes entièrement vides
    - Supprime les colonnes avec 1 seule valeur
    - Remplit éventuellement les NaN pour le LLM (avec '')
    """
    
    # --- Nettoyage des colonnes ---
    df.columns = df.columns.str.strip()
    
    # --- Supprimer colonnes entièrement vides ---
    df = df.dropna(axis=1, how="all")
    
    # --- Conversion des colonnes numériques ---
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass  # reste en object si conversion impossible
    
    # --- Supprimer colonnes numériques constantes ---
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        if df[col].nunique() <= 1:
            df = df.drop(columns=[col])
    
    # --- Supprimer colonnes catégorielles constantes ---
    categorical_cols = df.select_dtypes(exclude="number").columns
    for col in categorical_cols:
        if df[col].nunique() <= 1:
            df = df.drop(columns=[col])
    
    # --- Remplacer les NaN restants pour le LLM ---
    df = df.fillna("")
    
    return df
