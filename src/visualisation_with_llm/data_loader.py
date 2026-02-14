#data_loader.py
import pandas as pd
from pathlib import Path

# data_loader.py
import pandas as pd

def load_dataset(file) -> pd.DataFrame:
    """
    Charge un dataset CSV depuis un fichier uploadé ou un chemin.
    """
    try:
        df = pd.read_csv(file)

        if df.empty:
            return df

        # Drop colonnes inutiles
        cols_to_drop = [c for c in df.columns if 'unnamed' in c.lower()]
        df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

        # Tentative conversion numérique
        for col in df.select_dtypes(include=["object"]).columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass

        return df

    except Exception as e:
        print(f"Erreur chargement dataset : {e}")
        return pd.DataFrame()



# ------------------------------
# Test rapide du loader
# ------------------------------
if __name__ == "__main__":
    base_path = Path(__file__).parent.parent.parent  # src/visualisation_with_llm/ -> remonter à la racine
    test_file = base_path / "data" / "spotify-tracks-dataset" / "dataset.csv"

    # Charger le dataset
    df_test = load_dataset(test_file)

    # Afficher les 5 premières lignes pour vérifier
    if not df_test.empty:
        print("\nAperçu du dataset :")
        print(df_test.head())
    else:
        print("Le dataset n'a pas pu être chargé.")
