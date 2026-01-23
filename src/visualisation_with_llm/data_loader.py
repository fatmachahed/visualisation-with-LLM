import pandas as pd
from pathlib import Path

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Charge un dataset CSV et retourne un DataFrame pandas.
    
    Paramètres
    ----------
    file_path : str
        Chemin vers le fichier CSV.

    Retour
    ------
    pd.DataFrame
        DataFrame contenant le dataset chargé.
    """
    try:
        # Charger le CSV
        df = pd.read_csv(file_path)

        # Vérifier qu'il n'est pas vide
        if df.empty:
            print(f"Attention : le dataset {file_path} est vide.")
            return df

        # Nettoyage de base 
        cols_to_drop = [col for col in df.columns if 'unnamed' in col.lower()]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
        
        # Nettoyage de type : convertir les colonnes numériques
        for col in df.select_dtypes(include=['object']).columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass  # si conversion impossible, on garde le texte

        print(f"Dataset chargé avec succès : {df.shape[0]} lignes, {df.shape[1]} colonnes")
        return df

    except FileNotFoundError:
        print(f"Erreur : le fichier {file_path} n'existe pas.")
        return pd.DataFrame()
    except pd.errors.ParserError:
        print(f"Erreur : problème lors de la lecture du CSV {file_path}.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Erreur inattendue : {e}")
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
