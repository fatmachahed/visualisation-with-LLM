from pathlib import Path
import pandas as pd

from src.visualisation_with_llm.data_loader import load_dataset
from src.visualisation_with_llm.viz_utils import plot_bar, plot_scatter, plot_line

if __name__ == "__main__":

    # 1️⃣ Charger le dataset
    base_path = Path(__file__).parent
    dataset_path = base_path / "data" / "spotify-tracks-dataset" / "dataset.csv"
    df = load_dataset(dataset_path)

    if df.empty:
        print("Erreur : dataset vide")
        exit(1)

    print("Dataset chargé :")
    print(df.head())
    print("Colonnes disponibles :", df.columns.tolist())

    # 2️⃣ Top 10 artistes par popularité moyenne
    top_artists = (
        df.groupby("artists")["popularity"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    print("\nTop 10 artistes par popularité moyenne :")
    print(top_artists)

    # 3️⃣ Exemple de bar plot
    plot_bar(top_artists, x="artists", y="popularity", title="Top 10 artistes par popularité moyenne")

    # 4️⃣ Exemple de scatter plot : popularité vs durée
    plot_scatter(df, x="duration_ms", y="popularity", title="Popularité vs durée des tracks")

    # 5️⃣ Exemple de line plot : popularité moyenne par album (juste pour tester)
    album_pop = df.groupby("album_name")["popularity"].mean().sort_values(ascending=False).head(10).reset_index()
    plot_line(album_pop, x="album_name", y="popularity", title="Popularité moyenne par album (Top 10)")
