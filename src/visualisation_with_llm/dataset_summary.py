# dataset_summary.py
import pandas as pd

def summarize_dataset(df: pd.DataFrame, max_rows: int = 5) -> str:
    summary = []
    summary.append(f"Nombre de lignes : {df.shape[0]}")
    summary.append(f"Nombre de colonnes : {df.shape[1]}")
    summary.append("Colonnes :")

    for col in df.columns:
        summary.append(
            f"- {col} ({df[col].dtype}), valeurs non nulles : {df[col].notna().sum()}"
        )

    summary.append("\nExemples de données :")
    summary.append(df.head(max_rows).to_string())

    return "\n".join(summary)
