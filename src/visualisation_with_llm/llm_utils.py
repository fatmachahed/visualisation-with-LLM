import os
import json
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv()

# ==========================================================
# CONFIGURATION DES TYPES ET PALETTES
# ==========================================================
ALLOWED_TYPES = {
    "bar", "scatter", "line", "boxplot", "histogram", "heatmap",
    "count", "pairplot", "violin", "stacked_bar"
}
ALLOWED_PALETTES = {"deep", "muted", "colorblind"}

# ==========================================================
# INITIALISATION DU LLM
# ==========================================================
def init_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY manquant dans .env")

    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        api_key=api_key,
        temperature=0.2,
        max_tokens=1500
    )

# ==========================================================
# PROMPT ENGINEERING
# ==========================================================
def build_prompt(problem_statement: str, dataset_summary: str) -> str:
    return f"""
Tu es un expert académique en Data Visualization.

OBJECTIF :
Répondre à une problématique métier en choisissant des visualisations pertinentes.

PROBLEMATIQUE :
{problem_statement}

DATASET :
{dataset_summary}

ANALYSE DU DATASET :
1. Identifie d'abord les colonnes numériques et catégorielles
2. Vérifie si la heatmap est appropriée (au moins 2 colonnes numériques)
3. Si moins de 2 colonnes numériques, NE PROPOSE PAS de heatmap

INSTRUCTIONS :
1️⃣ Analyse brièvement la problématique.
2️⃣ Identifie les types de variables pertinentes.
3️⃣ Choisis EXACTEMENT 3 visualisations différentes.
4️⃣ Justifie chaque choix selon :
   - Pertinence analytique
   - Bonnes pratiques
   - Type de données
5️⃣ Retourne UNIQUEMENT un JSON STRICT.

FORMAT STRICT :
[
  {{
    "type": "bar | scatter | line | boxplot | histogram | heatmap | count | pairplot | violin | stacked_bar",
    "x": "nom exact d'une colonne ou null",
    "y": "nom exact d'une colonne ou null",
    "hue": "nom exact d'une colonne ou null",
    "bins": 20,
    "orientation": "v | h",
    "title": "titre clair",
    "justification": "explication concise",
    "palette": "deep | muted | colorblind"
  }}
]

RÈGLES PAR TYPE DE GRAPHIQUE :
- HEATMAP : AU MOINS 2 colonnes numériques, x=null, y=null, hue=null
- HISTOGRAM : x = colonne numérique, y = null
- BOXPLOT : x = colonne catégorielle, y = colonne numérique
- BAR : x = colonne catégorielle, y = colonne numérique
- SCATTER : x et y = colonnes numériques
- LINE : x et y = colonnes numériques
- COUNT : x = colonne catégorielle, y=null
- PAIRPLOT : toutes les colonnes numériques
- VIOLIN : x=catégorielle, y=numérique
- STACKED_BAR : x=catégorielle, y=numérique, hue=catégorielle

IMPORTANT :
- EXACTEMENT 3 objets
- JSON valide
- null doit être un vrai null JSON
"""

# ==========================================================
# GENERATION DES VISUALISATIONS
# ==========================================================
def generate_visualization_specs(llm, problem_statement: str, dataset_summary: str) -> List[Dict[str, Any]]:
    prompt = build_prompt(problem_statement, dataset_summary)
    response = llm.invoke(prompt)
    content = response.content.strip()

    # Nettoyage markdown éventuel
    if content.startswith("```"):
        content = content.split("```")[1]

    # Extraction JSON
    start = content.find("[")
    end = content.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("Réponse LLM invalide : JSON non trouvé")
    json_str = content[start:end+1]

    try:
        specs = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON invalide retourné par le LLM : {e}")

    if not isinstance(specs, list) or len(specs) != 3:
        raise ValueError("Le LLM doit retourner EXACTEMENT 3 visualisations")

    normalized = []
    for spec in specs:
        normalized.append(normalize_and_validate_spec(spec))

    return normalized

# ==========================================================
# NORMALISATION + VALIDATION FORTE
# ==========================================================
def normalize_and_validate_spec(spec: Dict[str, Any], df_columns=None) -> Dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError("Chaque visualisation doit être un objet JSON")

    # --- TYPE ---
    plot_type = str(spec.get("type", "")).lower().strip()
    if plot_type not in ALLOWED_TYPES:
        raise ValueError(f"Type de graphique non supporté : {plot_type}")
    spec["type"] = plot_type

    # --- X / Y / HUE ---
    for key in ["x", "y", "hue"]:
        value = spec.get(key)
        if value in ["null", "", "None"] or value is None:
            spec[key] = None
        else:
            spec[key] = str(value).strip()

    # --- BINS / ORIENTATION ---
    spec["bins"] = int(spec.get("bins", 20))
    spec["orientation"] = str(spec.get("orientation", "v")).lower()
    if spec["orientation"] not in ["v", "h"]:
        spec["orientation"] = "v"

    # --- HEATMAP ---
    if plot_type == "heatmap":
        spec["x"] = spec["y"] = spec["hue"] = None
        if "heatmap" not in spec.get("justification", "").lower():
            spec["justification"] = f"Heatmap de corrélation - {spec.get('justification','')}"

    # --- TITLE ---
    if not spec.get("title"):
        if plot_type == "heatmap":
            spec["title"] = "Matrice de Corrélation"
        else:
            spec["title"] = f"{plot_type.capitalize()} - {spec.get('x', '')} vs {spec.get('y', '')}"
    else:
        spec["title"] = str(spec.get("title")).strip()

    # --- JUSTIFICATION ---
    spec["justification"] = str(spec.get("justification", "")).strip()
    if not spec["justification"]:
        spec["justification"] = f"Visualisation {plot_type} pour analyser la relation entre les variables"

    # --- PALETTE ---
    palette = str(spec.get("palette", "deep")).lower().strip()
    if palette not in ALLOWED_PALETTES:
        palette = "deep"
    spec["palette"] = palette

    return spec
