# src/visualisation_with_llm/llm_utils.py

import os
import json
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()


# ==========================================================
# INITIALISATION LLM
# ==========================================================

def init_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2,  # faible pour stabilité
        max_tokens=1000
    )
    
# ==========================================================
# PROMPT ENGINEERING (Scaffolded Reasoning)
# ==========================================================

def build_prompt(problem_statement: str, dataset_summary: str) -> str:
    """
    Construit un prompt structuré avec raisonnement en étapes.
    """

    return f"""
Tu es un expert académique en Data Visualization.

OBJECTIF :
Répondre à une problématique métier en choisissant des visualisations pertinentes.

PROBLEMATIQUE :
{problem_statement}

DATASET :
{dataset_summary}

INSTRUCTIONS :

1️⃣ Analyse brièvement la problématique.
2️⃣ Identifie les types de variables pertinentes.
3️⃣ Choisis EXACTEMENT 3 visualisations différentes.
4️⃣ Justifie chaque choix selon :
   - Pertinence analytique
   - Bonnes pratiques (lisibilité, data-ink ratio, absence de chartjunk)
   - Type de données
5️⃣ Retourne UNIQUEMENT un JSON STRICT (pas de texte hors JSON).

FORMAT STRICT ATTENDU :

[
  {{
    "type": "bar | scatter | line | boxplot | histogram | heatmap",
    "x": "colonne ou null",
    "y": "colonne ou null",
    "hue": "colonne ou null",
    "title": "titre clair et académique",
    "justification": "explication claire et concise",
    "palette": "deep | muted | colorblind"
  }}
]

IMPORTANT :
- EXACTEMENT 3 objets
- JSON valide
- Pas de texte avant ou après
"""


# ==========================================================
# GENERATION DES VISUALISATIONS
# ==========================================================

def generate_visualization_specs(llm, problem_statement, dataset_summary):
    prompt = build_prompt(problem_statement, dataset_summary)
    response = llm.invoke(prompt)
    content = response.content.strip()

    # Retirer ```json``` ou ``` au début et à la fin
    if content.startswith("```"):
        content = content.split("```")[1]

    # Retirer tout texte avant le premier [
    idx = content.find("[")
    if idx >= 0:
        content = content[idx:]
    else:
        raise ValueError("Aucune liste JSON trouvée dans la réponse LLM")

    # Retirer tout texte après le dernier ]
    idx = content.rfind("]")
    if idx >= 0:
        content = content[:idx+1]
    else:
        raise ValueError("Aucune fin de JSON trouvée dans la réponse LLM")

    # Convertir en JSON
    try:
        specs = json.loads(content)
        if not isinstance(specs, list) or len(specs) != 3:
            raise ValueError("Le LLM doit retourner EXACTEMENT 3 visualisations")
        return specs
    except json.JSONDecodeError:
        raise ValueError("Le LLM n'a pas retourné un JSON valide")

