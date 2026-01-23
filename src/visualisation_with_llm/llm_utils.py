import os
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

def init_llm():
    """
    Initialise le LLM Google Generative AI.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Clé API Google manquante ! Vérifiez votre fichier .env")
    
    # Initialisation du modèle
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        api_key=api_key,
        temperature=0.3,
        max_tokens=512
    )
    return llm


def generate_visualization_proposals(llm, problem_statement: str, dataset_summary: str) -> list:
    """
    Génère trois propositions de visualisation pour une problématique donnée.
    
    Paramètres
    ----------
    llm : ChatGoogleGenerativeAI
        Le modèle LLM initialisé.
    problem_statement : str
        La problématique textuelle (ex: "Quels facteurs influencent le prix des logements à Paris ?")
    dataset_summary : str
        Une description ou résumé du dataset (colonnes, types, etc.)
    
    Retour
    ------
    list[str]
        Liste de 3 propositions textuelles de visualisation.
    """
    prompt = f"""
    Vous êtes un expert en data visualisation.
    Voici la problématique : {problem_statement}
    Voici un résumé du dataset : {dataset_summary}
    Proposez 3 visualisations pertinentes avec type de graphique et justification pour chacune.
    """

    response = llm.generate([{"role": "user", "content": prompt}])
    # Retourner chaque proposition séparée (suppose que le LLM sépare par ligne)
    proposals = response.content.split("\n")
    # Nettoyage de base
    proposals = [p.strip() for p in proposals if p.strip()]
    return proposals[:3]  # on garde les 3 premières
