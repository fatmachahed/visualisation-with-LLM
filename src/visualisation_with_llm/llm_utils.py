import os
from dotenv import load_dotenv
import re

load_dotenv()


def init_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Clé API manquante")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        api_key=api_key,
        temperature=0.2,
        max_output_tokens=2048
    )
    return llm


def generate_visualization_proposals(llm, problem_statement, dataset_summary, preferred_types=None, num_proposals=3, allow_duplicates=False):
    num_proposals = max(3, num_proposals)
    allowed_types = None
    
    if preferred_types:
        if isinstance(preferred_types, str):
            allowed_types = [t.strip().lower() for t in preferred_types.split(",")]
        else:
            allowed_types = [t.strip().lower() for t in preferred_types]
        
        if len(allowed_types) < num_proposals:
            allow_duplicates = True
        
        types_constraint = f"""
CONTRAINTE : Tu DOIS UNIQUEMENT utiliser ces types : {', '.join(allowed_types).upper()}
{"Tu PEUX répéter le même type avec des colonnes différentes" if allow_duplicates else ""}
"""
    else:
        types_constraint = ""
        allowed_types = ["scatter", "bar", "line", "histogram", "boxplot", "heatmap", "count"]
    
    columns_info = extract_columns_from_summary(dataset_summary)
    
    prompt = f"""
Tu es un expert en data visualisation.

PROBLÉMATIQUE : {problem_statement}
DATASET : {dataset_summary}
COLONNES : {columns_info}
{types_constraint}

TÂCHE : Propose EXACTEMENT {num_proposals} visualisations.

FORMAT ({num_proposals} lignes) :
type: <type>, x: <col>, y: <col>, title: <titre>, justification: <texte>

TYPES : scatter, bar, line, histogram, boxplot, heatmap, count
"""
    
    try:
        response = llm.invoke(prompt)
        text = response.content.strip()
        specs = parse_all_specs(text, dataset_summary, allow_duplicates)
        
        if allowed_types:
            specs = [s for s in specs if s.get("type") in allowed_types]
        
        if len(specs) < num_proposals:
            specs = complete_to_n_specs(specs, dataset_summary, allowed_types, num_proposals, allow_duplicates)
        
        return specs[:num_proposals]
    
    except Exception as e:
        print(f"Erreur LLM: {e}")
        return generate_smart_fallback_specs(dataset_summary, allowed_types, num_proposals, allow_duplicates)


def extract_columns_from_summary(summary):
    lines = summary.split('\n')
    columns = []
    for line in lines:
        if any(dtype in line.lower() for dtype in ['int', 'float', 'object', 'string']):
            parts = line.split()
            if parts:
                columns.append(parts[0].strip())
    return "Colonnes : " + ", ".join(columns) if columns else ""


def parse_all_specs(text, dataset_summary, allow_duplicates):
    specs = []
    seen_combos = set()
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    
    for line in lines:
        if "type:" not in line.lower():
            continue
        spec = parse_single_spec(line)
        if spec and spec.get("type"):
            signature = f"{spec['type']}_{spec.get('x', '')}_{spec.get('y', '')}"
            if signature not in seen_combos or allow_duplicates:
                specs.append(spec)
                seen_combos.add(signature)
    return specs


def parse_single_spec(line):
    spec = {"type": None, "x": None, "y": None, "title": "Visualisation", "justification": ""}
    parts = re.split(r',\s*(?=\w+:)', line)
    for part in parts:
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if value.lower() in ["null", "none", "", "n/a"]:
            value = None
        if key in spec:
            spec[key] = value
    return spec


def complete_to_n_specs(specs, dataset_summary, allowed_types, num_proposals, allow_duplicates):
    existing_combos = {f"{s['type']}_{s.get('x', '')}_{s.get('y', '')}" for s in specs}
    fallback_specs = generate_smart_fallback_specs(dataset_summary, allowed_types, num_proposals, allow_duplicates)
    for fb_spec in fallback_specs:
        if len(specs) >= num_proposals:
            break
        signature = f"{fb_spec['type']}_{fb_spec.get('x', '')}_{fb_spec.get('y', '')}"
        if signature not in existing_combos or allow_duplicates:
            specs.append(fb_spec)
            existing_combos.add(signature)
    return specs


def generate_smart_fallback_specs(dataset_summary, allowed_types=None, num_proposals=3, allow_duplicates=False):
    numeric_cols = []
    categorical_cols = []
    
    for line in dataset_summary.split("\n"):
        line_lower = line.lower()
        if not line.strip() or "column" in line_lower:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        col_name = parts[0].strip()
        if col_name in ["-", "Column", "Type"]:
            continue
        if any(dtype in line_lower for dtype in ['int64', 'float64']):
            numeric_cols.append(col_name)
        elif any(dtype in line_lower for dtype in ['object', 'string']):
            categorical_cols.append(col_name)
    
    if allowed_types:
        types_pool = allowed_types
    else:
        types_pool = ["scatter", "bar", "histogram", "boxplot", "heatmap"]
    
    specs = []
    used_combos = set()
    attempts = 0
    
    while len(specs) < num_proposals and attempts < num_proposals * 10:
        attempts += 1
        for t in types_pool:
            if len(specs) >= num_proposals:
                break
            spec = None
            
            if t == "scatter" and len(numeric_cols) >= 2:
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        combo = f"scatter_{numeric_cols[i]}_{numeric_cols[j]}"
                        if combo not in used_combos or allow_duplicates:
                            spec = {
                                "type": "scatter",
                                "x": numeric_cols[i],
                                "y": numeric_cols[j],
                                "title": f"Relation {numeric_cols[i]} vs {numeric_cols[j]}",
                                "justification": f"Corrélation entre {numeric_cols[i]} et {numeric_cols[j]}"
                            }
                            used_combos.add(combo)
                            break
                    if spec:
                        break
            
            elif t == "bar" and categorical_cols and numeric_cols:
                for cat in categorical_cols:
                    for num in numeric_cols:
                        combo = f"bar_{cat}_{num}"
                        if combo not in used_combos or allow_duplicates:
                            spec = {
                                "type": "bar",
                                "x": cat,
                                "y": num,
                                "title": f"{num} par {cat}",
                                "justification": f"Compare {num} selon {cat}"
                            }
                            used_combos.add(combo)
                            break
                    if spec:
                        break
            
            elif t == "histogram" and numeric_cols:
                for num in numeric_cols:
                    combo = f"histogram_{num}_null"
                    if combo not in used_combos or allow_duplicates:
                        spec = {
                            "type": "histogram",
                            "x": num,
                            "y": None,
                            "title": f"Distribution de {num}",
                            "justification": f"Répartition de {num}"
                        }
                        used_combos.add(combo)
                        break
            
            elif t == "boxplot" and categorical_cols and numeric_cols:
                for cat in categorical_cols:
                    for num in numeric_cols:
                        combo = f"boxplot_{cat}_{num}"
                        if combo not in used_combos or allow_duplicates:
                            spec = {
                                "type": "boxplot",
                                "x": cat,
                                "y": num,
                                "title": f"Distribution {num} par {cat}",
                                "justification": f"Compare distributions"
                            }
                            used_combos.add(combo)
                            break
                    if spec:
                        break
            
            elif t == "heatmap" and len(numeric_cols) >= 3:
                combo = "heatmap_null_null"
                if combo not in used_combos or allow_duplicates:
                    spec = {
                        "type": "heatmap",
                        "x": None,
                        "y": None,
                        "title": "Matrice de corrélation",
                        "justification": "Corrélations entre variables"
                    }
                    used_combos.add(combo)
            
            if spec:
                specs.append(spec)
    
    return specs[:num_proposals]
