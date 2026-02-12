from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import traceback
import json

app = Flask(__name__, template_folder="templates")
CORS(app)

# ==============================
# Import modules personnalisés
# ==============================
try:
    from src.visualisation_with_llm.dataset_summary import summarize_dataset
    from src.visualisation_with_llm.llm_utils import init_llm, generate_visualization_specs
    from src.visualisation_with_llm.data_loader import load_dataset
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False
    print("⚠️ Modules personnalisés non disponibles, mode standalone activé")

# ==============================
# ROUTES
# ==============================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    try:
        problem = request.form.get("problem")
        file = request.files.get("dataset")

        if not problem or not file:
            return jsonify({"error": "Problématique ou fichier manquant"}), 400

        # 🔹 Lire le CSV correctement
        try:
            if MODULES_AVAILABLE:
                df = load_dataset(file)
            else:
                df = pd.read_csv(file.stream)
        except Exception as e:
            return jsonify({"error": f"Erreur lecture CSV: {str(e)}"}), 400

        if df is None or df.empty:
            return jsonify({"error": "Dataset vide ou invalide"}), 400

        # 🔹 Résumé
        if MODULES_AVAILABLE:
            summary = summarize_dataset(df)
        else:
            summary = f"{len(df)} lignes, {len(df.columns)} colonnes"

        # 🔹 Génération visualisations
        if os.getenv("GOOGLE_API_KEY") and MODULES_AVAILABLE:
            try:
                llm = init_llm()
                specs = generate_visualization_specs(llm, problem, summary)
            except Exception as e:
                print("⚠️ Erreur LLM:", e)
                specs = generate_smart_visualizations(df)
        else:
            specs = generate_smart_visualizations(df)

        # 🔹 Nettoyage JSON
        specs = sanitize(specs)

        return app.response_class(
            response=json.dumps(specs),
            status=200,
            mimetype="application/json"
        )

    except Exception as e:
        print("❌ ERREUR SERVEUR")
        print(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500


# ==============================
# UTILITAIRES
# ==============================

def sanitize(obj):
    if isinstance(obj, float) and pd.isna(obj):
        return None
    elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize(x) for x in obj]
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    else:
        return str(obj)


def generate_smart_visualizations(df):
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    specs = []

    # 1️⃣ Scatter
    if len(numeric_cols) >= 2:
        specs.append({
            "type": "scatter",
            "x": numeric_cols[0],
            "y": numeric_cols[1],
            "title": f"{numeric_cols[0]} vs {numeric_cols[1]}",
            "justification": "Relation entre deux variables numériques"
        })

    # 2️⃣ Bar
    if categorical_cols and numeric_cols:
        specs.append({
            "type": "bar",
            "x": categorical_cols[0],
            "y": numeric_cols[0],
            "title": f"Moyenne {numeric_cols[0]} par {categorical_cols[0]}",
            "justification": "Comparaison par catégorie"
        })

    # 3️⃣ Line
    if len(numeric_cols) >= 1:
        specs.append({
            "type": "line",
            "x": df.columns[0],
            "y": numeric_cols[0],
            "title": f"Tendance {numeric_cols[0]}",
            "justification": "Évolution de la variable"
        })

    return specs[:3]


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    print("🚀 Serveur Flask démarré sur http://127.0.0.1:5000")
    app.run(debug=True, host="127.0.0.1", port=5000, use_reloader=False)
