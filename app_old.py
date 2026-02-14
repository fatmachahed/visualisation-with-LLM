from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import traceback

# === Import LLM Utils ===
from visualisation_with_llm.llm_utils_old import init_llm, generate_visualization_specs

app = Flask(__name__, template_folder="templates")
CORS(app)

# === Initialisation LLM ===
llm = init_llm()

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

        if not file:
            return jsonify({"error": "Fichier CSV manquant", "specs": [], "data": []}), 400

        try:
            df = pd.read_csv(file.stream)
        except Exception as e:
            return jsonify({"error": f"Erreur lecture CSV: {str(e)}", "specs": [], "data": []}), 400

        if df.empty:
            return jsonify({"error": "Dataset vide", "specs": [], "data": []}), 400

        # Nettoyage colonnes
        df.columns = df.columns.str.strip()

        # Résumé du dataset (simple pour LLM)
        dataset_summary = f"Colonnes : {', '.join(df.columns)} | Types : {df.dtypes.to_dict()} | Nombre de lignes : {len(df)}"

        # === Génération des visualisations via LLM ===
        specs = generate_visualization_specs(llm, problem, dataset_summary)

        return jsonify({
            "specs": specs,
            "data": df.fillna("").to_dict(orient="records")
        })

    except Exception as e:
        print("❌ ERREUR SERVEUR")
        print(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc(),
            "specs": [],
            "data": []
        }), 500

# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    print("🚀 Serveur Flask démarré sur http://127.0.0.1:5000")
    app.run(debug=True, host="127.0.0.1", port=5000)
