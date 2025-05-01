from flask import Flask, request, jsonify
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from flask_cors import CORS
import requests
import os
from dotenv import load_dotenv
import joblib

# Cargar variables de entorno
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"

# Inicializar Flask
app = Flask(__name__)
CORS(app)

# Cargar dataset
DATASET_URL = 'https://raw.githubusercontent.com/ronyvhell/pulmonary-prediction/master/datasetcovidcon2000campos.xlsx'
data = pd.read_excel(DATASET_URL)

# Separar características y etiqueta
X = data.drop(columns=['Flag_sospechoso'])
y = data['Flag_sospechoso']

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cargar o entrenar modelo
model_file = 'model.pkl'
if os.path.exists(model_file):
    model = joblib.load(model_file)
else:
    model = DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_file)

# Evaluar modelo
y_pred = model.predict(X_test)
print("Reporte de clasificación:\n", classification_report(y_test, y_pred))

def get_recommendation(probability):
    headers = {"Content-Type": "application/json"}
    prompt = f"La probabilidad de tener COVID es del {probability}%. Proporcione recomendaciones de cuidado personal y prevención de máximo 300 caracteres."
    data = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    try:
        response = requests.post(GEMINI_API_URL, json=data, headers=headers)
        response.raise_for_status()
        result = response.json()
        candidates = result.get("candidates", [])
        if not candidates:
            return "No se encontraron candidatos en la respuesta de Gemini."
        content = candidates[0].get("content", {}).get("parts", [])
        if not content:
            return "No se encontró contenido en la respuesta de Gemini."
        return content[0].get("text", "No se pudo extraer el texto.")
    except Exception as e:
        return f"Error al procesar respuesta de Gemini: {str(e)}"

@app.route('/predict', methods=['POST'])
def predict_covid():
    try:
        symptoms = request.json
        if not symptoms:
            return jsonify({'error': 'No se enviaron síntomas'}), 400

        # Validar valores
        for key, value in symptoms.items():
            if value not in [0, 1]:
                return jsonify({'error': f'Valor inválido para {key}: {value}'}), 400

        symptoms_df = pd.DataFrame([symptoms])
        missing_cols = [col for col in X.columns if col not in symptoms_df.columns]
        if missing_cols:
            return jsonify({'error': f'Faltan columnas: {missing_cols}'}), 400

        probability = model.predict_proba(symptoms_df)[0][1] * 100  # En porcentaje
        recommendation = get_recommendation(probability)

        return jsonify({
            'probability': round(probability, 2),
            'recommendation': recommendation,
            'message': "Recomendaciones basadas en la probabilidad calculada."
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)