from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY no está configurada en el archivo .env")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"

# Inicializar Flask
app = Flask(__name__)
CORS(app)

# Cargar dataset
DATASET_URL = 'https://raw.githubusercontent.com/ronyvhell/pulmonary-prediction/master/datasetcovidcon2000campos.xlsx'
data = pd.read_excel(DATASET_URL)

# Inspeccionar casos con todos los síntomas en 0
zero_symptoms = data[(data.drop(columns=['Flag_sospechoso']) == 0).all(axis=1)]
print("Casos con todos los síntomas en 0:")
print(zero_symptoms[['Flag_sospechoso']].value_counts())

# Inspeccionar balance de clases
print("Distribución de Flag_sospechoso:")
print(data['Flag_sospechoso'].value_counts())

# Separar características y etiqueta
X = data.drop(columns=['Flag_sospechoso'])
y = data['Flag_sospechoso']

# Imprimir columnas para depuración
print("Columnas del dataset:", X.columns.tolist())

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cargar o entrenar modelo
model_file = 'model.pkl'
if os.path.exists(model_file):
    model = joblib.load(model_file)
else:
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    joblib.dump(model, model_file)

# Evaluar modelo
y_pred = model.predict(X_test)
print("Reporte de clasificación:\n", classification_report(y_test, y_pred))

def get_recommendation(probability):
    headers = {"Content-Type": "application/json"}
    prompt = f"""
    Basado en una probabilidad de COVID-19 del {probability}%, proporcione recomendaciones breves (máximo 300 caracteres) para el cuidado personal y prevención, considerando síntomas respiratorios, digestivos y neurológicos. Incluya consejos prácticos y mencione consultar a un médico si es necesario.
    """
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

        # Convertir y validar valores
        symptoms_converted = {}
        for key, value in symptoms.items():
            try:
                converted_value = int(float(str(value)))
                if converted_value not in [0, 1]:
                    return jsonify({'error': f'Valor inválido para {key}: {value}'}), 400
                symptoms_converted[key] = converted_value
            except (ValueError, TypeError):
                return jsonify({'error': f'Valor inválido para {key}: {value}'}), 400

        print("Datos recibidos:", symptoms_converted)

        # Forzar probabilidad 0% si todos los síntomas son 0
        if all(value == 0 for value in symptoms_converted.values()):
            print("Todos los síntomas son 0, forzando probabilidad 0%")
            return jsonify({
                'probability': 0.0,
                'recommendation': "Sin síntomas reportados. Mantén medidas preventivas como lavado de manos y uso de mascarilla.",
                'message': "No se detectaron síntomas de COVID-19."
            })

        # Crear DataFrame con todas las columnas del modelo
        symptoms_df = pd.DataFrame([symptoms_converted])
        missing_cols = [col for col in X.columns if col not in symptoms_df.columns]
        if missing_cols:
            for col in missing_cols:
                symptoms_df[col] = 0
            print(f"Columnas faltantes rellenadas con 0: {missing_cols}")

        # Reordenar columnas
        symptoms_df = symptoms_df[X.columns]

        probability = model.predict_proba(symptoms_df)[0][1] * 100
        recommendation = get_recommendation(probability)

        print(f"Probabilidad calculada: {probability}%")

        return jsonify({
            'probability': round(probability, 2),
            'recommendation': recommendation,
            'message': "Recomendaciones basadas en la probabilidad calculada."
        })
    except Exception as e:
        print(f"Error en /predict: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)