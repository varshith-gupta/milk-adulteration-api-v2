from flask import Flask, request, jsonify
import joblib
import numpy as np
import traceback

app = Flask(__name__)

brand_model = joblib.load('brand_model.pkl')
water_model = joblib.load('water_model.pkl')
le = joblib.load('label_encoder.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        channels = np.array([data['channels']], dtype=float)

        brand_encoded = brand_model.predict(channels)[0]
        brand_name = le.inverse_transform([brand_encoded])[0]
        water_percent = round(float(water_model.predict(channels)[0]), 2)

        return jsonify({
            "brand": brand_name,
            "water_percent": water_percent,
            "status": "Pure" if water_percent == 0 else "Adulterated"
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "details": "Make sure ALL 18 fields have numbers (no empty boxes or only '.')"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
