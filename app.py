from flask import Flask, request, jsonify
from flask_cors import CORS

# Import your logic
from flaskcrop import crop_recommendation_logic
from flaskIrrigation import irrigation_predict_logic

# Create the Flask app

app = Flask(__name__)
CORS(app)

# -------------------------------
# Crop Recommendation API
# -------------------------------
@app.route('/crop_recommendation', methods=['POST'])
def crop_recommendation():
    try:
        data = request.get_json()

        result = crop_recommendation_logic(data)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------
# Fertilizer API (dummy for now)
# -------------------------------
@app.route('/fertilizer', methods=['POST'])
def fertilizer():
    data = request.get_json()

    return jsonify({
        "message": "Fertilizer API working",
        "input": data
    })


# -------------------------------
# Irrigation API (dummy for now)
# -------------------------------
@app.route("/irrigation_predict", methods=["POST"])
def irrigation_predict():
    data = request.get_json()
    result = irrigation_predict_logic(data)
    return jsonify(result)


# -------------------------------
# Run Server
import threading

def run_app():
    app.run(host="0.0.0.0", port=2004)

threading.Thread(target=run_app).start()




# from pyngrok import ngrok

# ngrok.set_auth_token("3CgHetjrfcyXc8OKl4Fb9Z2y0aN_6GTYfbgLFfF9cJrfYRMmp")
# public_url = ngrok.connect(2004)
# print(public_url)


# import os
# os._exit(0)