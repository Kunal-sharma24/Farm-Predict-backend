from flask import Flask, request, jsonify
from flask_cors import CORS

# Import your logic
from flaskcrop import crop_recommendation_logic

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
@app.route('/irrigation', methods=['POST'])
def irrigation():
    data = request.get_json()

    return jsonify({
        "message": "Irrigation API working",
        "input": data
    })


# -------------------------------
# Run Server
# -------------------------------
if __name__ == '__main__':
    app.run(debug=True)