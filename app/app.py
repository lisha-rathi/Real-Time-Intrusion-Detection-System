import json
import joblib
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime
from io import StringIO
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
import keras
import shap

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)
CORS(app)
limiter = Limiter(key_func=get_remote_address, default_limits=["100 per hour"])
limiter.init_app(app)

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'predictions.db')

RF_MODEL_PATH = os.path.join(MODEL_DIR, 'rf_model.pkl')
DL_MODEL_PATH = os.path.join(MODEL_DIR, 'dl_model.keras')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
FEATURES_PATH = os.path.join(MODEL_DIR, 'feature_names.pkl')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')
METRICS_PATH = os.path.join(MODEL_DIR, 'metrics.json')

rf_model = None
dl_model = None
scaler = None
feature_names = None
label_encoder = None
shap_explainer = None

MITRE_MAPPING = {
    'BENIGN': {'technique': 'N/A', 'name': 'Benign Traffic'},
    'DoS Hulk': {'technique': 'T1499', 'name': 'Endpoint Denial of Service'},
    'DoS GoldenEye': {'technique': 'T1499', 'name': 'Endpoint Denial of Service'},
    'DoS Slowhttptest': {'technique': 'T1499', 'name': 'Endpoint Denial of Service'},
    'DoS Slowloris': {'technique': 'T1499', 'name': 'Endpoint Denial of Service'},
    'Heartbleed': {'technique': 'T1040', 'name': 'Network Sniffing'},
    'Web Attack - Brute Force': {'technique': 'T1110', 'name': 'Brute Force'},
    'Web Attack - XSS': {'technique': 'T1190', 'name': 'Exploit Public-Facing Application'},
    'Web Attack - Sql Injection': {'technique': 'T1190', 'name': 'Exploit Public-Facing Application'},
    'Infiltration': {'technique': 'T1566', 'name': 'Phishing'},
    'Bot': {'technique': 'T1071', 'name': 'Application Layer Protocol'},
    'PortScan': {'technique': 'T1046', 'name': 'Network Service Discovery'},
    'DDoS': {'technique': 'T1499', 'name': 'Endpoint Denial of Service'},
}

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS predictions
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp TEXT NOT NULL,
                      prediction_label TEXT NOT NULL,
                      confidence REAL NOT NULL,
                      attack_probability REAL NOT NULL,
                      model_used TEXT NOT NULL,
                      features_json TEXT NOT NULL)''')
        conn.commit()
    except Exception as e:
        print(f"[ERROR] init_db failed: {e}")
    finally:
        if conn is not None:
            conn.close()

def store_prediction(label, confidence, attack_prob, model_used, features_dict):
    conn = get_db_connection()
    if conn is None:
        return
    try:
        c = conn.cursor()
        timestamp = datetime.now().isoformat()
        features_json = json.dumps(features_dict)
        c.execute('INSERT INTO predictions (timestamp, prediction_label, confidence, attack_probability, model_used, features_json) VALUES (?, ?, ?, ?, ?, ?)',
                  (timestamp, label, confidence, attack_prob, model_used, features_json))
        conn.commit()
    except Exception as e:
        print(f"[ERROR] Failed to store prediction: {e}")
    finally:
        conn.close()

def load_models():
    global rf_model, dl_model, scaler, feature_names, label_encoder, shap_explainer
    
    try:
        print('Loading models')

        if os.path.exists(RF_MODEL_PATH):
            rf_model = joblib.load(RF_MODEL_PATH)
            print('Loaded Random Forest model')
        else:
            print(f"[ERROR] Random Forest model not found at {RF_MODEL_PATH}")

        if os.path.exists(DL_MODEL_PATH):
            dl_model = keras.models.load_model(DL_MODEL_PATH)
            print('Loaded Deep Learning model')
        else:
            print(f"[ERROR] Deep Learning model not found at {DL_MODEL_PATH}")

        if not os.path.exists(SCALER_PATH) or not os.path.exists(FEATURES_PATH):
            raise FileNotFoundError('Scaler or features file not found!')

        scaler = joblib.load(SCALER_PATH)
        print('Loaded scaler')

        feature_names = joblib.load(FEATURES_PATH)
        print('Loaded features')

        if os.path.exists(LABEL_ENCODER_PATH):
            label_encoder = joblib.load(LABEL_ENCODER_PATH)
            print('Loaded label encoder')
        else:
            print(f"[ERROR] Label encoder not found at {LABEL_ENCODER_PATH}")

        if rf_model is not None:
            shap_explainer = shap.TreeExplainer(rf_model)

        return True
    except Exception as e:
        print(f"[ERROR] load_models failed: {e}")
        return False

@app.after_request
def set_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

def get_db_connection():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        print(f"[ERROR] Database connection failed: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/features', methods=['GET'])
def get_features():
    if feature_names is None:
        return jsonify({'error': 'Models not loaded'}), 500
    return jsonify({'features': feature_names, 'count': len(feature_names)})

@app.route('/api/sample', methods=['GET'])
def get_sample():
    if feature_names is None:
        return jsonify({'error': 'Models not loaded'}), 500
    
    try:
        data_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.csv')]
        if not data_files:
            return jsonify({'error': 'Dataset not found'}), 404
        
        csv_path = os.path.join(DATA_PATH, data_files[0])
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df[feature_names + ['Label']].dropna()

        row = df.sample(1).iloc[0]
        actual_label = str(row['Label'])
        values = {f: round(float(row[f]), 4) for f in feature_names}
        return jsonify({'values': values, 'actual_label': actual_label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    try:
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
        return jsonify(metrics)
    except FileNotFoundError:
        return jsonify({'error': 'Metrics not found. Please run train.py first.'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@limiter.limit("30 per minute")
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if scaler is None or feature_names is None or label_encoder is None:
            return jsonify({'error': 'Models not loaded'}), 500
        
        data = request.json
        model_type = data.get('model', 'rf')
        
        if model_type == 'rf' and rf_model is None:
            return jsonify({'error': 'Random Forest model not available'}), 500
        if model_type == 'dl' and dl_model is None:
            return jsonify({'error': 'Deep Learning model not available'}), 500
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        features_dict = {}
        features_list = []
        
        for feature in feature_names:
            if feature in data:
                try:
                    value = float(data[feature])
                    
                    if value < 0 and feature in ['Flow Duration', 'Flow Bytes/s', 'Flow Packets/s', 
                                                   'Total Fwd Packets', 'Total Backward Packets',
                                                   'Fwd Packet Length Mean', 'Fwd Packet Length Std',
                                                   'Bwd Packet Length Mean', 'Flow IAT Mean', 'Flow IAT Std',
                                                   'Destination Port', 'Fwd IAT Total']:
                        return jsonify({'error': f'Feature {feature} cannot be negative'}), 400
                    
                    features_dict[feature] = value
                    features_list.append(value)
                except (ValueError, TypeError):
                    return jsonify({'error': f'Invalid value for {feature}'}), 400
            else:
                features_dict[feature] = 0
                features_list.append(0)
        
        features_array = np.array(features_list).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        
        if model_type == 'rf':
            prediction_encoded = rf_model.predict(features_scaled)[0]
            prediction_proba = rf_model.predict_proba(features_scaled)[0]
            model_name = 'Random Forest'
        else:
            pred_proba = dl_model.predict(features_scaled, verbose=0)[0]
            prediction_encoded = np.argmax(pred_proba)
            prediction_proba = pred_proba
            model_name = 'Deep Learning'
        
        prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]
        confidence = float(max(prediction_proba)) * 100
        
        # Calculate attack probability as 1 - benign probability
        benign_index = list(label_encoder.classes_).index('BENIGN')
        benign_prob = float(prediction_proba[benign_index])
        attack_prob = (1 - benign_prob) * 100
        
        if prediction_label == 'BENIGN':
            severity = 'NONE'
        elif attack_prob > 90:
            severity = 'CRITICAL'
        elif attack_prob > 70:
            severity = 'HIGH'
        elif attack_prob > 40:
            severity = 'MEDIUM'
        else:
            severity = 'LOW'

        mitre = MITRE_MAPPING.get(prediction_label, {'technique': 'Unknown', 'name': 'Unknown'})
        
        shap_values = None
        shap_features = []
        if model_type == 'rf' and shap_explainer is not None:
            try:
                shap_vals = shap_explainer.shap_values(features_scaled)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[prediction_encoded]
                
                feature_importance = list(zip(feature_names, np.abs(shap_vals[0])))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                shap_features = [{'feature': f[0], 'value': round(float(f[1]), 4)} for f in feature_importance[:5]]
            except Exception as e:
                print(f"[WARNING] SHAP explanation failed: {e}")
        
        store_prediction(prediction_label, confidence, attack_prob, model_name, features_dict)
        
        return jsonify({
            'success': True,
            'prediction': prediction_label,
            'confidence': f'{confidence:.2f}%',
            'attack_probability': f'{attack_prob:.2f}%',
            'severity': severity,
            'model': model_name,
            'timestamp': pd.Timestamp.now().isoformat(),
            'mitre_technique': mitre['technique'],
            'mitre_name': mitre['name'],
            'shap_features': shap_features
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 400

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    conn = None
    try:
        limit = request.args.get('limit', 50, type=int)
        conn = get_db_connection()
        if conn is None:
            return jsonify({'error': 'Database unavailable'}), 500
        c = conn.cursor()
        c.execute('SELECT * FROM predictions ORDER BY id DESC LIMIT ?', (limit,))
        rows = c.fetchall()
        
        alerts = []
        for row in rows:
            alerts.append({
                'timestamp': row['timestamp'],
                'prediction_label': row['prediction_label'],
                'confidence': row['confidence'],
                'model_used': row['model_used']
            })
        
        return jsonify({'alerts': alerts, 'count': len(alerts)})
    except Exception as e:
        print(f"[ERROR] get_alerts failed: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        if conn is not None:
            conn.close()

@app.route('/api/analyze-batch', methods=['POST'])
def analyze_batch():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'File must be CSV format'}), 400
        
        # Get model selection from request
        model_type = request.form.get('model', 'rf')
        
        if model_type == 'rf' and rf_model is None:
            return jsonify({'error': 'Random Forest model not available'}), 500
        if model_type == 'dl' and dl_model is None:
            return jsonify({'error': 'Deep Learning model not available'}), 500
        
        try:
            stream = StringIO(file.stream.read().decode('utf-8'), newline=None)
            df = pd.read_csv(stream)
        except Exception as e:
            return jsonify({'error': f'Failed to parse CSV: {str(e)}'}), 400
        
        # Check row limit
        if len(df) > 1000:
            return jsonify({'error': 'CSV file too large. Maximum 1000 rows allowed.'}), 400
        
        df.columns = df.columns.str.strip()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        results = []
        attack_type_counts = {}
        benign_count = 0
        attack_count = 0
        
        for idx, row in df.iterrows():
            try:
                features_list = []
                for feature in feature_names:
                    if feature in row:
                        features_list.append(float(row[feature]) if pd.notna(row[feature]) else 0)
                    else:
                        features_list.append(0)
                
                features_array = np.array(features_list).reshape(1, -1)
                features_scaled = scaler.transform(features_array)
                
                if model_type == 'rf':
                    prediction_encoded = rf_model.predict(features_scaled)[0]
                    prediction_proba = rf_model.predict_proba(features_scaled)[0]
                    model_name = 'Random Forest'
                else:
                    pred_proba = dl_model.predict(features_scaled, verbose=0)[0]
                    prediction_encoded = np.argmax(pred_proba)
                    prediction_proba = pred_proba
                    model_name = 'Deep Learning'
                
                prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]
                confidence = float(max(prediction_proba)) * 100
                
                results.append({
                    'row_index': idx,
                    'prediction': prediction_label,
                    'confidence': f'{confidence:.2f}%'
                })
                
                if prediction_label == 'BENIGN':
                    benign_count += 1
                else:
                    attack_count += 1
                    attack_type_counts[prediction_label] = attack_type_counts.get(prediction_label, 0) + 1
            except Exception as e:
                results.append({
                    'row_index': idx,
                    'prediction': 'ERROR',
                    'confidence': '0%',
                    'error': str(e)
                })
        
        summary = {
            'total_rows': len(df),
            'benign_count': benign_count,
            'attack_count': attack_count,
            'attack_type_breakdown': attack_type_counts,
            'results': results
        }
        
        return jsonify(summary)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    conn = get_db_connection()
    total_preds = 0
    if conn is not None:
        try:
            c = conn.cursor()
            c.execute('SELECT COUNT(*) FROM predictions')
            total_preds = c.fetchone()[0]
        except Exception as e:
            print(f"[ERROR] Health check database query failed: {e}")
        finally:
            conn.close()
    
    status = 'healthy' if (rf_model or dl_model) else 'unhealthy'
    return jsonify({
        'status': status,
        'rf_loaded': rf_model is not None,
        'dl_loaded': dl_model is not None,
        'features_count': len(feature_names) if feature_names else 0,
        'total_predictions': total_preds
    })

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Server error'}), 500

if __name__ == '__main__':
    print('Starting Network Intrusion Detection System Dashboard...')
    init_db()
    load_models()
    print('Application startup complete.')
    app.run(debug=True, host='0.0.0.0', port=5000)