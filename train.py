import os
import json
import glob
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import label_binarize
import keras
from keras import layers

DL_MODEL_FILENAME = 'dl_model.keras'

SELECTED_FEATURES = [
    'Flow Duration',
    'Flow Bytes/s',
    'Flow Packets/s',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Fwd Packet Length Mean',
    'Fwd Packet Length Std',
    'Bwd Packet Length Mean',
    'Flow IAT Mean',
    'Flow IAT Std',
    'Destination Port',
    'Fwd IAT Total',
]

def get_data_files(data_dir):
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    return sorted(csv_files)

def load_and_preprocess(data_dir):
    csv_files = get_data_files(data_dir)
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    print(f"[INFO] Found {len(csv_files)} CSV file(s)")
    
    dfs = []
    for csv_file in csv_files:
        print(f"[INFO] Loading {os.path.basename(csv_file)}...")
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.strip()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Total rows after concatenation: {len(df)}")
    
    missing = [f for f in SELECTED_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features in dataset: {missing}")

    label_encoder = LabelEncoder()
    df['Label_Encoded'] = label_encoder.fit_transform(df['Label'])
    
    print(f"[INFO] Classes found: {label_encoder.classes_}")
    print(f"[INFO] Total classes: {len(label_encoder.classes_)}")

    X = df[SELECTED_FEATURES]
    y = df['Label_Encoded']

    feature_names = SELECTED_FEATURES
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_balanced, y_train_balanced = smote.fit_resample(
        X_train_scaled, y_train
    )
    
    return {
        'X_train': X_train_balanced,
        'X_test': X_test_scaled,
        'y_train': y_train_balanced,
        'y_test': y_test,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_names': feature_names,
        'num_classes': len(label_encoder.classes_)
    }

def train_random_forest(X_train, y_train, X_test, y_test, num_classes, model_dir):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    if num_classes == 2:
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    else:
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        y_bin = label_binarize(y_test, classes=range(num_classes))
        roc_auc = roc_auc_score(y_bin, y_pred_proba, average='weighted', multi_class='ovr')
    
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist()
    }
    
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, 'rf_model.pkl'))
    
    return model, metrics

def train_deep_learning(X_train, y_train, X_test, y_test, num_classes, model_dir):
    if num_classes == 2:
        model = keras.Sequential([
            layers.Input(shape=(X_train.shape[1],)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(2, activation='softmax')
        ])
        
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
    else:
        model = keras.Sequential([
            layers.Input(shape=(X_train.shape[1],)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
    
    model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=1, validation_split=0.1)
    
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    if num_classes == 2:
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    else:
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        y_bin = label_binarize(y_test, classes=range(num_classes))
        roc_auc = roc_auc_score(y_bin, y_pred_proba, average='weighted', multi_class='ovr')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
    }
    
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, DL_MODEL_FILENAME))
    
    return model, metrics

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    model_dir = os.path.join(current_dir, 'model')
    
    print("[INFO] Loading and preprocessing data...")
    data = load_and_preprocess(data_dir)
    
    print("[INFO] Saving scaler...")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(data['scaler'], os.path.join(model_dir, 'scaler.pkl'))
    
    print("[INFO] Saving label encoder...")
    joblib.dump(data['label_encoder'], os.path.join(model_dir, 'label_encoder.pkl'))
    
    print("[INFO] Saving feature names...")
    joblib.dump(data['feature_names'], os.path.join(model_dir, 'feature_names.pkl'))
    
    print("[INFO] Training Random Forest model...")
    rf_model, rf_metrics = train_random_forest(
        data['X_train'], data['y_train'],
        data['X_test'], data['y_test'],
        data['num_classes'],
        model_dir
    )
    
    print("\n" + "="*50)
    print("RANDOM FOREST METRICS")
    print("="*50)
    for key, value in rf_metrics.items():
        if key != 'confusion_matrix':
            print(f"{key.upper()}: {value:.4f}")
    print(f"Confusion Matrix:\n{np.array(rf_metrics['confusion_matrix'])}")
    print("="*50)
    
    print("\n[INFO] Training Deep Learning model...")
    dl_model, dl_metrics = train_deep_learning(
        data['X_train'], data['y_train'],
        data['X_test'], data['y_test'],
        data['num_classes'],
        model_dir
    )
    
    print("\n" + "="*50)
    print("DEEP LEARNING METRICS")
    print("="*50)
    for key, value in dl_metrics.items():
        print(f"{key.upper()}: {value:.4f}")
    print("="*50)

    print("\n[INFO] Saving metrics...")
    metrics_data = {
        'random_forest': {
            'accuracy':  round(rf_metrics['accuracy']  * 100, 2),
            'precision': round(rf_metrics['precision'] * 100, 2),
            'recall':    round(rf_metrics['recall']    * 100, 2),
            'f1_score':  round(rf_metrics['f1_score']  * 100, 2),
            'roc_auc':   round(rf_metrics['roc_auc']   * 100, 2),
            'confusion_matrix': rf_metrics['confusion_matrix']
        },
        'deep_learning': {
            'accuracy':  round(dl_metrics['accuracy']  * 100, 2),
            'precision': round(dl_metrics['precision'] * 100, 2),
            'recall':    round(dl_metrics['recall']    * 100, 2),
            'f1_score':  round(dl_metrics['f1_score']  * 100, 2),
            'roc_auc':   round(dl_metrics['roc_auc']   * 100, 2),
        }
    }
    with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print("[INFO] Metrics saved to model/metrics.json")

    print(f"\n[SUCCESS] All models and preprocessors saved to {model_dir}")

if __name__ == '__main__':
    main()