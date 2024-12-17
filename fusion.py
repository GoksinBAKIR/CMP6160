# Random Forest Fusion
# Neural Network Fusion
# Ensemble Learning Fusion

import numpy as np
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras import layers, models

class AudioFeatureFusion:
    def __init__(self):
        self.scaler = StandardScaler()
        self.chord_data = None
        self.rms_data = None
        self.combined_features = None
        
    def load_data(self):
        # Load chord data from JSON
        with open('chroma_features.json', 'r') as f:
            self.chord_data = json.load(f)
            
        # Load RMS data from JSON
        with open('rms_features.json', 'r') as f:
            self.rms_data = json.load(f)
            
    def prepare_features(self):
        # Extract features from chord data
        chord_features = []
        for frame in self.chord_data:
            features = list(frame['intensities'].values())
            chord_features.append(features)
            
        # Extract features from RMS data
        rms_features = []
        for frame in self.rms_data:
            features = [frame['rms'], frame['energy']]  # Adjust based on your RMS data structure
            rms_features.append(features)
            
        # Convert to numpy arrays
        chord_features = np.array(chord_features)
        rms_features = np.array(rms_features)
        
        # Combine features
        self.combined_features = np.concatenate((chord_features, rms_features), axis=1)
        self.combined_features = self.scaler.fit_transform(self.combined_features)
        
        return self.combined_features

    def random_forest_fusion(self):
        """Method 1: Random Forest Fusion"""
        print("Performing Random Forest Fusion...")
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        # Assuming we're predicting the dominant chord
        labels = [frame['dominant_note'] for frame in self.chord_data]
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.combined_features, labels, test_size=0.2, random_state=42
        )
        
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        print(f"Random Forest Accuracy: {accuracy}")
        
        return model

    def neural_network_fusion(self):
        """Method 2: Neural Network Fusion"""
        print("Performing Neural Network Fusion...")
        
        def create_model():
            model = models.Sequential([
                layers.Input(shape=(self.combined_features.shape[1],)),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.Dense(12, activation='softmax')  # 12 possible notes
            ])
            return model
        
        model = create_model()
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        # Convert labels to numerical values
        chord_to_num = {chord: i for i, chord in 
                       enumerate(['C', 'C#', 'D', 'D#', 'E', 'F', 
                                'F#', 'G', 'G#', 'A', 'A#', 'B'])}
        labels = np.array([chord_to_num[frame['dominant_note']] 
                          for frame in self.chord_data])
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.combined_features, labels, test_size=0.2, random_state=42
        )
        
        model.fit(X_train, y_train, 
                 epochs=10, 
                 validation_data=(X_test, y_test),
                 verbose=1)
        
        return model

    def ensemble_fusion(self):
        """Method 3: Ensemble Learning Fusion"""
        print("Performing Ensemble Fusion...")
        
        # Create base models
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        lr = LogisticRegression(random_state=42)
        svm = SVC(probability=True, random_state=42)
        
        # Create ensemble model
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('lr', lr),
                ('svm', svm)
            ],
            voting='soft'
        )
        
        # Prepare labels
        labels = [frame['dominant_note'] for frame in self.chord_data]
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.combined_features, labels, test_size=0.2, random_state=42
        )
        
        ensemble.fit(X_train, y_train)
        accuracy = ensemble.score(X_test, y_test)
        print(f"Ensemble Accuracy: {accuracy}")
        
        return ensemble

    def save_fusion_results(self, model, method_name):
        """Save fusion results to JSON"""
        predictions = model.predict(self.combined_features)
        
        results = []
        for i, (pred, chord_data, rms_data) in enumerate(zip(predictions, self.chord_data, self.rms_data)):
            result = {
                "time": chord_data["time"],
                "predicted_chord": pred,
                "original_chord": chord_data["dominant_note"],
                "rms_energy": rms_data["energy"],
                "confidence": float(max(model.predict_proba([self.combined_features[i]])[0]))
                if hasattr(model, 'predict_proba') else None
            }
            results.append(result)
            
        with open(f'fusion_results_{method_name}.json', 'w') as f:
            json.dump(results, f, indent=4)

def main():
    # Initialize fusion class
    fusion = AudioFeatureFusion()
    
    # Load and prepare data
    fusion.load_data()
    fusion.prepare_features()
    
    # Apply different fusion methods
    rf_model = fusion.random_forest_fusion()
    fusion.save_fusion_results(rf_model, "random_forest")
    
    nn_model = fusion.neural_network_fusion()
    # Note: Neural network results need special handling due to different prediction format
    
    ensemble_model = fusion.ensemble_fusion()
    fusion.save_fusion_results(ensemble_model, "ensemble")
    
    print("Fusion analysis complete. Results saved to JSON files.")

if __name__ == "__main__":
    main()

    
