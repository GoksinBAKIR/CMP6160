# Kalman Filtering
# Gaussian Mixture Models (GMM)
# Hidden Markov Models (HMM)
# Bayesian Fusion
# Ensemble Method

# ADD deep learning
# add temporal smoothing

# Kalman filtering
# GMM-based fusion
# HMM-based fusion
# Deep learning fusion
# Temporal smoothing
# Ensemble fusion
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
import tensorflow as tf
from scipy.stats import mode
from scipy.interpolate import interp1d

class AdvancedMusicFusion:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.gmm = GaussianMixture(n_components=12)
        self.rf_classifier = RandomForestClassifier(n_estimators=100)
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
    def load_and_process_audio(self, audio_path):
        """Load and process audio file"""
        y, sr = librosa.load(audio_path, sr=None)
        
        # Calculate RMS energy
        rms = librosa.feature.rms(y=y)[0]
        rms_normalized = (rms - rms.min()) / (rms.max() - rms.min())
        
        # Harmonic-percussive separation
        y_harmonic, _ = librosa.effects.hpss(y)
        
        # Compute chroma features
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
        
        return y, sr, chroma, rms_normalized

    def prepare_features(self, chroma, rms):
        """Prepare and combine features"""
        # Match RMS length to chroma
        old_times = np.linspace(0, len(rms), len(rms))
        new_times = np.linspace(0, len(rms), chroma.shape[1])
        rms_interpolator = interp1d(old_times, rms)
        rms_matched = rms_interpolator(new_times)
        
        # Combine features
        rms_reshaped = np.repeat(rms_matched.reshape(-1, 1), chroma.shape[0], axis=1).T
        combined_features = np.concatenate([chroma, rms_reshaped], axis=0)
        return self.scaler.fit_transform(combined_features), rms_matched

    def kalman_fusion(self, chroma, rms):
        """Kalman filter fusion"""
        R = 0.1  # Measurement noise
        Q = 0.1  # Process noise
        P = 1.0  # Initial error covariance
        X = chroma[:, 0]  # Initial state
        
        filtered_results = []
        for i in range(chroma.shape[1]):
            P = P + Q
            K = P / (P + R)
            X = X + K * (chroma[:, i] * rms[i] - X)
            P = (1 - K) * P
            filtered_results.append(X)
            
        return np.array(filtered_results).T

    def gmm_fusion(self, features):
        """GMM-based fusion"""
        self.gmm.fit(features.T)
        return self.gmm.predict_proba(features.T)

    def hmm_fusion(self, features):
        """HMM-based fusion"""
        model = hmm.GaussianHMM(n_components=12, covariance_type="full")
        model.fit(features.T)
        return model.predict(features.T)

    def deep_fusion(self, features):
        """Deep learning fusion"""
        # Create one-hot encoded labels (assuming 12 possible notes)
        labels = np.eye(12)[np.random.randint(0, 12, features.shape[1])]
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(12, activation='softmax')
        ])
        # train the model
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        model.fit(features.T, labels, epochs=10)
        print("10 epochs completed")

        return model.predict(features.T)

    def temporal_smoothing(self, predictions, window_size=5):
        """Apply temporal smoothing"""
        smoothed = np.zeros_like(predictions)
        for i in range(len(predictions)):
            start = max(0, i - window_size//2)
            end = min(len(predictions), i + window_size//2)
            smoothed[i] = np.mean(predictions[start:end], axis=0)
        return smoothed

    def ensemble_fusion(self, predictions_list):
        """Combine multiple predictions"""
        # First, ensure all predictions have the same shape (12 x T)
        processed_predictions = []
        
        for pred in predictions_list:
            # Convert to numpy array if not already
            pred = np.array(pred)
            
            # Reshape if necessary to ensure 2D array
            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)
                
            # Transpose if dimensions are swapped
            if pred.shape[0] != 12:
                pred = pred.T
                
            # Ensure predictions are non-negative
            pred = np.maximum(pred, 0)
            
            # Normalize each time step to sum to 1
            row_sums = pred.sum(axis=0, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            normalized = pred / row_sums
            
            processed_predictions.append(normalized)
        
        # Average the normalized predictions
        weights = [0.3, 0.3, 0.2, 0.2]  # Weights for each prediction method
        return np.average(processed_predictions, axis=0, weights=weights)

    def analyze_audio(self, audio_path):
        """Main analysis function"""
        # Load and process audio
        y, sr, chroma, rms = self.load_and_process_audio(audio_path)
        
        # Prepare features
        features, rms_matched = self.prepare_features(chroma, rms)
        
        # Apply different fusion methods

        # use kalman for temporal State Transition / next state covariance 
        kalman_results = self.kalman_fusion(chroma, rms_matched)
        print("kalman_results", kalman_results)
        #   use GMM for Clustering Prediction and anomaly detection  
        gmm_results = self.gmm_fusion(features)
        print("gmm_results", gmm_results)

        # use hmm for State Probability Prediction/anomaly detection 
        hmm_results = self.hmm_fusion(features)
        print("hmm_results", hmm_results)

        # use deep learning for temporal prediction

        deep_results = self.deep_fusion(features)
        print("deep_results", deep_results)
        print("deep_results.T", deep_results.T)
        
        print("combining all predictions")

        # Combine all predictions
        predictions = [
            kalman_results,
            gmm_results,
            np.eye(12)[hmm_results].T,
            deep_results.T
        ]
        print ("saving predictions to json")

        # Save predictions to JSON
        prediction_data = {
            'kalman': kalman_results.tolist(),
            'gmm': gmm_results.tolist(),
            'hmm': np.eye(12)[hmm_results].T.tolist(),
            'deep': deep_results.T.tolist()
        }
        
        with open('fusion_predictions.json', 'w') as f:
            json.dump(prediction_data, f, indent=4)
            
        # Plot predictions
        plt.figure(figsize=(15, 10))
        times = librosa.frames_to_time(range(len(kalman_results[0])), sr=sr)
        
        plt.subplot(4, 1, 1)
        plt.imshow(kalman_results, aspect='auto', origin='lower')
        plt.title('Kalman Filter Predictions')
        plt.colorbar()
        
        plt.subplot(4, 1, 2)
        plt.imshow(gmm_results.T, aspect='auto', origin='lower')
        plt.title('GMM Predictions')
        plt.colorbar()
        
        plt.subplot(4, 1, 3)
        plt.imshow(np.eye(12)[hmm_results].T, aspect='auto', origin='lower')
        plt.title('HMM Predictions')
        plt.colorbar()
        
        plt.subplot(4, 1, 4)
        plt.imshow(deep_results.T, aspect='auto', origin='lower')
        plt.title('Deep Learning Predictions')
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig('fusion_predictions.png')
        plt.close()



        print("predictions", predictions)
        print ("going to final results")
        # Get final results
        final_results = self.ensemble_fusion(predictions)
        
        print("final_results", final_results)

        smoothed_results = self.temporal_smoothing(final_results.T).T
        print("smoothed_results", smoothed_results)

        return self.process_results(smoothed_results, sr)

    def process_results(self, final_results, sr):
        """Process and format results"""
        print("processing results")

        results = []
        for t, frame in enumerate(final_results.T):
            time = librosa.frames_to_time(t, sr=sr)
            dominant_note = self.note_names[np.argmax(frame)]
            confidence = np.max(frame)
            
            results.append({
                'time': float(time),
                'dominant_note': dominant_note,
                'confidence': float(confidence),
                'probabilities': {note: float(prob) 
                                for note, prob in zip(self.note_names, frame)}
            })
        # Save results to JSON file
        with open('advanced_fusion_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        return results

    def visualize_results(self, results, sr):
        """Visualize the analysis results"""
        print("visualizing results")
        
        times = [r['time'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        plt.figure(figsize=(15, 8))
        
        # Plot note progression
        plt.subplot(2, 1, 1)
        plt.plot(times, [self.note_names.index(r['dominant_note']) for r in results], 'b.')
        plt.yticks(range(12), self.note_names)
        plt.title('Detected segments  Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Note')
        
        # Plot confidence levels
        plt.subplot(2, 1, 2)
        plt.plot(times, confidences)
        plt.title('Detection Confidence')
        plt.xlabel('Time (s)')
        plt.ylabel('Confidence')
        
        plt.tight_layout()
        plt.show()

def save_results(results, output_file):
    """Save results to JSON file"""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print("Results saved to advanced_fusion_results.json")


if __name__ == "__main__":
    # Example usage
    fusion = AdvancedMusicFusion()
    audio_path = './data/3_Seven Nation Army_The White Stripes.wav'
    
    try:
        # Analyze audio
        results = fusion.analyze_audio(audio_path)
        
        # Save results
        save_results(results, 'advanced_fusion_results.json')
        
        # Visualize results
        fusion.visualize_results(results, librosa.get_samplerate(audio_path))
        # Plot dominant note vs time
        plt.figure(figsize=(12, 6))
        times = [r['time'] for r in results]
        dominant_notes = [r['dominant_note'] for r in results]
        
        # plot dominant note again from final results  
        # Convert notes to numeric values for plotting
        note_to_num = {note: i for i, note in enumerate(fusion.note_names)}
        note_values = [note_to_num[note] for note in dominant_notes]
        
        plt.plot(times, note_values, 'b-', linewidth=2)
        plt.yticks(range(12), fusion.note_names)
        plt.grid(True, alpha=0.3)
        plt.title('Dominant Note Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Note')
        plt.show()
        print("Analysis complete! Results saved to advanced_fusion_results.json")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
       

