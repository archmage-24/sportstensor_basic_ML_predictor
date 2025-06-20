#!/usr/bin/env python3
"""
Basic ML Model for Sportstensor ML Predictor
Implements ensemble methods and basic ML algorithms
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import pickle
import json
from pathlib import Path
import logging
from datetime import datetime
import os

# ML imports (these will be installed via requirements)
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    import joblib
except ImportError as e:
    print(f"Warning: ML libraries not available: {e}")
    print("Install with: pip install scikit-learn joblib")

class BasicMLPredictor:
    """Basic ML Predictor using an ensemble of models"""
    
    def __init__(self, model_dir: str = "ml_predictor/models"):
        print("🤖 Initializing ML models...")
        self.model_dir = model_dir
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize models, scaler, encoder, and imputer
        self.models = {}
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='mean')
        self.feature_columns = None
        self.performance_history = []
        
        self._initialize_models()
        
        # Load models if they exist
        if os.path.exists(os.path.join(self.model_dir, "ensemble_model.pkl")):
            self.load_models()
    
    def _initialize_models(self):
        """Initialize all models to be used"""
        
        # Random Forest
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Gradient Boosting
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        # Logistic Regression
        self.models['logistic_regression'] = LogisticRegression(
            C=1.0,
            solver='liblinear',
            random_state=42
        )
        
        # Support Vector Machine
        self.models['svm'] = SVC(
            C=1.0,
            kernel='rbf',
            probability=True,
            random_state=42
        )
        
        # Create ensemble
        self.ensemble_model = VotingClassifier(
            estimators=[
                ('rf', self.models['random_forest']),
                ('gb', self.models['gradient_boosting']),
                ('lr', self.models['logistic_regression']),
                ('svm', self.models['svm'])
            ],
            voting='soft'  # Use probability predictions
        )
        
        print("✅ Models initialized successfully")
    
    def prepare_training_data(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training"""
        print("📊 Preparing training data...")
        
        # Separate numeric and non-numeric columns
        numeric_cols = features_df.select_dtypes(include=np.number).columns.tolist()
        
        # Get feature columns
        feature_columns = self._get_available_features(features_df)
        
        # We only want to train on numeric features
        self.feature_columns = [col for col in feature_columns if col in numeric_cols]
        
        # Extract features
        X = features_df[self.feature_columns].values
        
        # Create target variable (simplified for now)
        y = self._create_target_variable(features_df)
        
        # Handle missing values
        X = self.imputer.fit_transform(X)
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Encode target
        y = self.label_encoder.fit_transform(y)
        
        print(f"✅ Prepared {X.shape[0]} samples with {X.shape[1]} features")
        return X, y
    
    def _get_available_features(self, df: pd.DataFrame) -> List[str]:
        """Get list of available features that the model expects"""
        expected_features = [
            'league_nfl', 'league_nba', 'league_mlb', 'league_epl', 'league_mls',
            'win_rate_home', 'win_rate_away', 'win_rate_diff', 'strength_diff',
            'ppg_home', 'ppg_away', 'ppg_diff', 'papg_home', 'papg_away', 'papg_diff',
            'h2h_total_games', 'h2h_home_wins', 'h2h_away_wins', 'h2h_draws',
            'h2h_home_win_rate', 'h2h_avg_goals_home', 'h2h_avg_goals_away',
            'home_odds', 'away_odds', 'draw_odds', 'home_prob_odds', 'away_prob_odds',
            'draw_prob_odds', 'odds_margin', 'odds_favorite', 'odds_confidence',
            'match_day_of_week', 'match_month', 'match_year', 'match_hour',
            'is_weekend', 'is_weekday', 'is_evening', 'is_afternoon',
            'home_form_wins', 'home_form_losses', 'home_form_draws', 'home_form_score',
            'away_form_wins', 'away_form_losses', 'away_form_draws', 'away_form_score',
            'form_diff', 'home_venue_win_rate', 'away_venue_win_rate', 'venue_advantage',
            'home_games_played', 'away_games_played', 'strength_interaction',
            'form_strength_home', 'form_strength_away', 'odds_stats_alignment',
            'momentum_home', 'momentum_away', 'home_advantage_composite',
            'away_advantage_composite', 'advantage_differential'
        ]
        
        available_features = [col for col in expected_features if col in df.columns]
        return available_features

    def _create_target_variable(self, df: pd.DataFrame) -> np.ndarray:
        """Create a synthetic target variable for training based on odds and team stats."""
        targets = []
        for _, row in df.iterrows():
            home_prob = row.get('home_prob_odds', 0.33)
            away_prob = row.get('away_prob_odds', 0.33)
            home_strength = row.get('win_rate_home', 0.5)
            away_strength = row.get('win_rate_away', 0.5)

            combined_home_prob = (home_prob * 0.7) + (home_strength * 0.3)
            combined_away_prob = (away_prob * 0.7) + (away_strength * 0.3)

            if combined_home_prob > combined_away_prob * 1.15:
                targets.append('home_win')
            elif combined_away_prob > combined_home_prob * 1.15:
                targets.append('away_win')
            else:
                targets.append('draw')
        return np.array(targets)

    def train_models(self, features_df: pd.DataFrame, test_size: float = 0.2):
        """Train all models"""
        print("🎯 Training ML models...")
        
        # Prepare data
        X, y = self.prepare_training_data(features_df)
        
        unique_classes, counts = np.unique(y, return_counts=True)
        min_samples_per_class = min(counts) if counts.any() else 0
        
        if len(X) < 10:
            print(f"   ⚠️ Very small dataset detected ({len(X)} samples), using all data for training and testing")
            X_train, X_test, y_train, y_test = X, X, y, y
        elif min_samples_per_class < 2:
            print(f"   ⚠️ Not enough samples for stratification ({min_samples_per_class} samples in smallest class), using regular split")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        
        individual_scores = {}
        for name, model in self.models.items():
            print(f"   Training {name}...")
            model.fit(X_train, y_train)
            score = accuracy_score(y_test, model.predict(X_test))
            individual_scores[name] = score
            print(f"   ✅ {name} accuracy: {score:.4f}")
        
        print("   Training ensemble model...")
        self.ensemble_model.fit(X_train, y_train)
        ensemble_score = accuracy_score(y_test, self.ensemble_model.predict(X_test))
        individual_scores['ensemble'] = ensemble_score
        print(f"   ✅ Ensemble accuracy: {ensemble_score:.4f}")
        
        self.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'scores': individual_scores,
            'test_size': len(X_test),
            'train_size': len(X_train)
        })
        
        self.save_models()
        print("✅ All models trained and saved!")
        return individual_scores

    def predict_match(self, match_features: pd.DataFrame) -> Dict[str, Any]:
        """Predict outcome for a single match, ensuring feature alignment."""
        if self.ensemble_model is None or not self.feature_columns:
            raise ValueError("Models not trained or feature columns not set. Please train the model first.")
        
        # Align incoming features with the model's expected features
        aligned_features = pd.DataFrame(columns=self.feature_columns)
        aligned_features = pd.concat([aligned_features, match_features], ignore_index=True, sort=False)
        aligned_features = aligned_features[self.feature_columns].fillna(0)
        
        # Convert to numpy array
        X = aligned_features.values
        
        # Impute and scale features
        X = self.imputer.transform(X)
        X = self.scaler.transform(X)
        
        # Get predictions from all models
        predictions = {}
        probabilities = {}
        for name, model in self.models.items():
            pred = model.predict(X)[0]
            prob = model.predict_proba(X)[0]
            predictions[name] = pred
            probabilities[name] = prob
            
        ensemble_pred = self.ensemble_model.predict(X)[0]
        ensemble_prob = self.ensemble_model.predict_proba(X)[0]
        predictions['ensemble'] = ensemble_pred
        probabilities['ensemble'] = ensemble_prob
        
        original_labels = self.label_encoder.classes_
        prediction_labels = {name: original_labels[pred] for name, pred in predictions.items()}
        
        recommended_prediction = prediction_labels['ensemble']
        confidence = max(ensemble_prob)
        
        model_agreement = self._calculate_model_agreement(prediction_labels)
        
        return {
            "recommended_prediction": recommended_prediction,
            "confidence": confidence,
            "model_agreement": model_agreement,
            "individual_predictions": prediction_labels,
            "probabilities": {label: prob for label, prob in zip(original_labels, ensemble_prob)}
        }

    def _calculate_model_agreement(self, predictions: Dict[str, Any]) -> float:
        """Calculate the agreement between the individual models."""
        pred_values = [pred for name, pred in predictions.items() if name != 'ensemble']
        if not pred_values:
            return 1.0
        agreement = pd.Series(pred_values).value_counts(normalize=True).max()
        return agreement

    def evaluate_model(self, test_features_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate model performance on a test dataset."""
        if self.ensemble_model is None:
            raise ValueError("Models not trained yet!")
        
        X_test, y_test_true = self.prepare_training_data(test_features_df)
        
        # Get predictions
        y_pred = self.ensemble_model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_true, y_pred)
        f1 = f1_score(y_test_true, y_pred, average='weighted')
        
        # Get class-wise performance
        report = classification_report(y_test_true, y_pred, output_dict=True, zero_division=0)
        
        print(f"📊 Model Evaluation:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1-Score (weighted): {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'f1_score_weighted': f1,
            'classification_report': report
        }

    def save_models(self):
        """Save all models, scaler, and encoder"""
        print("💾 Saving models...")
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(self.model_dir, f"{name}_model.pkl"))
        joblib.dump(self.ensemble_model, os.path.join(self.model_dir, "ensemble_model.pkl"))
        joblib.dump(self.scaler, os.path.join(self.model_dir, "scaler.pkl"))
        joblib.dump(self.imputer, os.path.join(self.model_dir, "imputer.pkl"))
        joblib.dump(self.label_encoder, os.path.join(self.model_dir, "label_encoder.pkl"))
        with open(os.path.join(self.model_dir, "feature_columns.json"), 'w') as f:
            json.dump(self.feature_columns, f)
        with open(os.path.join(self.model_dir, "performance_history.json"), 'w') as f:
            json.dump(self.performance_history, f)
        print("✅ Models saved successfully")

    def load_models(self):
        """Load all models, scaler, and encoder"""
        print("📂 Loading models...")
        try:
            for name in self.models:
                self.models[name] = joblib.load(os.path.join(self.model_dir, f"{name}_model.pkl"))
            self.ensemble_model = joblib.load(os.path.join(self.model_dir, "ensemble_model.pkl"))
            self.scaler = joblib.load(os.path.join(self.model_dir, "scaler.pkl"))
            self.imputer = joblib.load(os.path.join(self.model_dir, "imputer.pkl"))
            self.label_encoder = joblib.load(os.path.join(self.model_dir, "label_encoder.pkl"))
            with open(os.path.join(self.model_dir, "feature_columns.json"), 'r') as f:
                self.feature_columns = json.load(f)
            if os.path.exists(os.path.join(self.model_dir, "performance_history.json")):
                with open(os.path.join(self.model_dir, "performance_history.json"), 'r') as f:
                    self.performance_history = json.load(f)
            print("✅ Models loaded successfully")
        except FileNotFoundError:
            print("   ⚠️ No saved models found. Please train the models first.")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained models."""
        if self.ensemble_model is None:
            return {"status": "Models not trained"}
        
        return {
            "status": "Models trained",
            "feature_count": len(self.feature_columns),
            "models_trained": list(self.models.keys()) + ['ensemble'],
            "label_classes": list(self.label_encoder.classes_)
        }

def main():
    """Main function to test the basic ML predictor"""
    
    # Create a dummy dataframe for testing
    data = {
        'win_rate_home': [0.6, 0.5, 0.7],
        'win_rate_away': [0.4, 0.5, 0.3],
        'win_rate_diff': [0.2, 0.0, 0.4],
        'strength_diff': [0.1, -0.1, 0.2],
        'home_prob_odds': [0.5, 0.4, 0.6],
        'away_prob_odds': [0.3, 0.4, 0.2],
        'form_diff': [0.5, 0.2, 0.8],
        'venue_advantage': [0.1, 0.05, 0.15],
        'league_nfl': [1, 0, 1],
        'league_epl': [0, 1, 0]
    }
    features_df = pd.DataFrame(data)
    
    # Initialize predictor
    predictor = BasicMLPredictor()
    
    # Train models
    scores = predictor.train_models(features_df)
    print("\n📊 Final Scores:", scores)
    
    # Test prediction
    test_match = pd.DataFrame({
        'win_rate_home': [0.55],
        'win_rate_away': [0.45],
        'win_rate_diff': [0.1],
        'strength_diff': [0.05],
        'home_prob_odds': [0.45],
        'away_prob_odds': [0.35],
        'form_diff': [0.3],
        'venue_advantage': [0.08],
        'league_nfl': [1],
        'league_epl': [0]
    })
    
    prediction = predictor.predict_match(test_match)
    print("\n🎯 Test Prediction:", prediction)
    
    # Get model info
    info = predictor.get_model_info()
    print("\n🤖 Model Info:", info)

if __name__ == '__main__':
    main() 