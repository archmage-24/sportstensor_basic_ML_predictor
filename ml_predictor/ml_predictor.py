#!/usr/bin/env python3
"""
Main ML Predictor for Sportstensor
Integrates data collection, feature engineering, and ML models
"""

import os
import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

# Import our modules
from data_collector import DataCollector
from feature_engineering import FeatureEngineer
from basic_ml_model import BasicMLPredictor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_predictor.log'),
        logging.StreamHandler()
    ]
)

class MLPredictor:
    """Main ML predictor that orchestrates the entire prediction pipeline"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.data_collector = DataCollector(self.config['data_dir'])
        self.feature_engineer = FeatureEngineer()
        self.ml_predictor = BasicMLPredictor(self.config['model_dir'])
        
        # State tracking
        self.is_trained = False
        self.last_training = None
        self.performance_metrics = {}
        
        # Load existing models if available
        self._load_existing_models()
        
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'data_dir': 'ml_predictor/data',
            'model_dir': 'ml_predictor/models',
            'leagues': ['NFL', 'NBA', 'MLB', 'EPL', 'MLS'],
            'training_frequency_days': 7,
            'prediction_confidence_threshold': 0.6,
            'max_prediction_age_hours': 24,
            'enable_auto_retraining': True,
            'enable_performance_monitoring': True
        }
    
    def _load_existing_models(self):
        """Load existing trained models"""
        try:
            if self.ml_predictor.load_models():
                self.is_trained = True
                model_info = self.ml_predictor.get_model_info()
                self.last_training = model_info.get('last_training')
                print(f"✅ Loaded existing models (trained: {self.last_training})")
            else:
                print("ℹ️ No existing models found - will train new ones")
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")
    
    async def collect_and_prepare_data(self) -> Dict[str, pd.DataFrame]:
        """Collect and prepare all data for training"""
        print("🚀 Starting data collection and preparation...")
        
        # Collect data
        await self.data_collector.collect_all_data()
        
        # Load collected data
        all_data = self.data_collector.load_all_data()
        
        # Separate different data types
        matches_data = {}
        team_stats_data = {}
        h2h_data = {}
        
        for filename, df in all_data.items():
            if 'historical_matches' in filename:
                sport_key = filename.split('_')[2]  # Extract sport key
                matches_data[sport_key] = df
            elif 'team_stats' in filename:
                league = filename.split('_')[2]  # Extract league
                team_stats_data[league] = df
            elif 'h2h_data' in filename:
                league = filename.split('_')[2]  # Extract league
                h2h_data[league] = df
        
        print(f"📊 Loaded data:")
        print(f"   Matches: {len(matches_data)} sports")
        print(f"   Team stats: {len(team_stats_data)} leagues")
        print(f"   H2H data: {len(h2h_data)} leagues")
        
        return {
            'matches': matches_data,
            'team_stats': team_stats_data,
            'h2h': h2h_data
        }
    
    def prepare_training_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Prepare features for training"""
        print("🔧 Preparing training features...")
        
        all_features = []
        
        # Process each sport/league
        for sport_key, matches_df in data['matches'].items():
            # Map sport key to league
            league = self._sport_key_to_league(sport_key)
            
            if league in data['team_stats'] and league in data['h2h']:
                # Engineer features for this league
                features_df = self.feature_engineer.engineer_match_features(
                    matches_df, 
                    data['team_stats'][league], 
                    data['h2h'][league]
                )
                
                # Add league identifier
                features_df['source_league'] = league
                features_df['source_sport'] = sport_key
                
                all_features.append(features_df)
                
                print(f"   ✅ Processed {league}: {len(features_df)} matches")
            else:
                print(f"   ⚠️ Skipping {league}: missing team stats or H2H data")
        
        # Combine all features
        if all_features:
            combined_features = pd.concat(all_features, ignore_index=True)
            print(f"✅ Combined {len(combined_features)} total matches with features")
            return combined_features
        else:
            raise ValueError("No features could be prepared!")
    
    def _sport_key_to_league(self, sport_key: str) -> str:
        """Convert sport key to league name"""
        mapping = {
            'americanfootball_nfl': 'NFL',
            'basketball_nba': 'NBA',
            'baseball_mlb': 'MLB',
            'soccer_epl': 'EPL',
            'soccer_usa_mls': 'MLS'
        }
        return mapping.get(sport_key, sport_key)
    
    async def train_models(self, force_retrain: bool = False) -> Dict[str, float]:
        """Train the ML models"""
        print("🎯 Training ML models...")
        
        # Check if retraining is needed
        if not force_retrain and self.is_trained:
            days_since_training = self._days_since_last_training()
            if days_since_training < self.config['training_frequency_days']:
                print(f"ℹ️ Models trained recently ({days_since_training} days ago), skipping retraining")
                return self.performance_metrics
        
        # Collect and prepare data
        data = await self.collect_and_prepare_data()
        features_df = self.prepare_training_features(data)
        
        # Train models
        scores = self.ml_predictor.train_models(features_df)
        
        # Update state
        self.is_trained = True
        self.last_training = datetime.now().isoformat()
        self.performance_metrics = scores
        
        print("✅ Model training completed!")
        return scores
    
    def _days_since_last_training(self) -> int:
        """Calculate days since last training"""
        if not self.last_training:
            return float('inf')
        
        last_training = datetime.fromisoformat(self.last_training)
        return (datetime.now() - last_training).days
    
    async def predict_match(self, match_data: Dict) -> Dict[str, Any]:
        """Predict outcome for a single match"""
        print(f"🎯 Making prediction for {match_data.get('home_team', 'Unknown')} vs {match_data.get('away_team', 'Unknown')}")
        
        # Check if models are trained
        if not self.is_trained:
            print("⚠️ Models not trained, training now...")
            await self.train_models()
        
        # Prepare features for this match
        try:
            # Load current data for feature engineering
            data = await self.collect_and_prepare_data()
            
            # Get relevant league data
            sport_key = match_data.get('sport_key', '')
            league = self._sport_key_to_league(sport_key)
            
            if league not in data['team_stats'] or league not in data['h2h']:
                raise ValueError(f"Missing data for league: {league}")
            
            # Create single match DataFrame
            match_df = pd.DataFrame([match_data])
            
            # Engineer features
            features_df = self.feature_engineer.engineer_match_features(
                match_df,
                data['team_stats'][league],
                data['h2h'][league]
            )
            
            # Make prediction
            prediction = self.ml_predictor.predict_match(features_df)
            
            # Add metadata
            prediction['match_info'] = {
                'home_team': match_data.get('home_team'),
                'away_team': match_data.get('away_team'),
                'sport': sport_key,
                'league': league,
                'prediction_timestamp': datetime.now().isoformat()
            }
            
            # Check confidence threshold
            if prediction['confidence'] < self.config['prediction_confidence_threshold']:
                prediction['warning'] = f"Low confidence prediction ({prediction['confidence']:.3f})"
            
            print(f"✅ Prediction: {prediction['recommended_prediction']} (confidence: {prediction['confidence']:.3f})")
            return prediction
            
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            return {
                'error': str(e),
                'fallback_prediction': self._get_fallback_prediction(match_data)
            }
    
    def _get_fallback_prediction(self, match_data: Dict) -> Dict[str, Any]:
        """Get fallback prediction when ML fails"""
        # Simple fallback based on odds
        home_odds = match_data.get('home_odds', 2.0)
        away_odds = match_data.get('away_odds', 2.0)
        
        home_prob = 1 / home_odds if home_odds > 0 else 0.5
        away_prob = 1 / away_odds if away_odds > 0 else 0.5
        
        if home_prob > away_prob:
            prediction = 'home_win'
            confidence = home_prob
        else:
            prediction = 'away_win'
            confidence = away_prob
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'method': 'fallback_odds_based'
        }
    
    async def batch_predict_matches(self, matches_data: List[Dict]) -> List[Dict[str, Any]]:
        """Predict outcomes for multiple matches"""
        print(f"🎯 Making batch predictions for {len(matches_data)} matches...")
        
        predictions = []
        
        for i, match_data in enumerate(matches_data):
            print(f"   Processing match {i+1}/{len(matches_data)}...")
            prediction = await self.predict_match(match_data)
            predictions.append(prediction)
            
            # Small delay to avoid overwhelming APIs
            await asyncio.sleep(0.1)
        
        print(f"✅ Completed {len(predictions)} predictions")
        return predictions
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get current model performance metrics"""
        if not self.is_trained:
            return {'status': 'not_trained'}
        
        model_info = self.ml_predictor.get_model_info()
        
        return {
            'status': 'trained',
            'last_training': self.last_training,
            'feature_count': model_info['feature_count'],
            'performance_history': model_info['performance_history'],
            'current_metrics': self.performance_metrics
        }
    
    async def evaluate_model_performance(self, test_data: List[Dict]) -> Dict[str, Any]:
        """Evaluate model performance on test data"""
        print("📈 Evaluating model performance...")
        
        # Prepare test features
        test_features = []
        for match_data in test_data:
            try:
                # Load data for feature engineering
                data = await self.collect_and_prepare_data()
                
                sport_key = match_data.get('sport_key', '')
                league = self._sport_key_to_league(sport_key)
                
                if league in data['team_stats'] and league in data['h2h']:
                    match_df = pd.DataFrame([match_data])
                    features_df = self.feature_engineer.engineer_match_features(
                        match_df,
                        data['team_stats'][league],
                        data['h2h'][league]
                    )
                    test_features.append(features_df)
            except Exception as e:
                print(f"⚠️ Skipping test match due to error: {e}")
        
        if test_features:
            combined_test_features = pd.concat(test_features, ignore_index=True)
            evaluation = self.ml_predictor.evaluate_model(combined_test_features)
            return evaluation
        else:
            return {'error': 'No valid test features could be prepared'}
    
    async def auto_retrain_if_needed(self):
        """Automatically retrain models if needed"""
        if not self.config['enable_auto_retraining']:
            return
        
        days_since_training = self._days_since_last_training()
        if days_since_training >= self.config['training_frequency_days']:
            print(f"🔄 Auto-retraining models (last training: {days_since_training} days ago)")
            await self.train_models(force_retrain=True)
    
    def get_prediction_summary(self, predictions: List[Dict]) -> Dict[str, Any]:
        """Get summary of batch predictions"""
        if not predictions:
            return {'error': 'No predictions provided'}
        
        # Count predictions by outcome
        outcome_counts = {}
        confidence_scores = []
        model_agreements = []
        
        for pred in predictions:
            if 'error' in pred:
                continue
                
            outcome = pred.get('recommended_prediction', 'unknown')
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
            
            confidence_scores.append(pred.get('confidence', 0))
            model_agreements.append(pred.get('model_agreement', 0))
        
        return {
            'total_predictions': len(predictions),
            'outcome_distribution': outcome_counts,
            'average_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'average_model_agreement': np.mean(model_agreements) if model_agreements else 0,
            'high_confidence_predictions': sum(1 for c in confidence_scores if c > 0.7),
            'low_confidence_predictions': sum(1 for c in confidence_scores if c < 0.5)
        }

async def main():
    """Test the ML predictor"""
    print("🚀 Testing ML Predictor...")
    
    # Initialize predictor
    predictor = MLPredictor()
    
    # Train models
    scores = await predictor.train_models()
    print(f"📊 Training scores: {scores}")
    
    # Test single prediction
    test_match = {
        'match_id': 'test_001',
        'sport_key': 'americanfootball_nfl',
        'home_team': 'Kansas City Chiefs',
        'away_team': 'Buffalo Bills',
        'match_date': '2024-01-20T20:00:00Z',
        'home_odds': 1.85,
        'away_odds': 2.10
    }
    
    prediction = await predictor.predict_match(test_match)
    print(f"🎯 Single prediction: {prediction}")
    
    # Test batch prediction
    test_matches = [
        {
            'match_id': 'test_002',
            'sport_key': 'basketball_nba',
            'home_team': 'Boston Celtics',
            'away_team': 'Los Angeles Lakers',
            'match_date': '2024-01-21T02:00:00Z',
            'home_odds': 1.95,
            'away_odds': 1.90
        },
        {
            'match_id': 'test_003',
            'sport_key': 'baseball_mlb',
            'home_team': 'Houston Astros',
            'away_team': 'Los Angeles Dodgers',
            'match_date': '2024-01-22T19:00:00Z',
            'home_odds': 2.20,
            'away_odds': 1.75
        }
    ]
    
    batch_predictions = await predictor.batch_predict_matches(test_matches)
    summary = predictor.get_prediction_summary(batch_predictions)
    print(f"📊 Batch prediction summary: {summary}")

if __name__ == "__main__":
    asyncio.run(main()) 