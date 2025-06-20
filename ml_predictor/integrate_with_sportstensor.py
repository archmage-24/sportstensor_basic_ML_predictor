#!/usr/bin/env python3
"""
Integration script to connect ML Predictor with Sportstensor
Replaces the basic prediction logic with advanced ML predictions
"""

import os
import sys
import asyncio
import json
from typing import Dict, Any, Optional
from pathlib import Path

# Add the parent directory to the path to import sportstensor modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import sportstensor modules
from common.data import MatchPrediction
from st.sport_prediction_model import make_match_prediction

# Import our ML predictor
from ml_predictor import MLPredictor

class MLEnhancedPredictor:
    """Enhanced predictor that uses ML models for better predictions"""
    
    def __init__(self):
        self.ml_predictor = MLPredictor()
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize the ML predictor"""
        print("🚀 Initializing ML Enhanced Predictor...")
        
        try:
            # Train models if needed
            await self.ml_predictor.train_models()
            self.is_initialized = True
            print("✅ ML Enhanced Predictor initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing ML predictor: {e}")
            print("⚠️ Falling back to basic prediction system")
            self.is_initialized = False
    
    async def predict_match_ml(self, match_prediction: MatchPrediction) -> MatchPrediction:
        """Make ML-enhanced prediction for a match"""
        if not self.is_initialized:
            print("⚠️ ML predictor not initialized, using fallback")
            return await self._fallback_prediction(match_prediction)
        
        try:
            # Convert MatchPrediction to dictionary format
            match_data = self._convert_to_match_data(match_prediction)
            
            # Get ML prediction
            ml_prediction = await self.ml_predictor.predict_match(match_data)
            
            # Update MatchPrediction with ML results
            updated_prediction = self._update_prediction_with_ml(match_prediction, ml_prediction)
            
            print(f"🤖 ML Prediction: {ml_prediction.get('recommended_prediction', 'unknown')} "
                  f"(confidence: {ml_prediction.get('confidence', 0):.3f})")
            
            return updated_prediction
            
        except Exception as e:
            print(f"❌ ML prediction failed: {e}")
            return await self._fallback_prediction(match_prediction)
    
    def _convert_to_match_data(self, match_prediction: MatchPrediction) -> Dict[str, Any]:
        """Convert MatchPrediction to dictionary format for ML predictor"""
        return {
            'match_id': match_prediction.matchId,
            'sport_key': self._get_sport_key(match_prediction.sport),
            'home_team': match_prediction.homeTeamName,
            'away_team': match_prediction.awayTeamName,
            'match_date': match_prediction.matchDate,
            'home_odds': self._extract_odds(match_prediction, 'home'),
            'away_odds': self._extract_odds(match_prediction, 'away'),
            'draw_odds': self._extract_odds(match_prediction, 'draw'),
            'sport': match_prediction.sport,
            'league': match_prediction.league
        }
    
    def _get_sport_key(self, sport: str) -> str:
        """Convert sport name to API sport key"""
        sport_mapping = {
            'Football': 'americanfootball_nfl',
            'Basketball': 'basketball_nba',
            'Baseball': 'baseball_mlb',
            'Soccer': 'soccer_epl',
            'Soccer MLS': 'soccer_usa_mls'
        }
        return sport_mapping.get(sport, sport.lower())
    
    def _extract_odds(self, match_prediction: MatchPrediction, team: str) -> float:
        """Extract odds from MatchPrediction"""
        # This is a simplified extraction - in real implementation,
        # you'd parse the actual odds data from the match prediction
        if team == 'home':
            return 2.0  # Default odds
        elif team == 'away':
            return 2.0  # Default odds
        elif team == 'draw':
            return 3.0  # Default draw odds
        return 2.0
    
    def _update_prediction_with_ml(self, match_prediction: MatchPrediction, 
                                 ml_prediction: Dict[str, Any]) -> MatchPrediction:
        """Update MatchPrediction with ML results"""
        # Get the recommended prediction
        recommended = ml_prediction.get('recommended_prediction', 'home_win')
        confidence = ml_prediction.get('confidence', 0.5)
        
        # Convert ML prediction to Sportstensor format
        if recommended == 'home_win':
            match_prediction.probabilityChoice = 'home'
            match_prediction.probability = confidence
        elif recommended == 'away_win':
            match_prediction.probabilityChoice = 'away'
            match_prediction.probability = confidence
        else:  # draw
            match_prediction.probabilityChoice = 'draw'
            match_prediction.probability = confidence
        
        # Add ML metadata
        if not hasattr(match_prediction, 'ml_metadata'):
            match_prediction.ml_metadata = {}
        
        match_prediction.ml_metadata.update({
            'ml_confidence': confidence,
            'model_agreement': ml_prediction.get('model_agreement', 0),
            'prediction_method': 'ml_enhanced',
            'all_predictions': ml_prediction.get('predictions', {}),
            'all_probabilities': ml_prediction.get('probabilities', {})
        })
        
        return match_prediction
    
    async def _fallback_prediction(self, match_prediction: MatchPrediction) -> MatchPrediction:
        """Fallback to basic prediction system"""
        print("🔄 Using fallback prediction system")
        return await make_match_prediction(match_prediction)

# Global instance
ml_enhanced_predictor = MLEnhancedPredictor()

async def initialize_ml_predictor():
    """Initialize the ML predictor (called at startup)"""
    await ml_enhanced_predictor.initialize()

async def make_ml_enhanced_prediction(match_prediction: MatchPrediction) -> MatchPrediction:
    """Main function to make ML-enhanced predictions"""
    return await ml_enhanced_predictor.predict_match_ml(match_prediction)

# Integration with existing sport_prediction_model.py
async def enhanced_make_match_prediction(match_prediction: MatchPrediction) -> MatchPrediction:
    """
    Enhanced version of make_match_prediction that uses ML models
    This function can replace the existing make_match_prediction function
    """
    try:
        # Try ML prediction first
        if ml_enhanced_predictor.is_initialized:
            return await make_ml_enhanced_prediction(match_prediction)
        else:
            # Fallback to original
            return await make_match_prediction(match_prediction)
    except Exception as e:
        print(f"❌ Enhanced prediction failed: {e}")
        # Fallback to original
        return await make_match_prediction(match_prediction)

# Performance monitoring
class MLPerformanceMonitor:
    """Monitor ML model performance"""
    
    def __init__(self):
        self.predictions_made = 0
        self.ml_predictions = 0
        self.fallback_predictions = 0
        self.errors = 0
        
    def record_prediction(self, used_ml: bool, success: bool):
        """Record a prediction attempt"""
        self.predictions_made += 1
        if used_ml:
            self.ml_predictions += 1
        else:
            self.fallback_predictions += 1
        
        if not success:
            self.errors += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'total_predictions': self.predictions_made,
            'ml_predictions': self.ml_predictions,
            'fallback_predictions': self.fallback_predictions,
            'errors': self.errors,
            'ml_usage_rate': self.ml_predictions / max(self.predictions_made, 1),
            'error_rate': self.errors / max(self.predictions_made, 1)
        }

# Global performance monitor
performance_monitor = MLPerformanceMonitor()

# Enhanced prediction function with monitoring
async def monitored_enhanced_prediction(match_prediction: MatchPrediction) -> MatchPrediction:
    """Enhanced prediction with performance monitoring"""
    try:
        if ml_enhanced_predictor.is_initialized:
            result = await make_ml_enhanced_prediction(match_prediction)
            performance_monitor.record_prediction(used_ml=True, success=True)
            return result
        else:
            result = await make_match_prediction(match_prediction)
            performance_monitor.record_prediction(used_ml=False, success=True)
            return result
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        performance_monitor.record_prediction(used_ml=False, success=False)
        # Final fallback
        return await make_match_prediction(match_prediction)

def get_ml_performance_stats() -> Dict[str, Any]:
    """Get ML performance statistics"""
    stats = performance_monitor.get_stats()
    
    # Add ML model performance if available
    if ml_enhanced_predictor.is_initialized:
        ml_stats = ml_enhanced_predictor.ml_predictor.get_model_performance()
        stats['ml_model_performance'] = ml_stats
    
    return stats

# Configuration management
class MLConfig:
    """Configuration for ML integration"""
    
    def __init__(self):
        self.enable_ml = True
        self.auto_retrain = True
        self.confidence_threshold = 0.6
        self.fallback_on_error = True
        self.monitor_performance = True
    
    def load_from_env(self):
        """Load configuration from environment variables"""
        self.enable_ml = os.getenv('ML_ENABLE', 'true').lower() == 'true'
        self.auto_retrain = os.getenv('ML_AUTO_RETRAIN', 'true').lower() == 'true'
        self.confidence_threshold = float(os.getenv('ML_CONFIDENCE_THRESHOLD', '0.6'))
        self.fallback_on_error = os.getenv('ML_FALLBACK_ON_ERROR', 'true').lower() == 'true'
        self.monitor_performance = os.getenv('ML_MONITOR_PERFORMANCE', 'true').lower() == 'true'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'enable_ml': self.enable_ml,
            'auto_retrain': self.auto_retrain,
            'confidence_threshold': self.confidence_threshold,
            'fallback_on_error': self.fallback_on_error,
            'monitor_performance': self.monitor_performance
        }

# Global configuration
ml_config = MLConfig()

async def main():
    """Test the ML integration"""
    print("🧪 Testing ML Integration...")
    
    # Load configuration
    ml_config.load_from_env()
    print(f"📋 ML Config: {ml_config.to_dict()}")
    
    # Initialize ML predictor
    await initialize_ml_predictor()
    
    # Create test match prediction
    test_match = MatchPrediction(
        matchId="test_ml_001",
        homeTeamName="Kansas City Chiefs",
        awayTeamName="Buffalo Bills",
        matchDate="2024-01-20T20:00:00Z",
        sport="Football",
        league="NFL",
        homeTeamScore=0,
        awayTeamScore=0,
        probabilityChoice="home",
        probability=0.5
    )
    
    # Test enhanced prediction
    enhanced_result = await monitored_enhanced_prediction(test_match)
    print(f"🎯 Enhanced Prediction: {enhanced_result.probabilityChoice} ({enhanced_result.probability:.3f})")
    
    # Get performance stats
    stats = get_ml_performance_stats()
    print(f"📊 Performance Stats: {stats}")

if __name__ == "__main__":
    asyncio.run(main()) 