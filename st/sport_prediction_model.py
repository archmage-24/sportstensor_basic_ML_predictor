from common.data import MatchPrediction, Sport, League, get_league_from_string, ProbabilityChoice
import bittensor as bt
from typing import Dict
import os

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)

# Try to import the ML Predictor components
try:
    from ml_predictor.ml_predictor import MLPredictor
    
    # Initialize the MLPredictor. This will be a global instance.
    # It handles loading models, feature engineering, and making predictions.
    # This is done once when the script is loaded to avoid reloading models on every request.
    logging.info("🧠 Initializing ML Predictor...")
    ML_PREDICTOR = MLPredictor()
    logging.info("✅ ML Predictor initialized successfully.")

except ImportError:
    logging.error("❌ Could not import ML Predictor components. Running in fallback mode.")
    ML_PREDICTOR = None
except Exception as e:
    logging.error(f"❌ Error initializing ML Predictor: {e}")
    ML_PREDICTOR = None


async def make_match_prediction(prediction: MatchPrediction) -> MatchPrediction:
    """
    Uses the integrated ML Predictor to make a prediction for a given match.
    If the ML Predictor is not available, it falls back to a default random prediction.
    """
    if ML_PREDICTOR:
        bt.logging.info("🧠 Using ML Predictor for match prediction.")
        try:
            # 1. Convert the MatchPrediction synapse to a dictionary for the ML Predictor
            sport_key_str = str(prediction.sport.name).lower()
            league_str = str(prediction.league).lower()
            
            match_data = {
                'match_id': prediction.matchId,
                'sport_key': f"{sport_key_str}_{league_str}",
                'home_team': prediction.homeTeamName,
                'away_team': prediction.awayTeamName,
                'match_date': prediction.matchDate,
                'home_odds': prediction.homeTeamOdds,
                'away_odds': prediction.awayTeamOdds,
                'draw_odds': prediction.drawOdds,
            }

            # 2. Call the ML predictor's predict_match method
            ml_prediction_result = await ML_PREDICTOR.predict_match(match_data)
            
            if 'error' in ml_prediction_result:
                bt.logging.error(f"ML Prediction failed: {ml_prediction_result['error']}. Falling back to default.")
                return _set_default_prediction(prediction)

            # 3. Map the ML prediction result back to the synapse object
            recommended_prediction = ml_prediction_result.get('recommended_prediction')
            confidence = ml_prediction_result.get('confidence', 0.5)

            if recommended_prediction == 'home_win':
                prediction.probabilityChoice = ProbabilityChoice.HOMETEAM
            elif recommended_prediction == 'away_win':
                prediction.probabilityChoice = ProbabilityChoice.AWAYTEAM
            elif recommended_prediction == 'draw':
                prediction.probabilityChoice = ProbabilityChoice.DRAW
            else:
                bt.logging.warning(f"Unknown prediction '{recommended_prediction}'. Falling back.")
                return _set_default_prediction(prediction)
            
            prediction.probability = float(confidence)
            bt.logging.success(f"ML Prediction successful: {prediction.probabilityChoice.value} with {prediction.probability:.2f} confidence.")

        except Exception as e:
            bt.logging.error(f"An exception occurred during ML prediction: {e}. Falling back to default.")
            return _set_default_prediction(prediction)
    else:
        # Fallback to default if ML_PREDICTOR is not available
        bt.logging.warning("ML Predictor not available. Using default prediction.")
        return _set_default_prediction(prediction)

    return prediction

def _set_default_prediction(prediction: MatchPrediction) -> MatchPrediction:
    """Sets a random default prediction as a fallback."""
    import random
    can_tie = prediction.sport in [Sport.SOCCER]
    
    if can_tie:
        probs = [random.uniform(0.1, 0.8) for _ in range(3)]
        total = sum(probs)
        probs = [p / total for p in probs]
        max_prob = max(probs)
        if probs[0] == max_prob:
            prediction.probabilityChoice = ProbabilityChoice.HOMETEAM
        elif probs[1] == max_prob:
            prediction.probabilityChoice = ProbabilityChoice.AWAYTEAM
        else:
            prediction.probabilityChoice = ProbabilityChoice.DRAW
        prediction.probability = max_prob
    else:
        prob_a = random.uniform(0.05, 0.95)
        prob_b = 1 - prob_a
        if prob_a > prob_b:
            prediction.probabilityChoice = ProbabilityChoice.HOMETEAM
        else:
            prediction.probabilityChoice = ProbabilityChoice.AWAYTEAM
        prediction.probability = max(prob_a, prob_b)
        
    bt.logging.info(f"Set default prediction: {prediction.probabilityChoice.value} with {prediction.probability:.2f} probability.")
    return prediction
