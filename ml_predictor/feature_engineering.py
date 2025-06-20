#!/usr/bin/env python3
"""
Feature Engineering for Sportstensor ML Predictor
Transforms raw data into ML-ready features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import re
from pathlib import Path

class FeatureEngineer:
    """Engineers features from raw sports data for ML models"""
    
    def __init__(self):
        self.feature_columns = []
        self.scaler = None
        
    def engineer_match_features(self, matches_df: pd.DataFrame, team_stats_df: pd.DataFrame, 
                              h2h_df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive features for match prediction"""
        print("🔧 Engineering match features...")
        
        features_df = matches_df.copy()
        
        # 1. Basic match features
        features_df = self._add_basic_features(features_df)
        
        # 2. Team statistics features
        features_df = self._add_team_stats_features(features_df, team_stats_df)
        
        # 3. Head-to-head features
        features_df = self._add_h2h_features(features_df, h2h_df)
        
        # 4. Odds-based features
        features_df = self._add_odds_features(features_df)
        
        # 5. Temporal features
        features_df = self._add_temporal_features(features_df)
        
        # 6. Form and momentum features
        features_df = self._add_form_features(features_df, team_stats_df)
        
        # 7. Venue features
        features_df = self._add_venue_features(features_df, team_stats_df)
        
        # 8. Derived features
        features_df = self._add_derived_features(features_df)
        
        # Final step: Remove duplicate columns
        features_df = features_df.loc[:,~features_df.columns.duplicated()]

        print(f"✅ Engineered {len(features_df.columns)} features for {len(features_df)} matches")
        return features_df
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic match features"""
        # League encoding
        df['league_nfl'] = (df['sport_key'] == 'americanfootball_nfl').astype(int)
        df['league_nba'] = (df['sport_key'] == 'basketball_nba').astype(int)
        df['league_mlb'] = (df['sport_key'] == 'baseball_mlb').astype(int)
        df['league_epl'] = (df['sport_key'] == 'soccer_epl').astype(int)
        df['league_mls'] = (df['sport_key'] == 'soccer_usa_mls').astype(int)
        
        # Team name features (length, complexity)
        df['home_team_length'] = df['home_team'].str.len()
        df['away_team_length'] = df['away_team'].str.len()
        df['team_name_diff'] = abs(df['home_team_length'] - df['away_team_length'])
        
        return df
    
    def _add_team_stats_features(self, df: pd.DataFrame, team_stats_df: pd.DataFrame) -> pd.DataFrame:
        """Add team statistics features"""
        # Merge home team stats
        df = df.merge(
            team_stats_df, 
            left_on='home_team', 
            right_on='team_name', 
            how='left', 
            suffixes=('', '_home')
        )
        
        # Merge away team stats
        df = df.merge(
            team_stats_df, 
            left_on='away_team', 
            right_on='team_name', 
            how='left', 
            suffixes=('', '_away')
        )
        
        # Calculate relative features
        df['win_rate_home'] = df['wins'] / (df['wins'] + df['losses'] + df['draws'])
        df['win_rate_away'] = df['wins_away'] / (df['wins_away'] + df['losses_away'] + df['draws_away'])
        
        df['win_rate_diff'] = df['win_rate_home'] - df['win_rate_away']
        df['strength_diff'] = df['strength_rating'] - df['strength_rating_away']
        
        # Points per game
        df['ppg_home'] = df['points_for'] / df['games_played']
        df['ppg_away'] = df['points_for_away'] / df['games_played_away']
        df['ppg_diff'] = df['ppg_home'] - df['ppg_away']
        
        # Defense (points against per game)
        df['papg_home'] = df['points_against'] / df['games_played']
        df['papg_away'] = df['points_against_away'] / df['games_played_away']
        df['papg_diff'] = df['papg_home'] - df['papg_away']
        
        return df
    
    def _add_h2h_features(self, df: pd.DataFrame, h2h_df: pd.DataFrame) -> pd.DataFrame:
        """Add head-to-head features"""
        # Create h2h lookup
        h2h_lookup = {}
        for _, row in h2h_df.iterrows():
            key1 = f"{row['team1']}_{row['team2']}"
            key2 = f"{row['team2']}_{row['team1']}"
            h2h_lookup[key1] = row
            h2h_lookup[key2] = row
        
        # Add h2h features
        h2h_features = []
        for _, row in df.iterrows():
            key = f"{row['home_team']}_{row['away_team']}"
            if key in h2h_lookup:
                h2h_data = h2h_lookup[key]
                h2h_features.append({
                    'h2h_total_games': h2h_data['total_games'],
                    'h2h_home_wins': h2h_data['team1_wins'] if h2h_data['team1'] == row['home_team'] else h2h_data['team2_wins'],
                    'h2h_away_wins': h2h_data['team2_wins'] if h2h_data['team1'] == row['home_team'] else h2h_data['team1_wins'],
                    'h2h_draws': h2h_data['draws'],
                    'h2h_home_win_rate': h2h_data['team1_wins'] / h2h_data['total_games'] if h2h_data['team1'] == row['home_team'] else h2h_data['team2_wins'] / h2h_data['total_games'],
                    'h2h_avg_goals_home': h2h_data['avg_goals_team1'] if h2h_data['team1'] == row['home_team'] else h2h_data['avg_goals_team2'],
                    'h2h_avg_goals_away': h2h_data['avg_goals_team2'] if h2h_data['team1'] == row['home_team'] else h2h_data['avg_goals_team1']
                })
            else:
                h2h_features.append({
                    'h2h_total_games': 0,
                    'h2h_home_wins': 0,
                    'h2h_away_wins': 0,
                    'h2h_draws': 0,
                    'h2h_home_win_rate': 0.5,  # Neutral when no h2h data
                    'h2h_avg_goals_home': 2.0,
                    'h2h_avg_goals_away': 2.0
                })
        
        h2h_df_features = pd.DataFrame(h2h_features)
        df = pd.concat([df, h2h_df_features], axis=1)
        
        return df
    
    def _add_odds_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features based on betting odds"""
        # Extract odds from odds_data column
        odds_features = []
        for _, row in df.iterrows():
            odds_data = row.get('odds_data', {})
            if isinstance(odds_data, str):
                try:
                    odds_data = json.loads(odds_data)
                except:
                    odds_data = {}
            
            home_odds = odds_data.get(row['home_team'], 2.0)
            away_odds = odds_data.get(row['away_team'], 2.0)
            draw_odds = odds_data.get('Draw', 3.0)
            
            # Convert odds to probabilities
            home_prob = 1 / home_odds if home_odds > 0 else 0.5
            away_prob = 1 / away_odds if away_odds > 0 else 0.5
            draw_prob = 1 / draw_odds if draw_odds > 0 else 0.1
            
            # Normalize
            total_prob = home_prob + away_prob + draw_prob
            home_prob /= total_prob
            away_prob /= total_prob
            draw_prob /= total_prob
            
            odds_features.append({
                'home_odds': home_odds,
                'away_odds': away_odds,
                'draw_odds': draw_odds,
                'home_prob_odds': home_prob,
                'away_prob_odds': away_prob,
                'draw_prob_odds': draw_prob,
                'odds_favorite': 'home' if home_prob > away_prob else 'away',
                'odds_margin': abs(home_prob - away_prob),
                'odds_confidence': max(home_prob, away_prob, draw_prob)
            })
        
        odds_df = pd.DataFrame(odds_features)
        df = pd.concat([df, odds_df], axis=1)
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        # Convert match_date to datetime
        df['match_date'] = pd.to_datetime(df['match_date'])
        
        # Extract temporal features
        df['match_day_of_week'] = df['match_date'].dt.dayofweek
        df['match_month'] = df['match_date'].dt.month
        df['match_year'] = df['match_date'].dt.year
        df['match_hour'] = df['match_date'].dt.hour
        
        # Season features
        df['is_weekend'] = df['match_day_of_week'].isin([5, 6]).astype(int)
        df['is_weekday'] = df['match_day_of_week'].isin([0, 1, 2, 3, 4]).astype(int)
        
        # Time of day
        df['is_evening'] = df['match_hour'].isin([18, 19, 20, 21, 22, 23]).astype(int)
        df['is_afternoon'] = df['match_hour'].isin([12, 13, 14, 15, 16, 17]).astype(int)
        
        return df
    
    def _add_form_features(self, df: pd.DataFrame, team_stats_df: pd.DataFrame) -> pd.DataFrame:
        """Add recent form and momentum features"""
        # Recent form analysis
        def analyze_form(form_string: str) -> Dict:
            if pd.isna(form_string) or form_string == '':
                return {'wins': 0, 'losses': 0, 'draws': 0, 'form_score': 0}
            
            wins = form_string.count('W')
            losses = form_string.count('L')
            draws = form_string.count('D')
            
            # Weight recent games more heavily
            form_score = 0
            for i, result in enumerate(form_string):
                weight = 1 + (i * 0.1)  # More recent games get higher weight
                if result == 'W':
                    form_score += weight
                elif result == 'L':
                    form_score -= weight
            
            return {
                'wins': wins,
                'losses': losses,
                'draws': draws,
                'form_score': form_score
            }
        
        # Analyze home team form
        home_form_features = []
        for _, row in df.iterrows():
            home_team_stats = team_stats_df[team_stats_df['team_name'] == row['home_team']]
            if not home_team_stats.empty:
                form_data = analyze_form(home_team_stats.iloc[0]['last_5_games'])
                home_form_features.append(form_data)
            else:
                home_form_features.append({'wins': 0, 'losses': 0, 'draws': 0, 'form_score': 0})
        
        home_form_df = pd.DataFrame(home_form_features)
        home_form_df.columns = ['home_form_wins', 'home_form_losses', 'home_form_draws', 'home_form_score']
        
        # Analyze away team form
        away_form_features = []
        for _, row in df.iterrows():
            away_team_stats = team_stats_df[team_stats_df['team_name'] == row['away_team']]
            if not away_team_stats.empty:
                form_data = analyze_form(away_team_stats.iloc[0]['last_5_games'])
                away_form_features.append(form_data)
            else:
                away_form_features.append({'wins': 0, 'losses': 0, 'draws': 0, 'form_score': 0})
        
        away_form_df = pd.DataFrame(away_form_features)
        away_form_df.columns = ['away_form_wins', 'away_form_losses', 'away_form_draws', 'away_form_score']
        
        # Combine form features
        df = pd.concat([df, home_form_df, away_form_df], axis=1)
        
        # Form differential
        df['form_diff'] = df['home_form_score'] - df['away_form_score']
        
        return df
    
    def _add_venue_features(self, df: pd.DataFrame, team_stats_df: pd.DataFrame) -> pd.DataFrame:
        """Add venue-specific features"""
        # Home venue advantage
        df['home_venue_win_rate'] = df['home_wins'] / (df['home_wins'] + df['home_losses'])
        df['away_venue_win_rate'] = df['away_wins'] / (df['away_wins'] + df['away_losses'])
        
        # Venue advantage differential
        df['venue_advantage'] = df['home_venue_win_rate'] - df['away_venue_win_rate']
        
        # Home/away game count
        df['home_games_played'] = df['home_wins'] + df['home_losses']
        df['away_games_played'] = df['away_wins'] + df['away_losses']
        
        return df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived and interaction features"""
        # Team strength interaction
        df['strength_interaction'] = df['strength_rating'] * df['strength_rating_away']
        
        # Form and strength interaction
        df['form_strength_home'] = df['home_form_score'] * df['strength_rating']
        df['form_strength_away'] = df['away_form_score'] * df['strength_rating_away']
        
        # Odds and stats interaction
        df['odds_stats_alignment'] = np.where(
            (df['home_prob_odds'] > df['away_prob_odds']) & (df['win_rate_home'] > df['win_rate_away']),
            1,  # Odds and stats agree home team is better
            np.where(
                (df['away_prob_odds'] > df['home_prob_odds']) & (df['win_rate_away'] > df['win_rate_home']),
                1,  # Odds and stats agree away team is better
                0   # Disagreement
            )
        )
        
        # Momentum features
        df['momentum_home'] = df['home_form_score'] * df['streak']
        df['momentum_away'] = df['away_form_score'] * df['streak_away']
        
        # Composite features
        df['home_advantage_composite'] = (
            df['win_rate_home'] * 0.3 +
            df['home_form_score'] * 0.2 +
            df['home_venue_win_rate'] * 0.2 +
            df['strength_rating'] * 0.3
        )
        
        df['away_advantage_composite'] = (
            df['win_rate_away'] * 0.3 +
            df['away_form_score'] * 0.2 +
            df['away_venue_win_rate'] * 0.2 +
            df['strength_rating_away'] * 0.3
        )
        
        df['advantage_differential'] = df['home_advantage_composite'] - df['away_advantage_composite']
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Get list of engineered feature columns"""
        return [
            # Basic features
            'league_nfl', 'league_nba', 'league_mlb', 'league_epl', 'league_mls',
            'home_team_length', 'away_team_length', 'team_name_diff',
            
            # Team stats
            'win_rate_home', 'win_rate_away', 'win_rate_diff', 'strength_diff',
            'ppg_home', 'ppg_away', 'ppg_diff', 'papg_home', 'papg_away', 'papg_diff',
            
            # H2H features
            'h2h_total_games', 'h2h_home_wins', 'h2h_away_wins', 'h2h_draws',
            'h2h_home_win_rate', 'h2h_avg_goals_home', 'h2h_avg_goals_away',
            
            # Odds features
            'home_odds', 'away_odds', 'draw_odds', 'home_prob_odds', 'away_prob_odds',
            'draw_prob_odds', 'odds_margin', 'odds_confidence',
            
            # Temporal features
            'match_day_of_week', 'match_month', 'match_year', 'match_hour',
            'is_weekend', 'is_weekday', 'is_evening', 'is_afternoon',
            
            # Form features
            'home_form_wins', 'home_form_losses', 'home_form_draws', 'home_form_score',
            'away_form_wins', 'away_form_losses', 'away_form_draws', 'away_form_score',
            'form_diff',
            
            # Venue features
            'home_venue_win_rate', 'away_venue_win_rate', 'venue_advantage',
            'home_games_played', 'away_games_played',
            
            # Derived features
            'strength_interaction', 'form_strength_home', 'form_strength_away',
            'odds_stats_alignment', 'momentum_home', 'momentum_away',
            'home_advantage_composite', 'away_advantage_composite', 'advantage_differential'
        ]
    
    def prepare_features_for_prediction(self, match_data: Dict, team_stats_df: pd.DataFrame, 
                                      h2h_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for a single match prediction"""
        # Convert single match to DataFrame
        match_df = pd.DataFrame([match_data])
        
        # Apply feature engineering
        features_df = self.engineer_match_features(match_df, team_stats_df, h2h_df)
        
        # Get only the feature columns
        feature_columns = self.get_feature_columns()
        available_features = [col for col in feature_columns if col in features_df.columns]
        
        return features_df[available_features]

def main():
    """Test feature engineering"""
    print("🔧 Testing feature engineering...")
    
    # Create sample data
    sample_matches = pd.DataFrame({
        'match_id': ['1', '2'],
        'sport_key': ['americanfootball_nfl', 'basketball_nba'],
        'home_team': ['Kansas City Chiefs', 'Boston Celtics'],
        'away_team': ['Buffalo Bills', 'Los Angeles Lakers'],
        'match_date': ['2024-01-20T20:00:00Z', '2024-01-21T02:00:00Z'],
        'odds_data': ['{"Kansas City Chiefs": 1.85, "Buffalo Bills": 2.10}', 
                     '{"Boston Celtics": 1.95, "Los Angeles Lakers": 1.90}']
    })
    
    sample_team_stats = pd.DataFrame({
        'team_name': ['Kansas City Chiefs', 'Buffalo Bills', 'Boston Celtics', 'Los Angeles Lakers'],
        'league': ['NFL', 'NFL', 'NBA', 'NBA'],
        'wins': [11, 10, 25, 20],
        'losses': [6, 7, 5, 10],
        'draws': [0, 0, 0, 0],
        'points_for': [371, 350, 2800, 2500],
        'points_against': [294, 320, 2600, 2400],
        'home_wins': [6, 5, 15, 12],
        'home_losses': [2, 3, 2, 5],
        'away_wins': [5, 5, 10, 8],
        'away_losses': [4, 4, 3, 5],
        'last_5_games': ['WWLWW', 'LWWLW', 'WWWWW', 'WLWWL'],
        'strength_rating': [0.75, 0.70, 0.80, 0.65]
    })
    
    sample_h2h = pd.DataFrame({
        'team1': ['Kansas City Chiefs'],
        'team2': ['Buffalo Bills'],
        'league': ['NFL'],
        'total_games': [8],
        'team1_wins': [5],
        'team2_wins': [3],
        'draws': [0],
        'avg_goals_team1': [2.5],
        'avg_goals_team2': [2.2]
    })
    
    # Engineer features
    engineer = FeatureEngineer()
    features_df = engineer.engineer_match_features(sample_matches, sample_team_stats, sample_h2h)
    
    print(f"✅ Engineered {len(features_df.columns)} features")
    print(f"📊 Feature columns: {list(features_df.columns)}")

if __name__ == "__main__":
    main() 