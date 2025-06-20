#!/usr/bin/env python3
"""
Data Collector for Sportstensor ML Predictor
Collects historical match data, team statistics, and other relevant features
"""

import os
import json
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataCollector:
    """Collects and manages historical sports data for ML training"""
    
    def __init__(self, data_dir: str = "ml_predictor/data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # API configuration
        self.odds_api_key = os.getenv("ODDS_API_KEY")
        self.odds_api_url = "https://api.the-odds-api.com/v4/sports/"
        
        # Data storage
        self.matches_df = None
        self.team_stats_df = None
        self.player_stats_df = None
        
    async def collect_historical_matches(self, sport_key: str, days_back: int = 365) -> pd.DataFrame:
        """Collect historical match data from the odds API"""
        print(f"🔍 Collecting historical matches for {sport_key} (last {days_back} days)")
        
        # Check if API key is available
        if not self.odds_api_key:
            print(f"   ⚠️ ODDS_API_KEY not set, using sample data for {sport_key}")
            return self._create_sample_matches(sport_key, days_back)
        
        matches_data = []
        current_date = datetime.now()
        
        # Collect data in chunks to avoid API limits
        chunk_size = 30  # days per request
        
        for i in range(0, days_back, chunk_size):
            end_date = current_date - timedelta(days=i)
            start_date = end_date - timedelta(days=min(chunk_size, days_back - i))
            
            url = f"{self.odds_api_url}{sport_key}/odds/"
            params = {
                "apiKey": self.odds_api_key,
                "regions": "us,eu",
                "bookmakers": "pinnacle",
                "markets": "h2h",
                "dateFormat": "iso",
                "oddsFormat": "decimal"
            }
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            for match in data:
                                match_info = {
                                    'match_id': match.get('id'),
                                    'sport_key': sport_key,
                                    'home_team': match.get('home_team'),
                                    'away_team': match.get('away_team'),
                                    'match_date': match.get('commence_time'),
                                    'last_update': match.get('last_update'),
                                    'bookmakers': match.get('bookmakers', []),
                                    'odds_data': self._extract_odds(match.get('bookmakers', [])),
                                    'raw_data': json.dumps(match)
                                }
                                matches_data.append(match_info)
                            
                            print(f"   ✅ Collected {len(data)} matches for {start_date.date()} to {end_date.date()}")
                            
                        else:
                            print(f"   ❌ API Error: {response.status}")
                            
            except Exception as e:
                print(f"   ❌ Error collecting data: {e}")
                
            # Rate limiting
            await asyncio.sleep(1)
        
        # Convert to DataFrame
        df = pd.DataFrame(matches_data)
        
        # Save to file
        filename = f"historical_matches_{sport_key}_{datetime.now().strftime('%Y%m%d')}.csv"
        filepath = self.data_dir / filename
        df.to_csv(filepath, index=False)
        
        print(f"📊 Saved {len(df)} matches to {filepath}")
        return df
    
    def _create_sample_matches(self, sport_key: str, days_back: int = 365) -> pd.DataFrame:
        """Create sample match data when API is not available"""
        print(f"   📝 Creating sample matches for {sport_key}")
        
        # Get teams for this sport
        teams = self._get_league_teams(self._sport_key_to_league(sport_key))
        
        # Create sample matches
        matches_data = []
        for i in range(10):  # Create 10 sample matches
            home_team = teams[i % len(teams)]
            away_team = teams[(i + 1) % len(teams)]
            
            match_date = datetime.now() - timedelta(days=i*7)  # One match per week
            
            match_info = {
                'match_id': f'sample_{sport_key}_{i}',
                'sport_key': sport_key,
                'home_team': home_team,
                'away_team': away_team,
                'match_date': match_date.isoformat(),
                'last_update': datetime.now().isoformat(),
                'odds_data': json.dumps({home_team: 2.0, away_team: 2.0}),
                'raw_data': json.dumps({'sample': True})
            }
            matches_data.append(match_info)
        
        df = pd.DataFrame(matches_data)
        
        # Save to file
        filename = f"historical_matches_{sport_key}_{datetime.now().strftime('%Y%m%d')}.csv"
        filepath = self.data_dir / filename
        df.to_csv(filepath, index=False)
        
        print(f"📊 Saved {len(df)} sample matches to {filepath}")
        return df
    
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
    
    def _extract_odds(self, bookmakers: List[Dict]) -> Dict:
        """Extract odds data from bookmakers"""
        odds_data = {}
        
        for bookmaker in bookmakers:
            if bookmaker.get('key') == 'pinnacle':
                for market in bookmaker.get('markets', []):
                    if market.get('key') == 'h2h':
                        for outcome in market.get('outcomes', []):
                            odds_data[outcome.get('name')] = outcome.get('price')
                        break
                break
        
        return odds_data
    
    def collect_team_statistics(self, league: str) -> pd.DataFrame:
        """Collect team statistics (placeholder for now)"""
        print(f"📈 Collecting team statistics for {league}")
        
        # This would integrate with league-specific APIs
        # For now, create sample data structure
        teams = self._get_league_teams(league)
        
        team_stats = []
        for team in teams:
            stats = {
                'team_name': team,
                'league': league,
                'season': '2024',
                'games_played': np.random.randint(10, 50),
                'wins': np.random.randint(5, 30),
                'losses': np.random.randint(5, 30),
                'draws': np.random.randint(0, 10),
                'points_for': np.random.randint(100, 500),
                'points_against': np.random.randint(100, 500),
                'home_wins': np.random.randint(3, 15),
                'home_losses': np.random.randint(2, 10),
                'away_wins': np.random.randint(2, 15),
                'away_losses': np.random.randint(3, 15),
                'last_5_games': self._generate_recent_form(),
                'last_10_games': self._generate_recent_form(10),
                'streak': np.random.randint(-5, 6),
                'strength_rating': np.random.uniform(0.3, 0.8)
            }
            team_stats.append(stats)
        
        df = pd.DataFrame(team_stats)
        
        # Save to file
        filename = f"team_stats_{league}_{datetime.now().strftime('%Y%m%d')}.csv"
        filepath = self.data_dir / filename
        df.to_csv(filepath, index=False)
        
        print(f"📊 Saved team statistics for {len(df)} teams to {filepath}")
        return df
    
    def _get_league_teams(self, league: str) -> List[str]:
        """Get teams for a specific league"""
        league_teams = {
            'NFL': [
                'Kansas City Chiefs', 'Buffalo Bills', 'Philadelphia Eagles', 'San Francisco 49ers',
                'Dallas Cowboys', 'Green Bay Packers', 'Baltimore Ravens', 'Miami Dolphins',
                'New England Patriots', 'New York Jets', 'Las Vegas Raiders', 'Denver Broncos'
            ],
            'NBA': [
                'Boston Celtics', 'Denver Nuggets', 'Milwaukee Bucks', 'Phoenix Suns',
                'Golden State Warriors', 'Miami Heat', 'Philadelphia 76ers', 'Los Angeles Lakers',
                'Dallas Mavericks', 'Memphis Grizzlies', 'Sacramento Kings', 'Cleveland Cavaliers'
            ],
            'MLB': [
                'Houston Astros', 'Los Angeles Dodgers', 'Atlanta Braves', 'New York Yankees',
                'Toronto Blue Jays', 'Philadelphia Phillies', 'San Diego Padres', 'New York Mets',
                'St. Louis Cardinals', 'Milwaukee Brewers', 'Chicago Cubs', 'San Francisco Giants'
            ],
            'EPL': [
                'Manchester City', 'Liverpool', 'Arsenal', 'Chelsea', 'Tottenham Hotspur',
                'Manchester United', 'Newcastle United', 'Brighton', 'West Ham United',
                'Aston Villa', 'Crystal Palace', 'Brentford'
            ],
            'MLS': [
                'LAFC', 'Philadelphia Union', 'New York City FC', 'Seattle Sounders',
                'Atlanta United', 'LA Galaxy', 'Portland Timbers', 'Inter Miami',
                'FC Cincinnati', 'Columbus Crew', 'Orlando City', 'Minnesota United'
            ]
        }
        return league_teams.get(league, [])
    
    def _generate_recent_form(self, games: int = 5) -> str:
        """Generate recent form string (W=Win, L=Loss, D=Draw)"""
        results = []
        for _ in range(games):
            result = np.random.choice(['W', 'L', 'D'], p=[0.4, 0.4, 0.2])
            results.append(result)
        return ''.join(results)
    
    def collect_head_to_head_data(self, league: str) -> pd.DataFrame:
        """Collect head-to-head statistics between teams"""
        print(f"🤝 Collecting head-to-head data for {league}")
        
        teams = self._get_league_teams(league)
        h2h_data = []
        
        for i, team1 in enumerate(teams):
            for team2 in teams[i+1:]:
                h2h_record = {
                    'team1': team1,
                    'team2': team2,
                    'league': league,
                    'total_games': np.random.randint(5, 30),
                    'team1_wins': np.random.randint(2, 15),
                    'team2_wins': np.random.randint(2, 15),
                    'draws': np.random.randint(0, 5),
                    'last_meeting': (datetime.now() - timedelta(days=np.random.randint(30, 365))).strftime('%Y-%m-%d'),
                    'last_result': np.random.choice(['W', 'L', 'D']),
                    'avg_goals_team1': np.random.uniform(1.5, 3.5),
                    'avg_goals_team2': np.random.uniform(1.5, 3.5)
                }
                h2h_data.append(h2h_record)
        
        df = pd.DataFrame(h2h_data)
        
        # Save to file
        filename = f"h2h_data_{league}_{datetime.now().strftime('%Y%m%d')}.csv"
        filepath = self.data_dir / filename
        df.to_csv(filepath, index=False)
        
        print(f"📊 Saved head-to-head data for {len(df)} team pairs to {filepath}")
        return df
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all collected data"""
        data = {}
        
        for file in self.data_dir.glob("*.csv"):
            df = pd.read_csv(file)
            data[file.stem] = df
            print(f"📂 Loaded {file.name}: {len(df)} records")
        
        return data
    
    async def collect_all_data(self):
        """Collect all data for all supported leagues"""
        print("🚀 Starting comprehensive data collection...")
        
        # Sport keys mapping
        sport_keys = {
            'NFL': 'americanfootball_nfl',
            'NBA': 'basketball_nba', 
            'MLB': 'baseball_mlb',
            'EPL': 'soccer_epl',
            'MLS': 'soccer_usa_mls'
        }
        
        # Collect historical matches
        for league, sport_key in sport_keys.items():
            try:
                await self.collect_historical_matches(sport_key, days_back=180)  # 6 months
            except Exception as e:
                print(f"❌ Error collecting matches for {league}: {e}")
        
        # Collect team statistics
        for league in sport_keys.keys():
            try:
                self.collect_team_statistics(league)
                self.collect_head_to_head_data(league)
            except Exception as e:
                print(f"❌ Error collecting stats for {league}: {e}")
        
        print("✅ Data collection completed!")

async def main():
    """Test the data collector"""
    collector = DataCollector()
    await collector.collect_all_data()

if __name__ == "__main__":
    asyncio.run(main()) 