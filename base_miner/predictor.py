import os
import pandas as pd
import numpy as np
from joblib import load
from datetime import datetime
import pytz

class FootballPredictor:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load the saved models and transformers
        self.model = load(os.path.join(base_dir, 'model/random_forest_model.joblib'))
        self.imputer = load(os.path.join(base_dir, 'model/imputer.joblib'))
        self.scaler = load(os.path.join(base_dir, 'model/feature_scaler.joblib'))
        self.le = load(os.path.join(base_dir, 'model/label_encoder.joblib'))

        # Load the dataset
        self.df = pd.read_csv(os.path.join(base_dir, 'model/football-features.csv'))
        self.df['Date'] = pd.to_datetime(self.df['Date'], utc=True)
        self.df['Season Start'] = pd.to_datetime(self.df['Season Start'], utc=True)
        self.df['Season End'] = pd.to_datetime(self.df['Season End'], utc=True)

    def get_competition(self, home_team, away_team, match_date):
        recent_match = self.df[((self.df['Home Team'] == home_team) | (self.df['Away Team'] == home_team) |
                           (self.df['Home Team'] == away_team) | (self.df['Away Team'] == away_team)) &
                          (self.df['Date'] < match_date)].sort_values('Date', ascending=False).iloc[0]
        return recent_match['Competition']

    def get_league_positions(self, date, season_start, season_end, competition):
        season_matches = self.df[(self.df['Date'] >= season_start) & 
                            (self.df['Date'] <= date) & 
                            (self.df['Date'] <= season_end) & 
                            (self.df['Competition'] == competition)]
        
        team_points = {}
        for _, match in season_matches.iterrows():
            home_team, away_team = match['Home Team'], match['Away Team']
            if match['Winner'] == 'HOME_TEAM':
                team_points[home_team] = team_points.get(home_team, 0) + 3
            elif match['Winner'] == 'AWAY_TEAM':
                team_points[away_team] = team_points.get(away_team, 0) + 3
            else:
                team_points[home_team] = team_points.get(home_team, 0) + 1
                team_points[away_team] = team_points.get(away_team, 0) + 1
        
        sorted_teams = sorted(team_points.items(), key=lambda x: x[1], reverse=True)
        positions = {team: pos+1 for pos, (team, _) in enumerate(sorted_teams)}
        
        default_position = len(positions) + 1 if positions else 1
        
        return lambda team: positions.get(team, default_position)

    def get_team_stats(self, team, date, competition):
        if date.tzinfo is None:
            date = date.replace(tzinfo=pytz.UTC)
        
        team_matches = self.df[(self.df['Home Team'] == team) | (self.df['Away Team'] == team)]
        team_matches = team_matches[team_matches['Date'] < date].sort_values('Date', ascending=False).head(5)
        
        last5_wins = sum(
            ((team_matches['Home Team'] == team) & (team_matches['Winner'] == 'HOME_TEAM')) |
            ((team_matches['Away Team'] == team) & (team_matches['Winner'] == 'AWAY_TEAM'))
        )
        
        h2h_matches = self.df[((self.df['Home Team'] == team) & (self.df['Away Team'] == team)) | ((self.df['Away Team'] == team) & (self.df['Home Team'] == team))]
        h2h_matches = h2h_matches[h2h_matches['Date'] < date].sort_values('Date', ascending=False).head(5)
        
        h2h_wins = sum(
            ((h2h_matches['Home Team'] == team) & (h2h_matches['Winner'] == 'HOME_TEAM')) |
            ((h2h_matches['Away Team'] == team) & (h2h_matches['Winner'] == 'AWAY_TEAM'))
        )
        h2h_win_ratio = h2h_wins / len(h2h_matches) if len(h2h_matches) > 0 else 0
        h2h_avg_goals = (h2h_matches['Full Time Home'] + h2h_matches['Full Time Away']).mean() if len(h2h_matches) > 0 else 0
        
        days_since_last_match = (date - team_matches['Date'].iloc[0]).days if not team_matches.empty else 30
        
        form = sum(team_matches['Winner'].map({'HOME_TEAM': 3, 'AWAY_TEAM': 0, 'DRAW': 1}))
        
        streak = 0
        for _, match in team_matches.iterrows():
            if (match['Home Team'] == team and match['Winner'] == 'HOME_TEAM') or (match['Away Team'] == team and match['Winner'] == 'AWAY_TEAM'):
                streak += 1
            elif (match['Home Team'] == team and match['Winner'] == 'AWAY_TEAM') or (match['Away Team'] == team and match['Winner'] == 'HOME_TEAM'):
                streak -= 1
            else:
                break
        
        performance = last5_wins / len(team_matches) if len(team_matches) > 0 else 0
        
        return {
            'last5_wins': last5_wins,
            'h2h_win_ratio': h2h_win_ratio,
            'h2h_avg_goals': h2h_avg_goals,
            'days_since_last_match': days_since_last_match,
            'form': form,
            'streak': streak,
            'performance': performance
        }

    def predict_winner(self, home_team, away_team, match_date):
        # Convert match_date to datetime if it's a string
        if isinstance(match_date, str):
            try:
                match_date = datetime.strptime(match_date, "%Y-%m-%d %H:%M:%S").replace(tzinfo=pytz.UTC)
            except ValueError:
                try:
                    match_date = datetime.strptime(match_date, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
                except ValueError:
                    raise ValueError("Invalid date format. Use 'YYYY-MM-DD HH:MM:SS' or 'YYYY-MM-DD'")

        # Ensure match_date is timezone-aware
        if match_date.tzinfo is None:
            match_date = match_date.replace(tzinfo=pytz.UTC)

        competition = self.get_competition(home_team, away_team, match_date)
        
        home_stats = self.get_team_stats(home_team, match_date, competition)
        away_stats = self.get_team_stats(away_team, match_date, competition)
        
        season_info = self.df[(self.df['Competition'] == competition) & (self.df['Date'] <= match_date)].iloc[-1]
        season_start = season_info['Season Start']
        season_end = season_info['Season End']
        
        get_positions = self.get_league_positions(match_date, season_start, season_end, competition)
        home_league_pos = get_positions(home_team)
        away_league_pos = get_positions(away_team)
        
        total_days = (season_end - season_start).total_seconds() / (24 * 3600)
        days_passed = (match_date - season_start).total_seconds() / (24 * 3600)
        season_progress = days_passed / total_days if total_days > 0 else 0
        
        feature_names = [
            'Home_Last5_Wins', 'Away_Last5_Wins',
            'H2H_Home_Win_Ratio', 'H2H_Avg_Goals', 'Home_Days_Since_Last_Match',
            'Away_Days_Since_Last_Match', 'Home_League_Pos', 'Away_League_Pos', 'League_Pos_Diff',
            'Home_Form', 'Away_Form',
            'Home_Team_Home_Performance', 'Away_Team_Away_Performance',
            'Home_Streak', 'Away_Streak', 'Season_Progress', 'Is_Weekend'
        ]
        
        features = pd.DataFrame([[
            home_stats['last5_wins'],
            away_stats['last5_wins'],
            home_stats['h2h_win_ratio'],
            home_stats['h2h_avg_goals'],
            home_stats['days_since_last_match'],
            away_stats['days_since_last_match'],
            home_league_pos,
            away_league_pos,
            home_league_pos - away_league_pos,
            home_stats['form'],
            away_stats['form'],
            home_stats['performance'],
            away_stats['performance'],
            home_stats['streak'],
            away_stats['streak'],
            season_progress,
            1 if match_date.weekday() >= 5 else 0  # Is_Weekend
        ]], columns=feature_names)
        
        features_imputed = self.imputer.transform(features)
        features_scaled = self.scaler.transform(features_imputed)
        
        # Get probabilities
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Define a threshold for "marginal" predictions
        threshold = 0.07
        
        # Randomly choose based on probabilities
        result = np.random.choice(self.le.classes_, p=probabilities)
        
        if result == 'HOME_TEAM':
            return home_team
        elif result == 'AWAY_TEAM':
            return away_team
        else:
            return 'DRAW'