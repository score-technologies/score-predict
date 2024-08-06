import os
import pandas as pd
import numpy as np
from joblib import load
from datetime import datetime, timedelta
import numpy as np
from sklearn.preprocessing import StandardScaler

class FootballPredictor:
    def __init__(self):
        # Get the directory of the current script
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construct absolute paths for the model files
        model_path = os.path.join(base_dir, 'model/random_forest_model.joblib')
        imputer_path = os.path.join(base_dir, 'model/imputer.joblib')
        scaler_path = os.path.join(base_dir, 'model/feature_scaler.joblib')
        le_path = os.path.join(base_dir, 'model/label_encoder.joblib')
        dataset_path = os.path.join(base_dir, 'model/soccer_matches_with_features.csv')
        
        # Load the saved models and transformers
        self.model = load(model_path)
        self.imputer = load(imputer_path)
        self.scaler = load(scaler_path)
        self.le = load(le_path)

        # Load the dataset
        self.df = pd.read_csv(dataset_path)
        self.random_state = np.random.RandomState()

    def get_team_stats(self, team, date):
        team_matches = self.df[(self.df['Home Team'] == team) | (self.df['Away Team'] == team)]
        team_matches = team_matches[team_matches['Date'] < date]
        team_matches = team_matches.sort_values('Date', ascending=False).head(5)
        
        if team_matches.empty:
            return 0, 0, 0, 365  # Default: 0 wins, 0 goals scored/conceded, 365 days since last match
        
        wins = sum((team_matches['Home Team'] == team) & (team_matches['Winner'] == 'HOME_TEAM') | 
                   (team_matches['Away Team'] == team) & (team_matches['Winner'] == 'AWAY_TEAM'))
        
        goals_scored = sum(team_matches[team_matches['Home Team'] == team]['Full Time Home']) + \
                       sum(team_matches[team_matches['Away Team'] == team]['Full Time Away'])
        
        goals_conceded = sum(team_matches[team_matches['Home Team'] == team]['Full Time Away']) + \
                         sum(team_matches[team_matches['Away Team'] == team]['Full Time Home'])
        
        days_since_last_match = (datetime.strptime(date, '%Y-%m-%d') - 
                                 datetime.strptime(team_matches.iloc[0]['Date'][:10], '%Y-%m-%d')).days
        
        return wins, goals_scored, goals_conceded, days_since_last_match

    def get_h2h_stats(self, home_team, away_team, date):
        h2h_matches = self.df[((self.df['Home Team'] == home_team) & (self.df['Away Team'] == away_team)) | 
                         ((self.df['Home Team'] == away_team) & (self.df['Away Team'] == home_team))]
        h2h_matches = h2h_matches[h2h_matches['Date'] < date]
        
        total_matches = len(h2h_matches)
        if total_matches == 0:
            return 0.5, 0  # Default values if no H2H matches
        
        home_wins = sum((h2h_matches['Home Team'] == home_team) & (h2h_matches['Winner'] == 'HOME_TEAM') | 
                        (h2h_matches['Away Team'] == home_team) & (h2h_matches['Winner'] == 'AWAY_TEAM'))
        
        home_win_ratio = home_wins / total_matches
        avg_goals = (h2h_matches['Full Time Home'] + h2h_matches['Full Time Away']).mean()
        
        return home_win_ratio, avg_goals

    def is_weekend(self, date):
        return datetime.strptime(date, '%Y-%m-%d').weekday() >= 5

    def get_league_positions(self, home_team, away_team, date):
        home_pos = self.df[self.df['Home Team'] == home_team]['Home_League_Pos'].mean()
        away_pos = self.df[self.df['Away Team'] == away_team]['Away_League_Pos'].mean()
        return home_pos, away_pos

    def predict_winner(self, home_team, away_team, date):
        home_stats = self.get_team_stats(home_team, date)
        away_stats = self.get_team_stats(away_team, date)
        h2h_stats = self.get_h2h_stats(home_team, away_team, date)
        home_pos, away_pos = self.get_league_positions(home_team, away_team, date)
        
        features = [
            home_stats[0], home_stats[1], home_stats[2],  # Home team stats
            away_stats[0], away_stats[1], away_stats[2],  # Away team stats
            h2h_stats[0], h2h_stats[1],                   # H2H stats
            home_stats[3], away_stats[3],                 # Days since last match
            home_pos, away_pos, home_pos - away_pos,      # League positions
            int(self.is_weekend(date))                    # Is weekend
        ]
        
        feature_names = [
            'Home_Last5_Wins', 'Home_Last5_Goals_Scored', 'Home_Last5_Goals_Conceded',
            'Away_Last5_Wins', 'Away_Last5_Goals_Scored', 'Away_Last5_Goals_Conceded',
            'H2H_Home_Win_Ratio', 'H2H_Avg_Goals', 'Home_Days_Since_Last_Match',
            'Away_Days_Since_Last_Match', 'Home_League_Pos', 'Away_League_Pos', 'League_Pos_Diff',
            'Is_Weekend'
        ]
        
        
        # Convert features to DataFrame with appropriate feature names
        X = pd.DataFrame([features], columns=feature_names)
        
        # Impute and scale the features
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)

        # Get probabilities instead of just the class
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        # Use probabilities to make a weighted random choice
        classes = self.le.classes_
        predicted_result = self.random_state.choice(classes, p=probabilities)
        
        # Map the prediction to the actual team names or "DRAW"
        if predicted_result == 'HOME_TEAM':
            return home_team
        elif predicted_result == 'AWAY_TEAM':
            return away_team
        else:
            return 'DRAW'

# Example usage (can be commented out or removed when used as a module)
# if __name__ == "__main__":
#     predictor = FootballPredictor()
#     home_team = "AFC Bournemouth"
#     away_team = "Aston Villa FC"
#     date = "2024-07-13"

#     result = predictor.predict_winner(home_team, away_team, date)
#     print(f"Predicted result: {result}")