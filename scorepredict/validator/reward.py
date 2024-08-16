# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
import bittensor as bt
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from typing import Tuple, List
from collections import defaultdict

from scorepredict.utils.utils import get_matches
from scorepredict.utils.utils import get_current_time
from scorepredict.constants import MINUTES_BEFORE_KICKOFF

def get_streak(c, miner_uid, prediction_time):
    """
    Calculate the current streak of correct predictions for a miner.
    """
    c.execute("""
        SELECT prediction, reward
        FROM predictions
        WHERE miner_uid = ? AND timestamp < ?
        ORDER BY timestamp DESC
        LIMIT 20
    """, (miner_uid, prediction_time))
    
    streak = 0
    for pred, reward in c.fetchall():
        if reward is not None and reward > 0.1:
            streak += 1
        else:
            break
    return streak

def get_streak_multiplier(streak):
    """
    Return the multiplier based on the current streak.
    """
    if streak >= 20:
        return 2.0
    elif streak >= 10:
        return 1.8
    elif streak >= 5:
        return 1.4
    elif streak >= 2:
        return 1.1
    else:
        return 1.0

def reward(prediction, match_data, prediction_time, c, miner_uid):
    """
    Reward the miner response to the match prediction request.
    """
    
    actual_winner_code = match_data['score']['winner']
    home_team_name = match_data['homeTeam']['name']
    away_team_name = match_data['awayTeam']['name']

    if actual_winner_code == "HOME_TEAM":
        actual_winner = home_team_name
    elif actual_winner_code == "AWAY_TEAM":
        actual_winner = away_team_name
    else:
        actual_winner = "DRAW"

    base_reward = 3.0 if prediction == actual_winner else 0
    
    match_time = datetime.fromisoformat(match_data['utcDate'].rstrip('Z'))
    
    # Calculate streak and apply multiplier
    streak = get_streak(c, miner_uid, prediction_time)
    streak_multiplier = get_streak_multiplier(streak)
    
    # Calculate time-based multiplier
    prediction_window_start = match_time - timedelta(minutes=MINUTES_BEFORE_KICKOFF)
    prediction_window_end = match_time - timedelta(minutes=1)
    
    if prediction_time <= prediction_window_start:
        time_multiplier = 1.5
    elif prediction_time >= prediction_window_end:
        time_multiplier = 1.0
    else:
        # Linear decay from 1.5 to 1.0
        total_minutes = MINUTES_BEFORE_KICKOFF - 1
        minutes_before = (prediction_window_end - prediction_time).total_seconds() / 60
        time_multiplier = 1.0 + (0.5 * minutes_before / total_minutes)
    
    total_reward = base_reward * streak_multiplier * time_multiplier

    return total_reward


def get_rewards(self) -> Tuple[torch.FloatTensor, List[int]]:
    if self.config.simulate_time:
        target_date = get_current_time(self) - timedelta(hours=24)
    else:
        target_date = get_current_time(self)

    finished_matches = get_matches(self, date_str=target_date, status='FINISHED')

    if finished_matches is None:
        bt.logging.info("No finished matches found. Returning empty reward tensor and miner UIDs.")
        return torch.FloatTensor([]), []

    db_name = f'predictions-{self.uid}.db'
    conn = sqlite3.connect(db_name)
    c = conn.cursor()

    # Get the total number of predictions in the last 7 days
    seven_days_ago = target_date - timedelta(days=7)
    c.execute("SELECT miner_uid, COUNT(*) FROM predictions WHERE timestamp > ? GROUP BY miner_uid", (seven_days_ago,))
    prediction_counts = dict(c.fetchall())

    total_predictions = sum(prediction_counts.values())
    avg_predictions = total_predictions / len(prediction_counts) if prediction_counts else 0

    aggregated_rewards = defaultdict(float)
    rewarded_miner_uids = set()

    for match_id, match in finished_matches.items():
        c.execute("SELECT miner_uid, prediction, timestamp, reward FROM predictions WHERE match_id=?", (match_id,))
        predictions = c.fetchall()
        
        for miner_uid, prediction, timestamp, reward_value_from_db in predictions:
            if reward_value_from_db is not None:
                bt.logging.info(f"Reward already processed for miner {miner_uid} for match {match_id}. Skipping.")
                continue

            prediction_time = datetime.fromisoformat(timestamp)
            base_reward = calculate_prediction_score(prediction, match)
            
            # Apply activity multiplier
            miner_prediction_count = prediction_counts.get(miner_uid, 0)
            activity_multiplier = min(1.0, miner_prediction_count / avg_predictions)
            
            # Apply time-based multiplier
            time_multiplier = calculate_time_multiplier(prediction_time, match['utcDate'])
            
            total_reward = base_reward * activity_multiplier * time_multiplier
            
            
            aggregated_rewards[miner_uid] += total_reward
            rewarded_miner_uids.add(miner_uid)

            # Update the reward in the database
            c.execute("UPDATE predictions SET reward=? WHERE match_id=? AND miner_uid=?", (total_reward, match_id, miner_uid))

    conn.commit()
    conn.close()

    # Normalize rewards
    total_rewards = sum(aggregated_rewards.values())
    if total_rewards > 0:
        normalized_rewards = [reward / total_rewards for reward in aggregated_rewards.values()]
    else:
        normalized_rewards = [0 for _ in aggregated_rewards]

    rewarded_miner_uids = list(rewarded_miner_uids)

    if normalized_rewards:
        rewards_tensor = torch.FloatTensor(normalized_rewards).to(self.device)
        bt.logging.info(f"Processed additional rewards for {len(rewarded_miner_uids)} miners.")
        return rewards_tensor, rewarded_miner_uids
    else:
        bt.logging.info("No additional rewards to process.")
        return torch.FloatTensor([]), []


def calculate_prediction_score(prediction, match_data):
    #actual_home_score = match_data['score']['fullTime']['homeTeam']
    #actual_away_score = match_data['score']['fullTime']['awayTeam']
    actual_result = match_data['result']
    print("Winner", actual_result)

    if prediction == actual_result:
        winner_accuracy = 1
    else:
        winner_accuracy = 0

    # # Score accuracy
    # score_diff = abs(predicted_home_score - actual_home_score) + abs(predicted_away_score - actual_away_score)
    # score_accuracy = max(0, 1 - (score_diff / 10))  # 10 is max difference we consider

    return (winner_accuracy) 
    #return (score_accuracy * 0.5) + (winner_accuracy * 0.5)  # Equal weight to score and winner prediction

def calculate_time_multiplier(prediction_time, match_time):
    match_time = datetime.fromisoformat(match_time.rstrip('Z'))
    time_diff = (match_time - prediction_time).total_seconds() / 60  # difference in minutes
    
    if time_diff >= MINUTES_BEFORE_KICKOFF:
        return 1.2
    elif time_diff <= 1:
        return 1.0
    else:
        return 1.0 + (0.5 * (time_diff - 1) / (MINUTES_BEFORE_KICKOFF - 1))