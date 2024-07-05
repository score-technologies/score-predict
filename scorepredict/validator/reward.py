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

import bittensor as bt
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from typing import Tuple, List

from scorepredict.utils.utils import get_matches
from scorepredict.utils.utils import get_current_time

def calculate_time_bonus(prediction_time, match_time):
    """Calculate bonus based on how early the prediction was made"""
    hours_before = (match_time - prediction_time).total_seconds() / 3600
    if hours_before >= 6:
        return 1.0  # Maximum bonus
    elif hours_before <= 1:
        return 0.0  # No bonus
    else:
        return (hours_before - 1) / 5  # Linear scale between 1 and 6 hours

def reward(prediction, match_data, prediction_time):
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

    base_reward = 3.0 if prediction == actual_winner else 0.0
    
    match_time = datetime.fromisoformat(match_data['utcDate'].rstrip('Z'))
    #time_bonus = calculate_time_bonus(prediction_time, match_time)
    
    #total_reward = base_reward * (1 + time_bonus)
    total_reward = base_reward

    return total_reward

def get_rewards(self) -> Tuple[np.ndarray, List[int]]:
    # We check for games completed 24 hours ago if simulating time for testing, otherwise just checks for games completed just now
    if self.config.simulate_time:
        target_date = get_current_time(self) - timedelta(hours=24)
    else:
        target_date = get_current_time(self)

    finished_matches = get_matches(self, date_str=target_date, status='FINISHED')

    #bt.logging.info(f"Finished matches: {finished_matches}")

    if finished_matches is None:
        bt.logging.info("No finished matches found. Returning empty reward array and miner UIDs.")
        return np.array([]), []

    rewards = []
    rewarded_miner_uids = []

    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()

    for match_id, match in finished_matches.items():
        c.execute("SELECT miner_uid, prediction, timestamp, reward FROM predictions WHERE match_id=?", (match_id,))
        predictions = c.fetchall()
        
        for miner_uid, prediction, timestamp, reward_value_from_db in predictions:
            if reward_value_from_db is not None:
                bt.logging.info(f"Reward already processed for miner {miner_uid} for match {match_id}. Skipping.")
                continue

            prediction_time = datetime.fromisoformat(timestamp)
            reward_value = reward(prediction, match, prediction_time)
            rewards.append(reward_value)
            rewarded_miner_uids.append(miner_uid)
            bt.logging.info(f"Reward for miner {miner_uid} for match {match_id}: {reward_value}")

            # Update the reward in the database
            c.execute("UPDATE predictions SET reward=? WHERE match_id=? AND miner_uid=?", (reward_value, match_id, miner_uid))

    conn.commit()
    conn.close()

    rewards_array = np.array(rewards)
    return rewards_array, rewarded_miner_uids if rewards_array.size > 0 else (np.array([]), [])