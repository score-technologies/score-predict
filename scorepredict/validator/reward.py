# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
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
        if reward is not None and reward > 0.5:
            streak += 1
        else:
            break
    return streak

def get_streak_multiplier(streak):
    """
    Return the multiplier based on the current streak.
    """
    if streak >= 20:
        return 1.8
    elif streak >= 10:
        return 1.5
    elif streak >= 5:
        return 1.3
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

    base_reward = 1.0 if prediction == actual_winner else 0.1
    
    match_time = datetime.fromisoformat(match_data['utcDate'].rstrip('Z'))
    
    # Calculate streak and apply multiplier
    streak = get_streak(c, miner_uid, prediction_time)
    streak_multiplier = get_streak_multiplier(streak)
    
    # Calculate time-based multiplier
    prediction_window_start = match_time - timedelta(minutes=MINUTES_BEFORE_KICKOFF)
    prediction_window_end = match_time - timedelta(minutes=1)
    
    if prediction_time <= prediction_window_start:
        time_multiplier = 1.3
    elif prediction_time >= prediction_window_end:
        time_multiplier = 1.0
    else:
        # Linear decay from 1.5 to 1.0
        total_minutes = MINUTES_BEFORE_KICKOFF - 1
        minutes_before = (prediction_window_end - prediction_time).total_seconds() / 60
        time_multiplier = 1.0 + (0.5 * minutes_before / total_minutes)
    
    total_reward = base_reward * streak_multiplier * time_multiplier

    return total_reward

def get_win_rate_multiplier(win_rate):
    """
    Calculate a multiplier based on the win rate.
    """
    if win_rate >= 0.7:
        return 1.3  # 30% bonus for excellent performance
    elif win_rate >= 0.6:
        return 1.2  # 20% bonus for very good performance
    elif win_rate >= 0.5:
        return 1.1  # 10% bonus for good performance
    elif win_rate >= 0.4:
        return 1.0  # No bonus for average performance
    else:
        return 0.9  # 10% penalty for below-average performance

def get_rewards(self) -> Tuple[torch.FloatTensor, List[int]]:
    bt.logging.debug("Entering get_rewards function")
    
    target_date = get_current_time(self) - timedelta(hours=24) if self.config.simulate_time else get_current_time(self)

    finished_matches = get_matches(self, date_str=target_date, status='FINISHED')

    if not finished_matches:
        bt.logging.info("No finished matches found. Returning empty reward tensor and miner UIDs.")
        return torch.FloatTensor([]), []

    db_name = f'predictions-{self.uid}.db'
    
    try:
        with sqlite3.connect(db_name) as conn:
            c = conn.cursor()

            seven_days_ago = target_date - timedelta(days=7)
            
            # Get prediction counts and win rates in a single query
            c.execute("""
                SELECT miner_uid, 
                       COUNT(*) as total_predictions,
                       SUM(CASE WHEN reward > 0.5 THEN 1 ELSE 0 END) as wins
                FROM predictions 
                WHERE timestamp > ?
                GROUP BY miner_uid
            """, (seven_days_ago,))
            
            miner_stats = {row[0]: {'predictions': row[1], 'wins': row[2]} for row in c.fetchall()}
            
            if not miner_stats:
                bt.logging.info("No predictions found in the last 7 days. Returning empty reward tensor and miner UIDs.")
                return torch.FloatTensor([]), []

            total_predictions = sum(stats['predictions'] for stats in miner_stats.values())
            avg_predictions = total_predictions / len(miner_stats) if len(miner_stats) > 0 else 0

            # Calculate win rates
            win_rates = {uid: stats['wins'] / stats['predictions'] if stats['predictions'] > 0 else 0 
                         for uid, stats in miner_stats.items()}

            # Process rewards
            miner_rewards = defaultdict(float)
            rewarded_miner_uids = []

            for match_id, match in finished_matches.items():
                c.execute("SELECT miner_uid, prediction, timestamp, reward FROM predictions WHERE match_id=? ORDER BY timestamp", (match_id,))
                predictions = c.fetchall()
                
                for miner_uid, prediction, timestamp, reward_value_from_db in predictions:
                    if reward_value_from_db is not None:
                        bt.logging.debug(f"Reward already processed for miner {miner_uid} for match {match_id}. Skipping.")
                        continue

                    prediction_time = datetime.fromisoformat(timestamp)
                    
                    # Calculate base reward
                    base_reward = reward(prediction, match, prediction_time, c, miner_uid)
                    
                    # Accumulate base reward for each miner
                    miner_rewards[miner_uid] += base_reward
                    
                    if miner_uid not in rewarded_miner_uids:
                        rewarded_miner_uids.append(miner_uid)

                    # Update the reward in the database
                    c.execute("UPDATE predictions SET reward=? WHERE match_id=? AND miner_uid=?", (base_reward, match_id, miner_uid))

            # Apply participation factor and win rate multiplier to accumulated rewards
            final_rewards = []
            for miner_uid in rewarded_miner_uids:
                base_reward = miner_rewards[miner_uid]
                
                # Apply participation factor
                miner_prediction_count = miner_stats.get(miner_uid, {'predictions': 0})['predictions']
                avg_predictions_for_participation = max(avg_predictions * 0.5, 1.0)
                participation_factor = min(1.0, miner_prediction_count / avg_predictions_for_participation)

                # Apply win rate multiplier
                win_rate = win_rates.get(miner_uid, 0)
                win_rate_multiplier = get_win_rate_multiplier(win_rate)

                total_reward = base_reward * participation_factor * win_rate_multiplier
                final_rewards.append(total_reward)

                # Log the results for each miner
                bt.logging.info(f"Miner {miner_uid}:")
                bt.logging.info(f"  Total Base Reward: {base_reward:.4f}")
                bt.logging.info(f"  Total Final Reward: {total_reward:.4f}")
                bt.logging.info(f"  Participation Factor: {participation_factor:.4f}")
                bt.logging.info(f"  Win Rate Multiplier: {win_rate_multiplier:.4f}")

            conn.commit()

        # Normalize rewards
        total_rewards = sum(final_rewards)

        bt.logging.debug(f"Total rewards: {total_rewards}, Rewarded miner UIDs: {rewarded_miner_uids}")

        if total_rewards > 0:
            normalized_rewards = [reward / total_rewards for reward in final_rewards]
            rewards_tensor = torch.FloatTensor(normalized_rewards).to(self.device)
            bt.logging.debug(f"Rewards tensor: {rewards_tensor}")
            bt.logging.info(f"Processed additional rewards for {len(rewarded_miner_uids)} miners.")
            return rewards_tensor, rewarded_miner_uids
        else:
            bt.logging.info("No additional rewards to process or all rewards are zero.")
            return torch.FloatTensor([]), []

    except sqlite3.Error as e:
        bt.logging.error(f"Database error: {e}")
        return torch.FloatTensor([]), []
    except Exception as e:
        bt.logging.error(f"Unexpected error in get_rewards: {e}")
        return torch.FloatTensor([]), []