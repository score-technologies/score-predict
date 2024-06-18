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

import requests
import torch
from typing import Tuple, List
import bittensor as bt
import time
from datetime import datetime, timedelta
import collections
import numpy as np

#from scorepredict.validator.forward import MinerSubmissions
from scorepredict.utils.utils import get_matches

# Initialize the start date for fetching matches
current_fetch_date = datetime(2023, 4, 1)

def reward(submission, match_data):
    """
    Reward the miner response to the match prediction request. This method returns a reward
    value for the miner, which is used to update the miner's score.

    Returns:
    - float: The reward value for the miner.
    """
    actual_winner_code = match_data['score']['winner']
    home_team_name = match_data['homeTeam']['name']
    away_team_name = match_data['awayTeam']['name']

    # Map actual_winner_code to the team name
    if actual_winner_code == "HOME_TEAM":
        actual_winner = home_team_name
    elif actual_winner_code == "AWAY_TEAM":
        actual_winner = away_team_name
    else:
        actual_winner = "DRAW"

    # Extract the predicted winner from the submission directly
    predicted_winner = submission['predicted_winner']

    # Check if the predicted winner is correct
    if predicted_winner == actual_winner:
        return 3.0  # Assign 3 points for correct winner prediction
    else:
        return 0.0  # Assign 0 points for incorrect winner prediction

def get_rewards(self, submissions: dict) -> Tuple[np.ndarray, List[str]]:
    target_date = datetime.utcnow().strftime('%Y-%m-%d')
    finished_matches = get_matches(self, date_str=target_date, status='FINISHED')

    bt.logging.info(f"Finished matches: {finished_matches}")
    bt.logging.info(f"Submissions: {submissions}")

    if finished_matches is None:
        bt.logging.info("No finished matches found. Returning empty reward array and miner UIDs.")
        return np.array([]), []

    rewards = []
    rewarded_miner_uids = []
    rewarded_submissions = []  # List to store rewarded submissions

    for match_id, match in finished_matches.items():
        for (miner_uid, submission_match_id), submission_data in submissions.items():
            if submission_match_id == match_id:
                submission = submission_data['prediction']
                reward_value = reward(submission, match)
                rewards.append(reward_value)
                rewarded_miner_uids.append(miner_uid)
                rewarded_submissions.append((miner_uid, submission_match_id))  # Add rewarded submission to the list
                bt.logging.info(f"Reward for miner {miner_uid} for match {match_id}: {reward_value}")
                with open("predictions_log.txt", "a") as log_file:
                    log_file.write(f"Reward for miner {miner_uid} for match {match_id}: {reward_value}\n")

    # Remove rewarded submissions from the submissions dictionary
    for miner_uid, submission_match_id in rewarded_submissions:
        del submissions[(miner_uid, submission_match_id)]

    rewards_array = np.array(rewards)
    return rewards_array, rewarded_miner_uids if rewards_array.size > 0 else (np.array([]), [])
