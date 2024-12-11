# The MIT License (MIT)
# Copyright © 2024 Yuma Rao
# Copyright © 2024 Score Protocol

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

import time
import requests
import pandas as pd
import bittensor as bt
import numpy as np
from typing import Optional
import json

from scorepredict.protocol import Prediction
from scorepredict.validator.reward import get_rewards
from scorepredict.utils.uids import get_random_uids
from scorepredict.utils.utils import assign_challenges_to_validators, get_all_validators
from scorepredict.utils.utils import get_current_time, advance_time, set_simulated_time
from scorepredict.utils.utils import send_predictions_to_website
from scorepredict.utils.utils import get_matches, get_all_miners, get_random_uids

import sqlite3
import collections
from datetime import datetime, timedelta, time as dt_time

from scorepredict.constants import (
    APP_SYNC_IN_BLOCKS,
    VALIDATOR_SET_WEIGHTS_IN_BLOCKS,
    MINUTES_BEFORE_KICKOFF,
    SCORE_PREDICT_API_URL,
    REWARD_FOR_RESPONSE
)

pending_weight_update = False
cached_rewards: Optional[tuple] = None

async def forward(self):
    global pending_weight_update, cached_rewards
    """
    The forward function is called by the validator every time step.
    It is responsible for querying the network and scoring the responses.
    """    

    """ FOR TESTING FAST FORWARD THROUGH TIME """
    bt.logging.info("simulate time: " + str(self.config.simulate_time))

    if self.config.simulate_time:
        # Advance simulated time by x minutes each iteration
        advance_time(self, 20)
        current_time = get_current_time(self)
        bt.logging.debug(f"Current simulated time: {current_time}")

        # If it's past 8 PM UTC, reset the simulated time to 2 PM of the next day
        if current_time.hour >= 23:
            next_day = current_time.date() + timedelta(days=1)  # Use timedelta directly
            new_time = datetime.combine(next_day, dt_time(10, 0))  # 2 PM
            set_simulated_time(new_time)
            bt.logging.debug(f"Reset simulated time to: {new_time}")
    else:
        current_time = get_current_time(self)
        bt.logging.debug(f"Current time: {current_time}")
        bt.logging.debug(f"UTC time: {datetime.utcnow()}")
    
    
    """ PERIODICALLY SEND PREDICTIONS TO APP """
    if self.step % APP_SYNC_IN_BLOCKS == 0:
        bt.logging.debug(f"Send Predictions To App - Step: {self.step}")
        send_predictions_to_website(self)

    """ PERIODICALLY KEEP VALIDATORS SETTING WEIGHTS AND PROCESS REWARDS """
    if self.step % VALIDATOR_SET_WEIGHTS_IN_BLOCKS == 0 or pending_weight_update:
        bt.logging.debug(f"Set Weights and Process Rewards - Step: {self.step}")
        try:
            if cached_rewards:
                bt.logging.debug(f"We have cached rewards, using them.")
                rewards, rewarded_miner_uids = cached_rewards
                self.update_scores(rewards, rewarded_miner_uids)
                # Don't clear cached_rewards yet
            else:
                bt.logging.debug(f"No cached rewards, fetching new rewards.")
                rewards, rewarded_miner_uids = get_rewards(self)
                if len(rewards) > 0:
                    bt.logging.info(f"Processed rewards for {len(rewarded_miner_uids)} miners.")
                    self.update_scores(rewards, rewarded_miner_uids)
                else:
                    bt.logging.debug(f"No new rewards to process.")
            
            self.set_weights()
            # Only clear cached_rewards if set_weights() succeeds
            cached_rewards = None
            pending_weight_update = False
        except Exception as e:
            bt.logging.error(f"Failed to process rewards or set weights: {e}")
            pending_weight_update = True
            if not cached_rewards:  # Only cache rewards if we don't already have cached ones
                cached_rewards = (rewards, rewarded_miner_uids) if 'rewards' in locals() else None

    """ FETCH UPCOMING MATCHES """
    matches = get_matches(self, date_str=current_time, minutes_before_kickoff=MINUTES_BEFORE_KICKOFF)
    
    if not matches:
        bt.logging.info("No upcoming matches found.")
        return

    """ FETCH VALID MIDERS """
    miner_uids = get_random_uids(self, k=30)
    bt.logging.info(f"Random Miner UIDs: {miner_uids}")
    
    if not miner_uids:
        bt.logging.info("No miners available.")
        return

    bt.logging.info(f"Found {len(matches)} matches and {len(miner_uids)} miners.")

    """ INITIALIZE LOCAL STORAGE """  
    db_name = f'predictions-{self.uid}.db'
    conn = sqlite3.connect(db_name)
    c = conn.cursor()

    # Create table if not exists
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                (miner_uid INTEGER, match_id INTEGER, prediction TEXT, timestamp DATETIME, 
                reward REAL, sentWebsite INTEGER, competition TEXT)''')
    conn.commit()

    # Check if the competition column exists
    c.execute("PRAGMA table_info(predictions)")
    columns = [column[1] for column in c.fetchall()]

    if 'competition' not in columns:
        # Add the competition column if it doesn't exist
        c.execute("ALTER TABLE predictions ADD COLUMN competition TEXT")
        conn.commit()
        bt.logging.info("Added 'competition' column to predictions table")

    rewards = []
    rewarded_miner_uids = []

    """ PROCESS EACH MATCH """  
    for match_id, match in matches.items():
        home_team = match['homeTeam']['name']
        away_team = match['awayTeam']['name']
        match_date = match['utcDate']
        deadline = match['utcDate']  
        competition = match['competition']['name']

        # Create the Prediction synapse
        prediction = Prediction(
            match_id=int(match_id), 
            home_team=home_team,
            away_team=away_team,
            match_date=match_date,
            deadline=deadline,
            competition=competition
        )

        # Query all miners for this match
        responses = await self.dendrite(
            axons=[self.metagraph.axons[uid] for uid in miner_uids],
            synapse=prediction,
            deserialize=True,
        )

        bt.logging.info(f"Received {len(responses)} responses for match {match_id}")


        """ PROCESS RESPONSES """ 
        active_miner_uids = []
        non_responsive_miner_uids = []

        for miner_uid, response in zip(miner_uids, responses):
            # Check if prediction already exists in the database
            c.execute("SELECT * FROM predictions WHERE miner_uid=? AND match_id=?", (miner_uid, match_id))
            existing_prediction = c.fetchone()

            if existing_prediction:
                bt.logging.debug(f"Prediction already exists for miner {miner_uid} and match {match_id}")
                active_miner_uids.append(miner_uid)  # Consider them active if they have a prediction
                continue

            # Check scorepredict API
            api_url = f"{SCORE_PREDICT_API_URL}/api/predictions/{match_id}?userId={miner_uid}"
            try:
                api_response = requests.get(api_url)
                if api_response.status_code == 200:
                    api_prediction = api_response.json().get('prediction', {})
                    if api_prediction:
                        predicted_winner = api_prediction.get('prediction')
                        sent_website = 1
                    else:
                        predicted_winner = response.get('predicted_winner')
                        sent_website = 0
                else:
                    predicted_winner = response.get('predicted_winner')
                    sent_website = 0
            except requests.RequestException as e:
                bt.logging.error(f"Error checking API for prediction: {e}")
                predicted_winner = response.get('predicted_winner')
                sent_website = 0

            if predicted_winner is None:
                bt.logging.debug(f"Received NULL prediction for miner {miner_uid} and match {match_id}")
                non_responsive_miner_uids.append(miner_uid)
                continue

            c.execute("INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?)",
                      (miner_uid, match_id, predicted_winner, datetime.now(), None, sent_website, competition))
            
            active_miner_uids.append(miner_uid)
            
        conn.commit()
        # Penalize only non-responsive miners
        # no_rewards = [0.0 for _ in non_responsive_miner_uids]
        # if non_responsive_miner_uids:
        #     bt.logging.info(f"Penalizing miners {non_responsive_miner_uids} that did not respond or returned NULL.")
        #     self.update_scores(torch.FloatTensor(no_rewards).to(self.device), non_responsive_miner_uids)

    conn.close()

    if not pending_weight_update:
        additional_rewards_array, additional_rewarded_miner_uids = get_rewards(self)

        if len(additional_rewards_array) > 0:
            bt.logging.debug(f"Additional scored responses array returned: {additional_rewards_array}")
            bt.logging.debug(f"Additional rewarded miner ids array returned: {additional_rewarded_miner_uids}")
            try:
                self.update_scores(additional_rewards_array.tolist(), additional_rewarded_miner_uids)
                self.set_weights()
            except Exception as e:
                bt.logging.error(f"Failed to update scores or set weights: {e}")
                pending_weight_update = True
                cached_rewards = (additional_rewards_array.tolist(), additional_rewarded_miner_uids)
        else:
            bt.logging.debug(f"No additional rewards to process.")
    else:
        bt.logging.info("Skipping get_rewards() due to pending weight update.")
