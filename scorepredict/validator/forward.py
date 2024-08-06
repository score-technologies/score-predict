# The MIT License (MIT)
# Copyright © 2024 Yuma Rao
# Copyright © 2024 Score Protocol

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

import time
import requests
import pandas as pd
import bittensor as bt
import wandb

from scorepredict.protocol import Prediction
from scorepredict.validator.reward import get_rewards
from scorepredict.utils.uids import get_random_uids
from scorepredict.utils.utils import assign_challenges_to_validators, get_all_validators
from scorepredict.utils.utils import get_current_time, advance_time, set_simulated_time
from scorepredict.utils.utils import send_predictions_to_website

import sqlite3
import collections
from datetime import datetime, timedelta, time as dt_time

async def forward(self):
    """
    The forward function is called by the validator every time step.
    It is responsible for querying the network and scoring the responses.
    """    
    time.sleep(1)


    bt.logging.info("simulate time: " + str(self.config.simulate_time))

    if self.config.simulate_time:
        # Advance simulated time by x minutes each iteration
        advance_time(self, 25)
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
    
    #current_time = datetime.now()
    

    if self.step % 10 == 0:
        bt.logging.debug(f"Send Predictions To Website - Step: {self.step}")
        #send_predictions_to_website(self) TODO turn back on

    if self.step % 100 == 0:
        bt.logging.debug(f"Keeping Validators Busy - Step: {self.step}")
        self.set_weights()

    # Initialize SQLite database connection
    db_name = f'predictions-{self.uid}.db'
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    
    # Create table if not exists
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (miner_uid INTEGER, match_id INTEGER, prediction TEXT, timestamp DATETIME, reward REAL, sentWebsite INTEGER)''')
    
    # Fetch all validators
    get_validators = get_all_validators(self)
    bt.logging.info(f"Validators: {get_validators}")
    
    # Fetch upcoming matches this validator should serve to miners
    bt.logging.info("🛜 Looking for upcoming matches")
    upcoming_matches = assign_challenges_to_validators(self, minutes_before_kickoff=60)
    bt.logging.debug(f"Upcoming matches: {upcoming_matches}")

    # Iterate through each miner's assigned matches
    for miner_uid, matches in upcoming_matches.items():
        time.sleep(1)
        bt.logging.debug(f"Processing matches for Miner UID {miner_uid}")

        for match_tuple in matches:
            match_id, match = match_tuple
            
            # Check if the miner has already made a prediction for this match
            c.execute("SELECT * FROM predictions WHERE miner_uid=? AND match_id=?", (miner_uid, match_id))
            existing_prediction = c.fetchone()

            if existing_prediction:
                bt.logging.debug(f"Miner UID {miner_uid} has already made a prediction for Match ID {match_id}. Skipping.")
                continue

            home_team = match['homeTeam']['name']
            away_team = match['awayTeam']['name']
            match_date = pd.to_datetime(match['utcDate']).isoformat()
            deadline = (pd.to_datetime(match['utcDate']) - timedelta(days=4)).isoformat()

            bt.logging.debug(f"Processing match: ID {match_id}, {home_team} vs {away_team} on {match_date}")

            # The dendrite client queries the network with the prediction synapse
            responses = await self.dendrite(
                axons=[self.metagraph.axons[miner_uid]],
                synapse=Prediction(
                    match_id=match_id,
                    home_team=home_team,
                    away_team=away_team,
                    match_date=match_date,
                    deadline=deadline
                ),
                deserialize=True,
            )

            # Log the received responses
            bt.logging.debug(f"Received responses for match {match_id}: {responses}")
            # Process each response in the list
            for response in responses:
                if response.get('predicted_winner') is None:
                    bt.logging.debug(f"No prediction received for match {match_id} from miner {miner_uid}. Ending process for this match.")
                    continue

                # Check if a prediction already exists
                c.execute("SELECT * FROM predictions WHERE miner_uid=? AND match_id=?", (miner_uid, match_id))
                existing_prediction = c.fetchone()

                if existing_prediction:
                    # Update existing prediction if new one is received
                    c.execute("UPDATE predictions SET prediction=?, timestamp=? WHERE miner_uid=? AND match_id=?",
                              (response['predicted_winner'], datetime.now(), miner_uid, match_id))
                else:
                    # Insert new prediction with reward set to None
                    c.execute("INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?)",
                              (miner_uid, match_id, response['predicted_winner'], datetime.now(), None, 0))

                conn.commit()                

                bt.logging.debug(f"Inserted prediction for match {match_id} from miner {miner_uid} at {current_time}")

            
            rewards_array = []  # Initialize rewards_array as an empty list when we are testing 
            
            rewards_array, rewarded_miner_uids = get_rewards(self)

            if len(rewards_array) > 0:
                # Split rewards_array back into rewards and miner_uids
                rewards = rewards_array.tolist()

                # Log and update scores based on the rewards
                bt.logging.debug(f"Scored responses array returned:{rewards_array}")
                bt.logging.debug(f"Rewarded miner ids array returned:{rewarded_miner_uids}")
                self.update_scores(rewards, rewarded_miner_uids)
            else:
                bt.logging.debug(f"No rewards to process. Skipping scoring.")

    
    conn.close()


