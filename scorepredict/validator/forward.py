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

import time
import requests
import pandas as pd
import bittensor as bt
import wandb

from scorepredict.protocol import Prediction
from scorepredict.validator.reward import get_rewards
from scorepredict.utils.uids import get_random_uids
from scorepredict.utils.utils import assign_challenges_to_validators, get_all_validators, keep_validators_alive
from scorepredict.utils.utils import get_current_time, advance_time, set_simulated_time

import sqlite3
import collections
from datetime import datetime, timedelta, time as dt_time

# Global set to track sent game IDs
sent_game_ids = set()

# Initialize the start date for fetching matches TESTING PURPOSES to simulate real world
current_fetch_date = datetime(2023, 4, 5)
  
submissions = {}  # Dictionary to store submissions

async def forward(self):
    """
    The forward function is called by the validator every time step.
    It is responsible for querying the network and scoring the responses.
    """    
    time.sleep(1)


    bt.logging.info("simulate time: " + str(self.config.simulate_time))

    if self.config.simulate_time:
        # Advance simulated time by x minutes each iteration
        advance_time(self, 15)
        current_time = get_current_time(self)
        bt.logging.info(f"Current simulated time: {current_time}")

        # If it's past 8 PM UTC, reset the simulated time to 2 PM of the next day
        if current_time.hour >= 20:
            next_day = current_time.date() + timedelta(days=1)  # Use timedelta directly
            new_time = datetime.combine(next_day, dt_time(13, 0))  # 2 PM
            set_simulated_time(new_time)
            bt.logging.info(f"Reset simulated time to: {new_time}")
    else:
        current_time = get_current_time(self)
        bt.logging.info(f"Current time: {current_time}")
    
    global current_fetch_date
    current_time = datetime.now()

    if self.step % 10 == 0:
        bt.logging.info(f"Keeping Alive - Step: {self.step}")
        self.set_weights()

    # Initialize SQLite database connection
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    
    # Create table if not exists
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (miner_uid INTEGER, match_id INTEGER, prediction TEXT, timestamp DATETIME, reward REAL)''')
    
    
    # Fetch all validators
    get_validators = get_all_validators(self)
    bt.logging.info(f"Validators: {get_validators}")
    
    # Fetch upcoming matches this validator should serve to miners
    bt.logging.info("🛜 Looking for upcoming matches")
    upcoming_matches = assign_challenges_to_validators(self, minutes_before_kickoff=60)
    #bt.logging.info(f"Upcoming matches: {upcoming_matches}")

    # If no upcoming matches, keep the validator alive
    # if not upcoming_matches:
    #     keep_validators_alive(self)
    #     return

    # Initialize or retrieve the set of sent game IDs
    if not hasattr(self, 'sent_game_ids'):
        self.sent_game_ids = set()

    # Iterate through each miner's assigned matches
    for miner_uid, matches in upcoming_matches.items():
        time.sleep(1)
        bt.logging.info(f"Processing matches for Miner UID {miner_uid}")

        for match_tuple in matches:
            match_id, match = match_tuple
            
            # Check if the miner has already made a prediction for this match
            c.execute("SELECT * FROM predictions WHERE miner_uid=? AND match_id=?", (miner_uid, match_id))
            existing_prediction = c.fetchone()

            if existing_prediction:
                bt.logging.info(f"Miner UID {miner_uid} has already made a prediction for Match ID {match_id}. Skipping.")
                continue

            home_team = match['homeTeam']['name']
            away_team = match['awayTeam']['name']
            match_date = pd.to_datetime(match['utcDate']).isoformat()
            deadline = (pd.to_datetime(match['utcDate']) - timedelta(days=4)).isoformat()

            bt.logging.info(f"Processing match: ID {match_id}, {home_team} vs {away_team} on {match_date}")

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
            bt.logging.info(f"Received responses for match {match_id}: {responses}")
            # Process each response in the list
            for response in responses:
                if response.get('predicted_winner') is None:
                    bt.logging.info(f"No prediction received for match {match_id} from miner {miner_uid}. Ending process for this match.")
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
                    c.execute("INSERT INTO predictions VALUES (?, ?, ?, ?, ?)",
                              (miner_uid, match_id, response['predicted_winner'], datetime.now(), None))

                conn.commit()                

                bt.logging.info(f"Inserted prediction for match {match_id} from miner {miner_uid} at {current_time}")
            #     log_data = {
            #         "miner_uid": miner_uid,
            #         "match_id": match_id,
            #         "home_team": home_team,
            #         "away_team": away_team,
            #         "match_date": match_date,
            #         "prediction": response['predicted_winner']
            #     }
            #    # wandb.log(log_data)
            #     # Log to a text file
            #     with open("predictions_log.txt", "a") as log_file:
            #         log_file.write(f"{log_data}\n")
                
                            
            # bt.logging.info(f"Submissions: {submissions}")
            
            rewards_array = []  # Initialize rewards_array as an empty list when we are testing 
            
            rewards_array, rewarded_miner_uids = get_rewards(self)
            
            # if len(submissions) > 3: #TODO remove this when we have a real test
            #     # Calculate rewards based on the responses
            #     rewards_array, rewarded_miner_uids = get_rewards(self, submissions)

            if len(rewards_array) > 0:
                # Split rewards_array back into rewards and miner_uids
                rewards = rewards_array.tolist()

                # Log and update scores based on the rewards
                bt.logging.info(f"Scored responses array returned:{rewards_array}")
                bt.logging.info(f"Rewarded miner ids array returned:{rewarded_miner_uids}")
                self.update_scores(rewards, rewarded_miner_uids)
            else:
                bt.logging.info(f"No rewards to process. Skipping scoring.")

            # Add the match ID and miner UID to the global set of sent game IDs
            #self.sent_game_ids.add(game_key)   
    
    #bt.logging.info(f"Unprocessed Submissions: {len(submissions.submissions)}")
    #bt.logging.info(f"Submissions: {submissions}")
    
    # TODO remove code for rapdily cycling through days to test
    #current_fetch_date += timedelta(days=1)
    bt.logging.info("All matches processed for date: " + (current_fetch_date - timedelta(days=1)).strftime('%Y-%m-%d'))
    #keep_validators_alive(self)
    
    conn.close()
    #bt.logging.info("Next fetch date set to: " + current_fetch_date.strftime('%Y-%m-%d'))


# TODO Create a class to store submissions & check for dupes etc
# class MinerSubmissions:
#     def __init__(self):
#         self.submissions = {}  # Dictionary to store submissions by match_id and miner_uid
    
#     def __str__(self):
#         submissions_str = []
#         for match_id, miners in self.submissions.items():
#             for miner_uid, submission in miners.items():
#                 submissions_str.append(f"Match ID: {match_id}, Miner UID: {miner_uid}, Submission: {submission}")
#         return "\n".join(submissions_str)

#     def insert(self, match_id, miner_uid, submission):
#         if match_id not in self.submissions:
#             self.submissions[match_id] = {}
#         self.submissions[match_id][miner_uid] = submission

#     def get(self, match_id, miner_uid):
#         miner_submissions = self.submissions.get(match_id, {})
#         return miner_submissions.get(miner_uid)  # Return the submission if found, None otherwise
    
#     def get_all_for_match(self, match_id):
#         """Retrieve all submissions for a given match ID."""
#         return self.submissions.get(match_id, {})      