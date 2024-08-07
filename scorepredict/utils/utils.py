# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 philanthrope

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

import os
import math
import time
import torch
import functools
import numpy as np
import random as pyrandom
import sqlite3

from Crypto.Random import random
from itertools import combinations, cycle
from typing import List, Union

import bittensor as bt
import datetime
import requests
import pandas as pd

from dotenv import load_dotenv

load_dotenv()

# Global variable to store the simulated current time
simulated_current_time = datetime.datetime.utcnow()

def send_predictions_to_website(self):
    """
    Checks for predictions in the database, sends them to the website API,
    and updates the 'sentWebsite' field in the database.
    """

    bt.logging.debug(f"Netuid: {self.config.netuid}")

        # Check if running on mainnet
    if self.config.netuid != 44:
        bt.logging.info("Not running on mainnet. Skipping sending predictions to website.")
        return


    db_name = f'predictions-{self.uid}.db'
    conn = sqlite3.connect(db_name)
    c = conn.cursor()

    # Create the predictions table if it doesn't exist
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            miner_uid INTEGER,
            match_id INTEGER,
            prediction TEXT,
            timestamp DATETIME,
            reward REAL,
            sentWebsite INTEGER DEFAULT 0
        )
    ''')
    conn.commit()

    # Fetch unsent predictions
    c.execute('''
        SELECT miner_uid, match_id, prediction, timestamp, reward
        FROM predictions
        WHERE sentWebsite = 0
    ''')
    unsent_predictions = c.fetchall()

    api_url = "https://app.scorepredict.io/api/predictions" 

    for miner_uid, match_id, prediction, timestamp, reward in unsent_predictions:
        payload = {
            "matchId": int(match_id),
            "userId": str(miner_uid),
            "prediction": prediction,
            "timestamp": timestamp,
            "reward": reward,
            "userType": "miner"
        }

        try:
            response = requests.post(api_url, json=payload)
            print(f"Sent payload: {payload}")  # Log the sent data
            print(f"Response status: {response.status_code}")
            print(f"Response content: {response.text}")  # Log the response content

            if response.status_code == 201:
                # Update the sentWebsite field
                c.execute('''
                    UPDATE predictions
                    SET sentWebsite = 1
                    WHERE miner_uid = ? AND match_id = ?
                ''', (miner_uid, match_id))
                conn.commit()
                bt.logging.debug(f"Prediction for match {match_id} by miner {miner_uid} sent successfully.")
            else:
                bt.logging.error(f"Failed to send prediction. Status: {response.status_code}, Content: {response.text}")
        except requests.RequestException as e:
            bt.logging.error(f"Error sending prediction to API: {e}")
            continue  # Skip to the next prediction if an error occurs

    conn.close()

def get_current_time(self):
    if self.config.simulate_time:
        return simulated_current_time
    else:
        return datetime.datetime.utcnow()

def set_simulated_time(new_time):
    global simulated_current_time
    simulated_current_time = new_time

def advance_time(self, minutes):
    global simulated_current_time
    if self.config.simulate_time:
        simulated_current_time += datetime.timedelta(minutes=minutes)

def check_uid_availability(
    metagraph: "bt.metagraph.Metagraph", uid: int, vpermit_tao_limit: int
) -> bool:
    """Check if uid is available. The UID should be available if it is serving and has less than vpermit_tao_limit stake
    Args:
        metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
        uid (int): uid to be checked
        vpermit_tao_limit (int): Validator permit tao limit
    Returns:
        bool: True if uid is available, False otherwise
    """
    # Filter validator permit > 1024 stake.
    if metagraph.validator_permit[uid]:
        if metagraph.S[uid] > vpermit_tao_limit:
            return False
    # Disallow non-serving miner axons by letting them fail early. (Excludes validators)
    if metagraph.axons[uid].ip == '0.0.0.0':
        return True
    # Available otherwise.
    return True


def current_block_hash(self):
    """
    Get the current block hash with caching.

    Args:
        subtensor (bittensor.subtensor.Subtensor): The subtensor instance to use for getting the current block hash.

    Returns:
        str: The current block hash.
    """
    try:
        block_hash: str = self.subtensor.get_block_hash(self.subtensor.get_current_block())
        if block_hash is not None:
            return block_hash
    except Exception as e:
        bt.logging.warning(f"Failed to get block hash: {e}. Returning a random hash value.")
    return str(random.randint(2 << 32, 2 << 64))


def get_block_seed(self):
    """
    Get the block seed for the current block.

    Args:
        subtensor (bittensor.subtensor.Subtensor): The subtensor instance to use for getting the block seed.

    Returns:
        int: The block seed.
    """
    block_hash = current_block_hash(self)
    bt.logging.trace(f"block hash in get_block_seed: {block_hash}")
    return int(block_hash, 16)


def get_pseudorandom_uids(self, uids, k):
    """
    Get a list of pseudorandom uids from the given list of uids.

    Args:
        subtensor (bittensor.subtensor.Subtensor): The subtensor instance to use for getting the block_seed.
        uids (list): The list of uids to generate pseudorandom uids from.

    Returns:
        list: A list of pseudorandom uids.
    """
    block_seed = get_block_seed(self)
    pyrandom.seed(block_seed)

    # Ensure k is not larger than the number of uids
    k = min(k, len(uids))

    sampled = pyrandom.sample(uids, k=k)
    bt.logging.debug(f"get_pseudorandom_uids() sampled: {k} | {sampled}")
    return sampled


def get_available_uids(self, exclude: list = None):
    """Returns all available uids from the metagraph.

    Returns:
        uids (torch.LongTensor): All available uids.
    """
    avail_uids = []

    for uid in range(self.metagraph.n.item()):
        uid_is_available = check_uid_availability(
            self.metagraph, uid, self.config.neuron.vpermit_tao_limit
        )
        if uid_is_available and (exclude is None or uid not in exclude):
            avail_uids.append(uid)
    bt.logging.debug(f"returning available uids: {avail_uids}")
    return avail_uids


# TODO: update this to use the block hash seed paradigm so that we don't get uids that are unavailable
def get_random_uids(
    self, k: int, exclude: List[int] = None, seed: int = None
) -> torch.LongTensor:
    """Returns k available random uids from the metagraph.
    Args:
        k (int): Number of uids to return.
        exclude (List[int]): List of uids to exclude from the random sampling.
    Returns:
        uids (torch.LongTensor): Randomly sampled available uids.
    Notes:
        If `k` is larger than the number of available `uids`, set `k` to the number of available `uids`.
    """
    candidate_uids = []
    avail_uids = []

    for uid in range(self.metagraph.n.item()):
        uid_is_available = check_uid_availability(
            self.metagraph, uid, self.config.neuron.vpermit_tao_limit
        )
        uid_is_not_excluded = exclude is None or uid not in exclude

        if uid_is_available and uid_is_not_excluded:
            candidate_uids.append(uid)
        elif uid_is_available:
            avail_uids.append(uid)

    # If not enough candidate_uids, supplement from avail_uids, ensuring they're not in exclude list
    if len(candidate_uids) < k:
        additional_uids_needed = k - len(candidate_uids)
        filtered_avail_uids = [uid for uid in avail_uids if uid not in exclude]
        additional_uids = random.sample(
            filtered_avail_uids, min(additional_uids_needed, len(filtered_avail_uids))
        )
        candidate_uids.extend(additional_uids)

    # Safeguard against trying to sample more than what is available
    num_to_sample = min(k, len(candidate_uids))
    if seed:  # use block hash seed if provided
        random.seed(seed)
    uids = random.sample(candidate_uids, num_to_sample)
    bt.logging.debug(f"returning available uids: {uids}")
    return uids

def get_current_epoch(self):
    """
    Calculate the current epoch based on the current block number.
    An epoch is defined as the ceiling of the division of the current block number by 3600.
    This results in epochs changing roughly twice a day.

    Returns:
        int: The current epoch number.
    """
    try:
        current_block = self.subtensor.get_current_block()
        current_epoch = math.ceil(current_block / 3600)
        bt.logging.info(f"Current block: {current_block}, Current epoch: {current_epoch}")
        return current_epoch
    except Exception as e:
        bt.logging.error(f"Failed to calculate current epoch: {e}")
        return None

def get_validators_and_shares(self, vpermit_tao_limit: int, vtrust_threshold: float = 0.0):
    """
    Retrieves the UIDs of all validators in the network and their share of the total stake.
    Qualifications for validator peers:
        - validator permit
        - stake > vpermit_tao_limit
        - validator trust score > vtrust_threshold
    
    Returns:
        Dict[int, float]: A dictionary mapping each validator's UID to their share of the total stake.
    """
    # Ensure vpermits is a torch.Tensor
    vpermits = torch.tensor(self.metagraph.validator_permit)
    vpermit_uids = torch.where(vpermits)[0]
    
    # Convert S and validator_trust to torch.Tensor
    S_tensor = torch.tensor(self.metagraph.S)
    vtrust_tensor = torch.tensor(self.metagraph.validator_trust)
    
    # Filter UIDs based on stake and vtrust
    query_idxs = torch.where(
        (S_tensor[vpermit_uids] > vpermit_tao_limit) &
        (vtrust_tensor[vpermit_uids] >= vtrust_threshold)
    )[0]
    validator_uids = vpermit_uids[query_idxs].tolist()
    
    # Calculate total stake of selected validators
    total_stake = sum(S_tensor[validator_uids].tolist())
    
    # Calculate each validator's share
    validator_shares = {uid: (S_tensor[uid] / total_stake).item() for uid in validator_uids}

    bt.logging.info(f"Validator UIDs: {validator_uids}")
    bt.logging.info(f"Total stake: {total_stake}")
    bt.logging.info(f"Validator shares: {validator_shares}")
    
    return validator_shares



def get_matches(self, date_str, status: str = None, minutes_before_kickoff: int = 60):
    """
    Fetches upcoming matches from a football data API and returns them as a dictionary.
    Each match is keyed by its match ID, making it easy to reference specific matches.

    Returns:
        dict or None: A dictionary where each key is a match ID and each value is a dictionary of match details, or None if no matches are found.

    Raises:
        Exception: If the API call fails or returns an unexpected response.
    """
    API_KEY = os.getenv('FOOTBALL_API_KEY')
    if self.config.score_api == True:
        url = 'http://api.scorepredict.io/matches'
        bt.logging.debug("Using score API")
    else:
        url = 'https://api.football-data.org/v4/matches'
    headers = {'X-Auth-Token': API_KEY}
    
    params = {
        'dateFrom': date_str.strftime('%Y-%m-%d'),
        'dateTo': date_str.strftime('%Y-%m-%d'),
        'status': status
    }

    bt.logging.debug(f"Params: {params}")
    bt.logging.debug(f"date_str: {date_str}")
    
    while True:
        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                break
            else:
                bt.logging.error(f"Failed to fetch matches: HTTP {response.status_code}")
        except Exception as e:
            bt.logging.error(f"API call failed: {e}")
        
        bt.logging.info("Retrying in 5 seconds...")
        time.sleep(5)

    data = response.json()
    matches = data.get('matches', [])
    bt.logging.debug(f"Matches Found: {matches}")
    
    if status == 'FINISHED':
        return {match['id']: match for match in matches}
    
    # Filter matches to be between 0 and specified minutes before their kickoff time
    upcoming_matches = {}
    for match in matches:
        match_time = datetime.datetime.fromisoformat(match['utcDate'].rstrip('Z'))
        
        # Convert date_str to datetime object if it's not already
        if isinstance(date_str, str):
            current_time = datetime.datetime.fromisoformat(date_str)
        else:
            current_time = date_str

        # Calculate time difference in minutes
        time_difference = (match_time - current_time).total_seconds() / 60

        bt.logging.debug(f"Time difference: {time_difference}")

        if 0 <= time_difference <= minutes_before_kickoff:
            upcoming_matches[match['id']] = match

    bt.logging.debug(f"Fetched {len(upcoming_matches)} upcoming matches within {minutes_before_kickoff} minutes before kickoff.")
    
    # Return None if no upcoming matches are found
    return upcoming_matches if upcoming_matches else None

def assign_challenges_to_validators(self, minutes_before_kickoff: int = 60):
    """
    Assigns miners to validators based on validators' stake, then assigns all matches to these miners.
    The assignment is deterministic and consistent across all validators for a given epoch.
    
    Returns:
        dict: A dictionary mapping miner UIDs to their assigned matches for the current validator.
    """
    # Retrieve validator UIDs and their stake amounts
    validator_shares = get_validators_and_shares(self, self.config.neuron.vpermit_tao_limit)
    
    if not validator_shares:
        bt.logging.error("No validators available to assign challenges.")
        return {}

    # Sort validators based on stake amounts in descending order
    sorted_validators = sorted(validator_shares.items(), key=lambda x: x[1], reverse=True)
    
    # Retrieve upcoming matches for today
    target_date = get_current_time(self)
    matches_dict = get_matches(self, date_str=target_date, minutes_before_kickoff=minutes_before_kickoff)

    if not matches_dict:
        bt.logging.debug("No upcoming matches found to assign to validators.")
        return {}

    # Get all miner UIDs
    miner_uids = get_all_miners(self)
    bt.logging.info(f"⛏️ Miner UIDs: {miner_uids}")
    
    # Create a deterministic ordering of miners
    block_hash = get_current_epoch(self)
    miner_seed = f"{block_hash}_miners"
    pyrandom.seed(miner_seed)
    shuffled_miners = miner_uids.copy()
    pyrandom.shuffle(shuffled_miners)

    # Assign miners to validators deterministically
    validator_miners = {}
    miner_index = 0
    total_stake = sum(stake for _, stake in sorted_validators)
    
    for validator_uid, stake in sorted_validators:
        # Calculate number of miners for this validator based on stake
        num_miners = max(1, int((stake / total_stake) * len(shuffled_miners)))
        validator_miners[validator_uid] = []
        
        for _ in range(num_miners):
            if miner_index < len(shuffled_miners):
                validator_miners[validator_uid].append(shuffled_miners[miner_index])
                miner_index += 1
            else:
                break  # All miners have been assigned

    # Assign all matches to each validator's miners
    validator_challenges = {}
    for validator_uid, assigned_miners in validator_miners.items():
        validator_challenges[validator_uid] = {
            miner_uid: list(matches_dict.items()) for miner_uid in assigned_miners
        }
    
    bt.logging.debug(f"Validator challenges: {validator_challenges}")
    bt.logging.debug(f"My UID: {self.uid}")
    
    # Return challenges for the current validator
    return validator_challenges.get(self.uid, {})

def get_all_validators_vtrust(
    self,
    vpermit_tao_limit: int,
    vtrust_threshold: float = 0.0,
    return_hotkeys: bool = False,
):
    """
    Retrieves the hotkeys of all validators in the network. This method is used to
    identify the validators and their corresponding hotkeys, which are essential
    for various network operations, including blacklisting and peer validation.
    Qualifications for validator peers:
        - stake > threshold (e.g. 500, may vary per subnet)
        - validator permit (implied with vtrust score)
        - validator trust score > threshold (e.g. 0.5)
    Returns:
        List[str]: A list of hotkeys corresponding to all the validators in the network.
    """
    vtrusted_uids = [
        uid for uid in torch.where(self.metagraph.validator_trust > vtrust_threshold)[0]
    ]
    stake_uids = [
        uid for uid in vtrusted_uids if self.metagraph.S[uid] > vpermit_tao_limit
    ]
    return (
        [self.metagraph.hotkeys[uid] for uid in stake_uids]
        if return_hotkeys
        else stake_uids
    )


def get_all_validators(self, return_hotkeys=False):
    """
    Retrieve all validator UIDs from the metagraph. Optionally, return their hotkeys instead.
    Args:
        return_hotkeys (bool): If True, returns the hotkeys of the validators; otherwise, returns the UIDs.

    Returns:
        list: A list of validator UIDs or hotkeys, depending on the value of return_hotkeys.
    """
    # Ensure vpermits is a torch.Tensor
    vpermits = torch.tensor(self.metagraph.validator_permit)  # Convert to PyTorch tensor if not already
    vpermit_uids = torch.where(vpermits)[0]
    
    # Convert self.metagraph.S to a torch.Tensor
    S_tensor = torch.tensor(self.metagraph.S)
    
    query_idxs = torch.where(
        S_tensor[vpermit_uids] > self.config.neuron.vpermit_tao_limit
    )[0]
    query_uids = vpermit_uids[query_idxs].tolist()

    return (
        [self.metagraph.hotkeys[uid] for uid in query_uids]
        if return_hotkeys
        else query_uids
    )

def get_all_miners(self):
    """
    Retrieve all miner UIDs from the metagraph, excluding those that are validators.

    Returns:
        list: A list of UIDs of miners.
    """
    # Determine miner axons to query from metagraph
    vuids = get_all_validators(self)
    return [uid for uid in self.metagraph.uids.tolist() if uid not in vuids]







######


def get_query_miners(self, k=20, exlucde=None):
    """
    Obtain a list of miner UIDs selected pseudorandomly based on the current block hash.

    Args:
        k (int): The number of miner UIDs to retrieve.

    Returns:
        list: A list of pseudorandomly selected miner UIDs.
    """
    # Determine miner axons to query from metagraph with pseudorandom block_hash seed
    muids = get_all_miners(self)
    if exlucde is not None:
        muids = [muid for muid in muids if muid not in exlucde]
    return get_pseudorandom_uids(self, muids, k=k)


def get_query_validators(self, k=3):
    """
    Obtain a list of available validator UIDs selected pseudorandomly based on the current block hash.

    Args:
        k (int): The number of available miner UIDs to retreive.

    Returns:
        list: A list of pseudorandomly selected available validator UIDs
    """
    vuids = get_all_validators(self)
    return get_pseudorandom_uids(self, uids=vuids, k=k)


async def get_available_query_miners(
    self, k, exclude: List[int] = None, exclude_full: bool = False
):
    """
    Obtain a list of available miner UIDs selected pseudorandomly based on the current block hash.

    Args:
        k (int): The number of available miner UIDs to retrieve.

    Returns:
        list: A list of pseudorandomly selected available miner UIDs.
    """
    # Determine miner axons to query from metagraph with pseudorandom block_hash seed
    muids = get_available_uids(self, exclude=exclude)
    bt.logging.debug(f"get_available_query_miners() available uids: {muids}")
    if exclude_full:
        muids_nonfull = [
            uid
            for uid in muids
        ]
        bt.logging.debug(f"available uids nonfull: {muids_nonfull}")
    return get_pseudorandom_uids(self, muids, k=k)


def get_current_validator_uid_pseudorandom(self):
    """
    Retrieve a single validator UID selected pseudorandomly based on the current block hash.

    Returns:
        int: A pseudorandomly selected validator UID.
    """
    block_seed = get_block_seed(self)
    pyrandom.seed(block_seed)
    vuids = get_query_validators(self)
    return pyrandom.choice(vuids)


def get_current_validtor_uid_round_robin(self):
    """
    Retrieve a validator UID using a round-robin selection based on the current block and epoch length.

    Returns:
        int: The UID of the validator selected via round-robin.
    """
    vuids = get_all_validators(self)
    vidx = self.subtensor.get_current_block() // 100 % len(vuids)
    return vuids[vidx]


def generate_efficient_combinations(available_uids, R):
    """
    Generates all possible combinations of UIDs for a given redundancy factor.

    Args:
        available_uids (list): A list of UIDs that are available for storing data.
        R (int): The redundancy factor specifying the number of UIDs to be used for each chunk of data.

    Returns:
        list: A list of tuples, where each tuple contains a combination of UIDs.

    Raises:
        ValueError: If the redundancy factor is greater than the number of available UIDs.
    """

    if R > len(available_uids):
        raise ValueError(
            "Redundancy factor cannot be greater than the number of available UIDs."
        )

    # Generate all combinations of available UIDs for the redundancy factor
    uid_combinations = list(combinations(available_uids, R))

    return uid_combinations


def assign_combinations_to_hashes_by_block_hash(self, hashes, combinations):
    """
    Assigns combinations of UIDs to each data chunk hash based on a pseudorandom seed derived from the blockchain's current block hash.

    Args:
        subtensor: The subtensor instance used to obtain the current block hash for pseudorandom seed generation.
        hashes (list): A list of hashes, where each hash represents a unique data chunk.
        combinations (list): A list of UID combinations, where each combination is a tuple of UIDs.

    Returns:
        dict: A dictionary mapping each chunk hash to a pseudorandomly selected combination of UIDs.

    Raises:
        ValueError: If there are not enough unique UID combinations for the number of data chunk hashes.
    """

    if len(hashes) > len(combinations):
        raise ValueError(
            "Not enough unique UID combinations for the given redundancy factor and number of hashes."
        )
    block_seed = get_block_seed(self)
    pyrandom.seed(block_seed)

    # Shuffle once and then iterate in order for assignment
    pyrandom.shuffle(combinations)
    return {hash_val: combinations[i] for i, hash_val in enumerate(hashes)}


def assign_combinations_to_hashes(hashes, combinations):
    """
    Assigns combinations of UIDs to each data chunk hash in a pseudorandom manner.

    Args:
        hashes (list): A list of hashes, where each hash represents a unique data chunk.
        combinations (list): A list of UID combinations, where each combination is a tuple of UIDs.

    Returns:
        dict: A dictionary mapping each chunk hash to a pseudorandomly selected combination of UIDs.

    Raises:
        ValueError: If there are not enough unique UID combinations for the number of data chunk hashes.
    """

    if len(hashes) > len(combinations):
        raise ValueError(
            "Not enough unique UID combinations for the given redundancy factor and number of hashes."
        )

    # Shuffle once and then iterate in order for assignment
    pyrandom.shuffle(combinations)
    return {hash_val: combinations[i] for i, hash_val in enumerate(hashes)}


