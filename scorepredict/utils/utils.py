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

from Crypto.Random import random
from itertools import combinations, cycle
from typing import List, Union

import bittensor as bt
import datetime
import requests
import pandas as pd

from dotenv import load_dotenv

load_dotenv()


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
    An epoch is defined as the ceiling of the division of the current block number by 360.

    Returns:
        int: The current epoch number.
    """
    try:
        current_block = self.subtensor.get_current_block()  
        # TODO find actual block rather than this method
        current_epoch = math.ceil(current_block / 360)
        bt.logging.info(f"Current block: {current_block}, Current epoch: {current_epoch}")
        return current_epoch
    except Exception as e:
        bt.logging.error(f"Failed to calculate current epoch: {e}")
        return None

def get_validators_and_shares(self, vpermit_tao_limit: int, vtrust_threshold: float = 0.0):
    """
    Retrieves the UIDs of all validators in the network and their share of the total stake.
    Qualifications for validator peers:
        - stake > threshold (e.g., 500, may vary per subnet)
        - validator permit (implied with vtrust score)
        - validator trust score > threshold (e.g., 0.5)
    Returns:
        Dict[int, float]: A dictionary mapping each validator's UID to their share of the total stake.
    """
    # Convert to PyTorch tensor if necessary and apply threshold
    validator_trust_tensor = torch.tensor(self.metagraph.validator_trust)
    vtrusted_indices = torch.where(validator_trust_tensor >= vtrust_threshold)[0].tolist()
    bt.logging.info(f"Vtrusted uids: {vtrusted_indices}")

    # Filter UIDs based on stake and convert indices to actual UIDs
    stake_uids = [uid for uid in vtrusted_indices if self.metagraph.S[uid] > vpermit_tao_limit]
    bt.logging.info(f"Validators uids with minimum required stake: {stake_uids}")

    # Calculate total stake
    total_stake = sum(self.metagraph.S[uid] for uid in stake_uids)
    bt.logging.info(f"Total stake: {total_stake}")

    # Calculate each validator's share
    validator_shares = {uid: self.metagraph.S[uid] / total_stake for uid in stake_uids}
    bt.logging.info(f"Validator shares: {validator_shares}")

    return validator_shares



def get_matches(self, date_str, status: str = None):
    """
    Fetches upcoming matches from a football data API and returns them as a dictionary.
    Each match is keyed by its match ID, making it easy to reference specific matches.

    Returns:
        dict or None: A dictionary where each key is a match ID and each value is a dictionary of match details, or None if no matches are found.

    Raises:
        Exception: If the API call fails or returns an unexpected response.
    """
    API_KEY = os.getenv('FOOTBALL_API_KEY')
    if self.config.test_api == True:
        url = 'http://170.64.240.149:5000/matches'
        bt.logging.info("Using test API")
    else:
        url = 'https://api.football-data.org/v4/matches'
    headers = {'X-Auth-Token': API_KEY}
    
    params = {
        'dateFrom': date_str,
        'dateTo': date_str,
        'status': status
    }
    
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        bt.logging.error(f"Failed to fetch matches: HTTP {response.status_code}")
        raise Exception(f"API call failed with status {response.status_code}")

    data = response.json()
    matches = data.get('matches', [])
    
    # TODO Check for status = FINSIHED and return all games if so
    if status == 'FINISHED':
        return {match['id']: match for match in matches}

    # Current UTC time
    current_utc = datetime.datetime.utcnow()
    bt.logging.info(f"Current UTC time: {current_utc}")
    # Filter matches to be between 60 and 75 minutes before their kickoff time
    upcoming_matches = {}
    for match in matches:
        match_time = datetime.datetime.fromisoformat(match['utcDate'].rstrip('Z'))
        time_difference = (match_time - current_utc).total_seconds() / 60  # Convert seconds to minutes

        if 1020 <= time_difference <= 1300:
            upcoming_matches[match['id']] = match

    bt.logging.info(f"Fetched {len(upcoming_matches)} upcoming matches between 60 and 75 minutes before kickoff.")
    
    # Return None if no upcoming matches are found
    return upcoming_matches if upcoming_matches else None

def assign_challenges_to_validators(self):
    """
    Assigns upcoming matches to validators based on their current stake. Higher stake gets more matches to send to miners.

    The distribution criteria include:
        - Validator's stake: Validators with a higher stake will receive more matches as they have more at risk in the network.
        - Validator's past performance: Validators with a history of accurate predictions and timely responses may be prioritized.

    Steps involved:
        1. Retrieve a list of all available validators with their current stake and performance metrics.
        2. Fetch the list of upcoming matches that need to be assigned.
        3. Distribute matches to validators based on the criteria mentioned above, ensuring an even and fair distribution.

    Returns:
        dict: A dictionary mapping miner UIDs to their assigned challenges for the current validator.

    Raises:
        Exception: If there are no available validators or matches, or if an error occurs during the assignment process.
    """    
    # Retrieve validator UIDs and their stake amounts
    validator_shares = get_validators_and_shares(self, self.config.neuron.vpermit_tao_limit)
    
    # Sort validators based on stake amounts in descending order
    sorted_validators = sorted(validator_shares.items(), key=lambda x: x[1], reverse=True)
    
    # Assign sequence numbers to validators
    validator_sequences = {uid: i for i, (uid, _) in enumerate(sorted_validators)}
    
    # Retrieve upcoming matches for today as a dictionary with match IDs as keys
    target_date = datetime.datetime.utcnow().strftime('%Y-%m-%d')
    matches_dict = get_matches(self, date_str=target_date)
    
    # Check if matches_dict is None or empty
    if not matches_dict:
        bt.logging.info("No upcoming matches found to assign to validators.")
        return {}

    # Get all miner UIDs
    miner_uids = get_all_miners(self)
    bt.logging.info(f"⛏️ Miner UIDs: {miner_uids}")
    # Number of challenges per validator
    challenges_per_validator = len(matches_dict) // len(validator_sequences)
    
    validator_challenges = {}
    
    for validator_uid in validator_sequences:
        miner_challenges = {}
        
        for miner_uid in miner_uids:
            # Get the current block hash
            block_hash = get_current_epoch(self)
            
            # Create a unique seed for the random generator
            seed = f"{block_hash}_{miner_uid}"
            pyrandom.seed(seed)

            # Shuffle the match IDs
            shuffled_match_ids = list(matches_dict.keys())
            pyrandom.shuffle(shuffled_match_ids)
            
            # Assign challenges to the current validator
            sequence = validator_sequences[validator_uid]
            start_index = sequence * challenges_per_validator
            end_index = start_index + challenges_per_validator
            miner_challenges[miner_uid] = [
                (match_id, matches_dict[match_id])
                for match_id in shuffled_match_ids[start_index:end_index]
            ]
        
        validator_challenges[validator_uid] = miner_challenges
    
    bt.logging.info(f"Validator challenges: {validator_challenges}")
    bt.logging.info(f"My UID: {self.uid}")
    
    # Filter challenges for the current validator
    current_validator_challenges = validator_challenges.get(self.uid, {})
    
    return current_validator_challenges


def keep_validators_alive(self):
    """
    Periodically syncs the metagraph and sets weights to keep validators alive and updated.
    """
    current_time = datetime.datetime.now()
    bt.logging.info("Keeping validators busy setting weights...")

    # Perform the sync and weight setting at the top of the hour or just before
    minutes = current_time.minute
    if 15 <= minutes or minutes < 25:
        self.resync_metagraph()
        self.set_weights()
        bt.logging.info("Performed periodic resync and set weights at: " + current_time.strftime('%Y-%m-%d %H:%M:%S'))
        time.sleep(60)  # Sleep for 300 seconds or 5 minutes
    else:
        bt.logging.info("No action needed. Next check in 5 minutes.")
        time.sleep(60)  # Sleep for 300 seconds or 5 minutes


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


