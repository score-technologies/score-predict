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

from typing import Optional, List
from datetime import datetime
import bittensor as bt
from pydantic import BaseModel, Field, validator

class Prediction(bt.Synapse, BaseModel):
    """
    A protocol representation for handling football match predictions, using Pydantic for data validation.

    Attributes:
    - match_id: Unique identifier for the match.
    - home_team: Name of the home team.
    - away_team: Name of the away team.
    - match_date: Date and time of the match.
    - deadline: Deadline for submitting predictions.
    - predicted_winner: Predicted winning team (optional).
    - predicted_score_home: Predicted number of goals by the home team (optional).
    - predicted_score_away: Predicted number of goals by the away team (optional).
    """

    match_id: int = Field(..., title="Match ID", description="Unique identifier for the football match")
    home_team: str = Field(..., title="Home Team", description="Name of the home team")
    away_team: str = Field(..., title="Away Team", description="Name of the away team")
    match_date: str = Field(..., title="Match Date", description="Date and time of the match (ISO 8601 string format)")
    deadline: str = Field(..., title="Prediction Deadline", description="Deadline for submitting predictions (ISO 8601 string format)")
    predicted_winner: Optional[str] = Field(None, title="Predicted Winner", description="Predicted winning team")
    

    def deserialize(self) -> dict:
        """
        Deserialize the prediction data. This method retrieves the prediction details from
        the miner and returns them as a dictionary.

        Returns:
        - dict: The deserialized prediction data.
        """
        return {
            'match_id': self.match_id,
            'home_team': self.home_team,
            'away_team': self.away_team,
            'match_date': self.match_date,
            'deadline': self.deadline,
            'predicted_winner': self.predicted_winner,
        }