
# CONSTANTS

# How often to send predictions to the app, measured in steps
APP_SYNC_IN_BLOCKS = 5

# How often to keep validators setting weights, measured in steps
VALIDATOR_SET_WEIGHTS_IN_BLOCKS = 200

# Number of minutes before kickoff to fetch upcoming matches
MINUTES_BEFORE_KICKOFF = 60

# App Prediction API URL - where predictions are fetched and sent
#SCORE_PREDICT_API_URL = "https://app.scorepredict.io/"
SCORE_PREDICT_API_URL = "http://localhost:3000"

# Reward for a response
REWARD_FOR_RESPONSE = 0.1

# Score Match API URL - where games are fetched from
#SCORE_MATCH_API = "http://api.scorepredict.io"
SCORE_MATCH_API = "http://localhost:5001"