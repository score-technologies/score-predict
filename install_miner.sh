#!/bin/bash
set -euxo pipefail

if [ $# -ne 2 ]
then
  >&2 echo "USAGE: ./install_miner.sh SSH_DESTINATION HOTKEY_PATH"
  exit 1
fi

SSH_DESTINATION="$1"
LOCAL_HOTKEY_PATH=$(realpath "$2")
LOCAL_COLDKEY_PUB_PATH=$(dirname "$(dirname "$LOCAL_HOTKEY_PATH")")/coldkeypub.txt

if [ ! -f "$LOCAL_HOTKEY_PATH" ]; then
  >&2 echo "Given HOTKEY_PATH does not exist"
  exit 1
fi

HOTKEY_NAME=$(basename "$LOCAL_HOTKEY_PATH")
WALLET_NAME=$(basename "$(dirname "$(dirname "$LOCAL_HOTKEY_PATH")")")

# set default names if they contain special characters
[[ $HOTKEY_NAME =~ ['$#!;*?&()<>'\"\'] ]] && HOTKEY_NAME=default
[[ $WALLET_NAME =~ ['$#!;*?&()<>'\"\'] ]] && WALLET_NAME=mywallet

REMOTE_HOTKEY_PATH=".bittensor/wallets/$WALLET_NAME/hotkeys/$HOTKEY_NAME"
REMOTE_COLDKEY_PUB_PATH=".bittensor/wallets/$WALLET_NAME/coldkeypub.txt"
REMOTE_HOTKEY_DIR=$(dirname "$REMOTE_HOTKEY_PATH")

# Copy the wallet files to the server
ssh "$SSH_DESTINATION" "mkdir -p $REMOTE_HOTKEY_DIR"
scp "$LOCAL_HOTKEY_PATH" "$SSH_DESTINATION:$REMOTE_HOTKEY_PATH"
scp "$LOCAL_COLDKEY_PUB_PATH" "$SSH_DESTINATION:$REMOTE_COLDKEY_PUB_PATH"

# Set PYTHONPATH locally, ensuring it's always defined
export PYTHONPATH="${HOME}/score-predict${PYTHONPATH:+:$PYTHONPATH}"

# Install necessary software and set up the miner
ssh "$SSH_DESTINATION" <<ENDSSH
set -euxo pipefail

WALLET_NAME="$WALLET_NAME"
HOTKEY_NAME="$HOTKEY_NAME"
REMOTE_PYTHONPATH="\$HOME/score-predict\${PYTHONPATH:+:\$PYTHONPATH}"

# Update and install dependencies
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv git

# Clone the repository
git clone https://github.com/score-protocol/score-predict.git
cd score-predict

# Set up virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Manual patch for bittensor
cd ..
git clone https://github.com/opentensor/bittensor.git
cd bittensor
git checkout release/7.2.1
pip install -e .
cd ../score-predict

# Set up PYTHONPATH
echo "export PYTHONPATH=\"\$REMOTE_PYTHONPATH\"" >> ~/.bashrc
source ~/.bashrc

# Install pm2
sudo apt-get install -y nodejs npm
sudo npm install pm2 -g

# Create a startup script
cat > start_miner.sh <<EOF
#!/bin/bash
source \$HOME/score-predict/venv/bin/activate
export PYTHONPATH="\$REMOTE_PYTHONPATH"
python neurons/miner.py --netuid 180 --subtensor.network test --wallet.name $WALLET_NAME --wallet.hotkey $HOTKEY_NAME --axon.port 8089
EOF

chmod +x start_miner.sh

# Start the miner using the startup script
pm2 start ./start_miner.sh --name miner

# Save the pm2 configuration
pm2 save

# Set up pm2 to start on boot
pm2 startup
ENDSSH

echo "Miner installed and started successfully!"