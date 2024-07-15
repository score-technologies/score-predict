module.exports = {
    apps: [
      {
        name: 'validator',
        script: 'python3',
        args: './neurons/validator.py --netuid 180 --logging.debug --logging.trace --subtensor.network test --wallet.name validator --wallet.hotkey hotkeyName'
      },
    ],
  };