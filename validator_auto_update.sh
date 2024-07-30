#!/bin/bash

PM2_PROCESS_NAME=$1

while true; do
  sleep 5

  VERSION=$(git rev-parse HEAD)

  git pull --rebase --autostash

  NEW_VERSION=$(git rev-parse HEAD)

  if [ $VERSION != $NEW_VERSION ]; then
    pip install -r requirements.txt
    pm2 restart $PM2_PROCESS_NAME
  fi
done