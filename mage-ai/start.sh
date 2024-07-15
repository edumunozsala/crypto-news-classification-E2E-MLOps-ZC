#!/bin/bash

PROJECT_NAME=machine_learning \
  MAGE_CODE_PATH=/home/src \
  MLFLOW_PORT=5005 \
  SMTP_EMAIL=$SMTP_EMAIL \
  SMTP_PASSWORD=$SMTP_PASSWORD \
  docker compose up --build
