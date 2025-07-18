#!/bin/bash

# Cleaning up Mininet
echo "Cleaning up Mininet..."
sudo mn -c

# Stopping all Docker containers
echo "Stopping all running Docker containers..."
docker stop $(docker ps -aq)

# Removing all stopped Docker containers
echo "Removing all stopped Docker containers..."
docker container prune -f

echo "Cleanup complete."
