#!/bin/bash

# Updating package lists
apt-get update -y

# Installing useful tools
apt-get install -y --no-install-recommends \
    bash \
    python3 \
    bash-completion \
    curl \
    net-tools

# Installing the packages required for streaming videos and dumping traffic.
apt-get install -y \
    ffmpeg \
    tcpdump \
    nano

