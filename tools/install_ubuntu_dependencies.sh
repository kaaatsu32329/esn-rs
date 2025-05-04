#!/bin/bash

set -euox pipefail

sudo apt-get update
sudo apt-get install -y \
    pkg-config \
    libfontconfig1-dev
