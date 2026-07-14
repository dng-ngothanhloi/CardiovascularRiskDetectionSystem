#!/bin/bash
# Helper script to run make commands from any directory
# Usage: ./make.sh [make_target]

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to project root directory
cd "$SCRIPT_DIR" || exit 1

# Check if Makefile exists
if [ ! -f "Makefile" ]; then
    echo "Error: Makefile not found in project root: $SCRIPT_DIR"
    exit 1
fi

# Run make with all arguments
make "$@"

