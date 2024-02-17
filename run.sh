#!/bin/bash

# Start the Python scripts in the background
poetry run python3 ./scripts/update_input_csv.py &
PID_PYTHON_INPUT=$!
echo "Started update_input_csv.py with PID $PID_PYTHON_INPUT"

poetry run python3 ./scripts/update_output_google_sheet.py &
PID_PYTHON_OUTPUT=$!
echo "Started update_output_google_sheet.py with PID $PID_PYTHON_OUTPUT"

# Build and run the Rust project
cargo build --release
echo "Starting cargo run..."
cargo run &

# Wait for the cargo run process to complete, if ever
# Alternatively, you could remove the wait command if you don't want the script to wait for the cargo process
wait $!

# If you reach this point and want to terminate the Python scripts because cargo run is done (or interrupted),
# you can use the stored PIDs to kill the Python processes
# Uncomment the lines below if you want to automatically stop the Python scripts when the cargo run process ends

# echo "Stopping Python scripts..."
# kill $PID_PYTHON_INPUT $PID_PYTHON_OUTPUT

echo "All tasks completed."
