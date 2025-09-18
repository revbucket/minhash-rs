import datetime
import subprocess
import sys
import threading
import time

import matplotlib.pyplot as plt
import psutil


def profile(command, interval_ms):
    # Command to run (replace with your actual command)

    # Lists to store the data
    timestamps = []
    memory_percent = []
    memory_used_gb = []
    memory_available_gb = []

    # Flag to control monitoring
    monitoring = True

    def monitor_memory():
        print("Starting memory monitoring...")
        while monitoring:
            # Get memory stats
            memory = psutil.virtual_memory()

            # Record timestamp and memory values
            timestamps.append(datetime.datetime.now())
            memory_percent.append(memory.percent)
            memory_used_gb.append(memory.used / (1024**2))  # Convert bytes to MB
            memory_available_gb.append(
                memory.available / (1024**2)
            )  # Convert bytes to MB

            # Sample every 0.5 seconds for more granular data
            time.sleep(interval_ms / 1000)

    # Start the monitoring thread
    monitor_thread = threading.Thread(target=monitor_memory)
    monitor_thread.start()

    print(f"Running command: {command}")
    start_time = time.time()

    try:
        # Run the command and capture output
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Stream output in real-time (optional)
        for line in process.stdout:
            sys.stdout.write(line)

        # Wait for the command to complete
        process.wait()
        return_code = process.returncode

        # Stop monitoring
        monitoring = False
        monitor_thread.join()

        end_time = time.time()
        duration = end_time - start_time

        print(f"\nCommand completed with return code {return_code}")
        print(f"Execution time: {duration:.2f} seconds")
        print(f"Collected {len(timestamps)} memory samples")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        monitoring = False
        monitor_thread.join()

    except Exception as e:
        print(f"\nError: {e}")
        monitoring = False
        monitor_thread.join()
    return memory_percent, memory_used_gb, memory_available_gb


if __name__ == "__main__":
    cmd = sys.argv[1]
    output_file = sys.argv[2]
    _, used, _ = profile(cmd, 10)
    with open(output_file, "w") as f:
        f.write(json.dumps(used))
