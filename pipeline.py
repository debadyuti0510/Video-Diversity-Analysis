
import sys
import time
from utils import setup_experiment, run_experiment
from visualization_utils import generate_visualizations, generate_csv

VALID_ARGS = ["-v", "-V", "-d", "-D"]

if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in VALID_ARGS:
        print("You need to pass a flag after the file path!")
        exit()
    # Start timing
    start_time = time.time()
    # Extract demography data
    setup_experiment()
    run_experiment(sys.argv[2])
    # End time
    end_time = time.time()
    time_taken = (end_time - start_time) / 60 # In minutes
    print(f"Took {time_taken:.3f} mins to run pipeline.")
    # Check if visualization flag is used
    if sys.argv[1] in ["-v", "-V"]:
        generate_visualizations()
    # Check if csv flag is used
    elif sys.argv[1] in ["-d", "-D"]:
        generate_csv()
    







