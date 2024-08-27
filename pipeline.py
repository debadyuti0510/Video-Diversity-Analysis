
import sys
import time
from utils import setup_experiment, run_experiment
from visualization_utils import generate_visualizations

if __name__ == "__main__":
    # Start timing
    start_time = time.time()
    # Extract demography data
    setup_experiment()
    run_experiment(sys.argv[1])
    # End time
    end_time = time.time()
    time_taken = (end_time - start_time) / 60 # In minutes
    print(f"Took {time_taken:.3f} mins to run pipeline.")
    generate_visualizations()
    







