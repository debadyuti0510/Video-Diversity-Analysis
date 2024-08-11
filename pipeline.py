
import sys
from utils import setup_experiment, run_experiment
from visualization_utils import generate_visualizations

if __name__ == "__main__":
    # Extract demography data
    setup_experiment()
    run_experiment(sys.argv[1])
    generate_visualizations()







