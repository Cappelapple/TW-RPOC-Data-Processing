import os
import sys

# Make `cell_kinetics` importable regardless of which directory pytest is
# invoked from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
