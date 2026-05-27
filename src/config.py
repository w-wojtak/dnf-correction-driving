# config.py
"""
Configuration for hybrid DNF-LLM correction system.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# ============================================================================
# API Configuration
# ============================================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)  # Get from environment or .env file
USE_API = GROQ_API_KEY is not None

# ============================================================================
# Path Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results_driving_correction"
LEARNING_DIR = PROJECT_ROOT / "results_driving_learning"

# Create directories if they don't exist
RESULTS_DIR.mkdir(exist_ok=True)
LEARNING_DIR.mkdir(exist_ok=True)

# ============================================================================
# DNF Parameters (from your existing code)
# ============================================================================

X_LIM = 80
T_LIM = 60
DX = 0.05
DT = 0.05

# Field parameters
KERNEL_PARS_DEST = [1.5, 0.8, 0.1]
KERNEL_PARS_WM = [1.75, 0.5, 0.8]

H_0_WM = -1.0
THETA_WM = 0.8
TAU_H_DEST = 20
THETA_DEST = 1.5

# Feedback field parameters
FEEDBACK_KERNEL_PARS = [2.0, 0.8, 0.05]
FEEDBACK_H_0 = -1.0
FEEDBACK_THETA = 0.5
FEEDBACK_TAU = 1.0
INPUT_DURATION = 50
TRIGGER_STEP = 100

# Correction coupling
EARLY_LATE_COUPLING = 0.5

# ============================================================================
# Experiment Configuration
# ============================================================================

# Default destinations
DEFAULT_DESTINATIONS = {
    'names': ["coffee", "work", "gym", "home"],
    'positions': [-60, -30, 0, 30],
    'colors': ['tab:brown', 'tab:blue', 'tab:green', 'tab:red']
}