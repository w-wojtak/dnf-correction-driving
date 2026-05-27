# run_experiments.py
"""
Run hybrid correction experiments.
"""

from hybrid_correction import HybridCorrectionSystem, load_and_correct
from config import GROQ_API_KEY, DEFAULT_DESTINATIONS
import numpy as np

# ============================================================================
# Example 1: Quick correction
# ============================================================================

def quick_test():
    """Test with a single command"""
    
    system = load_and_correct(
        "I go to gym before work now",
        driver="A",
        day="monday",
        api_key=GROQ_API_KEY  # From .env file
    )
    
    print("\n✅ Correction applied!")
    return system


# ============================================================================
# Example 2: Multiple corrections
# ============================================================================

def test_multiple_corrections():
    """Test several natural language commands"""
    
    commands = [
        "I don't want coffee anymore",
        "Can I do gym before work?",
        "I need to get to work earlier",
        "Always keep home as my last stop",
    ]
    
    system = load_and_correct(commands[0], api_key=GROQ_API_KEY)
    
    for cmd in commands[1:]:
        system.correct_natural_language(cmd)
    
    system.save_state(iteration=1)
    print("\n✅ All corrections applied and saved!")
    
    return system


# ============================================================================
# Example 3: Locality experiment
# ============================================================================

def locality_experiment():
    """
    Test that corrections don't affect unrelated destinations.
    This is the publishable result!
    """
    
    system = load_and_correct(
        "dummy",  # Just to load
        api_key=GROQ_API_KEY
    )
    
    # Record state
    coffee_idx = np.argmin(np.abs(system.x - system.destination_positions[0]))
    home_idx = np.argmin(np.abs(system.x - system.destination_positions[-1]))
    
    coffee_before = system.u_dest_memory[coffee_idx]
    home_before = system.u_dest_memory[home_idx]
    
    # Apply correction in the middle
    system.correct_natural_language("I do gym before work now")
    
    # Check if coffee and home changed
    coffee_after = system.u_dest_memory[coffee_idx]
    home_after = system.u_dest_memory[home_idx]
    
    coffee_change = abs(coffee_after - coffee_before)
    home_change = abs(home_after - home_before)
    
    print(f"\n🔬 LOCALITY TEST RESULTS:")
    print(f"   Coffee change: {coffee_change:.15f}")
    print(f"   Home change:   {home_change:.15f}")
    
    # SKIP THIS???
    if coffee_change < 1e-10 and home_change < 1e-10:
        print(f"\n   ✅ LOCALITY GUARANTEED!")
    else:
        print(f"\n   ❌ Locality violated")
    
    return coffee_change, home_change


if __name__ == "__main__":
    print("="*70)
    print("HYBRID DNF-LLM CORRECTION EXPERIMENTS")
    print("="*70)
    
    # Run quick test
    quick_test()
    
    # Uncomment to run other experiments:
    # test_multiple_corrections()
    # locality_experiment()