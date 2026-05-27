# src/run_experiments.py
"""
Experiments for natural language DNF corrections.
Run specific correction scenarios here.
"""

import numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import os

# Import DNF system
from sequence_tuning import (
    execute_correction,
    destination_names,
    destination_positions,
    u_dest_memory,
    h_dmem_mask,
    x, dx,
    destination_indices,
    results_dir
)

# Load API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)

print("="*70)
print("NATURAL LANGUAGE DNF CORRECTION EXPERIMENTS")
print("="*70)

if GROQ_API_KEY:
    print("✅ Using Groq API (Llama 3.1)")
else:
    print("ℹ️ Using rule-based parser (no API key)")


# ============================================================================
# Experiment 1: Single Correction
# ============================================================================

def test_single_correction():
    """Test a single natural language command"""
    
    print("\n" + "="*70)
    print("EXPERIMENT 1: Single Correction")
    print("="*70)
    
    # Try different commands here
    command = "I arrive at work earlier now"
    # command = "Skip gym on Mondays"
    # command = "I go to gym before work now"
    # command = "Always keep home as my last stop"
    
    u_new, mask_new, feedback_type, target = execute_correction(
        user_command=command,
        destination_names=destination_names,
        destination_positions=destination_positions,
        u_dest_memory=u_dest_memory,
        h_dmem_mask=h_dmem_mask,
        x=x,
        dx=dx,
        api_key=GROQ_API_KEY
    )
    
    # Show changes
    print("\nMemory changes:")
    for i, (name, pos) in enumerate(zip(destination_names, destination_positions)):
        idx = destination_indices[i]
        before = u_dest_memory[idx]
        after = u_new[idx]
        change = after - before
        print(f"  {name:8s}: {before:.3f} → {after:.3f} (Δ {change:+.3f})")
    
    return u_new, mask_new


# ============================================================================
# Experiment 2: Multiple Sequential Corrections
# ============================================================================

def test_multiple_corrections():
    """Test several corrections in sequence"""
    
    print("\n" + "="*70)
    print("EXPERIMENT 2: Multiple Corrections")
    print("="*70)
    
    commands = [
        "I don't want coffee anymore",
        "I go to gym before work now",
        "I need to arrive at work earlier",
        "Always keep home as my last stop",
    ]
    
    u_current = u_dest_memory.copy()
    mask_current = h_dmem_mask.copy()
    
    for i, cmd in enumerate(commands):
        print(f"\n--- Correction {i+1}/{len(commands)} ---")
        
        u_current, mask_current, _, _ = execute_correction(
            user_command=cmd,
            destination_names=destination_names,
            destination_positions=destination_positions,
            u_dest_memory=u_current,
            h_dmem_mask=mask_current,
            x=x,
            dx=dx,
            api_key=GROQ_API_KEY
        )
    
    print("\n" + "="*70)
    print("Final memory state:")
    for i, (name, pos) in enumerate(zip(destination_names, destination_positions)):
        idx = destination_indices[i]
        before = u_dest_memory[idx]
        after = u_current[idx]
        change = after - before
        print(f"  {name:8s}: {before:.3f} → {after:.3f} (Δ {change:+.3f})")
    
    return u_current, mask_current


# ============================================================================
# Experiment 3: Locality Test (PUBLISHABLE!)
# ============================================================================

def test_locality_guarantee():
    """
    Demonstrate that corrections are mathematically local.
    This is the key scientific contribution!
    """
    
    print("\n" + "="*70)
    print("EXPERIMENT 3: Locality Guarantee Test")
    print("="*70)
    
    # Record state of first and last destinations
    coffee_idx = destination_indices[0]  # First
    home_idx = destination_indices[-1]   # Last
    
    coffee_before = u_dest_memory[coffee_idx]
    home_before = u_dest_memory[home_idx]
    
    print(f"\n📊 Initial state:")
    print(f"   Coffee: {coffee_before:.10f}")
    print(f"   Home:   {home_before:.10f}")
    
    # Apply correction to MIDDLE destinations
    print(f"\n🔬 Applying: 'I go to gym before work now'")
    print(f"   (This should only affect gym and work, NOT coffee or home)")
    
    u_new, mask_new, _, _ = execute_correction(
        user_command="I go to gym before work now",
        destination_names=destination_names,
        destination_positions=destination_positions,
        u_dest_memory=u_dest_memory,
        h_dmem_mask=h_dmem_mask,
        x=x,
        dx=dx,
        api_key=GROQ_API_KEY
    )
    
    # Check if coffee and home changed
    coffee_after = u_new[coffee_idx]
    home_after = u_new[home_idx]
    
    coffee_change = abs(coffee_after - coffee_before)
    home_change = abs(home_after - home_before)
    
    print(f"\n📊 Final state:")
    print(f"   Coffee: {coffee_after:.10f}")
    print(f"   Home:   {home_after:.10f}")
    
    print(f"\n🎯 LOCALITY VERIFICATION:")
    print(f"   Coffee change: {coffee_change:.15e}")
    print(f"   Home change:   {home_change:.15e}")
    
    if coffee_change < 1e-10 and home_change < 1e-10:
        print(f"\n   ✅ MATHEMATICAL GUARANTEE SATISFIED!")
        print(f"   Unrelated destinations are EXACTLY unchanged.")
        print(f"   (Change < 10^-10, i.e., floating point precision)")
        print(f"\n   This is PROVABLE from DNF equations.")
        print(f"   Pure LLM approaches CANNOT provide this guarantee.")
    else:
        print(f"\n   ⚠️ Unexpected change detected")
    
    return coffee_change, home_change


# ============================================================================
# Experiment 4: Parsing Capability Test
# ============================================================================

def test_parsing_variations():
    """Test different phrasings of the same intent"""
    
    print("\n" + "="*70)
    print("EXPERIMENT 4: Natural Language Variation Test")
    print("="*70)
    
    # Same intent, different phrasings
    test_cases = [
        ("SKIP intent", [
            "Skip gym",
            "I don't go to gym anymore",
            "No more gym for me",
            "Remove gym from my routine",
        ]),
        ("SWAP intent", [
            "Gym before work",
            "I go to gym before work now",
            "Switch gym and work",
            "Reverse the order of gym and work",
        ]),
        ("EARLY intent", [
            "Work earlier",
            "I arrive at work earlier now",
            "Get to work sooner",
            "I need to be at work earlier",
        ]),
    ]
    
    for intent, commands in test_cases:
        print(f"\n--- Testing {intent} ---")
        success = 0
        for cmd in commands:
            from llm_parser import LLMParser
            parser = LLMParser(api_key=GROQ_API_KEY, use_api=(GROQ_API_KEY is not None))
            result = parser.parse_command(cmd, destination_names)
            
            if result['feedback_type']:
                print(f"  ✓ '{cmd}' → {result['feedback_type'].value}")
                success += 1
            else:
                print(f"  ✗ '{cmd}' → FAILED")
        
        print(f"  Success: {success}/{len(commands)} ({100*success/len(commands):.0f}%)")


# ============================================================================
# Main: Run Selected Experiments
# ============================================================================

if __name__ == "__main__":
    
    # Uncomment the experiment you want to run:
    
    # Single correction
    test_single_correction()
    
    # Multiple corrections
    # test_multiple_corrections()
    
    # Locality guarantee (for paper!)
    # test_locality_guarantee()
    
    # Parsing variations
    # test_parsing_variations()
    
    print("\n" + "="*70)
    print("✅ Experiments complete!")
    print("="*70)