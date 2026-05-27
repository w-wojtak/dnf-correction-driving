# hybrid_correction.py
"""
Hybrid DNF-LLM correction system.
Combines your existing DNF correction code with LLM natural language parsing.
"""

import numpy as np
from pathlib import Path
from datetime import datetime

# Import your existing DNF code
from sequence_tuning import (
    FeedbackType, FeedbackField, FeedbackFieldParams,
    apply_skip_from_field, apply_lock_from_field,
    apply_early_from_field, apply_late_from_field,
    apply_swap_from_field,
    compute_convolution,
    find_latest_file_with_prefix
)

from utils import *

# Import new LLM parser
from llm_parser import LLMParser

# Import configuration
from config import *


class HybridCorrectionSystem:
    """
    Integrates LLM natural language parsing with DNF correction operations.
    """
    
    def __init__(self, 
                 destination_names,
                 destination_positions,
                 u_dest_memory,
                 api_key=None):
        
        self.destination_names = destination_names
        self.destination_positions = destination_positions
        self.u_dest_memory = u_dest_memory
        
        # Initialize spatial field
        self.x = np.arange(-X_LIM, X_LIM + DX, DX)
        self.dx = DX
        
        # Initialize LLM parser
        self.parser = LLMParser(api_key=api_key, use_api=(api_key is not None))
        
        # Initialize feedback fields (from your existing code)
        self.feedback_fields = self._init_feedback_fields()
        
        # Mask for locked destinations
        self.h_dmem_mask = np.ones(len(self.x))
        
        print(f"✅ Hybrid system initialized")
        print(f"   Destinations: {', '.join(destination_names)}")
        print(f"   Parser mode: {'API' if api_key else 'Rule-based'}")
    
    def _init_feedback_fields(self):
        """Initialize feedback fields (from your correction.py)"""
        
        params = FeedbackFieldParams(
            kernel_type="gauss",
            kernel_pars=FEEDBACK_KERNEL_PARS,
            h_0=FEEDBACK_H_0,
            theta=FEEDBACK_THETA,
            tau=FEEDBACK_TAU,
            transient=False
        )
        
        return {
            FeedbackType.SKIP:  FeedbackField(self.x, self.dx, params, INPUT_DURATION),
            FeedbackType.LOCK:  FeedbackField(self.x, self.dx, params, INPUT_DURATION),
            FeedbackType.EARLY: FeedbackField(self.x, self.dx, params, INPUT_DURATION),
            FeedbackType.LATE:  FeedbackField(self.x, self.dx, params, INPUT_DURATION),
            FeedbackType.SWAP:  FeedbackField(self.x, self.dx, params, INPUT_DURATION),
        }
    
    def correct_natural_language(self, user_command: str):
        """
        Main interface: accept natural language, apply DNF correction.
        
        This is the hybrid magic!
        """
        
        print(f"\n{'='*70}")
        print(f"💬 User: '{user_command}'")
        print(f"{'='*70}")
        
        # Step 1: Parse with LLM
        parsed = self.parser.parse_command(user_command, self.destination_names)
        
        if parsed['feedback_type'] is None:
            print(f"❌ Could not parse command")
            return False
        
        print(f"🤖 Parsed as: {parsed['feedback_type'].value.upper()}")
        print(f"   Target: {parsed['target']}")
        if parsed['target2']:
            print(f"   Target2: {parsed['target2']}")
        
        # Step 2: Execute DNF correction (your existing code)
        success = self._execute_dnf_correction(
            parsed['feedback_type'],
            parsed['target'],
            parsed.get('target2')
        )
        
        return success
    
    def _execute_dnf_correction(self, feedback_type, target_name, target2_name=None):
        """
        Execute DNF correction using your existing feedback field dynamics.
        """
        
        # Get spatial position
        try:
            idx = [n.lower() for n in self.destination_names].index(target_name.lower())
            target_center = self.destination_positions[idx]
        except ValueError:
            print(f"❌ Destination '{target_name}' not found")
            return False
        
        # Get feedback field
        ff = self.feedback_fields[feedback_type]
        
        # Run feedback field dynamics (your existing code)
        t_feedback = np.arange(0, T_FEEDBACK_LIM + DT, DT)
        
        for i in range(len(t_feedback)):
            if i == TRIGGER_STEP:
                ff.inject(center=target_center, amplitude=3.0, width=5.0, current_step=i)
                
                # For SWAP, inject second peak
                if feedback_type == FeedbackType.SWAP and target2_name:
                    try:
                        idx2 = [n.lower() for n in self.destination_names].index(target2_name.lower())
                        target2_center = self.destination_positions[idx2]
                        ff.inject_add(center=target2_center, amplitude=3.0, width=5.0)
                    except ValueError:
                        print(f"⚠️ Second target '{target2_name}' not found")
            
            # Update feedback field (your dynamics)
            conv_ff = compute_convolution(ff.u, ff.params.theta, ff.w_hat, self.dx)
            ff.u += (DT / ff.params.tau) * (-ff.u + conv_ff + ff.h + ff.s)
        
        # Apply correction to memory (your existing functions)
        u_before = self.u_dest_memory.copy()
        
        if feedback_type == FeedbackType.SKIP:
            self.u_dest_memory = apply_skip_from_field(self.u_dest_memory, ff)
            print(f"✓ Removed {target_name} from memory")
            
        elif feedback_type == FeedbackType.LOCK:
            self.h_dmem_mask = apply_lock_from_field(self.h_dmem_mask, ff)
            print(f"✓ Locked {target_name}")
            
        elif feedback_type == FeedbackType.EARLY:
            self.u_dest_memory = apply_early_from_field(self.u_dest_memory, ff, THETA_DEST)
            print(f"✓ {target_name} will be recalled earlier")
            
        elif feedback_type == FeedbackType.LATE:
            self.u_dest_memory = apply_late_from_field(self.u_dest_memory, ff, THETA_DEST)
            print(f"✓ {target_name} will be recalled later")
            
        elif feedback_type == FeedbackType.SWAP:
            self.u_dest_memory = apply_swap_from_field(self.u_dest_memory, ff)
            print(f"✓ Swapped {target_name} ↔ {target2_name}")
        
        # Reset feedback field
        ff.clear_input()
        ff.u = ff.params.h_0 * np.ones(len(self.x))
        
        return True
    
    def save_state(self, iteration):
        """Save current memory state"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        np.save(RESULTS_DIR / f"u_dest_memory_iter{iteration}_{timestamp}.npy", 
                self.u_dest_memory)
        np.save(RESULTS_DIR / f"h_dmem_mask_iter{iteration}_{timestamp}.npy", 
                self.h_dmem_mask)


# ============================================================================
# Convenience function
# ============================================================================

def load_and_correct(natural_language_command: str, 
                     driver="A", 
                     day="monday",
                     api_key=None):
    """
    Load latest memory and apply natural language correction.
    
    Example:
        load_and_correct("I go to gym before work now", api_key="gsk_...")
    """
    
    # Load memory (your existing code)
    folder = LEARNING_DIR
    file_dest, ts = find_latest_file_with_prefix(
        folder, f"u_dest_memory_driver{driver}_{day}_"
    )
    
    u_dest_memory = np.load(file_dest)
    
    # Load metadata
    file_meta, _ = find_latest_file_with_prefix(
        folder, f"metadata_driver{driver}_{day}_"
    )
    metadata = np.load(file_meta, allow_pickle=True).item()
    
    # Create hybrid system
    system = HybridCorrectionSystem(
        destination_names=metadata['destination_names'],
        destination_positions=metadata['destination_positions'],
        u_dest_memory=u_dest_memory,
        api_key=api_key
    )
    
    # Apply correction
    system.correct_natural_language(natural_language_command)
    
    return system