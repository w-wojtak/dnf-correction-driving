# llm_parser.py
"""
LLM-based natural language parser for DNF correction commands.
Converts human language to FeedbackType operations.
"""

import json
from typing import Optional, Dict, List
from enum import Enum
from dataclasses import dataclass

# Import your existing FeedbackType from correction.py
from correction import FeedbackType  # ← Use your existing enum


class LLMParser:
    """
    Natural language parser with Groq API and rule-based fallback.
    """
    
    def __init__(self, api_key: Optional[str] = None, use_api: bool = True):
        self.api_key = api_key
        self.use_api = use_api
        
        if use_api and api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=api_key)
                print("✅ Groq API initialized")
                self.provider = "groq"
            except ImportError:
                print("⚠️ groq package not installed. Run: pip install groq")
                self.provider = "mock"
            except Exception as e:
                print(f"⚠️ API initialization failed: {e}")
                self.provider = "mock"
        else:
            self.provider = "mock"
            print("ℹ️ Using rule-based parser")
    
    def parse_command(self, 
                     user_input: str, 
                     destination_names: List[str]) -> Dict[str, any]:
        """
        Parse natural language command into structured operation.
        
        Args:
            user_input: Natural language string
            destination_names: List of valid destination names
            
        Returns:
            {
                'feedback_type': FeedbackType enum value,
                'target': str (destination name),
                'target2': str or None (for SWAP)
            }
        """
        
        # Try API first if enabled
        if self.provider == "groq":
            try:
                result = self._groq_parse(user_input, destination_names)
                if result['feedback_type'] is not None:
                    return result
            except Exception as e:
                print(f"⚠️ API parsing failed: {e}, using fallback")
        
        # Fallback to rules
        return self._rule_based_parse(user_input, destination_names)
    
    def _groq_parse(self, user_input: str, destinations: List[str]) -> Dict:
        """Use Groq API (Llama 3.1)"""
        
        prompt = f"""Parse this routine modification command.

Available destinations: {', '.join(destinations)}

User command: "{user_input}"

Operations:
- SKIP: remove destination
- SWAP: exchange two destinations  
- EARLY: arrive earlier
- LATE: arrive later
- LOCK: protect from changes

Return JSON only:
{{
  "operation": "SKIP|SWAP|EARLY|LATE|LOCK",
  "target": "destination_name",
  "target2": "second_destination_or_null"
}}"""
        
        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Parse commands to JSON. No explanation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=100
        )
        
        text = response.choices[0].message.content.strip()
        
        # Extract JSON
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        parsed = json.loads(text)
        
        # Map to your FeedbackType enum
        op_map = {
            "SKIP": FeedbackType.SKIP,
            "SWAP": FeedbackType.SWAP,
            "EARLY": FeedbackType.EARLY,
            "LATE": FeedbackType.LATE,
            "LOCK": FeedbackType.LOCK
        }
        
        return {
            'feedback_type': op_map.get(parsed.get('operation', '').upper()),
            'target': parsed.get('target'),
            'target2': parsed.get('target2') if parsed.get('target2') != 'null' else None
        }
    
    def _rule_based_parse(self, user_input: str, destinations: List[str]) -> Dict:
        """Enhanced rule-based fallback parser"""
        
        lower = user_input.lower()
        
        # SKIP
        if any(kw in lower for kw in ["skip", "don't", "stop", "remove", "no more"]):
            for dest in destinations:
                if dest.lower() in lower:
                    return {
                        'feedback_type': FeedbackType.SKIP,
                        'target': dest,
                        'target2': None
                    }
        
        # SWAP
        if any(kw in lower for kw in ["before", "switch", "swap", "reverse", "flip"]):
            found = [d for d in destinations if d.lower() in lower]
            if len(found) >= 2:
                return {
                    'feedback_type': FeedbackType.SWAP,
                    'target': found[0],
                    'target2': found[1]
                }
        
        # EARLIER
        if any(kw in lower for kw in ["earlier", "sooner", "advance"]):
            for dest in destinations:
                if dest.lower() in lower:
                    return {
                        'feedback_type': FeedbackType.EARLY,
                        'target': dest,
                        'target2': None
                    }
        
        # LATER
        if any(kw in lower for kw in ["later", "delay", "postpone"]):
            for dest in destinations:
                if dest.lower() in lower:
                    return {
                        'feedback_type': FeedbackType.LATE,
                        'target': dest,
                        'target2': None
                    }
        
        # LOCK
        if any(kw in lower for kw in ["always", "lock", "protect", "keep", "final"]):
            for dest in destinations:
                if dest.lower() in lower:
                    return {
                        'feedback_type': FeedbackType.LOCK,
                        'target': dest,
                        'target2': None
                    }
        
        return {
            'feedback_type': None,
            'target': None,
            'target2': None,
            'error': 'Could not parse'
        }


# Convenience function for direct use
def parse_natural_language(command: str, 
                          destinations: List[str],
                          api_key: Optional[str] = None) -> Dict:
    """
    Quick parse function.
    
    Example:
        result = parse_natural_language(
            "I go to gym before work now",
            ["coffee", "work", "gym", "home"],
            api_key="gsk_..."
        )
    """
    parser = LLMParser(api_key=api_key, use_api=(api_key is not None))
    return parser.parse_command(command, destinations)


if __name__ == "__main__":
    # Test when run directly
    test_parser = LLMParser(use_api=False)
    
    test_commands = [
        "skip gym",
        "gym before work",
        "arrive at work earlier",
    ]
    
    destinations = ["coffee", "work", "gym", "home"]
    
    print("\n=== LLM Parser Test ===\n")
    for cmd in test_commands:
        result = test_parser.parse_command(cmd, destinations)
        print(f"'{cmd}' → {result['feedback_type']}")