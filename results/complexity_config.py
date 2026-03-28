# -*- coding: utf-8 -*-
"""
Complexity Configuration - VERSION 2
=====================================
Constraint threshold values for different difficulty levels.

USAGE:
    from complexity_config import set_complexity, COMPLEXITY_LEVELS
    
    # Hard problem
    set_complexity('HARD')
    
    # Custom values
    set_complexity('CUSTOM', z2=0.12, z3=0.010)
"""

from objectives import set_constraint_thresholds, get_constraint_thresholds

# ============================================================================
# COMPLEXITY LEVELS
# ============================================================================
# Z2 and Z3 threshold values per level
# Lower value = Harder problem (tighter constraints)

COMPLEXITY_LEVELS = {
    # =====================================================
    # EASY: Wide tolerance, most solutions will be feasible
    # =====================================================
    'EASY': {
        'Z2_THRESHOLD': 0.35,   # Accept fairness gap up to 35%
        'Z3_THRESHOLD': 0.040,  # Varyans 0.040'a kadar kabul
        'description': 'Easy - Wide constraint tolerance'
    },
    
    # =====================================================
    # MEDIUM: Medium difficulty, balanced
    # =====================================================
    'MEDIUM': {
        'Z2_THRESHOLD': 0.25,   # Accept fairness gap up to 25%
        'Z3_THRESHOLD': 0.025,  # Varyans 0.025'e kadar kabul
        'description': 'Medium - Balanced constraints'
    },
    
    # =====================================================
    # HARD: Tight constraints, harder optimization
    # =====================================================
    'HARD': {
        'Z2_THRESHOLD': 0.15,   # Accept fairness gap up to 15%
        'Z3_THRESHOLD': 0.015,  # Varyans 0.015'e kadar kabul
        'description': 'Hard - Tight constraints'
    },
    
    # =====================================================
    # VERY_HARD: Very tight, only very good solutions feasible
    # =====================================================
    'VERY_HARD': {
        'Z2_THRESHOLD': 0.10,   # Accept fairness gap up to 10%
        'Z3_THRESHOLD': 0.008,  # Varyans 0.008'e kadar kabul
        'description': 'Very Hard - Very tight constraints'
    },
    
    # =====================================================
    # EXTREME: Near-perfect fairness required
    # =====================================================
    'EXTREME': {
        'Z2_THRESHOLD': 0.05,   # Accept fairness gap up to 5%
        'Z3_THRESHOLD': 0.003,  # Varyans 0.003'e kadar kabul
        'description': 'Extreme - Near-perfect fairness required'
    }
}


def set_complexity(level: str, z2: float = None, z3: float = None) -> dict:
    """
    Sets complexity level.
    
    Args:
        level: 'EASY', 'MEDIUM', 'HARD', 'VERY_HARD', 'EXTREME' or 'CUSTOM'
        z2: Custom Z2 threshold (only when level='CUSTOM')
        z3: Custom Z3 threshold (only when level='CUSTOM')
    
    Returns:
        Dict containing the applied values
    """
    if level == 'CUSTOM':
        if z2 is None or z3 is None:
            raise ValueError("z2 and z3 values required for CUSTOM level")
        
        set_constraint_thresholds(z2, z3)
        
        return {
            'level': 'CUSTOM',
            'Z2_THRESHOLD': z2,
            'Z3_THRESHOLD': z3,
            'description': f'Custom - Z2≤{z2:.4f}, Z3≤{z3:.6f}'
        }
    
    if level not in COMPLEXITY_LEVELS:
        raise ValueError(f"Invalid complexity level: {level}. "
                        f"Valid values: {list(COMPLEXITY_LEVELS.keys())} or 'CUSTOM'")
    
    config = COMPLEXITY_LEVELS[level]
    set_constraint_thresholds(config['Z2_THRESHOLD'], config['Z3_THRESHOLD'])
    
    return {
        'level': level,
        'Z2_THRESHOLD': config['Z2_THRESHOLD'],
        'Z3_THRESHOLD': config['Z3_THRESHOLD'],
        'description': config['description']
    }


def get_complexity() -> dict:
    """Returns current complexity settings."""
    z2, z3 = get_constraint_thresholds()
    
    # Check which predefined level matches
    for level, config in COMPLEXITY_LEVELS.items():
        if abs(config['Z2_THRESHOLD'] - z2) < 0.001 and abs(config['Z3_THRESHOLD'] - z3) < 0.0001:
            return {
                'level': level,
                'Z2_THRESHOLD': z2,
                'Z3_THRESHOLD': z3,
                'description': config['description']
            }
    
    return {
        'level': 'CUSTOM',
        'Z2_THRESHOLD': z2,
        'Z3_THRESHOLD': z3,
        'description': f'Custom - Z2≤{z2:.4f}, Z3≤{z3:.6f}'
    }


def print_complexity_info():
    """Prints all complexity levels."""
    print("\n" + "="*70)
    print("COMPLEXITY LEVELS")
    print("="*70)
    
    current = get_complexity()
    
    for level, config in COMPLEXITY_LEVELS.items():
        marker = " ◀ ACTIVE" if level == current['level'] else ""
        print(f"\n   {level}:{marker}")
        print(f"      Z2 ≤ {config['Z2_THRESHOLD']:.4f} (Fairness Gap)")
        print(f"      Z3 ≤ {config['Z3_THRESHOLD']:.6f} (Workload Variance)")
        print(f"      {config['description']}")
    
    if current['level'] == 'CUSTOM':
        print(f"\n   CUSTOM: ◀ ACTIVE")
        print(f"      Z2 ≤ {current['Z2_THRESHOLD']:.4f}")
        print(f"      Z3 ≤ {current['Z3_THRESHOLD']:.6f}")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    print_complexity_info()
    
    print("\n--- Test: Switch to HARD level ---")
    result = set_complexity('HARD')
    print(f"Set: {result}")
    
    print("\n--- Test: CUSTOM values ---")
    result = set_complexity('CUSTOM', z2=0.12, z3=0.012)
    print(f"Set: {result}")
    
    print_complexity_info()
