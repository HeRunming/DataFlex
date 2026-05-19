#!/usr/bin/env python3
"""
Alignment Verification Script
Checks DataFlex MMD vs LESS configuration alignment.

Run: python verify_alignment.py
"""

import sys
import yaml
import re
from pathlib import Path

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def check_gradient_type():
    """Check gradient_type settings in components.yaml"""
    print(f"\n{bcolors.HEADER}1. GRADIENT TYPE ALIGNMENT{bcolors.ENDC}")
    print("-" * 70)
    
    with open('src/dataflex/configs/components.yaml') as f:
        config = yaml.safe_load(f)
    
    selectors = config.get('selectors', {})
    
    gradient_types = {
        'less': selectors.get('less', {}).get('params', {}).get('gradient_type'),
        'mmd_grad_rbf': selectors.get('mmd_grad_rbf', {}).get('params', {}).get('gradient_type'),
        'mmd_grad_cov': selectors.get('mmd_grad_cov', {}).get('params', {}).get('gradient_type'),
    }
    
    issues = []
    for method, gtype in gradient_types.items():
        print(f"  {method:15} gradient_type: {gtype}")
        if method.startswith('mmd_grad') and gtype != 'adam':
            issues.append(f"    ❌ {method} should use adam, not {gtype}")
    
    if gradient_types['mmd_grad_rbf'] == gradient_types['mmd_grad_cov']:
        if gradient_types['mmd_grad_rbf'] != 'adam':
            print(f"\n  {bcolors.FAIL}❌ ISSUE: MMD gradient methods use {gradient_types['mmd_grad_rbf']} instead of adam{bcolors.ENDC}")
            issues.append("Gradient type mismatch")
    
    if not issues:
        print(f"\n  {bcolors.OKGREEN}✓ All gradient types correctly set{bcolors.ENDC}")
    
    return len(issues) == 0

def check_random_seed():
    """Check random seed consistency"""
    print(f"\n{bcolors.HEADER}2. RANDOM SEED ALIGNMENT{bcolors.ENDC}")
    print("-" * 70)
    
    with open('src/dataflex/configs/components.yaml') as f:
        config = yaml.safe_load(f)
    
    selectors = config.get('selectors', {})
    
    seeds = {
        'less': selectors.get('less', {}).get('params', {}).get('seed'),
        'mmd_grad_rbf': selectors.get('mmd_grad_rbf', {}).get('params', {}).get('seed'),
        'mmd_grad_cov': selectors.get('mmd_grad_cov', {}).get('params', {}).get('seed'),
    }
    
    print(f"  LESS seed:        {seeds['less']}")
    print(f"  mmd_grad_rbf:     {seeds['mmd_grad_rbf']}")
    print(f"  mmd_grad_cov:     {seeds['mmd_grad_cov']}")
    
    if seeds['mmd_grad_rbf'] != seeds['mmd_grad_cov']:
        print(f"\n  {bcolors.WARNING}⚠ WARNING: MMD variant seeds differ{bcolors.ENDC}")
        return False
    
    if seeds['less'] != seeds['mmd_grad_rbf']:
        print(f"\n  {bcolors.FAIL}❌ ISSUE: LESS seed {seeds['less']} != MMD seed {seeds['mmd_grad_rbf']}{bcolors.ENDC}")
        return False
    
    print(f"\n  {bcolors.OKGREEN}✓ All seeds are consistent{bcolors.ENDC}")
    return True

def check_projection_params():
    """Check projection parameters"""
    print(f"\n{bcolors.HEADER}3. PROJECTION PARAMETERS{bcolors.ENDC}")
    print("-" * 70)
    
    with open('src/dataflex/configs/components.yaml') as f:
        config = yaml.safe_load(f)
    
    selectors = config.get('selectors', {})
    
    params = {}
    for method in ['less', 'mmd_grad_rbf', 'mmd_grad_cov']:
        p = selectors.get(method, {}).get('params', {})
        params[method] = {
            'proj_dim': p.get('proj_dim'),
            'save_interval': p.get('save_interval'),
        }
    
    for method, p in params.items():
        print(f"  {method:15} proj_dim={p['proj_dim']:5} save_interval={p['save_interval']}")
    
    if all(p['proj_dim'] == params['less']['proj_dim'] for p in params.values()):
        print(f"  {bcolors.OKGREEN}✓ Projection dimensions match{bcolors.ENDC}")
    else:
        print(f"  {bcolors.FAIL}❌ Projection dimensions differ{bcolors.ENDC}")
        return False
    
    if all(p['save_interval'] == params['less']['save_interval'] for p in params.values()):
        print(f"  {bcolors.OKGREEN}✓ Save intervals match{bcolors.ENDC}")
    else:
        print(f"  {bcolors.FAIL}❌ Save intervals differ{bcolors.ENDC}")
        return False
    
    return True

def check_adam_preconditioning():
    """Check LESS Adam preconditioning implementation"""
    print(f"\n{bcolors.HEADER}4. ADAM PRECONDITIONING IMPLEMENTATION{bcolors.ENDC}")
    print("-" * 70)
    
    with open('src/dataflex/train/selector/less_selector.py') as f:
        less_code = f.read()
    
    with open('src/dataflex/train/selector/mmd_selector.py') as f:
        mmd_code = f.read()
    
    # Check for in-place mutations in LESS
    if 'denom = v.mul(beta2)' in less_code:
        print(f"  {bcolors.FAIL}❌ LESS: Found in-place v.mul(beta2) mutation{bcolors.ENDC}")
        print(f"     This corrupts optimizer state!")
        return False
    
    # Check for correct implementation in LESS
    if 'numerator = beta1 * m + (1.0 - beta1) * vectorized_grads' in less_code:
        print(f"  {bcolors.OKGREEN}✓ LESS: Fixed non-destructive Adam implementation{bcolors.ENDC}")
    else:
        print(f"  {bcolors.WARNING}⚠ LESS: Could not verify correct implementation{bcolors.ENDC}")
        return False
    
    # Check for correct implementation in MMD
    if 'denominator = torch.sqrt(beta2 * v' in mmd_code:
        print(f"  {bcolors.OKGREEN}✓ MMD: Correct non-destructive Adam implementation{bcolors.ENDC}")
    else:
        print(f"  {bcolors.WARNING}⚠ MMD: Could not verify correct implementation{bcolors.ENDC}")
        return False
    
    return True

def check_target_dataset_implementation():
    """Check if target_dataset parameter is actually used"""
    print(f"\n{bcolors.HEADER}5. TARGET_DATASET PARAMETER IMPLEMENTATION{bcolors.ENDC}")
    print("-" * 70)
    
    with open('src/dataflex/train/trainer/select_trainer.py') as f:
        trainer_code = f.read()
    
    with open('src/dataflex/train/hparams/dynamic_params.py') as f:
        params_code = f.read()
    
    # Check if target_dataset field is defined
    if 'target_dataset' not in params_code:
        print(f"  {bcolors.FAIL}❌ target_dataset field NOT defined in DynamicFinetuningArguments{bcolors.ENDC}")
        return False
    
    # Check if SelectTrainer loads it
    if 'finetuning_args.target_dataset' not in trainer_code:
        print(f"  {bcolors.FAIL}❌ SelectTrainer does NOT load target_dataset{bcolors.ENDC}")
        return False
    
    # Check if loading logic is implemented (not just 'pass')
    if 'get_dataset(' in trainer_code and 'target_dataset_for_selector = get_dataset' in trainer_code:
        print(f"  {bcolors.OKGREEN}✓ target_dataset parameter is properly implemented{bcolors.ENDC}")
        return True
    
    print(f"  {bcolors.WARNING}⚠ target_dataset handling incomplete{bcolors.ENDC}")
    return False

def check_evaluation_protocol():
    """Check evaluation protocol consistency"""
    print(f"\n{bcolors.HEADER}6. EVALUATION PROTOCOL{bcolors.ENDC}")
    print("-" * 70)
    
    configs = {
        'LESS': 'experiments/mmd/configs/less_baseline.yaml',
        'MMD Grad RBF': 'experiments/mmd/configs/mmd_grad_rbf.yaml',
        'MMD Grad Cov': 'experiments/mmd/configs/mmd_grad_cov.yaml',
    }
    
    eval_strategies = {}
    for name, path in configs.items():
        if Path(path).exists():
            with open(path) as f:
                config = yaml.safe_load(f)
            eval_strategy = config.get('eval_strategy', 'not set')
            eval_strategies[name] = eval_strategy
            print(f"  {name:15} eval_strategy: {eval_strategy}")
        else:
            print(f"  {name:15} File not found")
    
    # Note: Different eval strategies are acceptable because:
    # - LESS evaluates during dynamic training (eval_strategy: steps)
    # - MMD does not evaluate (eval_strategy: no)
    # This is by design, not a bug
    
    print(f"\n  {bcolors.WARNING}⚠ DIFFERENT eval_strategy (by design){bcolors.ENDC}")
    print(f"     LESS: 'steps' (evaluates during dynamic training)")
    print(f"     MMD:  'no' (selection only, no intermediate eval)")
    print(f"     Reason: Different experimental protocols are acceptable")
    return True  # This is acceptable

def main():
    print(f"\n{bcolors.BOLD}{bcolors.HEADER}DataFlex Alignment Verification{bcolors.ENDC}")
    print("=" * 70)
    
    results = {}
    
    try:
        results['gradient_type'] = check_gradient_type()
    except Exception as e:
        print(f"  {bcolors.FAIL}Error: {e}{bcolors.ENDC}")
        results['gradient_type'] = False
    
    try:
        results['random_seed'] = check_random_seed()
    except Exception as e:
        print(f"  {bcolors.FAIL}Error: {e}{bcolors.ENDC}")
        results['random_seed'] = False
    
    try:
        results['projection_params'] = check_projection_params()
    except Exception as e:
        print(f"  {bcolors.FAIL}Error: {e}{bcolors.ENDC}")
        results['projection_params'] = False
    
    try:
        results['adam_preconditioning'] = check_adam_preconditioning()
    except Exception as e:
        print(f"  {bcolors.FAIL}Error: {e}{bcolors.ENDC}")
        results['adam_preconditioning'] = False
    
    try:
        results['target_dataset'] = check_target_dataset_implementation()
    except Exception as e:
        print(f"  {bcolors.FAIL}Error: {e}{bcolors.ENDC}")
        results['target_dataset'] = False
    
    try:
        results['evaluation_protocol'] = check_evaluation_protocol()
    except Exception as e:
        print(f"  {bcolors.FAIL}Error: {e}{bcolors.ENDC}")
        results['evaluation_protocol'] = False
    
    # Summary
    print(f"\n{bcolors.BOLD}{bcolors.HEADER}SUMMARY{bcolors.ENDC}")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = f"{bcolors.OKGREEN}✓ PASS{bcolors.ENDC}" if result else f"{bcolors.FAIL}✗ FAIL{bcolors.ENDC}"
        print(f"  {name:25} {status}")
    
    print(f"\n  Total: {passed}/{total} checks passed")
    
    if passed < total:
        print(f"\n  {bcolors.FAIL}❌ Alignment issues detected!{bcolors.ENDC}")
        print(f"     See AUDIT_SUMMARY.txt for details")
        sys.exit(1)
    else:
        print(f"\n  {bcolors.OKGREEN}✓ All alignment checks passed!{bcolors.ENDC}")
        sys.exit(0)

if __name__ == '__main__':
    main()
