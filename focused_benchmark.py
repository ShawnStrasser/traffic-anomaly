#!/usr/bin/env python3
"""
Focused benchmark for the optimized winsorized variance implementation.
"""

import time
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import traffic_anomaly
from traffic_anomaly import sample_data
from simple_optimized_changepoint import optimized_changepoint

def create_scaled_dataset(base_data: pd.DataFrame, multiplier: int) -> pd.DataFrame:
    """Create a scaled dataset for benchmarking."""
    print(f"Creating {multiplier}x scaled dataset...")
    
    original_entities = base_data['ID'].unique()
    max_entity_id = int(base_data['ID'].max())
    
    datasets = []
    for i in range(multiplier):
        data_copy = base_data.copy()
        entity_mapping = {
            old_id: int(old_id) + (i * (max_entity_id + 1000))
            for old_id in original_entities
        }
        data_copy['ID'] = data_copy['ID'].map(entity_mapping)
        datasets.append(data_copy)
    
    large_dataset = pd.concat(datasets, ignore_index=True)
    print(f"Scaled dataset shape: {large_dataset.shape}")
    print(f"Unique entities: {len(large_dataset['ID'].unique())}")
    
    return large_dataset

def benchmark_implementation(func, data, description, timeout=300, **kwargs):
    """Benchmark a single implementation with timeout."""
    print(f"\nBenchmarking {description}...")
    
    start_time = time.time()
    
    try:
        result = func(data, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        if execution_time > timeout:
            print(f"{description} TIMEOUT after {execution_time:.2f} seconds")
            return float('inf'), pd.DataFrame()
        
        print(f"{description} completed in {execution_time:.2f} seconds")
        if hasattr(result, 'shape'):
            print(f"Result shape: {result.shape}")
        
        return execution_time, result
        
    except Exception as e:
        print(f"Error in {description}: {e}")
        import traceback
        traceback.print_exc()
        return float('inf'), pd.DataFrame()

def main():
    """Run focused benchmark."""
    print("="*80)
    print("FOCUSED WINSORIZED VARIANCE OPTIMIZATION BENCHMARK")
    print("="*80)
    
    # Load sample data
    base_data = sample_data.changepoints_input.copy()
    print(f"Base data shape: {base_data.shape}")
    
    # Test parameters
    params = {
        'value_column': 'travel_time_seconds',
        'entity_grouping_column': 'ID',
        'datetime_column': 'TimeStamp',
        'score_threshold': 0.7,
        'rolling_window_days': 14,
        'robust': True
    }
    
    # Test with different dataset sizes
    multipliers = [1, 10, 25, 50]
    results = {}
    
    for mult in multipliers:
        print(f"\n{'='*60}")
        print(f"TESTING WITH {mult}x MULTIPLIER")
        print(f"{'='*60}")
        
        # Create test dataset
        if mult == 1:
            test_data = base_data.copy()
        else:
            test_data = create_scaled_dataset(base_data, mult)
        
        # Test baseline (robust=False)
        params_false = params.copy()
        params_false['robust'] = False
        time_false, result_false = benchmark_implementation(
            traffic_anomaly.changepoint, test_data, f"Original Ibis (robust=False) - {mult}x", **params_false
        )
        
        # Test original robust=True
        time_true, result_true = benchmark_implementation(
            traffic_anomaly.changepoint, test_data, f"Original Ibis (robust=True) - {mult}x", timeout=180, **params
        )
        
        # Test optimized implementation
        time_optimized, result_optimized = benchmark_implementation(
            optimized_changepoint, test_data, f"Optimized Implementation - {mult}x", **params
        )
        
        # Store results
        results[f"{mult}x"] = {
            'dataset_size': test_data.shape,
            'entities': len(test_data['ID'].unique()),
            'robust_false_time': time_false,
            'robust_true_time': time_true,
            'optimized_time': time_optimized,
            'robust_false_shape': result_false.shape if hasattr(result_false, 'shape') else (0, 0),
            'robust_true_shape': result_true.shape if hasattr(result_true, 'shape') else (0, 0),
            'optimized_shape': result_optimized.shape if hasattr(result_optimized, 'shape') else (0, 0)
        }
        
        # Print summary for this multiplier
        print(f"\nSUMMARY FOR {mult}x MULTIPLIER:")
        print(f"Dataset: {test_data.shape}, Entities: {len(test_data['ID'].unique())}")
        print(f"robust=False:  {time_false:.2f}s -> {result_false.shape if hasattr(result_false, 'shape') else (0, 0)}")
        print(f"robust=True:   {time_true:.2f}s -> {result_true.shape if hasattr(result_true, 'shape') else (0, 0)}")
        print(f"Optimized:     {time_optimized:.2f}s -> {result_optimized.shape if hasattr(result_optimized, 'shape') else (0, 0)}")
        
        if time_true != float('inf') and time_optimized != float('inf') and time_optimized > 0:
            improvement = time_true / time_optimized
            print(f"Performance improvement: {improvement:.2f}x")
        
        # Stop if original is taking too long
        if time_true > 120:  # 2 minutes
            print(f"Stopping tests as original robust=True is taking {time_true:.1f}s")
            break
    
    # Generate final report
    print(f"\n{'='*80}")
    print("FINAL BENCHMARK RESULTS")
    print(f"{'='*80}")
    
    print(f"{'Multiplier':<12} {'Dataset Size':<15} {'Entities':<10} {'robust=False':<12} {'robust=True':<12} {'Optimized':<12} {'Improvement':<12}")
    print("-" * 95)
    
    best_improvement = 0
    best_config = None
    
    for config, data in results.items():
        false_time = data['robust_false_time']
        true_time = data['robust_true_time']
        opt_time = data['optimized_time']
        
        false_str = f"{false_time:.2f}s" if false_time != float('inf') else "TIMEOUT"
        true_str = f"{true_time:.2f}s" if true_time != float('inf') else "TIMEOUT"
        opt_str = f"{opt_time:.2f}s" if opt_time != float('inf') else "TIMEOUT"
        
        if true_time != float('inf') and opt_time != float('inf') and opt_time > 0:
            improvement = true_time / opt_time
            improvement_str = f"{improvement:.2f}x"
            if improvement > best_improvement:
                best_improvement = improvement
                best_config = config
        else:
            improvement_str = "N/A"
        
        print(f"{config:<12} {str(data['dataset_size']):<15} {data['entities']:<10} {false_str:<12} {true_str:<12} {opt_str:<12} {improvement_str:<12}")
    
    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}")
    
    if best_improvement > 1.5:
        print(f"🎉 SIGNIFICANT IMPROVEMENT FOUND!")
        print(f"Best improvement: {best_improvement:.2f}x faster with {best_config} dataset")
        print(f"Recommendation: Implement the optimized version")
    elif best_improvement > 1.1:
        print(f"✅ Moderate improvement found: {best_improvement:.2f}x faster")
        print(f"Recommendation: Consider implementing if the use case justifies it")
    else:
        print(f"❌ No significant improvement found")
        print(f"Recommendation: Original implementation is already well optimized")
    
    return results

if __name__ == "__main__":
    results = main()