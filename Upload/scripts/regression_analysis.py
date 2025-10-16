
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("STARTING REGRESSION ANALYSIS...")
    try:
        from src.regression import run_regression
        manual_results, sklearn_results, feature_names = run_regression()
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE!")
        print("="*50)
        best_manual = min(manual_results.keys(), key=lambda x: manual_results[x]['mse'])
        best_sklearn = min(sklearn_results.keys(), key=lambda x: sklearn_results[x]['mse'])
        print(f"Best Manual Model: {best_manual.upper()} (MSE: {manual_results[best_manual]['mse']:.4f})")
        print(f"Best Sklearn Model: {best_sklearn.upper()} (MSE: {sklearn_results[best_sklearn]['mse']:.4f})")
        diff = abs(manual_results[best_manual]['mse'] - sklearn_results[best_sklearn]['mse'])
        print(f"Implementation Difference: {diff:.6f}")
        
        if diff < 0.001:
            print(" manual implementation matches scikit-learn closely!")
        elif diff < 0.01:
            print(" implementation is very close to scikit-learn!")
        else:
            print("there's some difference from scikit-learn.")
    except ImportError as e:
        print(f"Import Error: {e}")
        print("Make sure your folder structure is correct:")
        print("id123456_assignment/")
        print("├── src/")
        print("│   ├── regression.py")
        print("│   └── models.py")
        print("└── scripts/")
        print("    └── regression_analysis.py")
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()