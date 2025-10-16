#!/usr/bin/env python3
"""
Run this file to execute your regression part only
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("MACHINE LEARNING ASSIGNMENT - REGRESSION PART")
    print("=" * 50)
    print("Student: [Your Name]")
    print("Running Regression Analysis...")
    print()
    
    try:
        from scripts.regression_analysis import main as run_analysis
        run_analysis()
        
        print("\n" + "="*50)
        print("REGRESSION PART COMPLETED SUCCESSFULLY!")
        print("="*50)
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting steps:")
        print("1. Make sure all files are in the correct locations")
        print("2. Make sure you have installed: numpy, pandas, matplotlib, scikit-learn")
        print("3. Check that your data file is at: data/california_houses/California_Houses.csv")

if __name__ == "__main__":
    main()