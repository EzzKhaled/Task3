import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .models import ManualLinearRegression

def run_regression():
    print("=" * 60)
    print("REGRESSION PROBLEM: California Houses Dataset")
    print("=" * 60)
    X, y, feature_names = load_california_houses_data()
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"Target: Median_House_Value")
    print(f"Target range: ${y.min():.0f} - ${y.max():.0f}")
    X_train, X_val, X_test, y_train, y_val, y_test = split_regression_data(X, y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    X_train_bias = np.c_[np.ones(X_train_scaled.shape[0]), X_train_scaled]
    X_val_bias = np.c_[np.ones(X_val_scaled.shape[0]), X_val_scaled]
    X_test_bias = np.c_[np.ones(X_test_scaled.shape[0]), X_test_scaled]

    #  Manual implementations
    print("\n" + "="*50)
    print("1. MANUAL REGRESSION IMPLEMENTATIONS")
    print("="*50)
    manual_results = run_manual_regressions(X_train_bias, X_val_bias, y_train, y_val, X_test_bias, y_test)
    # Scikit-Learn implementations
    print("\n" + "="*50)
    print("2. SCIKIT-LEARN REGRESSION IMPLEMENTATIONS")
    print("="*50)
    sklearn_results = run_sklearn_regressions(X_train_scaled, X_val_scaled, y_train, y_val, X_test_scaled, y_test)
    # Comparison
    print("\n" + "="*50)
    print("3. COMPARISON AND ANALYSIS")
    print("="*50)
    compare_regression_results(manual_results, sklearn_results, feature_names)
    return manual_results, sklearn_results, feature_names

def load_california_houses_data():
    try:
        file_path = "data/california_houses/California_Houses.csv"
        data = pd.read_csv(file_path)
        print("Dataset loaded successfully!")
        print(f"Original data shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        feature_columns = [col for col in data.columns if col != 'Median_House_Value']
        X = data[feature_columns].values
        y = data['Median_House_Value'].values
        print(f"Features used: {feature_columns}")
        print(f"X shape: {X.shape}, y shape: {y.shape}") 
        return X, y, feature_columns
    except Exception as e:
        print(f"Error loading your dataset: {e}")
        print("Using synthetic data as fallback...")
        return create_synthetic_data()

def create_synthetic_data():
    """Create synthetic data if real data cannot be loaded"""
    print("Creating synthetic California housing data...")
    np.random.seed(42)
    n_samples = 20000
    X = np.column_stack([
        np.random.normal(3.0, 1.0, n_samples),  # Median_Income
        np.random.normal(30, 15, n_samples),     # Median_Age
        np.random.normal(1500, 800, n_samples),  # Tot_Rooms
        np.random.normal(300, 150, n_samples),   # Tot_Bedrooms
        np.random.normal(1500, 800, n_samples),  # Population
        np.random.normal(500, 250, n_samples),   # Households
        np.random.uniform(32, 42, n_samples),    # Latitude
        np.random.uniform(-124, -114, n_samples), # Longitude
        np.random.normal(50, 25, n_samples),     # Distance_to_coast
        np.random.normal(100, 50, n_samples),    # Distance_to_LA
        np.random.normal(150, 75, n_samples),    # Distance_to_SanDiego
        np.random.normal(80, 40, n_samples),     # Distance_to_SanJose
        np.random.normal(60, 30, n_samples)      # Distance_to_SanFrancisco
    ])
  
    y = (X[:, 0] * 50000 +  
         X[:, 1] * 1000 +   
         X[:, 2] * 10 +     
         np.random.normal(200000, 50000, n_samples))  
    feature_names = [
        'Median_Income', 'Median_Age', 'Tot_Rooms', 'Tot_Bedrooms', 'Population',
        'Households', 'Latitude', 'Longitude', 'Distance_to_coast', 
        'Distance_to_LA', 'Distance_to_SanDiego', 'Distance_to_SanJose', 
        'Distance_to_SanFrancisco'
    ]
    return X, y, feature_names

def split_regression_data(X, y, test_size=0.3, val_size=0.15):
    # First split: separate test data (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    # Second split: separate validation from train 
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=42
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples") 
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Target stats - Mean: ${y_train.mean():.0f}, Std: ${y_train.std():.0f}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def run_manual_regressions(X_train, X_val, y_train, y_val, X_test, y_test):
    """Run manual regression implementations"""
    alphas = [0, 0.001, 0.01, 0.1, 1, 10, 100]
    results = {}
    print("\n--- Manual Linear Regression (No Regularization) ---")
    linear_model = ManualLinearRegression(alpha=0, regularization='none')
    linear_model.fit(X_train, y_train, method='normal')
    # Test performance
    y_test_pred = linear_model.predict(X_test)
    results['linear'] = evaluate_regression_model(y_test, y_test_pred, "Manual Linear Regression")
    results['linear']['model'] = linear_model
    results['linear']['coefficients'] = linear_model.get_coefficients()
    print("\n--- Manual Ridge Regression (L2 Regularization) ---")
    ridge_val_errors = []
    best_ridge_alpha = alphas[1]
    best_ridge_mse = float('inf')
    for alpha in alphas[1:]:  
        ridge_model = ManualLinearRegression(alpha=alpha, regularization='l2')
        ridge_model.fit(X_train, y_train, method='normal')
        y_val_pred = ridge_model.predict(X_val)
        mse_val = mean_squared_error(y_val, y_val_pred)
        ridge_val_errors.append(mse_val) 
        if mse_val < best_ridge_mse:
            best_ridge_mse = mse_val
            best_ridge_alpha = alpha      
        if alpha in [0.01, 0.1, 1, 10]:
            print(f"  alpha={alpha}: Validation MSE = {mse_val:.4f}")
    print(f"Best alpha for Ridge: {best_ridge_alpha} (MSE: {best_ridge_mse:.4f})")
    
    # Train final Ridge model with best alpha
    final_ridge = ManualLinearRegression(alpha=best_ridge_alpha, regularization='l2')
    final_ridge.fit(X_train, y_train, method='normal')
    y_test_pred = final_ridge.predict(X_test)
    results['ridge'] = evaluate_regression_model(y_test, y_test_pred, f"Manual Ridge (α={best_ridge_alpha})")
    results['ridge']['model'] = final_ridge
    results['ridge']['val_errors'] = ridge_val_errors
    results['ridge']['best_alpha'] = best_ridge_alpha
    results['ridge']['coefficients'] = final_ridge.get_coefficients()
    print("\n--- Manual Lasso Regression (L1 Regularization) ---")
    lasso_val_errors = []
    best_lasso_alpha = alphas[1]
    best_lasso_mse = float('inf')
    for alpha in alphas[1:]:  
        lasso_model = ManualLinearRegression(alpha=alpha, regularization='l1')
        lasso_model.fit(X_train, y_train, method='normal')
        y_val_pred = lasso_model.predict(X_val)
        mse_val = mean_squared_error(y_val, y_val_pred)
        lasso_val_errors.append(mse_val)
        if mse_val < best_lasso_mse:
            best_lasso_mse = mse_val
            best_lasso_alpha = alpha 
        if alpha in [0.01, 0.1, 1, 10]:
            print(f"  alpha={alpha}: Validation MSE = {mse_val:.4f}")
    print(f"Best alpha for Lasso: {best_lasso_alpha} (MSE: {best_lasso_mse:.4f})")

    # Train final Lasso model with best alpha
    final_lasso = ManualLinearRegression(alpha=best_lasso_alpha, regularization='l1')
    final_lasso.fit(X_train, y_train, method='normal')
    y_test_pred = final_lasso.predict(X_test)
    results['lasso'] = evaluate_regression_model(y_test, y_test_pred, f"Manual Lasso (α={best_lasso_alpha})")
    results['lasso']['model'] = final_lasso
    results['lasso']['val_errors'] = lasso_val_errors
    results['lasso']['best_alpha'] = best_lasso_alpha
    results['lasso']['coefficients'] = final_lasso.get_coefficients()
    
    return results

def run_sklearn_regressions(X_train, X_val, y_train, y_val, X_test, y_test):
    """Run scikit-learn regression implementations for comparison"""
    alphas = [0.001, 0.01, 0.1, 1, 10, 100]
    results = {}
    print("\n--- Scikit-Learn Linear Regression ---")
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_test_pred = linear_model.predict(X_test)
    results['linear'] = evaluate_regression_model(y_test, y_test_pred, "Sklearn Linear Regression")
    results['linear']['model'] = linear_model
    results['linear']['coefficients'] = np.concatenate([[linear_model.intercept_], linear_model.coef_])
    print("\n--- Scikit-Learn Ridge Regression ---")
    ridge_val_errors = []
    best_ridge_alpha = alphas[0]
    best_ridge_mse = float('inf')
    for alpha in alphas:
        ridge_model = Ridge(alpha=alpha, random_state=42)
        ridge_model.fit(X_train, y_train)
        y_val_pred = ridge_model.predict(X_val)
        mse_val = mean_squared_error(y_val, y_val_pred)
        ridge_val_errors.append(mse_val)
        if mse_val < best_ridge_mse:
            best_ridge_mse = mse_val
            best_ridge_alpha = alpha  
        if alpha in [0.1, 1, 10]:
            print(f"  alpha={alpha}: Validation MSE = {mse_val:.4f}")
    print(f"Best alpha for Ridge: {best_ridge_alpha} (MSE: {best_ridge_mse:.4f})")
    final_ridge = Ridge(alpha=best_ridge_alpha, random_state=42)
    final_ridge.fit(X_train, y_train)
    y_test_pred = final_ridge.predict(X_test)
    results['ridge'] = evaluate_regression_model(y_test, y_test_pred, f"Sklearn Ridge (α={best_ridge_alpha})")
    results['ridge']['model'] = final_ridge
    results['ridge']['val_errors'] = ridge_val_errors
    results['ridge']['best_alpha'] = best_ridge_alpha
    results['ridge']['coefficients'] = np.concatenate([[final_ridge.intercept_], final_ridge.coef_])
    print("\n--- Scikit-Learn Lasso Regression ---")
    lasso_val_errors = []
    best_lasso_alpha = alphas[0]
    best_lasso_mse = float('inf')
    for alpha in alphas:
        lasso_model = Lasso(alpha=alpha, random_state=42, max_iter=10000)
        lasso_model.fit(X_train, y_train)
        y_val_pred = lasso_model.predict(X_val)
        mse_val = mean_squared_error(y_val, y_val_pred)
        lasso_val_errors.append(mse_val)
        if mse_val < best_lasso_mse:
            best_lasso_mse = mse_val
            best_lasso_alpha = alpha  
        if alpha in [0.1, 1, 10]:
            print(f"  alpha={alpha}: Validation MSE = {mse_val:.4f}")
    print(f"Best alpha for Lasso: {best_lasso_alpha} (MSE: {best_lasso_mse:.4f})")
    final_lasso = Lasso(alpha=best_lasso_alpha, random_state=42, max_iter=10000)
    final_lasso.fit(X_train, y_train)
    y_test_pred = final_lasso.predict(X_test)
    results['lasso'] = evaluate_regression_model(y_test, y_test_pred, f"Sklearn Lasso (α={best_lasso_alpha})")
    results['lasso']['model'] = final_lasso
    results['lasso']['val_errors'] = lasso_val_errors
    results['lasso']['best_alpha'] = best_lasso_alpha
    results['lasso']['coefficients'] = np.concatenate([[final_lasso.intercept_], final_lasso.coef_])
    return results


def evaluate_regression_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) 
    print(f"{model_name} - Test Set Results:")
    print(f"  Mean Squared Error:  {mse:.4f}")
    print(f"  Mean Absolute Error: ${mae:.2f}")
    print(f"  R² Score:           {r2:.4f}")
    return {
        'mse': mse,
        'mae': mae, 
        'r2': r2,
        'model_name': model_name
    }

def compare_regression_results(manual_results, sklearn_results, feature_names):
    print("\nREGRESSION MODELS COMPARISON:")
    print("=" * 80)
    comparison_data = []
    for model_type in ['linear', 'ridge', 'lasso']:
        manual = manual_results[model_type]
        sklearn = sklearn_results[model_type]
        comparison_data.append({
            'Model Type': model_type.upper(),
            'Implementation': 'Manual',
            'MSE': f"{manual['mse']:.4f}",
            'MAE': f"${manual['mae']:.2f}",
            'R²': f"{manual['r2']:.4f}",
            'Best Alpha': manual.get('best_alpha', 'N/A')
        })
        comparison_data.append({
            'Model Type': model_type.upper(), 
            'Implementation': 'Scikit-Learn',
            'MSE': f"{sklearn['mse']:.4f}",
            'MAE': f"${sklearn['mae']:.2f}",
            'R²': f"{sklearn['r2']:.4f}",
            'Best Alpha': sklearn.get('best_alpha', 'N/A')
        })
    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))
    models = ['Linear', 'Ridge', 'Lasso']
    manual_mses = [manual_results['linear']['mse'], manual_results['ridge']['mse'], manual_results['lasso']['mse']]
    sklearn_mses = [sklearn_results['linear']['mse'], sklearn_results['ridge']['mse'], sklearn_results['lasso']['mse']]
    x = np.arange(len(models))
    width = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.bar(x - width/2, manual_mses, width, label='Manual', alpha=0.7, color='blue')
    ax1.bar(x + width/2, sklearn_mses, width, label='Scikit-Learn', alpha=0.7, color='red')
    ax1.set_xlabel('Regression Model')
    ax1.set_ylabel('Mean Squared Error (Lower is Better)')
    ax1.set_title('MSE Comparison: Manual vs Scikit-Learn')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    manual_r2 = [manual_results['linear']['r2'], manual_results['ridge']['r2'], manual_results['lasso']['r2']]
    sklearn_r2 = [sklearn_results['linear']['r2'], sklearn_results['ridge']['r2'], sklearn_results['lasso']['r2']]
    ax2.bar(x - width/2, manual_r2, width, label='Manual', alpha=0.7, color='blue')
    ax2.bar(x + width/2, sklearn_r2, width, label='Scikit-Learn', alpha=0.7, color='red')
    ax2.set_xlabel('Regression Model')
    ax2.set_ylabel('R² Score (Higher is Better)')
    ax2.set_title('R² Comparison: Manual vs Scikit-Learn')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    print("\nKEY INSIGHTS:")
    print("• All models use your California Houses dataset with 13 features")
    print("• Manual and Scikit-Learn implementations should give similar results")
    print("• Regularization helps prevent overfitting to your specific dataset")
    print("• The best alpha value depends on your dataset's characteristics")