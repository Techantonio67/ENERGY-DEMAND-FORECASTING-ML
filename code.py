import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

print(">>> INITIATING 'GOD MODE 2.0' ANALYSIS <<<")
print(">>> INTEGRATING ALL ENERGY, ECONOMIC & CLIMATE VARIABLES... <<<")

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
file_name = 'data.xlsx'
try:
    df = pd.read_excel(file_name, sheet_name=None)
    sheet_to_use = 'All Vs' if 'All Vs' in df.keys() else list(df.keys())[0]
    df = df[sheet_to_use]
except Exception as e:
    print(f"Error: {e}")
    exit()

print(f">>> Columns in dataset: {df.columns.tolist()}")

df.dropna(how='all', inplace=True)
if 'Month' in df.columns:
    df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
    df.set_index('Month', inplace=True)

# Interpolate to fix small holes
df = df.interpolate(method='time').ffill().bfill()
target_col = 'Natural Gas'

# ---------------------------------------------------------
# 2. ADVANCED FEATURE ENGINEERING (THE ULTIMATE FORMULA)
# ---------------------------------------------------------

# A. Fourier Terms (ریاضیات امواج - برای گرفتن دقیق فصل‌ها)
print(">>> Adding Fourier Terms for Seasonality...")
for k in [1, 2, 3, 4]:  # 4 harmonics
    df[f'sin_{k}'] = np.sin(2 * np.pi * k * df.index.month / 12)
    df[f'cos_{k}'] = np.cos(2 * np.pi * k * df.index.month / 12)

# B. Advanced Temperature Physics
print(">>> Creating Advanced Temperature Features...")
if 'Temp' in df.columns:
    # Heating Degree Days (HDD)
    df['Temp_clean'] = df['Temp'].clip(lower=-50, upper=50)  # Clip extreme temperatures
    df['HDD'] = (15.5 - df['Temp_clean']).clip(lower=0)
    df['HDD_power'] = df['HDD'] ** 1.5
    df['HDD_squared'] = df['HDD'] ** 2

# C. Advanced Price Effects (کشش قیمتی پیشرفته)
print(">>> Creating Advanced Price Features...")
if 'RP-Gas' in df.columns:
    # Clean price data - remove zeros or very small values for log
    df['RP-Gas_clean'] = df['RP-Gas'].clip(lower=0.1)
    
    # Logarithmic prices (add small epsilon to avoid log(0))
    df['Log_Gas_Price'] = np.log1p(df['RP-Gas_clean'])
    
    # Moving averages of prices
    df['Gas_Price_MA3'] = df['RP-Gas_clean'].rolling(window=3, min_periods=1).mean()
    df['Gas_Price_MA12'] = df['RP-Gas_clean'].rolling(window=12, min_periods=1).mean()
    
    # Interaction with temperature
    if 'Temp' in df.columns:
        df['Price_Cold_Interaction'] = df['Log_Gas_Price'] * df['HDD']

# D. Energy Mix and Substitution Effects
print(">>> Creating Energy Mix Features...")
# Get the correct column name for renewables
renewables_col = None
for col in df.columns:
    if 'Renewables' in col or 'renewables' in col:
        renewables_col = col
        break

if renewables_col:
    print(f">>> Found renewables column: {renewables_col}")
    # Clean energy data - remove negative values
    energy_cols = ['Petroleum', 'Natural Gas', 'Bioenergy&Waste', 'Nuclear', renewables_col]
    for col in energy_cols:
        if col in df.columns:
            df[f'{col}_clean'] = df[col].clip(lower=0)
    
    # Total primary energy (approximate)
    df['Total_Primary_Energy'] = (
        df.get('Petroleum_clean', df.get('Petroleum', 0)) + 
        df.get('Natural Gas_clean', df.get('Natural Gas', 0)) + 
        df.get('Bioenergy&Waste_clean', df.get('Bioenergy&Waste', 0)) + 
        df.get('Nuclear_clean', df.get('Nuclear', 0)) + 
        df.get(f'{renewables_col}_clean', df.get(renewables_col, 0)) + 0.001
    )
    
    # Energy shares (avoid division by zero)
    df['Gas_Share'] = df.get('Natural Gas_clean', df.get('Natural Gas', 0)) / df['Total_Primary_Energy'].replace(0, np.nan).fillna(1)
    df['Renewables_Share'] = (
        (df.get(f'{renewables_col}_clean', df.get(renewables_col, 0)) + 
         df.get('Bioenergy&Waste_clean', df.get('Bioenergy&Waste', 0)))
        / df['Total_Primary_Energy'].replace(0, np.nan).fillna(1)
    )
    df['Fossil_Share'] = (
        (df.get('Petroleum_clean', df.get('Petroleum', 0)) + 
         df.get('Natural Gas_clean', df.get('Natural Gas', 0)))
        / df['Total_Primary_Energy'].replace(0, np.nan).fillna(1)
    )
    
    # Energy substitution indicators
    renewables_data = df.get(f'{renewables_col}_clean', df.get(renewables_col, df[renewables_col]))
    df['Renewables_Growth'] = renewables_data.pct_change(12).fillna(0)  # Annual growth rate
    df['Renewables_Growth'] = df['Renewables_Growth'].clip(lower=-1, upper=1)  # Clip extreme growth rates
else:
    print(">>> Warning: Renewables column not found, skipping energy mix features")
    # Use simpler total energy without renewables
    df['Total_Primary_Energy'] = (
        df.get('Petroleum', 0) + 
        df.get('Natural Gas', 0) + 
        df.get('Bioenergy&Waste', 0) + 
        df.get('Nuclear', 0) + 0.001
    )
    df['Gas_Share'] = df.get('Natural Gas', 0) / df['Total_Primary_Energy'].replace(0, np.nan).fillna(1)

# E. Economic and Demographic Features
print(">>> Creating Economic & Demographic Features...")
if 'GDP' in df.columns and 'Pop' in df.columns:
    # Clean GDP and Population data
    df['GDP_clean'] = df['GDP'].clip(lower=0.1)
    df['Pop_clean'] = df['Pop'].clip(lower=0.1)
    
    # Per capita measures
    df['GDP_per_Capita'] = df['GDP_clean'] / df['Pop_clean']
    df['Gas_per_Capita'] = df.get('Natural Gas_clean', df.get('Natural Gas', 0)) / df['Pop_clean']
    
    # Intensity measures (avoid division by zero)
    df['Gas_Intensity'] = df.get('Natural Gas_clean', df.get('Natural Gas', 0)) / df['GDP_clean'].replace(0, np.nan).fillna(1)
    
    # Economic trends
    df['GDP_Growth_Annual'] = df['GDP_clean'].pct_change(12).fillna(0).clip(lower=-0.5, upper=0.5)
    df['GDP_Log'] = np.log1p(df['GDP_clean'])
    
    # Wealth effect (price affordability)
    df['Gas_Price_to_GDP_Ratio'] = df.get('RP-Gas_clean', df.get('RP-Gas', 0)) / df['GDP_per_Capita'].replace(0, np.nan).fillna(1)

# F. Price Comparison (if electricity price exists)
print(">>> Creating Price Comparison Features...")
if 'RP-Gas' in df.columns and 'RP-Elec' in df.columns:
    # Clean electricity price
    df['RP-Elec_clean'] = df['RP-Elec'].clip(lower=0.1)
    
    # Ratios (avoid division by zero)
    df['Gas_Elec_Price_Ratio'] = df.get('RP-Gas_clean', df.get('RP-Gas', 0)) / df['RP-Elec_clean'].replace(0, np.nan).fillna(1)
    df['Price_Difference'] = df.get('RP-Gas_clean', df.get('RP-Gas', 0)) - df['RP-Elec_clean']

# G. Advanced Weather Features
print(">>> Creating Advanced Weather Features...")
if 'Rainfall' in df.columns and 'Temp' in df.columns:
    # Clean rainfall data
    df['Rainfall_clean'] = df['Rainfall'].clip(lower=0, upper=500)  # Clip extreme rainfall
    
    # Cold rainfall effect
    df['Cold_Rain_Effect'] = np.where(df['Temp_clean'] < 8, df['Rainfall_clean'] * (8 - df['Temp_clean']), 0)
    
if 'WinsSpeed' in df.columns and 'Temp' in df.columns:
    # Clean wind speed data
    df['WinsSpeed_clean'] = df['WinsSpeed'].clip(lower=0, upper=50)  # Clip extreme wind speeds
    
    # Simple wind effect
    df['Wind_Effect'] = df['WinsSpeed_clean'] * df['HDD']

# H. Time-Based Features
print(">>> Creating Time-Based Features...")
df['Time_Index'] = np.arange(len(df))  # Linear trend
df['Year'] = df.index.year
df['Month_num'] = df.index.month

# Non-linear time trend
df['Time_Squared'] = df['Time_Index'] ** 2

# I. Structural Breaks and Crisis Dummies
print(">>> Creating Crisis Dummy Variables...")
# COVID-19 pandemic
df['Covid_2020'] = ((df.index >= '2020-03-01') & (df.index <= '2021-06-01')).astype(int)

# Energy crisis 2021-2023
df['Energy_Crisis_2021'] = ((df.index >= '2021-09-01') & (df.index <= '2023-03-01')).astype(int)

# Seasonal dummies
df['Winter'] = df.index.month.isin([12, 1, 2]).astype(int)

# J. Lag Features (Historical Memory)
print(">>> Creating Lag Features...")
# Clean target variable
df['Natural Gas_clean'] = df[target_col].clip(lower=0.1)
df['Log_Target'] = np.log1p(df['Natural Gas_clean'])  # Log transform for normalization

# Short-term lags (recent months)
for lag in [1, 2, 3, 6, 12]:
    df[f'Lag_{lag}'] = df['Log_Target'].shift(lag)

# Moving averages
df['MA_3'] = df['Log_Target'].rolling(window=3, min_periods=1).mean()
df['MA_12'] = df['Log_Target'].rolling(window=12, min_periods=1).mean()

# K. Interaction Features
print(">>> Creating Interaction Features...")
if 'HDD' in df.columns and 'Log_Gas_Price' in df.columns:
    df['HDD_Price_Interaction'] = df['HDD'] * df['Log_Gas_Price']
    
if 'GDP_per_Capita' in df.columns and 'Log_Gas_Price' in df.columns:
    df['Wealth_Price_Interaction'] = df['GDP_per_Capita'] * df['Log_Gas_Price']

# ---------------------------------------------------------
# 3. DATA CLEANING - REMOVE INFINITE VALUES
# ---------------------------------------------------------
print(">>> Cleaning Data (Removing Infinite/NaN values)...")

# Replace infinite values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with NaN values in key features
df_clean = df.dropna(subset=[target_col, 'Log_Target'])

# Fill remaining NaN values with column medians
df_clean = df_clean.fillna(df_clean.median())

# Clip extreme values in all numeric columns (except target)
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
exclude_from_clipping = [target_col, 'Log_Target', 'Natural Gas_clean']
for col in numeric_cols:
    if col not in exclude_from_clipping:
        # Calculate IQR for outlier detection
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR > 0:  # Only clip if there's variation
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)

print(f">>> Total Features Created: {len(df_clean.columns)}")

# ---------------------------------------------------------
# 4. DATA SPLIT & SCALING
# ---------------------------------------------------------
print("\n>>> Splitting Data...")
train_size = int(len(df_clean) * 0.85)
train_df = df_clean.iloc[:train_size]
test_df = df_clean.iloc[train_size:]

print(f"Training set: {len(train_df)} months ({train_df.index[0]} to {train_df.index[-1]})")
print(f"Test set: {len(test_df)} months ({test_df.index[0]} to {test_df.index[-1]})")

# Identify features to exclude
features_to_drop = [target_col, 'Log_Target', 'Natural Gas_clean', 'Total_Primary_Energy', 'Year', 'Month_num']
# Also drop any columns that might still have infinite values
for col in df_clean.columns:
    if df_clean[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
        if df_clean[col].isna().any() or np.isinf(df_clean[col]).any():
            features_to_drop.append(col)
            print(f"Warning: Dropping column '{col}' due to NaN/inf values")

features_to_drop = [col for col in features_to_drop if col in df_clean.columns]

X_train = train_df.drop(columns=features_to_drop)
y_train = train_df['Log_Target']
X_test = test_df.drop(columns=features_to_drop)
y_test_actual = test_df[target_col]

print(f"Number of features: {X_train.shape[1]}")
print(f"Sample feature names: {list(X_train.columns)[:10]}...")

# Check for any remaining infinite or NaN values
print(f">>> Checking for remaining issues...")
print(f"X_train NaN count: {X_train.isna().sum().sum()}")
print(f"X_train Inf count: {np.isinf(X_train.values).sum()}")
print(f"X_test NaN count: {X_test.isna().sum().sum()}")
print(f"X_test Inf count: {np.isinf(X_test.values).sum()}")

# Scaling
print(">>> Scaling Features...")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for feature importance analysis
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

# ---------------------------------------------------------
# 5. ADVANCED MODEL ENSEMBLE
# ---------------------------------------------------------
print("\n>>> Training Advanced Ensemble Model...")

# Model 1: Gradient Boosting
gbr = GradientBoostingRegressor(
    n_estimators=500,  # Reduced for faster training
    learning_rate=0.05,
    max_depth=3,
    min_samples_split=10,
    min_samples_leaf=5,
    loss='huber',
    subsample=0.8,
    random_state=42
)

# Model 2: Ridge Regression
ridge = Ridge(alpha=1.0, random_state=42)

# Model 3: Random Forest
rf = RandomForestRegressor(
    n_estimators=100,  # Reduced for faster training
    max_depth=6,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features=0.5,
    n_jobs=-1,
    random_state=42
)

# Voting Regressor
model = VotingRegressor([
    ('gbr', gbr), 
    ('ridge', ridge),
    ('rf', rf)
], weights=[3, 1, 2])

# Train the model
model.fit(X_train_scaled, y_train)

# ---------------------------------------------------------
# 6. PREDICTION & EVALUATION
# ---------------------------------------------------------
print("\n>>> Making Predictions...")
y_pred_log = model.predict(X_test_scaled)
y_pred = np.expm1(y_pred_log)  # Convert back from log

# Calculate metrics
r2 = r2_score(y_test_actual, y_pred)
mae = mean_absolute_error(y_test_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
mape = np.mean(np.abs((y_test_actual - y_pred) / y_test_actual.clip(lower=0.1))) * 100

print("\n" + "#"*70)
print(f"🚀 GOD MODE 2.0 - FINAL RESULTS 🚀")
print("#"*70)
print(f"R² Score          : {r2:.6f}")
print(f"MAE Error         : {mae:.6f}")
print(f"RMSE              : {rmse:.6f}")
print(f"MAPE              : {mape:.2f}%")
print("#"*70)

# ---------------------------------------------------------
# 7. FEATURE IMPORTANCE ANALYSIS
# ---------------------------------------------------------
print("\n>>> Analyzing Feature Importance...")
try:
    # Get feature importances from Random Forest
    rf_model = model.named_estimators_['rf']
    importances = rf_model.feature_importances_
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print("\n📊 TOP 15 MOST IMPORTANT FEATURES:")
    print(importance_df.head(15).to_string(index=False))
    
    # Plot top features
    plt.figure(figsize=(10, 6))
    top_features = importance_df.head(10)
    plt.barh(range(len(top_features)), top_features['Importance'][::-1])
    plt.yticks(range(len(top_features)), top_features['Feature'][::-1])
    plt.xlabel('Feature Importance (Random Forest)')
    plt.title('Top 10 Most Important Features')
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"Could not calculate feature importance: {e}")

# ---------------------------------------------------------
# 8. VISUALIZATION
# ---------------------------------------------------------
print("\n>>> Creating Visualizations...")

plt.figure(figsize=(15, 8))

# Plot 1: Actual vs Predicted
plt.subplot(2, 2, 1)
plt.plot(y_test_actual.index, y_test_actual, label='ACTUAL', color='black', linewidth=2)
plt.plot(y_test_actual.index, y_pred, label='PREDICTED', color='#00FF00', linestyle='--', linewidth=1.5)
plt.fill_between(y_test_actual.index, y_test_actual, y_pred, color='red', alpha=0.15)
plt.title(f'Actual vs Predicted (R² = {r2:.4f})', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Plot 2: Residuals
residuals = y_test_actual - y_pred
plt.subplot(2, 2, 2)
plt.scatter(y_pred, residuals, alpha=0.6, color='blue')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)

# Plot 3: Error Distribution
plt.subplot(2, 2, 3)
plt.hist(residuals, bins=20, edgecolor='black', alpha=0.7, color='purple')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Error Distribution')
plt.grid(True, alpha=0.3)

# Plot 4: Zoom on Last 24 Months
plt.subplot(2, 2, 4)
if len(y_test_actual) >= 24:
    zoom_period = y_test_actual.index[-24:]
else:
    zoom_period = y_test_actual.index
    
y_actual_zoom = y_test_actual.loc[zoom_period]
y_pred_zoom = pd.Series(y_pred[-len(zoom_period):], index=zoom_period)

plt.plot(zoom_period, y_actual_zoom, label='ACTUAL', color='black', linewidth=3)
plt.plot(zoom_period, y_pred_zoom, label='PREDICTED', color='#00FF00', 
         linestyle='--', linewidth=2, marker='o', markersize=4)
plt.title('Zoom: Recent Predictions', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 9. PERFORMANCE ANALYSIS
# ---------------------------------------------------------
print("\n" + "="*70)
print("📊 PERFORMANCE ANALYSIS")
print("="*70)

if r2 > 0.95:
    print("✅ STATUS: EXCELLENT PRECISION (>95%)")
    print("   - Model performance is outstanding")
elif r2 > 0.90:
    print("✅ STATUS: VERY GOOD PRECISION (90-95%)")
    print("   - Model captures most patterns well")
elif r2 > 0.85:
    print("⚠️ STATUS: GOOD PRECISION (85-90%)")
    print("   - Model is working but has room for improvement")
elif r2 > 0.80:
    print("⚠️ STATUS: MODERATE PRECISION (80-85%)")
    print("   - Model captures basic patterns")
else:
    print("❌ STATUS: LOW PRECISION (<80%)")
    print("   - Model is missing key patterns")

print(f"\n📈 Prediction Error Statistics:")
print(f"   - Average Error: {residuals.mean():.4f}")
print(f"   - Std of Errors: {residuals.std():.4f}")
print(f"   - Max Over-prediction: {residuals.max():.4f}")
print(f"   - Max Under-prediction: {residuals.min():.4f}")

# ---------------------------------------------------------
# 10. SAVE RESULTS
# ---------------------------------------------------------
print("\n>>> Saving Results...")

# Create results DataFrame
results_df = pd.DataFrame({
    'Date': y_test_actual.index,
    'Actual': y_test_actual.values,
    'Predicted': y_pred,
    'Error': residuals,
    'Error_Percent': np.abs(residuals / y_test_actual.clip(lower=0.1)) * 100
})

# Save to CSV
results_df.to_csv('god_mode_2.0_results.csv', index=False)
print("✅ Results saved to 'god_mode_2.0_results.csv'")

# Save feature importances
if 'importance_df' in locals():
    importance_df.to_csv('feature_importances.csv', index=False)
    print("✅ Feature importances saved to 'feature_importances.csv'")

print("\n" + "="*70)
print("🎯 GOD MODE 2.0 ANALYSIS COMPLETE")
print("="*70)