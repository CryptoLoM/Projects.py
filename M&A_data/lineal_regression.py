import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data = pd.read_csv("cleaned_dataset.csv")
energy_acousticness = data[['Energy', 'Acousticness']].dropna()

# Define X (Acousticness) and y (Energy)
X = energy_acousticness['Acousticness']
y = energy_acousticness['Energy']

# Add constant for regression (intercept)
X_with_const = sm.add_constant(X)

# Fit a linear regression model
model = sm.OLS(y, X_with_const).fit()

# Regression line prediction
y_pred = model.predict(X_with_const)

# Calculate error metrics
rss = np.sum((y - y_pred) ** 2)  # Residual Sum of Squares
rmse = np.sqrt(mean_squared_error(y, y_pred))  # Root Mean Squared Error
rse = np.sqrt(rss / (len(y) - 2))  # Residual Standard Error
r_squared = r2_score(y, y_pred)  # Coefficient of Determination
t_b = model.tvalues  # t-statistics for coefficients

# Visualize regression line
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label="Observed Data", alpha=0.6)
plt.plot(X, y_pred, color='red', label="Regression Line")
plt.title("Energy vs Acousticness with Regression Line")
plt.xlabel("Acousticness")
plt.ylabel("Energy")
plt.legend()
plt.grid()
plt.show()

# Display regression equation and metrics
regression_equation = f"Equation: Energy = {model.params[0]:.4f} + {model.params[1]:.4f} * Acousticness"

print(regression_equation)

print("RSS:", rss, 'RMSE:', rmse, 'RSE:', rse, 'R^2:', r_squared, 't_b:',  t_b)
# Create new Acousticness values for prediction
# Redo predictions for future steps and recalculate metrics
try:
    # Future Acousticness for 3 steps ahead (linear extrapolation within range)
    future_acousticness = np.linspace(X.min(), X.max(), 3)
    future_X_with_const = sm.add_constant(future_acousticness)

    # Predict future Energy values
    future_predictions = model.predict(future_X_with_const)

    # Confidence intervals for future predictions
    prediction_summary = model.get_prediction(future_X_with_const).summary_frame()

    # Display prediction results
    future_results = pd.DataFrame({
        'Acousticness': future_acousticness,
        'Predicted Energy': future_predictions,
        'CI Lower Bound': prediction_summary['mean_ci_lower'],
        'CI Upper Bound': prediction_summary['mean_ci_upper']
    })

except Exception as e:
    str(e)


# Visualize predictions and intervals
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label="Observed Data", alpha=0.6)
plt.plot(X, y_pred, color='red', label="Regression Line")
plt.scatter(future_acousticness, future_predictions, color='green', label="Future Predictions")
plt.fill_between(
    future_acousticness,
    prediction_summary['mean_ci_lower'],
    prediction_summary['mean_ci_upper'],
    color='green',
    alpha=0.2,
    label="Prediction Interval"
)
plt.title("Energy Prediction for Future Acousticness Values")
plt.xlabel("Acousticness")
plt.ylabel("Energy")
plt.legend()
plt.grid()
plt.show()