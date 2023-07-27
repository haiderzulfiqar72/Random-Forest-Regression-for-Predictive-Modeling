import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


X_train = np.loadtxt("X_train.dat")
Y_train = np.loadtxt("Y_train.dat")

X_test = np.loadtxt("X_test.dat")
Y_test = np.loadtxt("Y_test.dat")

print(f'\nShape of training data:{X_train.shape}')

## a)

#Computing baseline regression
baseline= np.mean(Y_train)
print("Baseline:", baseline)
mae = np.mean(np.abs(Y_test - baseline))
print("Baseline accuracy: Mean Absolute Error (MAE):", mae)


## b)

print(f'Shape of training data: {X_train.shape}')

# Baseline accuracy
baseline = np.mean(Y_train)
mae_baseline = np.mean(np.abs(Y_test - baseline))
print(f"Baseline accuracy (MAE): {mae_baseline:.3f}")

# Random Forest regressor
rf = RandomForestRegressor()
rf.fit(X_train, Y_train)
Y_pred = rf.predict(X_test)
mae_rf = np.mean(np.abs(Y_test - Y_pred))
print(f"RF regr. accuracy (MAE): {mae_rf:.3f}")

## c)

# Define the depths to try
depths = range(1, 13)

# Train the random forest regressor with different depths and compute the MAE
mae_means = []
mae_stds = []
for depth in depths:
    maes = [np.mean(np.abs(Y_test - RandomForestRegressor(max_depth=depth).fit(X_train, Y_train).predict(X_test))) for i in range(10)]
    mae_means.append(np.mean(maes))
    mae_stds.append(np.std(maes))

# Plot the results
plt.errorbar(depths, mae_means, yerr=mae_stds, fmt='o-', capsize=4)
plt.xlabel("Depth")
plt.ylabel("Mean Absolute Error (MAE)")
plt.title("Random Forest Regressor Performance vs. Max Depth")
plt.show()

print("Plots ready.")
