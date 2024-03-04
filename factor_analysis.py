import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load data
factor_data = pd.read_csv('path_to_factor_data.csv')  # Update path
stock_returns = pd.read_csv('path_to_stock_returns.csv')  # Update path
fund_returns = pd.read_csv('path_to_fund_returns.csv')  # Update path
market_caps = pd.read_csv('path_to_market_caps.csv')  # Update path

# Convert monthly returns to annual
factor_data['Annual'] = factor_data['Monthly'] * 12
stock_returns['Annual'] = stock_returns['Monthly'] * 12
fund_returns['Annual'] = fund_returns['Monthly'] * 12

# Plot factor returns over time
plt.figure(figsize=(12, 8))
plt.plot(factor_data['Date'], factor_data['Market_Excess_Return'], label='Market Excess Return')
plt.plot(factor_data['Date'], factor_data['SMB'], label='SMB')
plt.plot(factor_data['Date'], factor_data['HML'], label='HML')
plt.xlabel('Date')
plt.ylabel('Annual Returns')
plt.title('Factor Returns Over Time')
plt.legend()
plt.show()

# Calculate mean returns and variance-covariance matrix
mean_returns = factor_data[['Market_Excess_Return', 'SMB', 'HML']].mean()
cov_matrix = factor_data[['Market_Excess_Return', 'SMB', 'HML']].cov()

print("Mean Returns:\n", mean_returns)
print("Variance-Covariance Matrix:\n", cov_matrix)

# Scatter plot of stock returns with Capital Market Line
# Assuming risk_free_rate and market_return are defined
risk_free_rate = 0.02  # Example value
market_return = 0.1  # Example value
average_returns = stock_returns.mean()
std_devs = stock_returns.std()

plt.figure(figsize=(10, 6))
plt.scatter(std_devs, average_returns, label='Stocks')

slope = (market_return - risk_free_rate) / std_devs['Market']  # Assuming 'Market' is a column in stock_returns
cml_x = np.linspace(0, max(std_devs), 100)
cml_y = risk_free_rate + slope * cml_x
plt.plot(cml_x, cml_y, color='red', label='Capital Market Line')

plt.xlabel('Standard Deviation')
plt.ylabel('Average Return')
plt.title('Risk vs. Return with CML')
plt.legend()
plt.show()

# Estimate factor betas for each stock under CAPM and Fama-French 3-Factor model
# This is a simplified example for a single stock. You will need to loop over stocks in your dataset.
X_capm = sm.add_constant(factor_data['Market_Excess_Return'])
X_ff3 = sm.add_constant(factor_data[['Market_Excess_Return', 'SMB', 'HML']])

# Example regression for one stock
y = stock_returns['Stock_1']  # Replace 'Stock_1' with actual stock column names
model_capm = sm.OLS(y, X_capm).fit()
model_ff3 = sm.OLS(y, X_ff3).fit()

print("CAPM Beta for Stock 1:", model_capm.params)
print("Fama-French 3-Factor Betas for Stock 1:", model_ff3.params)

