import yfinance as yf
import jax.numpy as jnp
from jax import random, jit, vmap
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from functools import partial

console = Console()

@partial(jit, static_argnums=(2,))
def simulate_price_path(key, params, time_horizon):
    annual_return, annual_volatility, last_price = params
    daily_returns = random.normal(key, shape=(time_horizon,)) * (annual_volatility/jnp.sqrt(252)) + (annual_return/252)
    return last_price * jnp.exp(jnp.cumsum(daily_returns))

def main():
    # Download NVDA stock data
    ticker = "TSLA"
    start_date = "2022-01-01"
    end_date = "2023-08-07"
    nvda = yf.Ticker(ticker)
    hist = nvda.history(start=start_date, end=end_date)

    # Get current stock price
    current_price = nvda.info['currentPrice']

    # Check for any missing values
    hist.isnull().sum()

    # Calculate daily log returns using numpy, then convert to jax array
    log_returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna().values
    log_returns = jnp.array(log_returns)

    # Calculate annualized mean and volatility
    annual_return = log_returns.mean() * 252
    annual_volatility = log_returns.std() * jnp.sqrt(252)

    # Set up Monte Carlo simulation parameters
    num_simulations = 500000
    time_horizon = 252  # One trading year
    last_price = hist['Close'].iloc[-1]

    # Perform Monte Carlo simulation using JAX
    key = random.PRNGKey(0)
    keys = random.split(key, num_simulations)
    
    params = (annual_return, annual_volatility, last_price)
    simulate_batch = vmap(simulate_price_path, in_axes=(0, None, None))
    simulations = simulate_batch(keys, params, time_horizon)

    # Calculate statistics
    final_prices = simulations[:, -1]
    mean_price = jnp.mean(final_prices)
    median_price = jnp.median(final_prices)
    std_dev = jnp.std(final_prices)
    percentile_5 = jnp.percentile(final_prices, 5)
    percentile_95 = jnp.percentile(final_prices, 95)

    # Calculate upside potential
    upside_potential = (mean_price - current_price) / current_price * 100

    # Create a rich table for results
    table = Table(title=f"Monte Carlo Analysis Results for {ticker}")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    table.add_row("Current Price", f"${current_price:.2f}")
    table.add_row("Starting Price", f"${last_price:.2f}")
    table.add_row("Mean Projected Price", f"${mean_price:.2f}")
    table.add_row("Median Projected Price", f"${median_price:.2f}")
    table.add_row("Standard Deviation", f"${std_dev:.2f}")
    table.add_row("5th Percentile", f"${percentile_5:.2f}")
    table.add_row("95th Percentile", f"${percentile_95:.2f}")
    table.add_row("Upside Potential", f"{upside_potential:.2f}%")

    console.print(Panel(table, expand=False))

    # Calculate potential return
    
    potential_return_current = (mean_price - current_price) / current_price * 100
    console.print(f"Potential Return from Current Price {current_price}: [bold green]{potential_return_current:.2f}%[/bold green]")

    # Plot results
    plt.figure(figsize=(12, 8))
    plt.plot(simulations.T, alpha=0.1, color='gray')
    plt.plot(jnp.median(simulations, axis=0), color='blue', linewidth=2, label='Median Projection')
    plt.fill_between(range(time_horizon), 
                     jnp.percentile(simulations, 5, axis=0), 
                     jnp.percentile(simulations, 95, axis=0), 
                     color='gray', alpha=0.3, label='90% Confidence Interval')

    plt.axhline(y=mean_price, color='g', linestyle='--', label='Mean Projected Price')
    plt.axhline(y=percentile_5, color='r', linestyle='--', label='5th Percentile')
    plt.axhline(y=percentile_95, color='r', linestyle='--', label='95th Percentile')
    plt.axhline(y=last_price, color='black', linestyle='-', label='Starting Price')
    plt.axhline(y=current_price, color='purple', linestyle='--', label='Current Price')

    plt.title(f'Monte Carlo Simulation: {ticker} Stock Price Projection', fontsize=16)
    plt.xlabel('Trading Days', fontsize=12)
    plt.ylabel('Stock Price ($)', fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)

    # Add annotations
    plt.annotate(f'Start: ${last_price:.2f}', (0, last_price), xytext=(5, 5), textcoords='offset points')
    plt.annotate(f'Current: ${current_price:.2f}', (0, current_price), xytext=(5, 20), textcoords='offset points')
    plt.annotate(f'Mean: ${mean_price:.2f}', (time_horizon-1, mean_price), xytext=(5, 5), textcoords='offset points')
    plt.annotate(f'5th %ile: ${percentile_5:.2f}', (time_horizon-1, percentile_5), xytext=(5, 5), textcoords='offset points')
    plt.annotate(f'95th %ile: ${percentile_95:.2f}', (time_horizon-1, percentile_95), xytext=(5, 5), textcoords='offset points')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()