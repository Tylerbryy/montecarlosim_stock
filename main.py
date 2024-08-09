import yfinance as yf
import jax.numpy as jnp
from jax import random, jit, vmap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from functools import partial
from datetime import datetime
import pandas as pd
console = Console()

plt.style.use("Solarize_Light2")

@partial(jit, static_argnums=(2,))
def simulate_price_path(key, params, time_horizon):
    annual_return, annual_volatility, last_price = params
    daily_returns = random.normal(key, shape=(time_horizon,)) * (annual_volatility/jnp.sqrt(252)) + (annual_return/252)
    return last_price * jnp.exp(jnp.cumsum(daily_returns))


def plot_monte_carlo_results(simulations, time_horizon, current_price, mean_price, percentile_5, percentile_95, ticker):
    # Get the present date
    present_date = datetime.now().date()
    
    # Create a date range for x-axis using business days, starting from the present date
    date_range = pd.date_range(start=present_date, periods=time_horizon, freq='B')

    fig, ax = plt.subplots(figsize=(20, 12))  # Increased figure size to accommodate legend
    
    # Plot simulations with reduced alpha for clarity
    ax.plot(date_range, simulations.T, alpha=0.02, color='lightgray')
    
    # Plot median projection
    median_projection = np.median(simulations, axis=0)
    ax.plot(date_range, median_projection, color='blue', linewidth=2, label='Median Projection')
    
    # Plot confidence interval
    ax.fill_between(date_range, 
                    np.percentile(simulations, 5, axis=0), 
                    np.percentile(simulations, 95, axis=0), 
                    color='skyblue', alpha=0.3, label='90% Confidence Interval')

    # Plot key price levels
    ax.axhline(y=mean_price, color='green', linestyle='--', linewidth=2, label='Mean Projected Price')
    ax.axhline(y=percentile_5, color='red', linestyle='--', linewidth=2, label='5th Percentile')
    ax.axhline(y=percentile_95, color='purple', linestyle='--', linewidth=2, label='95th Percentile')
    ax.axhline(y=current_price, color='orange', linestyle='-', linewidth=2, label='Starting Price (Current)')

    # Customize the plot
    ax.set_title(f'Monte Carlo Simulation: {ticker} Stock Price Projection', fontsize=20, fontweight='bold')
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Stock Price ($)', fontsize=14)
    
    # Format x-axis to show dates and set limits
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.set_xlim(date_range[0], date_range[-1])
    fig.autofmt_xdate()  # Rotate and align the tick labels

    # Format y-axis to show dollar values
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Customize grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # Function to add side annotations with arrows
    def add_side_annotation(y, label, color):
        return ax.annotate(label, xy=(1.02, y), xycoords=('axes fraction', 'data'),
                    va='center', ha='left', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', fc='white', ec=color, alpha=0.8),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color=color))

    # Create annotations
    annotations = [
        (current_price, 'Starting Price (Current)', 'orange'),
        (mean_price, 'Mean Projected Price', 'green'),
        (percentile_5, '5th Percentile', 'red'),
        (percentile_95, '95th Percentile', 'purple'),
        (median_projection[-1], 'Median Projection', 'blue'),
        ((percentile_5 + percentile_95) / 2, '90% Confidence Interval', 'skyblue')
    ]

    # Sort annotations by y-value
    annotations.sort(key=lambda x: x[0])

    # Add annotations with overlap prevention
    added_annotations = []
    min_gap = 0.03  # Minimum gap between annotations (in axes coordinates)
    for y, label, color in annotations:
        annotation = add_side_annotation(y, label, color)
        
        # Check for overlap and adjust position if necessary
        if added_annotations:
            last_anno = added_annotations[-1]
            current_pos = annotation.get_position()[1]
            last_pos = last_anno.get_position()[1]
            if abs(current_pos - last_pos) < min_gap:
                new_y = last_pos - min_gap
                annotation.set_position((annotation.get_position()[0], new_y))
                annotation.xy = (annotation.xy[0], new_y)
        
        added_annotations.append(annotation)

    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(0.05, 0.95), fontsize=12, 
              fancybox=True, shadow=True, ncol=1)

    # Add text to clearly indicate future projection
    ax.text(0.5, 0.02, f'Future Price Projections ({time_horizon} trading days)', transform=ax.transAxes, 
            fontsize=16, color='red', ha='center', va='bottom',
            bbox=dict(facecolor='white', edgecolor='red', alpha=0.8))

    plt.tight_layout()
    plt.show()

def main():
    # Download stock data
    ticker = "SPY"
    start_date = "2022-01-01"
    end_date = "2024-08-07"
    num_simulations = 10000
    time_horizon = 252  # One trading year
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, end=end_date)

    # Get current stock price
    current_price = stock.info.get('currentPrice', stock.info['regularMarketPreviousClose'])

    # Check for any missing values
    hist.isnull().sum()

    # Calculate daily log returns using numpy, then convert to jax array
    log_returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna().values
    log_returns = jnp.array(log_returns)

    # Calculate annualized mean and volatility
    annual_return = log_returns.mean() * 252
    annual_volatility = log_returns.std() * jnp.sqrt(252)

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

    # Create a table for Monte Carlo configuration
    config_table = Table(title="Monte Carlo Configuration")
    config_table.add_column("Parameter", style="cyan", no_wrap=True)
    config_table.add_column("Value", style="magenta")
    config_table.add_row("Number of Simulations", f"{num_simulations:,}")
    config_table.add_row("Time Horizon (Trading Days)", f"{time_horizon}")
    config_table.add_row("Annual Return", f"{annual_return:.4f}")
    config_table.add_row("Annual Volatility", f"{annual_volatility:.4f}")

    layout = Layout()
    layout.split_row(
        Layout(name="table", ratio=2),
        Layout(name="config_table", ratio=1)
    )
    layout["table"].update(Panel(table, title=f"Monte Carlo Analysis for {ticker}"))
    layout["config_table"].update(Panel(config_table))

    console.print(layout)

    # Calculate potential return
    potential_return_current = (mean_price - current_price) / current_price * 100
    console.print(f"Potential Return from Current Price {current_price}: [bold green]{potential_return_current:.2f}%[/bold green]")

    # Plot results with improved visualization
    plot_monte_carlo_results(simulations, time_horizon, current_price, mean_price, percentile_5, percentile_95, ticker)

if __name__ == '__main__':
    main()