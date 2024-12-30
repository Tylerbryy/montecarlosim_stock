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
import yaml
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import json

console = Console()

plt.style.use("seaborn-v0_8-muted")

# Initialize FastAPI
app = FastAPI()
templates = Jinja2Templates(directory="templates")

@partial(jit, static_argnums=(2,))
def simulate_price_path(key, params, time_horizon):
    """
    Simulate a single price path using geometric Brownian motion.

    Args:
    key (PRNGKey): JAX random key for generating random numbers
    params (tuple): Annual return, annual volatility, and last price
    time_horizon (int): Number of days to simulate

    Returns:
    jnp.array: Simulated price path
    """
    annual_return, annual_volatility, last_price = params
    daily_returns = random.normal(key, shape=(time_horizon,)) * (annual_volatility/jnp.sqrt(252)) + (annual_return/252)
    return last_price * jnp.exp(jnp.cumsum(daily_returns))

def plot_monte_carlo_results(simulations, time_horizon, current_price, mean_price, median_price, std_dev, percentile_5, percentile_95, upside_potential, ticker, num_simulations):
    """
    Plot the results of the Monte Carlo simulation.

    Args:
    simulations (np.array): Array of simulated price paths
    time_horizon (int): Number of days simulated
    current_price (float): Current stock price
    mean_price (float): Mean projected price
    median_price (float): Median projected price
    std_dev (float): Standard deviation of projected prices
    percentile_5 (float): 5th percentile of projected prices
    percentile_95 (float): 95th percentile of projected prices
    upside_potential (float): Upside potential in percentage
    ticker (str): Stock ticker symbol
    num_simulations (int): Number of simulations run
    """
    present_date = datetime.now().date()
    date_range = pd.date_range(start=present_date, periods=time_horizon, freq='B')

    fig, ax = plt.subplots(figsize=(20, 12))
    
    # Plot a subset of simulations for better performance
    subset_size = min(1000, simulations.shape[0])
    ax.plot(date_range, simulations[:subset_size].T, alpha=0.02, color='lightgray')
    
    median_projection = np.median(simulations, axis=0)
    ax.plot(date_range, median_projection, color='blue', linewidth=2, label='Median Projection')
    
    ax.fill_between(date_range, 
                    np.percentile(simulations, 5, axis=0), 
                    np.percentile(simulations, 95, axis=0), 
                    color='skyblue', alpha=0.3, label='90% Confidence Interval')

    ax.axhline(y=mean_price, color='green', linestyle='--', linewidth=2, label='Mean Projected Price')
    ax.axhline(y=percentile_5, color='red', linestyle='--', linewidth=2, label='5th Percentile')
    ax.axhline(y=percentile_95, color='purple', linestyle='--', linewidth=2, label='95th Percentile')
    ax.axhline(y=current_price, color='orange', linestyle='-', linewidth=2, label='Current Price')

    ax.set_title(f'Monte Carlo Simulation: {ticker} Stock Price Projection\n(Number of Simulations: {num_simulations})', fontsize=18, fontweight='bold')
    ax.set_xlabel('Date', fontsize=16)
    ax.set_ylabel('Stock Price ($)', fontsize=16)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.set_xlim(date_range[0], date_range[-1])
    fig.autofmt_xdate()

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add text annotations for key stats
    stats = [
        (0.95, f'Current Price: ${current_price:.2f}', 'orange'),
        (0.90, f'Mean Projected Price: ${mean_price:.2f}', 'green'),
        (0.85, f'5th Percentile: ${percentile_5:.2f}', 'red'),
        (0.80, f'95th Percentile: ${percentile_95:.2f}', 'purple'),
        (0.75, f'Median Projection: ${median_projection[-1]:.2f}', 'blue'),
        (0.70, f'Standard Deviation: ${std_dev:.2f}', 'black'),
        (0.65, f'Upside Potential: {upside_potential:.2f}%', 'brown')
    ]
    
    for y, text, color in stats:
        ax.text(0.95, y, text, transform=ax.transAxes, fontsize=14, va='top', ha='right', 
                bbox=dict(facecolor='white', edgecolor=color, alpha=0.8))

    ax.legend(loc='upper left', bbox_to_anchor=(0.05, 0.95), fontsize=14, 
              fancybox=True, shadow=True, ncol=1)

    ax.text(0.5, 0.02, f'Future Price Projections ({time_horizon} trading days)', transform=ax.transAxes, 
            fontsize=18, color='red', ha='center', va='bottom',
            bbox=dict(facecolor='white', edgecolor='red', alpha=0.8))

    plt.tight_layout()
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Save to plots directory
    fig.savefig(f"plots/{ticker}_monte_carlo_projection.png", dpi=300, bbox_inches='tight')

def load_config(file_path):
    """
    Load configuration from a YAML file.

    Args:
    file_path (str): Path to the YAML configuration file

    Returns:
    dict: Loaded configuration
    """
    try:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        console.print(f"[bold red]Error: Configuration file '{file_path}' not found.[/bold red]")
        exit(1)
    except yaml.YAMLError as e:
        console.print(f"[bold red]Error parsing YAML file: {e}[/bold red]")
        exit(1)

def get_stock_data(ticker, start_date, end_date):
    """
    Download stock data using yfinance.

    Args:
    ticker (str): Stock ticker symbol
    start_date (str): Start date for historical data
    end_date (str): End date for historical data

    Returns:
    tuple: Stock history, current price
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)
        current_price = stock.info.get('currentPrice', stock.info['regularMarketPreviousClose'])
        return hist, current_price
    except Exception as e:
        console.print(f"[bold red]Error downloading stock data for {ticker}: {e}[/bold red]")
        return None, None

def calculate_returns(hist):
    """
    Calculate log returns and annual statistics.

    Args:
    hist (pd.DataFrame): Historical stock data

    Returns:
    tuple: Log returns, annual return, annual volatility
    """
    log_returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna().values
    log_returns = jnp.array(log_returns)
    annual_return = log_returns.mean() * 252
    annual_volatility = log_returns.std() * jnp.sqrt(252)
    return log_returns, annual_return, annual_volatility

def run_monte_carlo(params, num_simulations, time_horizon):
    """
    Run Monte Carlo simulation.

    Args:
    params (tuple): Simulation parameters
    num_simulations (int): Number of simulations to run
    time_horizon (int): Number of days to simulate

    Returns:
    jnp.array: Simulated price paths
    """
    key = random.PRNGKey(0)
    keys = random.split(key, num_simulations)
    simulate_batch = vmap(simulate_price_path, in_axes=(0, None, None))
    return simulate_batch(keys, params, time_horizon)

def calculate_statistics(simulations):
    """
    Calculate statistics from simulation results.

    Args:
    simulations (jnp.array): Simulated price paths

    Returns:
    tuple: Mean price, median price, standard deviation, 5th percentile, 95th percentile
    """
    final_prices = simulations[:, -1]
    return (
        jnp.mean(final_prices),
        jnp.median(final_prices),
        jnp.std(final_prices),
        jnp.percentile(final_prices, 5),
        jnp.percentile(final_prices, 95)
    )

def create_results_table(ticker, current_price, last_price, mean_price, median_price, std_dev, percentile_5, percentile_95, upside_potential):
    """
    Create a rich table with Monte Carlo analysis results.

    Args:
    Various statistical results

    Returns:
    rich.table.Table: Formatted table with results
    """
    table = Table(title=f"Monte Carlo Analysis Results for {ticker}")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    rows = [
        ("Current Price", f"${current_price:.2f}"),
        ("Starting Price", f"${last_price:.2f}"),
        ("Mean Projected Price", f"${mean_price:.2f}"),
        ("Median Projected Price", f"${median_price:.2f}"),
        ("Standard Deviation", f"${std_dev:.2f}"),
        ("5th Percentile", f"${percentile_5:.2f}"),
        ("95th Percentile", f"${percentile_95:.2f}"),
        ("Upside Potential", f"{upside_potential:.2f}%")
    ]

    for row in rows:
        table.add_row(*row)

    return table

def create_config_table(num_simulations, time_horizon, annual_return, annual_volatility):
    """
    Create a rich table with Monte Carlo configuration.

    Args:
    Various configuration parameters

    Returns:
    rich.table.Table: Formatted table with configuration
    """
    config_table = Table(title="Monte Carlo Configuration")
    config_table.add_column("Parameter", style="cyan", no_wrap=True)
    config_table.add_column("Value", style="magenta")

    rows = [
        ("Number of Simulations", f"{num_simulations:,}"),
        ("Time Horizon (Trading Days)", f"{time_horizon}"),
        ("Annual Return", f"{annual_return:.4f}"),
        ("Annual Volatility", f"{annual_volatility:.4f}")
    ]

    for row in rows:
        config_table.add_row(*row)

    return config_table

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/simulation-data")
async def simulation_data():
    config = load_config("config.yaml")
    all_stocks_data = []
    
    for stock_config in config['stocks']:
        ticker = stock_config['ticker']
        hist, current_price = get_stock_data(ticker, stock_config['start_date'], stock_config['end_date'])
        if hist is None or current_price is None:
            continue
            
        log_returns, annual_return, annual_volatility = calculate_returns(hist)
        last_price = hist['Close'].iloc[-1]
        
        params = (annual_return, annual_volatility, last_price)
        simulations = run_monte_carlo(params, config['num_simulations'], config['time_horizon'])
        
        mean_price, median_price, std_dev, percentile_5, percentile_95 = calculate_statistics(simulations)
        upside_potential = (mean_price - current_price) / current_price * 100
        
        present_date = datetime.now().date()
        date_range = pd.date_range(start=present_date, periods=config['time_horizon'], freq='B')
        dates = date_range.strftime('%Y-%m-%d').tolist()
        
        all_stocks_data.append({
            "ticker": ticker,
            "dates": dates,
            "simulations": simulations.tolist(),
            "median_projection": np.median(simulations, axis=0).tolist(),
            "mean_price": float(mean_price),
            "current_price": float(current_price),
            "percentile_5": np.percentile(simulations, 5, axis=0).tolist(),
            "percentile_95": np.percentile(simulations, 95, axis=0).tolist(),
            "std_dev": float(std_dev),
            "upside_potential": float(upside_potential),
            "last_price": float(last_price)
        })
    
    return all_stocks_data

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == '__main__':
    main()