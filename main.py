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
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from statsmodels.tsa.stattools import adfuller
from arch import arch_model
import warnings
import os
warnings.filterwarnings('ignore')

console = Console()

plt.style.use("seaborn-v0_8-muted")

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

def plot_monte_carlo_results(simulations, time_horizon, current_price, mean_price, median_price, std_dev, percentile_5, percentile_95, upside_potential, ticker, num_simulations, run_dir, ml_metrics=None):
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
    run_dir (str): Directory to save the plot
    ml_metrics (dict): Machine learning metrics

    Returns:
    str: Path to saved plot
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

    # Add text annotations for key stats with indicators
    upside_indicator = "↑" if upside_potential > 0 else "↓"
    volatility_indicator = "↓" if std_dev/current_price < 0.3 else "↑"  # Lower volatility is generally better
    price_trend = "↑" if mean_price > current_price else "↓"
    
    stats = [
        (0.95, f'Current Price: ${current_price:.2f}', 'orange', ''),
        (0.90, f'Mean Projected Price: ${mean_price:.2f} {price_trend}', 'green' if mean_price > current_price else 'red', 
         '(Bullish)' if mean_price > current_price else '(Bearish)'),
        (0.85, f'5th Percentile: ${percentile_5:.2f}', 'red', '(Worst Case)'),
        (0.80, f'95th Percentile: ${percentile_95:.2f}', 'purple', '(Best Case)'),
        (0.75, f'Median Projection: ${median_projection[-1]:.2f} {price_trend}', 'blue', 
         '(Bullish)' if median_projection[-1] > current_price else '(Bearish)'),
        (0.70, f'Volatility: ${std_dev:.2f} {volatility_indicator}', 
         'green' if std_dev/current_price < 0.3 else 'red', 
         '(Low Risk)' if std_dev/current_price < 0.3 else '(High Risk)'),
        (0.65, f'Upside Potential: {upside_potential:.2f}% {upside_indicator}', 
         'green' if upside_potential > 0 else 'red',
         '(Buy Signal)' if upside_potential > 10 else '(Hold)' if upside_potential > 0 else '(Sell Signal)')
    ]
    
    # Add ML metrics if available
    if ml_metrics:
        ml_stats = [
            (0.60, f"ML Model Performance:", 'black', ''),
            (0.55, f"Linear Regression R²: {ml_metrics['Linear']['R2']:.4f} {'↑' if ml_metrics['Linear']['R2'] > 0.7 else '↓'}", 
             'green' if ml_metrics['Linear']['R2'] > 0.7 else 'red',
             '(Good Fit)' if ml_metrics['Linear']['R2'] > 0.7 else '(Poor Fit)'),
            (0.50, f"Ridge R²: {ml_metrics['Ridge']['R2']:.4f} {'↑' if ml_metrics['Ridge']['R2'] > 0.7 else '↓'}", 
             'green' if ml_metrics['Ridge']['R2'] > 0.7 else 'red',
             '(Good Fit)' if ml_metrics['Ridge']['R2'] > 0.7 else '(Poor Fit)'),
            (0.45, f"Lasso R²: {ml_metrics['Lasso']['R2']:.4f} {'↑' if ml_metrics['Lasso']['R2'] > 0.7 else '↓'}", 
             'green' if ml_metrics['Lasso']['R2'] > 0.7 else 'red',
             '(Good Fit)' if ml_metrics['Lasso']['R2'] > 0.7 else '(Poor Fit)'),
            (0.40, f"Random Forest R²: {ml_metrics['RandomForest']['R2']:.4f} {'↑' if ml_metrics['RandomForest']['R2'] > 0.7 else '↓'}", 
             'green' if ml_metrics['RandomForest']['R2'] > 0.7 else 'red',
             '(Good Fit)' if ml_metrics['RandomForest']['R2'] > 0.7 else '(Poor Fit)')
        ]
        stats.extend(ml_stats)

    # Modified text box rendering to include interpretation
    for y, text, color, interpretation in stats:
        if interpretation:
            text = f"{text} {interpretation}"
        ax.text(0.95, y, text, transform=ax.transAxes, fontsize=14, va='top', ha='right', 
                bbox=dict(facecolor='white', edgecolor=color, alpha=0.8))

    # Add legend explaining indicators
    ax.text(0.02, 0.02, 
            "Indicators Guide:\n"
            "↑ = Positive/Bullish Signal\n"
            "↓ = Negative/Bearish Signal\n"
            "R² > 0.7 = Good Model Fit\n"
            "Volatility < 30% = Lower Risk",
            transform=ax.transAxes, fontsize=12, va='bottom', ha='left',
            bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))

    ax.legend(loc='upper left', bbox_to_anchor=(0.05, 0.95), fontsize=14, 
              fancybox=True, shadow=True, ncol=1)

    ax.text(0.5, 0.02, f'Future Price Projections ({time_horizon} trading days)', transform=ax.transAxes, 
            fontsize=18, color='red', ha='center', va='bottom',
            bbox=dict(facecolor='white', edgecolor='red', alpha=0.8))

    plt.tight_layout()
    plot_path = os.path.join(run_dir, f"{ticker}_monte_carlo_projection.png")
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    return plot_path

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

def create_features(hist, time_horizon):
    """Create technical indicators for ML models"""
    df = hist.copy()
    
    # More sophisticated technical indicators
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()  # Long-term trend
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['Volatility'] = df['Close'].rolling(window=20).std()
    
    # Price momentum and trend indicators
    df['Returns'] = df['Close'].pct_change()
    df['Return_20d'] = df['Close'].pct_change(periods=20)
    df['Return_50d'] = df['Close'].pct_change(periods=50)
    
    # Price levels and support/resistance
    df['Distance_From_SMA50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
    df['Distance_From_SMA200'] = (df['Close'] - df['SMA_200']) / df['SMA_200']
    
    # Volume indicators
    df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
    
    # Target variable with shorter horizon
    df['Target'] = df['Close'].shift(-min(time_horizon, 30))  # Use shorter prediction horizon
    
    return df.dropna()

def calculate_rsi(prices, period=14):
    """Calculate RSI technical indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def train_ml_models(hist, time_horizon):
    """Train multiple ML models for price prediction"""
    df = create_features(hist, time_horizon)
    
    feature_cols = [
        'SMA_20', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'MACD_Signal',
        'Volatility', 'Returns', 'Return_20d', 'Return_50d',
        'Distance_From_SMA50', 'Distance_From_SMA200',
        'Volume_SMA_20', 'Volume_Ratio'
    ]
    
    X = df[feature_cols].fillna(method='ffill')
    y = df['Target'].fillna(method='ffill')
    
    # Use more recent data for training
    train_size = int(len(df) * 0.8)
    X_train = X[-train_size:]
    y_train = y[-train_size:]
    X_test = X[-int(len(df) * 0.2):]
    y_test = y[-int(len(df) * 0.2):]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=0.1),  # Reduced alpha for less regularization
        'Lasso': Lasso(alpha=0.01),  # Reduced alpha for less regularization
        'RandomForest': RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
    }
    
    predictions = {}
    metrics = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        predictions[name] = pred
        metrics[name] = {
            'R2': r2_score(y_test, pred),
            'RMSE': mean_squared_error(y_test, pred, squared=False)
        }
        
        if name == 'RandomForest':
            # Get feature importance for Random Forest
            importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            console.print(f"\n[cyan]Feature Importance for {name}:[/cyan]")
            console.print(importance.head())
    
    return predictions, metrics, scaler, models

def check_stationarity(returns):
    """Test for stationarity using Augmented Dickey-Fuller test"""
    adf_result = adfuller(returns)
    return adf_result[1] < 0.05  # Returns True if stationary

def estimate_garch_parameters(returns):
    """Estimate GARCH(1,1) parameters for better volatility forecasting"""
    try:
        model = arch_model(returns, vol='Garch', p=1, q=1)
        results = model.fit(disp='off')
        forecast = results.forecast(horizon=1)
        return np.sqrt(forecast.variance.values[-1][0])
    except:
        return None

def calculate_jump_diffusion(returns, window=252):
    """Estimate jump parameters using a rolling window"""
    # Convert JAX array to numpy if needed
    if hasattr(returns, 'device_buffer'):
        returns = np.array(returns)
    
    rolling_std = pd.Series(returns).rolling(window=window).std()
    jump_threshold = 3 * rolling_std
    
    # Convert to numpy array for boolean indexing
    returns_np = np.array(returns)
    jumps = returns_np[np.abs(returns_np) > jump_threshold]
    
    jump_intensity = len(jumps) / len(returns)
    jump_mean = np.mean(jumps) if len(jumps) > 0 else 0
    jump_std = np.std(jumps) if len(jumps) > 0 else 0
    
    return jump_intensity, jump_mean, jump_std

@partial(jit, static_argnums=(2,))
def simulate_price_path_enhanced(key, params, time_horizon):
    """Enhanced simulation with jumps and GARCH volatility"""
    annual_return, annual_volatility, last_price, jump_intensity, jump_mean, jump_std = params
    
    # Generate two sets of random numbers - one for diffusion, one for jumps
    key1, key2, key3 = random.split(key, 3)
    
    # Normal diffusion process
    daily_returns = random.normal(key1, shape=(time_horizon,)) * (annual_volatility/jnp.sqrt(252)) + (annual_return/252)
    
    # Jump process
    jump_occurs = random.uniform(key2, shape=(time_horizon,)) < (jump_intensity/252)
    jumps = random.normal(key3, shape=(time_horizon,)) * jump_std + jump_mean
    
    # Combine processes
    total_returns = daily_returns + jump_occurs * jumps
    return last_price * jnp.exp(jnp.cumsum(total_returns))

def run_monte_carlo_enhanced(hist, params, num_simulations, time_horizon):
    """Enhanced Monte Carlo with additional market checks"""
    log_returns, annual_return, annual_volatility = calculate_returns(hist)
    last_price = hist['Close'].iloc[-1]
    
    # Check for stationarity
    is_stationary = check_stationarity(log_returns)
    if not is_stationary:
        console.print("[yellow]Warning: Returns are not stationary, results may be less reliable[/yellow]")
    
    # Estimate GARCH volatility
    garch_vol = estimate_garch_parameters(log_returns)
    if garch_vol is not None:
        annual_volatility = garch_vol * np.sqrt(252)
    
    # Estimate jump parameters
    jump_intensity, jump_mean, jump_std = calculate_jump_diffusion(log_returns)
    
    enhanced_params = (annual_return, annual_volatility, last_price, jump_intensity, jump_mean, jump_std)
    
    key = random.PRNGKey(0)
    keys = random.split(key, num_simulations)
    simulate_batch = vmap(simulate_price_path_enhanced, in_axes=(0, None, None))
    
    return simulate_batch(keys, enhanced_params, time_horizon)

def create_run_directory():
    """Create a directory for the current run using timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"simulation_runs/{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def main():
    # Create directory for this run
    run_dir = create_run_directory()
    console.print(f"[green]Saving results to: {run_dir}[/green]")

    config = load_config("config.yaml")
    num_simulations = config['num_simulations']
    time_horizon = config['time_horizon']

    for stock_config in config['stocks']:
        ticker = stock_config['ticker']
        start_date = stock_config['start_date']
        end_date = stock_config['end_date']

        hist, current_price = get_stock_data(ticker, start_date, end_date)
        if hist is None or current_price is None:
            continue

        log_returns, annual_return, annual_volatility = calculate_returns(hist)
        last_price = hist['Close'].iloc[-1]

        params = (annual_return, annual_volatility, last_price)
        simulations = run_monte_carlo_enhanced(hist, params, num_simulations, time_horizon)

        mean_price, median_price, std_dev, percentile_5, percentile_95 = calculate_statistics(simulations)
        upside_potential = (mean_price - current_price) / current_price * 100

        results_table = create_results_table(ticker, current_price, last_price, mean_price, median_price, std_dev, percentile_5, percentile_95, upside_potential)
        config_table = create_config_table(num_simulations, time_horizon, annual_return, annual_volatility)

        # Train ML models first
        predictions, metrics, scaler, models = train_ml_models(hist, time_horizon)

        # Create layout and tables
        layout = Layout()
        layout.split_column(
            Layout(name="upper", ratio=2),
            Layout(name="lower")
        )
        layout["upper"].split_row(
            Layout(name="table", ratio=2),
            Layout(name="config_table", ratio=1)
        )
        layout["upper"]["table"].update(Panel(results_table))
        layout["upper"]["config_table"].update(Panel(config_table))

        # Create and add ML table
        ml_table = Table(title="Machine Learning Model Performance")
        ml_table.add_column("Model", style="cyan")
        ml_table.add_column("R² Score", style="magenta")
        ml_table.add_column("RMSE", style="magenta")
        
        for model_name, metric in metrics.items():
            ml_table.add_row(
                model_name,
                f"{metric['R2']:.4f}",
                f"${metric['RMSE']:.2f}"
            )
        
        layout["lower"].update(Panel(ml_table))
        console.print(layout)

        potential_return_current = (mean_price - current_price) / current_price * 100
        console.print(f"Potential Return from Current Price {current_price}: [bold green]{potential_return_current:.2f}%[/bold green]")

        # Now plot with ML metrics
        plot_path = plot_monte_carlo_results(
            simulations, time_horizon, current_price, mean_price, 
            median_price, std_dev, percentile_5, percentile_95, 
            upside_potential, ticker, num_simulations, run_dir,
            ml_metrics=metrics
        )
        
        console.print(f"[green]Saved plot for {ticker} to: {plot_path}[/green]")

if __name__ == '__main__':
    main()