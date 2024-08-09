import yfinance as yf

def get_stock_info(ticker):
    """
    Returns all available information for a given stock ticker.
    
    Args:
        ticker (str): The stock ticker symbol.
        
    Returns:
        dict: A dictionary containing all available information for the stock.
    """
    stock = yf.Ticker(ticker)
    return stock.info


print(get_stock_info('AAPL'))