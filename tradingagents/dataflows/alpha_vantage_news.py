import json

from .alpha_vantage_common import _make_api_request, format_datetime_for_api
from .errors import NoNewsError


def _raise_if_empty_feed(payload, message: str) -> None:
    """Raise ``NoNewsError`` when NEWS_SENTIMENT answered with no articles.

    An empty window is a *successful* response whose ``feed`` is an empty list
    (``{"items": "0", ..., "feed": []}``). Handed back as-is it reads to the
    router as a served request and ends the vendor chain, so a configured
    fallback — e.g. ``tool_vendors={"get_news": "alpha_vantage,firecrawl"}`` —
    would never run.

    Only a positively identified empty feed raises. A non-JSON body, a payload
    without a ``feed`` key, or any shape this does not recognize passes through
    untouched: mislabeling an unexpected response as "no news" would hide real
    articles, which is worse than the missed fallback this prevents.
    """
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return
    elif isinstance(payload, dict):
        parsed = payload
    else:
        return

    feed = parsed.get("feed") if isinstance(parsed, dict) else None
    if isinstance(feed, list) and not feed:
        raise NoNewsError(message)


def get_news(ticker, start_date, end_date) -> dict[str, str] | str:
    """Returns live and historical market news & sentiment data from premier news outlets worldwide.

    Covers stocks, cryptocurrencies, forex, and topics like fiscal policy, mergers & acquisitions, IPOs.

    Args:
        ticker: Stock symbol for news articles.
        start_date: Start date for news search.
        end_date: End date for news search.

    Returns:
        Dictionary containing news sentiment data or JSON string.

    Raises:
        NoNewsError: The feed came back empty, so the router can try the next
            vendor in the chain.
    """

    params = {
        "tickers": ticker,
        "time_from": format_datetime_for_api(start_date),
        "time_to": format_datetime_for_api(end_date),
    }

    response = _make_api_request("NEWS_SENTIMENT", params)
    _raise_if_empty_feed(
        response, f"No news found for {ticker} between {start_date} and {end_date}"
    )
    return response

def get_global_news(curr_date, look_back_days: int = 7, limit: int = 50) -> dict[str, str] | str:
    """Returns global market news & sentiment data without ticker-specific filtering.

    Covers broad market topics like financial markets, economy, and more.

    Args:
        curr_date: Current date in yyyy-mm-dd format.
        look_back_days: Number of days to look back (default 7).
        limit: Maximum number of articles (default 50).

    Returns:
        Dictionary containing global news sentiment data or JSON string.

    Raises:
        NoNewsError: The feed came back empty, so the router can try the next
            vendor in the chain.
    """
    from datetime import datetime, timedelta

    # Calculate start date
    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = curr_dt - timedelta(days=look_back_days)
    start_date = start_dt.strftime("%Y-%m-%d")

    params = {
        "topics": "financial_markets,economy_macro,economy_monetary",
        "time_from": format_datetime_for_api(start_date),
        "time_to": format_datetime_for_api(curr_date),
        "limit": str(limit),
    }

    response = _make_api_request("NEWS_SENTIMENT", params)
    _raise_if_empty_feed(
        response, f"No global news found between {start_date} and {curr_date}"
    )
    return response


def get_insider_transactions(symbol: str) -> dict[str, str] | str:
    """Returns latest and historical insider transactions by key stakeholders.

    Covers transactions by founders, executives, board members, etc.

    Args:
        symbol: Ticker symbol. Example: "IBM".

    Returns:
        Dictionary containing insider transaction data or JSON string.
    """

    params = {
        "symbol": symbol,
    }

    return _make_api_request("INSIDER_TRANSACTIONS", params)
