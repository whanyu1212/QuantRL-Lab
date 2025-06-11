from datetime import datetime
from typing import Union

import pandas as pd
from loguru import logger


def convert_datetime_to_alpha_vantage_format(date_input: Union[str, datetime]) -> str:
    """
    Convert various date formats to Alpha Vantage's YYYYMMDDTHHMM
    format.

    Args:
        date_input: Date in formats like:
            - "2022-04-10" (date only, defaults to 00:00)
            - "2022-04-10 09:30" (date with time)
            - datetime object

    Returns:
        str: Date in YYYYMMDDTHHMM format (e.g., "20220410T0930")

    Examples:
        >>> convert_to_alpha_vantage_format("2022-04-10")
        "20220410T0000"

        >>> convert_to_alpha_vantage_format("2022-04-10 09:30")
        "20220410T0930"
    """
    if isinstance(date_input, str):
        try:
            # If date only (YYYY-MM-DD), add default time
            if len(date_input) == 10 and "T" not in date_input and ":" not in date_input:
                date_input += " 00:00"

            dt = pd.to_datetime(date_input)
        except Exception as e:
            logger.error(f"Failed to parse date '{date_input}': {e}")
            raise ValueError(f"Invalid date format: {date_input}")

    elif isinstance(date_input, datetime):
        dt = date_input
    else:
        raise ValueError(f"Date must be string or datetime object, got {type(date_input)}")

    # Convert to Alpha Vantage format: YYYYMMDDTHHMM
    return dt.strftime("%Y%m%dT%H%M")
