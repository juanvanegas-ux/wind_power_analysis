"""Download hourly wind data for La Guajira, Colombia, from the Open-Meteo
historical archive (free, no API key) and save it as a CSV.

La Guajira hosts Colombia's largest wind-energy resource. The dataset feeds the
exploratory analysis and the wind-power forecasting model in this repo.

Run:  python src/fetch_data.py
"""

import os

import pandas as pd
import requests

import config

# pulled from config so the site only lives in one place
LATITUDE = config.LATITUDE
LONGITUDE = config.LONGITUDE
START_DATE = config.START_DATE
END_DATE = config.END_DATE

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OUT = config.DATA


def fetch() -> pd.DataFrame:
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": "wind_speed_100m,wind_speed_10m,wind_direction_100m,temperature_2m",
        "wind_speed_unit": "ms",
        "timezone": "America/Bogota",
    }
    print(f"Requesting {START_DATE} to {END_DATE} for ({LATITUDE}, {LONGITUDE}) ...")
    resp = requests.get(ARCHIVE_URL, params=params, timeout=120)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]

    df = pd.DataFrame(hourly).rename(
        columns={
            "time": "timestamp",
            "wind_speed_100m": "wind_speed_100m_ms",
            "wind_speed_10m": "wind_speed_10m_ms",
            "wind_direction_100m": "wind_direction_100m_deg",
            "temperature_2m": "temperature_c",
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def main():
    df = fetch()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df):,} hourly rows -> {os.path.abspath(OUT)}")
    print(df.head())


if __name__ == "__main__":
    main()
