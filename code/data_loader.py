import pandas as pd
import os
from typing import Dict, Any

class DataLoader:
    def __init__(self, data_dir: str = "dataset"):
        self.data_dir = data_dir
        
        # DataFrames
        self.requests = pd.DataFrame()
        self.profiles = pd.DataFrame()
        self.events = pd.DataFrame()
        self.rates = pd.DataFrame()
        self.payment_options = pd.DataFrame()
        self.messages = pd.DataFrame()
        self.images = pd.DataFrame()
        
    def load_all(self):
        """Loads all required datasets."""
        self.requests = pd.read_csv(os.path.join(self.data_dir, "requests.csv"))
        self.profiles = pd.read_csv(os.path.join(self.data_dir, "financial_profiles.csv"))
        self.events = pd.read_csv(os.path.join(self.data_dir, "financial_events.csv"))
        self.rates = pd.read_csv(os.path.join(self.data_dir, "exchange_rates.csv"))
        self.payment_options = pd.read_csv(os.path.join(self.data_dir, "request_payment_options.csv"))
        self.messages = pd.read_csv(os.path.join(self.data_dir, "messages.csv"))
        self.images = pd.read_csv(os.path.join(self.data_dir, "images.csv"))
        
        self._normalize_currencies()
        
    def _get_exchange_rate(self, rate_date: str, from_currency: str, to_currency: str) -> float:
        if pd.isna(from_currency) or from_currency == to_currency:
            return 1.0
            
        rate_row = self.rates[
            (self.rates['rate_date'] == rate_date) & 
            (self.rates['from_currency'] == from_currency) & 
            (self.rates['to_currency'] == to_currency)
        ]
        
        if not rate_row.empty:
            return rate_row.iloc[0]['rate']
            
        # Try reverse rate if direct rate is not found
        reverse_rate_row = self.rates[
            (self.rates['rate_date'] == rate_date) & 
            (self.rates['from_currency'] == to_currency) & 
            (self.rates['to_currency'] == from_currency)
        ]
        if not reverse_rate_row.empty:
            return 1.0 / reverse_rate_row.iloc[0]['rate']
            
        raise ValueError(f"Missing exchange rate: {from_currency} -> {to_currency} on {rate_date}")

    def _normalize_currencies(self):
        """Converts all event amounts to the user's home currency."""
        # Create a mapping of user_id to home_currency
        user_home_currency = dict(zip(self.profiles['user_id'], self.profiles['home_currency']))
        
        # Add home_currency to events
        self.events['home_currency'] = self.events['user_id'].map(user_home_currency)
        
        # Convert event amounts
        converted_amounts = []
        for _, row in self.events.iterrows():
            if pd.isna(row['amount']):
                converted_amounts.append(pd.NA)
                continue
                
            rate = self._get_exchange_rate(
                rate_date=row['settlement_date'], 
                from_currency=row['currency'], 
                to_currency=row['home_currency']
            )
            converted_amounts.append(row['amount'] * rate)
            
        self.events['amount_home_currency'] = converted_amounts
        # Also convert minimum_allowed_amount if present
        converted_min_amounts = []
        for _, row in self.events.iterrows():
            if pd.isna(row['minimum_allowed_amount']):
                converted_min_amounts.append(pd.NA)
                continue
                
            rate = self._get_exchange_rate(
                rate_date=row['settlement_date'], 
                from_currency=row['currency'], 
                to_currency=row['home_currency']
            )
            converted_min_amounts.append(row['minimum_allowed_amount'] * rate)
            
        self.events['min_allowed_amount_home_currency'] = converted_min_amounts

if __name__ == "__main__":
    loader = DataLoader()
    loader.load_all()
    print("Data loaded successfully.")
