import sys
import os
import pandas as pd
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from forecaster import Forecaster

def test_forecaster_safe_amount():
    current_balance = 2000.0
    min_balance = 500.0
    
    # 90 day window: 2026-09-01 to 2026-11-30
    events_df = pd.DataFrame([
        # Settled but future date? (Shouldn't happen, but let's say pending debit)
        {'event_id': 'evt1', 'status': 'pending', 'direction': 'debit', 'amount_home_currency': 300.0, 'event_date': '2026-09-10', 'settlement_date': '2026-09-10', 'category': 'rent', 'flexibility': 'fixed'}
    ])
    
    recurring = [
        {'category': 'salary', 'direction': 'credit', 'pattern': 'monthly', 'forecast_amount': 2000.0, 'last_date': '2026-08-15', 'description': 'Salary'}
    ]
    
    forecaster = Forecaster(current_balance, min_balance, events_df, recurring)
    
    # Simulating 2026-09-01 for 90 days.
    # Initial balance = 2000
    # Day 10: -300 -> 1700
    # Next salary: 2026-09-15 (+2000 -> 3700)
    # Next salary: 2026-10-15 (+2000 -> 5700)
    # Next salary: 2026-11-15 (+2000 -> 7700)
    # Min projected balance without any payment = 1700 (on Sept 10)
    # So surplus = 1700 - 500 = 1200
    # If requested_amount = 1500, amount_safe_to_pay = 1200.
    
    amt_safe = forecaster.calculate_amount_safe_to_pay('2026-09-01', 1500.0)
    assert amt_safe == 1200.0

    # If requested_amount = 1000, amount_safe_to_pay = 1000
    amt_safe_2 = forecaster.calculate_amount_safe_to_pay('2026-09-01', 1000.0)
    assert amt_safe_2 == 1000.0
