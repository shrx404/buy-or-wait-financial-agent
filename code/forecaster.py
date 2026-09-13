import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import dateutil.relativedelta

class Forecaster:
    def __init__(self, current_balance: float, min_balance: float, valid_events: pd.DataFrame, recurring_patterns: List[Dict[str, Any]]):
        self.initial_balance = current_balance
        self.min_balance = min_balance
        self.valid_events = valid_events
        self.recurring_patterns = recurring_patterns
        
    def _generate_timeline(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generates all future cash flows between start_date and end_date (inclusive)."""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        flows = []
        
        # 1. Add pending future events
        # We assume valid_events only contains the events we should include.
        # "Count confirmed salary only on its settlement date".
        # If an event is pending but valid (e.g. pending debit or confirmed salary), its settlement_date or event_date determines when it hits.
        for _, row in self.valid_events.iterrows():
            date_col = row['settlement_date'] if pd.notna(row['settlement_date']) else row['event_date']
            d = pd.to_datetime(date_col)
            if start <= d <= end:
                # If it's a settled event from the past, it's already in the current_balance.
                # Only include events that are strictly in the future relative to the "current time".
                # Wait, what is "current time"? 
                # "Forecast the user's balance for the next 90 days".
                # If the event_date > start_date (which is request_date), it hits the balance.
                # If an event has status='pending', it hasn't hit the balance yet, even if its date is today or in the past?
                # "Reserve pending debits" -> This means pending debits subtract from available balance immediately or on their scheduled date.
                # Let's place it on max(start, date_col).
                effective_date = max(start, d)
                
                # Only include pending or scheduled events, or settled events that somehow have future dates (unlikely).
                if row['status'] in ['pending', 'scheduled', 'forecast', 'estimate'] or (row['status'] == 'settled' and d > start):
                    amount = row['amount_home_currency']
                    if row['direction'] == 'debit':
                        amount = -amount
                    flows.append({'date': effective_date, 'amount': amount, 'event_id': row['event_id'], 'category': row['category'], 'is_flexible': row['flexibility'] == 'flexible'})

        # 2. Add recurring patterns
        for p in self.recurring_patterns:
            last_date = pd.to_datetime(p['last_date'])
            pattern = p['pattern']
            amt = p['forecast_amount']
            if p['direction'] == 'debit':
                amt = -amt
                
            curr_date = last_date
            while curr_date <= end:
                if pattern == 'monthly':
                    curr_date += dateutil.relativedelta.relativedelta(months=1)
                elif pattern == 'weekly':
                    curr_date += timedelta(days=7)
                elif pattern == 'biweekly':
                    curr_date += timedelta(days=14)
                else:
                    break
                    
                if start <= curr_date <= end:
                    # Determine flexibility: assume fixed unless we know it's flexible.
                    # We can refine this if we pass expense_categories_user_is_willing_to_reduce
                    flows.append({'date': curr_date, 'amount': amt, 'event_id': f"recurring_{p['category']}_{curr_date.strftime('%Y%m%d')}", 'category': p['category'], 'is_flexible': False})

        df_flows = pd.DataFrame(flows)
        if not df_flows.empty:
            df_flows = df_flows.sort_values('date')
        return df_flows

    def simulate(self, start_date: str, days: int = 90, candidate_plan: Optional[List[Tuple[str, float]]] = None, spending_changes: Optional[List[Dict[str, Any]]] = None) -> Tuple[bool, float]:
        """
        Simulate the balance for `days` starting from `start_date`.
        candidate_plan: list of (date_str, payment_amount_to_subtract)
        spending_changes: list of {'action': 'stop'|'reduce', 'category'/'event_id': ..., 'new_amount': ...}
        
        Returns:
            (is_safe, min_projected_balance)
        """
        start = pd.to_datetime(start_date)
        end = start + timedelta(days=days)
        
        flows = self._generate_timeline(start_date, end.strftime('%Y-%m-%d'))
        
        balance = self.initial_balance
        min_projected = balance
        
        # We can aggregate flows by day
        daily_flows = {}
        if not flows.empty:
            for _, row in flows.iterrows():
                d = row['date']
                amt = row['amount']
                
                # Apply spending changes
                if spending_changes and row['direction'] == 'debit':
                    # Check if this flow is stopped or reduced
                    pass # We will implement this if needed. Currently amount_safe_to_pay is BEFORE optional spending changes.
                    
                if d not in daily_flows:
                    daily_flows[d] = 0.0
                daily_flows[d] += amt
                
        if candidate_plan:
            for d_str, amt in candidate_plan:
                d = pd.to_datetime(d_str)
                if d not in daily_flows:
                    daily_flows[d] = 0.0
                daily_flows[d] -= amt # Payment is a debit
                
        # Simulate day by day
        curr_date = start
        while curr_date <= end:
            if curr_date in daily_flows:
                balance += daily_flows[curr_date]
            
            if balance < min_projected:
                min_projected = balance
                
            if balance < self.min_balance:
                # Early return if we drop below minimum
                return False, balance
                
            curr_date += timedelta(days=1)
            
        return True, min_projected

    def calculate_amount_safe_to_pay(self, request_date: str, requested_amount: float) -> float:
        """
        `amount_safe_to_pay`: the most the user can pay today before optional spending changes 
        without breaking the 90-day safety check, capped at `requested_amount`.
        """
        # Binary search or simple math?
        # Since we just subtract a single lump sum on request_date, it shifts the entire curve down by exactly that amount.
        # So we can simulate with 0 payment, find the minimum projected balance.
        # The amount safe to pay is min_projected - min_balance (capped at requested_amount).
        is_safe, min_proj = self.simulate(request_date, days=90)
        
        if not is_safe:
            return 0.0
            
        surplus = min_proj - self.min_balance
        if surplus <= 0:
            return 0.0
            
        return min(surplus, requested_amount)
        
    def find_earliest_date_for_full_payment(self, request_date: str, requested_amount: float, days: int = 90) -> str:
        """
        Finds the first date where a single payment of `requested_amount` is safe.
        Returns empty string if none found.
        """
        start = pd.to_datetime(request_date)
        end = start + timedelta(days=days)
        
        # Test each date
        curr = start
        while curr <= end:
            # Plan: pay requested_amount on curr
            is_safe, _ = self.simulate(request_date, days=days, candidate_plan=[(curr.strftime('%Y-%m-%d'), requested_amount)])
            if is_safe:
                return curr.strftime('%Y-%m-%d')
            curr += timedelta(days=1)
            
        return ""
