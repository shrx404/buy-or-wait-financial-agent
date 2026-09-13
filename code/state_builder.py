import pandas as pd
from typing import List, Dict, Any, Optional

class StateBuilder:
    def __init__(self, user_events: pd.DataFrame, user_profile: pd.Series, extracted_facts: Optional[Dict[str, Any]] = None):
        """
        user_events: DataFrame of financial_events for a specific user, with amount_home_currency added.
        user_profile: Series of financial_profile for the user.
        extracted_facts: Dictionary of facts extracted from messages/images.
            e.g. {
                'event_01': {'status': 'cancelled'},
                'event_02': {'amount': 150.0},
                'event_03': {'status': 'settled', 'amount': 100.0}
            }
        """
        self.events = user_events.copy()
        self.profile = user_profile
        self.facts = extracted_facts or {}
        
        # We will parse events into standard objects or just a processed DataFrame.
        self._apply_facts()
        self._resolve_lifecycle_links()
        self._filter_valid_cash_flows()
        self._detect_recurrences()
        
    def _apply_facts(self):
        """Apply amendments, cancellations, or extracted amounts from messages/images."""
        # For each event, if we have extracted facts, update it.
        # This covers Conflict Resolution (1) Explicit cancellation, settlement, or amendment.
        for event_id, fact_updates in self.facts.items():
            idx = self.events.index[self.events['event_id'] == event_id]
            if not idx.empty:
                for k, v in fact_updates.items():
                    if k == 'amount':
                        self.events.loc[idx, 'amount_home_currency'] = v
                    elif k == 'status':
                        self.events.loc[idx, 'status'] = v
                    elif k == 'settlement_date':
                        self.events.loc[idx, 'settlement_date'] = v
                        
    def _resolve_lifecycle_links(self):
        """
        De-duplicate repeated representations.
        Conflict Resolution:
        (1) Explicit cancellation/settlement/amendment (handled in facts or by status)
        (2) Newer record from the same source (event_id order or event_date)
        (3) Settled event over estimate or forecast.
        (4) Financially safer.
        """
        # Build a map of linked events
        # A linked_event_id points to an EARLIER event in the same transaction.
        # We want to group by these chains and pick the active one.
        
        # Find all root events
        chain_map = {}
        for _, row in self.events.iterrows():
            eid = row['event_id']
            lid = row['linked_event_id']
            
            # Follow links to find the root
            curr = lid
            while pd.notna(curr) and curr in self.events['event_id'].values:
                # Find the parent's linked_event_id
                parent_row = self.events[self.events['event_id'] == curr].iloc[0]
                if pd.notna(parent_row['linked_event_id']):
                    curr = parent_row['linked_event_id']
                else:
                    break
                    
            root = curr if pd.notna(curr) else eid
            if root not in chain_map:
                chain_map[root] = []
            chain_map[root].append(row.to_dict())
            
        resolved_events = []
        for root, chain in chain_map.items():
            if len(chain) == 1:
                resolved_events.append(chain[0])
                continue
                
            # We have multiple events in a lifecycle chain.
            # 1. Prefer cancelled/failed to mark the whole chain dead, OR prefer settled.
            # If any is settled, use the settled one.
            settled_events = [e for e in chain if e['status'] == 'settled']
            if settled_events:
                # If multiple settled (rare), pick the latest by event_date or highest amount if debit (safer)
                # Sort by event_date desc
                settled_events.sort(key=lambda x: x['event_date'], reverse=True)
                resolved_events.append(settled_events[0])
                continue
                
            cancelled_events = [e for e in chain if e['status'] in ['cancelled', 'failed']]
            if cancelled_events:
                # Chain is cancelled
                cancelled_events.sort(key=lambda x: x['event_date'], reverse=True)
                resolved_events.append(cancelled_events[0])
                continue
                
            # Otherwise, pick the newest record (by event_date, then event_id string sort as fallback)
            chain.sort(key=lambda x: (x['event_date'], x['event_id']), reverse=True)
            resolved_events.append(chain[0])
            
        self.events = pd.DataFrame(resolved_events)

    def _filter_valid_cash_flows(self):
        """
        Apply inclusion/exclusion rules:
        - Reserve pending debits.
        - Exclude pending credits, bonuses, commissions, refunds, lottery, investment gains until settled.
        - Exclude unrealized/non-cash records.
        """
        valid_indices = []
        for idx, row in self.events.iterrows():
            if pd.isna(row['amount_home_currency']):
                # Missing amount and no extracted fact
                continue
                
            # Exclude unrealized non-cash
            if row['status'] == 'unrealized':
                continue
                
            # Exclude failed/cancelled
            if row['status'] in ['cancelled', 'failed']:
                continue
                
            # If pending/scheduled
            if row['status'] in ['pending', 'scheduled', 'forecast', 'estimate']:
                if row['direction'] == 'debit':
                    # Reserve pending debits
                    valid_indices.append(idx)
                else:
                    # Pending credit
                    # Exclude pending credits until settled (salary is a credit, reserve it only on settlement_date, meaning if it's pending in the future, we DO count it if it's confirmed salary. Wait!
                    # "Count confirmed salary only on its settlement date, never earlier."
                    # If it's a future scheduled salary, do we count it?
                    # The prompt says: "Count confirmed salary only on its settlement date, never earlier."
                    # This means we DO include it in the forecast for that future date.
                    # But "Exclude pending credits, bonuses, commissions, refunds, lottery proceeds, and investment gains until they are actually settled."
                    # Salary is typically category='salary'. If it's salary, and it's scheduled/pending, we CAN count it on its settlement_date?
                    # "Exclude pending credits ... until they are actually settled."
                    # Wait, if we exclude them from the forecast completely, we assume 0 income?
                    # Ah! "Count confirmed salary only on its settlement date" means in the FORECAST, it appears on the settlement date.
                    # If it's a bonus, we do NOT forecast it.
                    if row['category'] == 'salary':
                        valid_indices.append(idx)
                    else:
                        pass # Ignore other pending credits
            elif row['status'] == 'settled':
                valid_indices.append(idx)
                
        self.events = self.events.loc[valid_indices].copy()

    def _detect_recurrences(self):
        """
        Detect recurrence only where the event history actually supports it.
        We will group by category, description, direction and look for regular intervals.
        """
        self.recurring_patterns = []
        
        # Group settled events
        settled = self.events[self.events['status'] == 'settled'].copy()
        settled['event_date_ts'] = pd.to_datetime(settled['event_date'])
        
        groups = settled.groupby(['category', 'direction'])
        for (category, direction), group in groups:
            if len(group) >= 2:
                # Sort by date
                group = group.sort_values('event_date_ts')
                # Calculate diffs in days
                diffs = group['event_date_ts'].diff().dt.days.dropna()
                # If there is a consistent pattern (e.g. ~30 days for monthly, ~7 for weekly)
                avg_diff = diffs.mean()
                if 25 <= avg_diff <= 35:
                    pattern = 'monthly'
                elif 6 <= avg_diff <= 8:
                    pattern = 'weekly'
                elif 13 <= avg_diff <= 15:
                    pattern = 'biweekly'
                else:
                    pattern = 'irregular'
                    
                if pattern != 'irregular':
                    # Determine average/max amount
                    # "Forecast essential variable spending conservatively" -> max or 90th percentile
                    # For debits, conservative = max. For credits (salary), conservative = min.
                    amounts = group['amount_home_currency']
                    if direction == 'debit':
                        forecast_amt = amounts.max()
                    else:
                        forecast_amt = amounts.min()
                        
                    self.recurring_patterns.append({
                        'category': category,
                        'direction': direction,
                        'pattern': pattern,
                        'forecast_amount': forecast_amt,
                        'last_date': group['event_date_ts'].max().strftime('%Y-%m-%d'),
                        'description': group.iloc[-1]['description']
                    })

    def get_valid_events(self) -> pd.DataFrame:
        return self.events
        
    def get_recurring_patterns(self) -> List[Dict[str, Any]]:
        return self.recurring_patterns

if __name__ == "__main__":
    import numpy as np
    # Quick test logic
    df = pd.DataFrame([
        {'event_id': 'e1', 'user_id': 'u1', 'event_type': 'income', 'description': 'Salary', 'category': 'salary', 'direction': 'credit', 'amount_home_currency': 1000, 'event_date': '2023-01-01', 'settlement_date': '2023-01-01', 'status': 'settled', 'linked_event_id': np.nan, 'flexibility': 'fixed'},
        {'event_id': 'e2', 'user_id': 'u1', 'event_type': 'income', 'description': 'Salary', 'category': 'salary', 'direction': 'credit', 'amount_home_currency': 1000, 'event_date': '2023-02-01', 'settlement_date': '2023-02-01', 'status': 'settled', 'linked_event_id': np.nan, 'flexibility': 'fixed'},
        {'event_id': 'e3', 'user_id': 'u1', 'event_type': 'expense', 'description': 'Rent', 'category': 'rent', 'direction': 'debit', 'amount_home_currency': 500, 'event_date': '2023-02-01', 'settlement_date': '2023-02-01', 'status': 'pending', 'linked_event_id': np.nan, 'flexibility': 'fixed'}
    ])
    builder = StateBuilder(df, pd.Series({'user_id': 'u1'}))
    print("Valid events:")
    print(builder.get_valid_events()[['event_id', 'status', 'amount_home_currency']])
    print("Recurring:")
    print(builder.get_recurring_patterns())
