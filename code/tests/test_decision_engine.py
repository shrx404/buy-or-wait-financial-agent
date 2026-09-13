import sys
import os
import pandas as pd
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from forecaster import Forecaster
from decision_engine import DecisionEngine

class MockForecaster(Forecaster):
    def __init__(self, amount_safe, earliest_full):
        self.amount_safe = amount_safe
        self.earliest_full = earliest_full
        
    def calculate_amount_safe_to_pay(self, *args, **kwargs):
        return self.amount_safe
        
    def find_earliest_date_for_full_payment(self, *args, **kwargs):
        return self.earliest_full
        
    def simulate(self, *args, **kwargs):
        return True, 1000.0
        
    def _generate_timeline(self, start, end):
        return pd.DataFrame()

def test_partial_payment_exact_sum():
    request_row = pd.Series({
        'request_id': 'req1',
        'request_date': '2026-09-01',
        'requested_amount': 1000.0,
        'desired_completion_date': '2026-09-30',
        'allows_partial_payment': True
    })
    
    profile_row = pd.Series({
        'payment_methods_user_will_consider': 'partial_payment',
        'max_installment_months': ''
    })
    
    payment_options = pd.DataFrame()
    
    forecaster = MockForecaster(amount_safe=400.0, earliest_full='2026-09-15')
    engine = DecisionEngine(request_row, profile_row, payment_options, forecaster)
    
    decision = engine.decide()
    
    assert decision['recommended_payment_method'] == 'partial_payment'
    plan_parts = decision['payment_plan'].split('|')
    assert len(plan_parts) == 2
    sum_paid = 0
    for p in plan_parts:
        date, amt = p.split(':')
        sum_paid += float(amt)
    
    assert sum_paid == 1000.0
    assert plan_parts[0] == '2026-09-01:400.0'
    assert plan_parts[1] == '2026-09-15:600.0'

def test_installments_exact_match():
    request_row = pd.Series({
        'request_id': 'req2',
        'request_date': '2026-09-01',
        'requested_amount': 1000.0,
        'desired_completion_date': '2026-12-01',
        'allows_partial_payment': False
    })
    
    profile_row = pd.Series({
        'payment_methods_user_will_consider': 'installments',
        'max_installment_months': '3'
    })
    
    payment_options = pd.DataFrame([
        {
            'request_id': 'req2',
            'payment_option_id': 'opt1',
            'payment_method': 'installments',
            'number_of_payments': 3,
            'payment_frequency_days': 30,
            'first_payment_date': '2026-09-01',
            'payment_amount': 350.0,
            'total_payable_amount': 1050.0
        }
    ])
    
    forecaster = MockForecaster(amount_safe=0.0, earliest_full='')
    engine = DecisionEngine(request_row, profile_row, payment_options, forecaster)
    
    decision = engine.decide()
    assert decision['recommended_payment_method'] == 'installments'
    
    plan_parts = decision['payment_plan'].split('|')
    assert len(plan_parts) == 3
    assert plan_parts[0] == '2026-09-01:350.0'
    # 30 days from 09-01 is 10-01
    assert plan_parts[1] == '2026-10-01:350.0'

def test_spending_changes_never_conflict():
    # If a spending change is generated, it should not have both stop and reduce_to for the same event
    # We implemented `stop` only, so this is trivially true, but we assert the structure.
    request_row = pd.Series({
        'request_id': 'req3',
        'request_date': '2026-09-01',
        'requested_amount': 1000.0,
        'desired_completion_date': '2026-09-10',
        'allows_partial_payment': False
    })
    profile_row = pd.Series({
        'payment_methods_user_will_consider': 'full_payment',
        'max_installment_months': ''
    })
    
    class UnsafeForecaster(MockForecaster):
        def simulate(self, start_date, days=90, candidate_plan=None, spending_changes=None):
            # Safe only if we have spending changes
            if spending_changes:
                return True, 1000.0
            return False, 0.0
            
        def _generate_timeline(self, start, end):
            # return one flexible event
            return pd.DataFrame([{'is_flexible': True, 'amount': -200, 'event_id': 'flex1'}])
            
    forecaster = UnsafeForecaster(amount_safe=0.0, earliest_full='')
    engine = DecisionEngine(request_row, profile_row, pd.DataFrame(), forecaster)
    decision = engine.decide()
    
    # It should have recommended a spending change
    sc = decision['spending_changes_needed']
    assert sc != 'none'
    assert sc.startswith('stop:')
    assert 'reduce_to:' not in sc
