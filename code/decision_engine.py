import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from forecaster import Forecaster
from datetime import datetime

class DecisionEngine:
    def __init__(self, request_row: pd.Series, profile_row: pd.Series, payment_options: pd.DataFrame, forecaster: Forecaster):
        self.request = request_row
        self.profile = profile_row
        self.payment_options = payment_options
        self.forecaster = forecaster
        
        self.req_id = self.request['request_id']
        self.req_date = self.request['request_date']
        self.req_amt = self.request['requested_amount']
        self.desired_date = self.request['desired_completion_date']
        self.allows_partial = str(self.request['allows_partial_payment']).lower() == 'true'
        
        methods_str = str(self.profile['payment_methods_user_will_consider'])
        self.accepted_methods = methods_str.split('|') if pd.notna(methods_str) else []
        
        max_inst_months = self.profile['max_installment_months']
        self.max_installments = int(max_inst_months) if pd.notna(max_inst_months) and str(max_inst_months).strip() != '' else None
        
        self.amount_safe_to_pay = self.forecaster.calculate_amount_safe_to_pay(self.req_date, self.req_amt)
        self.earliest_full = self.forecaster.find_earliest_date_for_full_payment(self.req_date, self.req_amt)
        
    def _is_safe(self, plan: List[Tuple[str, float]], spending_changes: Optional[List[Dict[str, Any]]] = None) -> bool:
        is_safe, _ = self.forecaster.simulate(self.req_date, days=90, candidate_plan=plan, spending_changes=spending_changes)
        return is_safe

    def get_candidate_plans(self, spending_changes: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        candidates = []
        
        has_changes = bool(spending_changes)
        change_str = 'none'
        if has_changes:
            sc = spending_changes[0]
            if sc['action'] == 'stop':
                change_str = f"stop:{sc['event_id']}"
            elif sc['action'] == 'reduce_to':
                change_str = f"reduce_to:{sc['event_id']}:{sc['new_amount']}"
        
        # 1. full_payment
        if 'full_payment' in self.accepted_methods:
            plan = [(self.req_date, self.req_amt)]
            if self._is_safe(plan, spending_changes):
                candidates.append({
                    'method': 'full_payment',
                    'plan': plan,
                    'total_paid': self.req_amt,
                    'completion_date': self.req_date,
                    'spending_changes': has_changes,
                    'spending_change_str': change_str,
                    'first_payment_date': self.req_date,
                    'num_payments': 1,
                    'payment_option_id': ''
                })
                
        # 2. partial_payment
        if 'partial_payment' in self.accepted_methods and self.allows_partial:
            if 0 < self.amount_safe_to_pay < self.req_amt and self.earliest_full != '':
                if pd.to_datetime(self.earliest_full) <= pd.to_datetime(self.desired_date):
                    remainder = self.req_amt - self.amount_safe_to_pay
                    plan = [(self.req_date, self.amount_safe_to_pay), (self.earliest_full, remainder)]
                    if self._is_safe(plan, spending_changes):
                        candidates.append({
                            'method': 'partial_payment',
                            'plan': plan,
                            'total_paid': self.req_amt,
                            'completion_date': self.earliest_full,
                            'spending_changes': has_changes,
                            'spending_change_str': change_str,
                            'first_payment_date': self.req_date,
                            'num_payments': 2,
                            'payment_option_id': ''
                        })
                        
        # 3. installments
        if 'installments' in self.accepted_methods and self.max_installments is not None:
            # check available options
            req_options = self.payment_options[self.payment_options['request_id'] == self.req_id]
            for _, opt in req_options.iterrows():
                if opt['payment_method'] == 'installments':
                    months = opt['number_of_payments'] # rough assumption based on standard rules
                    # Wait, max_installment_months is max months.
                    # If frequency is monthly, number_of_payments <= max_installment_months.
                    # Or we calculate total duration in months.
                    # freq = opt['payment_frequency_days']
                    freq_days = opt['payment_frequency_days']
                    num_pay = opt['number_of_payments']
                    total_days = (num_pay - 1) * freq_days
                    total_months = total_days / 30.44
                    
                    if total_months <= self.max_installments:
                        # build plan
                        amt = opt['payment_amount']
                        start = pd.to_datetime(opt['first_payment_date'])
                        plan = []
                        curr = start
                        for i in range(num_pay):
                            plan.append((curr.strftime('%Y-%m-%d'), amt))
                            curr += pd.Timedelta(days=freq_days)
                            
                        if self._is_safe(plan, spending_changes):
                            candidates.append({
                                'method': 'installments',
                                'plan': plan,
                                'total_paid': opt['total_payable_amount'],
                                'completion_date': plan[-1][0],
                                'spending_changes': has_changes,
                                'spending_change_str': change_str,
                                'first_payment_date': plan[0][0],
                                'num_payments': num_pay,
                                'payment_option_id': opt['payment_option_id']
                            })
                            
        # 4. wait
        if 'full_payment' in self.accepted_methods and self.earliest_full != '' and pd.to_datetime(self.earliest_full) > pd.to_datetime(self.req_date):
            # wait implies we just wait until earliest_full and pay full.
            # But the 'wait' payment plan is usually empty or none?
            # Problem says: "Use none when no payment is recommended. Installment plans must exactly match..."
            # Wait, if method is 'wait', what is the payment plan? Output meaning says: "use 'none' when no payment is recommended."
            # Actually, `wait` means we don't pay today. Wait, if `wait` is selected, `payment_plan` = `none`?
            # Let's check sample_requests.csv for `wait`.
            # We will use 'none' for wait.
            candidates.append({
                'method': 'wait',
                'plan': [],
                'total_paid': 0, # wait has no payments now
                'completion_date': self.earliest_full, # logically completes later
                'spending_changes': has_changes,
                'spending_change_str': change_str,
                'first_payment_date': '9999-12-31', # sort last
                'num_payments': 0,
                'payment_option_id': ''
            })

        return candidates

    def rank_candidates(self, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not candidates:
            return None
            
        def sort_key(c):
            # 1. Completes by desired date (True is better, so negate)
            completes_in_time = pd.to_datetime(c['completion_date']) <= pd.to_datetime(self.desired_date)
            # 2. Requires no spending changes (False is better, so just use boolean since False < True)
            has_changes = c['spending_changes']
            # 3. Minimizes total amount paid
            total_paid = c['total_paid']
            # 4. Starts payment earlier (date string compare)
            starts = c['first_payment_date']
            # 5. Uses fewer payments
            num_pay = c['num_payments']
            # 6. Lowest payment_option_id
            opt_id = c['payment_option_id'] or 'zzzzzz'
            
            return (not completes_in_time, has_changes, total_paid, starts, num_pay, opt_id)
            
        candidates.sort(key=sort_key)
        return candidates[0]

    def decide(self) -> Dict[str, Any]:
        baseline_candidates = self.get_candidate_plans()
        best = self.rank_candidates(baseline_candidates)
        
        # Check if unsafe or incomplete (wait is considered incomplete for 'affordable_now' or 'affordable_with_plan', 
        # or if completion_date > desired_date)
        needs_fix = False
        if not best:
            needs_fix = True
        elif best['method'] == 'wait':
            needs_fix = True
        elif pd.to_datetime(best['completion_date']) > pd.to_datetime(self.desired_date):
            needs_fix = True
            
        if needs_fix:
            from datetime import timedelta
            
            # Find adjustable and protected categories
            adj_cat_str = str(self.profile.get('adjustable_categories', ''))
            allowed_cats = [c.strip() for c in adj_cat_str.split('|')] if pd.notna(adj_cat_str) and adj_cat_str.strip() else []
            
            prot_cat_str = str(self.profile.get('protected_categories', ''))
            protected_cats = [c.strip() for c in prot_cat_str.split('|')] if pd.notna(prot_cat_str) and prot_cat_str.strip() else []
            
            # Find flexible recurring events to try and stop/reduce
            end_date = (pd.to_datetime(self.req_date) + timedelta(days=90)).strftime('%Y-%m-%d')
            flows = self.forecaster._generate_timeline(self.req_date, end_date)
            flex_events = []
            if not flows.empty:
                flex_mask = (
                    (flows['is_flexible'] == True) & 
                    (flows['amount'] < 0) & 
                    (flows['category'].isin(allowed_cats)) &
                    (~flows['category'].isin(protected_cats))
                )
                flex_events = flows[flex_mask]['event_id'].unique().tolist()
                
            for flex_evt in flex_events:
                # Try reduce_to 50% first
                orig_amt = abs(flows[flows['event_id'] == flex_evt]['amount'].iloc[0])
                new_amt = round(orig_amt * 0.5, 2)
                sc_reduce = [{'action': 'reduce_to', 'event_id': flex_evt, 'new_amount': new_amt}]
                
                sc_candidates_reduce = self.get_candidate_plans(spending_changes=sc_reduce)
                best_sc_reduce = self.rank_candidates(sc_candidates_reduce)
                
                if best_sc_reduce and best_sc_reduce['method'] != 'wait' and pd.to_datetime(best_sc_reduce['completion_date']) <= pd.to_datetime(self.desired_date):
                    best = best_sc_reduce
                    break
                
                # If reduce_to isn't enough, try stop
                sc_stop = [{'action': 'stop', 'event_id': flex_evt}]
                sc_candidates_stop = self.get_candidate_plans(spending_changes=sc_stop)
                best_sc_stop = self.rank_candidates(sc_candidates_stop)
                
                if best_sc_stop and best_sc_stop['method'] != 'wait' and pd.to_datetime(best_sc_stop['completion_date']) <= pd.to_datetime(self.desired_date):
                    best = best_sc_stop
                    break
                    
        affordability = 'not_affordable'
        method = 'not_recommended'
        plan_str = 'none'
        spending_changes_needed = 'none'
        
        if best:
            method = best['method']
            # Round amounts for display
            rounded_plan = [(d, round(a, 2)) for d, a in best['plan']]
            
            if method == 'full_payment':
                affordability = 'affordable_now'
                plan_str = "|".join([f"{d}:{a}" for d, a in rounded_plan])
            elif method == 'partial_payment' or method == 'installments':
                affordability = 'affordable_with_plan'
                plan_str = "|".join([f"{d}:{a}" for d, a in rounded_plan])
            elif method == 'wait':
                affordability = 'affordable_later'
                plan_str = 'none'
        else:
            if self.earliest_full != '':
                if self.earliest_full == self.req_date:
                    affordability = 'affordable_now'
                else:
                    affordability = 'affordable_later'
            else:
                affordability = 'not_affordable'
                
        return {
            'amount_safe_to_pay': round(self.amount_safe_to_pay, 2),
            'affordability_status': affordability,
            'recommended_payment_method': method,
            'payment_plan': plan_str,
            'earliest_date_for_full_payment': self.earliest_full,
            'spending_changes_needed': best['spending_change_str'] if best else 'none',
            'decision_explanation': f"Recommended {method} based on 90-day safety check."
        }
