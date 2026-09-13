import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from data_loader import DataLoader
from state_builder import StateBuilder
from forecaster import Forecaster
from decision_engine import DecisionEngine
from untrusted_parser import UntrustedParser
import json

def main():
    print("Loading data...")
    loader = DataLoader()
    loader.load_all()
    
    print("Initializing parser...")
    parser = UntrustedParser()
    
    # Process all messages
    print("Extracting facts from messages...")
    message_facts = parser.parse_messages(loader.messages)
    
    # Process all images
    print("Extracting facts from images...")
    image_facts = parser.parse_images(loader.images, loader.events)
    
    # Merge facts
    all_facts = {}
    for eid, f in message_facts.items():
        all_facts[eid] = f
    for eid, f in image_facts.items():
        if eid not in all_facts:
            all_facts[eid] = {}
        all_facts[eid].update(f)
        
    outputs = []
    total_requests = len(loader.requests)
    
    print(f"Processing {total_requests} requests...")
    for i, (idx, req) in enumerate(loader.requests.iterrows()):
        req_id = req['request_id']
        user_id = req['user_id']
        
        # Get user specific data
        profile = loader.profiles[loader.profiles['user_id'] == user_id].iloc[0]
        user_events = loader.events[loader.events['user_id'] == user_id]
        
        # Build financial state
        state_builder = StateBuilder(user_events, profile, extracted_facts=all_facts)
        valid_events = state_builder.get_valid_events()
        recurring = state_builder.get_recurring_patterns()
        
        # Forecaster
        current_balance = float(profile['current_available_balance'])
        min_balance = float(profile['minimum_balance_to_keep'])
        forecaster = Forecaster(current_balance, min_balance, valid_events, recurring)
        
        # Decision Engine
        engine = DecisionEngine(req, profile, loader.payment_options, forecaster)
        decision = engine.decide()
        
        output_row = {
            'request_id': req_id,
            'amount_safe_to_pay': decision['amount_safe_to_pay'],
            'affordability_status': decision['affordability_status'],
            'recommended_payment_method': decision['recommended_payment_method'],
            'payment_plan': decision['payment_plan'],
            'earliest_date_for_full_payment': decision['earliest_date_for_full_payment'],
            'spending_changes_needed': decision['spending_changes_needed'],
            'decision_explanation': decision['decision_explanation']
        }
        outputs.append(output_row)
        
        if (i + 1) % 25 == 0:
            print(f"Processed {i + 1}/{total_requests} requests.")
            
    # Write output.csv
    print("Writing output.csv...")
    output_df = pd.DataFrame(outputs)
    # Required columns in exact order
    columns = [
        'request_id', 'amount_safe_to_pay', 'affordability_status', 'recommended_payment_method',
        'payment_plan', 'earliest_date_for_full_payment', 'spending_changes_needed', 'decision_explanation'
    ]
    output_df = output_df[columns]
    output_df.to_csv("output.csv", index=False)
    
    # Write usage report
    print("Writing usage report...")
    os.makedirs("evaluation", exist_ok=True)
    usage = parser.usage
    avg_tokens = (usage['input_tokens'] + usage['output_tokens']) / max(total_requests, 1)
    avg_cost = usage['cost_estimate'] / max(total_requests, 1)
    
    with open("evaluation/usage_report.md", "w") as f:
        f.write("# Model Usage Report\n\n")
        f.write("This report summarizes the LLM usage for the final full-dataset run.\n\n")
        f.write(f"**Models Used**: {', '.join(usage['models_used']) if usage['models_used'] else 'None'}\n")
        f.write(f"**Total Model Calls**: {usage['model_calls']}\n")
        f.write(f"**Total Input Tokens**: {usage['input_tokens']}\n")
        f.write(f"**Total Output Tokens**: {usage['output_tokens']}\n")
        f.write(f"**Average Tokens per Request**: {avg_tokens:.2f}\n")
        f.write(f"**Estimated Total Cost**: ${usage['cost_estimate']:.4f}\n")
        f.write(f"**Estimated Cost per Request**: ${avg_cost:.4f}\n")
        
    print("Done!")

if __name__ == "__main__":
    main()
