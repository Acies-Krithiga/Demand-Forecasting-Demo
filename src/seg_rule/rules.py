import pandas as pd
import sys
import os

# Add the project root to the path to import config
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.config import SEGMENTATION_CONFIG

class ForecastRuleAssigner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def assign_rules(self):
        rules = []
        algorithms = []

        for _, row in self.df.iterrows():
            # Get and clean the data for the current row
            intermittency = str(row.get("Intermitent", "")).strip().lower()
            plc_status = str(row.get("PLC_Status", "")).strip().lower()
            cycle_classification = str(row.get("Cycle_Classification", "")).strip().lower()
            variability = str(row.get("Variability_Segment", "")).strip().upper()
            volume = str(row.get("Volume_Segment", "")).strip().upper()
            seasonality = str(row.get("Seasonal", "")).strip().lower()
            
            # Convert trend to 'yes'/'no'
            trend_input = str(row.get("Trend", "")).strip().lower()
            trend = "yes" if trend_input in ["upward", "downward"] else "no"
            
            # --- Default values ---
            rule_assigned = "No Rule Found"
            algorithms_assigned = "N/A"

            # --- Rule Assignment Logic based on the CORRECT image ---
            if intermittency == "yes":
                rule_assigned = "Rule 1"
                algorithms_assigned = "Croston, Seasonal Naive YoY, Weighted AOA"
            
            elif plc_status == "disco":
                rule_assigned = "Rule 2"
                algorithms_assigned = "Naive Random Walk"
            
            elif plc_status == "npi":
                rule_assigned = "Rule 3"
                algorithms_assigned = "Moving Average, SES, Naive Random Walk"

            elif plc_status == "mature":
                is_long_series = 'greater' in cycle_classification or '>= 2' in cycle_classification
                
                if not is_long_series: # Length of Series < 2 Cycles
                    rule_assigned = "Rule 4"
                    algorithms_assigned = "ETS, DES, Simple Snaive, Weighted Snaive Simple AOA, Weighted AOA"
                
                else: # Length of Series >= 2 Cycles (More than 2 cycles)
                    if variability == 'X':
                        rule_assigned = "Rule 5"
                        algorithms_assigned = "AR-NNET, Auto ARIMA, Prophet, sARIMA, STLF, TBATS, TES, Theta, Weighted Snaive, Growth Snaive, Simple AOA, Weighted AOA"
                    
                    elif variability == 'Y':
                        if volume == 'B':
                            rule_assigned = "Rule 6"
                            algorithms_assigned = "SES, Moving Average, TES, Weighted Snaive, Growth Snaive, Weighted AOA"
                        
                        elif volume == 'A':
                            if trend == 'yes':
                                rule_assigned = "Rule 7"
                                algorithms_assigned = "sARIMA, STLF, TES, Theta, Simple Snaive, Weighted Snaive"
                            else: # trend == 'no'
                                if seasonality == 'yes':
                                    rule_assigned = "Rule 8"
                                    algorithms_assigned = "sARIMA, STLF, TBATS, TES, Weighted Snaive, Growth Snaive, Weighted AOA, Growth AOA"
                                else:
                                    # This case isn't explicitly shown in the tree, but we'll assign Rule 7 as fallback
                                    rule_assigned = "Rule 7"
                                    algorithms_assigned = "sARIMA, STLF, TES, Theta, Simple Snaive, Weighted Snaive"
            
            rules.append(rule_assigned)
            algorithms.append(algorithms_assigned)
            
        self.df["Rule"] = rules
        self.df["Algorithms_Used"] = algorithms
        
        # Remove intermediate columns used for rule assignment
        intermediate_cols = ["Intermitent", "PLC_Status", "Cycle_Classification", 
                           "Variability_Segment", "Volume_Segment", "Seasonal", "Trend"]
        self.df.drop(columns=intermediate_cols, inplace=True)
        
        return self.df

