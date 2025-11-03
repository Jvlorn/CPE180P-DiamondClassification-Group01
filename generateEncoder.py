# create_encoders_from_csv.py
import pandas as pd
import json
import os

def create_label_encoder(values):
    """Create label encoder mapping from a list of values"""
    uniques = sorted(list(pd.Series(values).dropna().unique()))
    mapping = {v: i for i, v in enumerate(uniques)}
    inverse = {i: v for i, v in enumerate(uniques)}
    return {'mapping': mapping, 'inverse': inverse}

def main():
    csv_path = 'diamond_data.csv'
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"📊 Loaded dataset with {len(df)} samples")
    
    # Create encoders for each attribute
    encoders = {}
    attributes = ['polish', 'symmetry', 'fluorescence', 'clarity']
    
    for attr in attributes:
        if attr in df.columns:
            encoder_data = create_label_encoder(df[attr])
            encoders[attr] = encoder_data
            print(f"✅ Created {attr} encoder with {len(encoder_data['mapping'])} classes: {list(encoder_data['mapping'].keys())}")
        else:
            print(f"❌ Column '{attr}' not found in CSV")
    
    # Save encoder files
    for attr in attributes:
        if attr in encoders:
            filename = f"{attr}_encoder.json"
            with open(filename, 'w') as f:
                json.dump(encoders[attr], f, indent=2)
            print(f"💾 Saved {filename}")
    
    print("🎉 All encoder files created!")

if __name__ == "__main__":
    main()