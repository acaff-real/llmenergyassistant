import random
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import pandas as pd

def generate_sample_data(days=30, blocks_per_day=96):
    engine = create_engine('mysql+pymysql://root:password1234@localhost/goober_schema')
    
    data = []
    current_date = datetime.now() - timedelta(days=days)
    block_counter = 1
    
    for _ in range(days):
        for block in range(1, blocks_per_day + 1):
            # Randomly determine if this is a buy or sell order
            is_buy = random.choice([True, False])
            
            record = {
                'blocknumber': block_counter,
                'Date': current_date.date(),
                'BuySell': 'B' if is_buy else 'S',
                'Purchase offer': round(random.uniform(0, 1000), 2) if is_buy else 0,
                'Sell offer': 0 if is_buy else round(random.uniform(0, 1000), 2),
                'Quantity': round(random.uniform(1, 100), 2),
                'Price per unit': round(random.uniform(10, 100), 2)
            }
            record['Invoice'] = record['Quantity'] * record['Price per unit']
            
            data.append(record)
            block_counter += 1
        
        current_date += timedelta(days=1)
    
    df = pd.DataFrame(data)
    df.to_sql('electricity_bids2', engine, if_exists='replace', index=False)
    print(f"Generated {len(data)} records across {days} days")

if __name__ == "__main__":
    generate_sample_data(days=30, blocks_per_day=96)