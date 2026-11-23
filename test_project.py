from unittest.mock import patch

import pandas as pd
from app import app, current_results
from feature_engineering import engineer_features, validate_columns

def test_routes():
    client=app.test_client()
    for path in ['/', '/dashboard', '/anomalies', '/results', '/blog', '/upload']:
        r=client.get(path); assert r.status_code==200, (path,r.status_code)

def test_anomaly_pagination():
    client=app.test_client()
    rows=[{'Date':'2025-01-01','Product ID':f'SKU-{i}','Category':'Food','Store ID':'S-1','Inventory Level':100,'Units Sold':20,'Inventory_Turnover':.2,'Anomaly Score':.5,'Status':'ANOMALY','Recommended Action':'Review'} for i in range(60)]
    with patch('app.current_results', return_value=pd.DataFrame(rows)):
        first=client.get('/anomalies')
        second=client.get('/anomalies?page=2')
    assert first.status_code==200 and b'Page 1 of 2' in first.data and b'/details/0' in first.data
    assert second.status_code==200 and b'Page 2 of 2' in second.data and b'/details/50' in second.data

def test_schema_and_results():
    raw=pd.read_csv('data/retail_store_inventory.csv').head(20)
    assert validate_columns(raw)==[]
    df=engineer_features(raw)
    assert {'Stock_to_Sales_Ratio','Days_of_Inventory','Demand_Change'}.issubset(df.columns)
    result=current_results(); assert len(result)>0 and {'Anomaly Score','Status','Key Indicators'}.issubset(result.columns)

if __name__=='__main__':
    test_routes(); test_anomaly_pagination(); test_schema_and_results(); print('smoke tests passed')
