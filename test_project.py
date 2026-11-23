from io import BytesIO
from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
import app as app_module
from app import app, current_results
from feature_engineering import engineer_features, validate_columns

def test_routes():
    client=app.test_client()
    for path in ['/', '/dashboard', '/anomalies', '/results', '/blog', '/upload']:
        r=client.get(path); assert r.status_code==200, (path,r.status_code)
    assert client.get('/static/css/style.css').status_code==200

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

def test_generic_csv_upload_builds_dashboard_and_anomaly_charts():
    csv='Date,Category,Amount\n2025-01-01,A,10\n2025-01-02,A,11\n2025-01-03,B,12\n2025-01-04,B,100\n'
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        output_dir=root/'outputs'
        upload_dir=root/'uploads'
        upload_dir.mkdir()
        with patch.object(app_module,'OUTPUT_DIR',output_dir), \
             patch.object(app_module,'UPLOAD_DIR',upload_dir), \
             patch.object(app_module,'ANALYSIS_META_PATH',output_dir/'current_analysis.json'):
            response=app.test_client().post(
                '/upload',
                data={'file':(BytesIO(csv.encode()),'sample.csv')},
                content_type='multipart/form-data',
                follow_redirects=True,
            )
            assert response.status_code==200
            assert b'LIVE CSV' in response.data
            assert b'id="statusChart"' in response.data
            assert b'id="featureComparisonChart"' in response.data
            assert b'id="categoryChart"' in response.data
            assert b'id="trendChart"' in response.data
            assert b'Anomaly rate \xc2\xb7 Date' in response.data
            assert b'Accuracy cannot be measured without verified anomaly labels.' in response.data
            assert app.test_client().get('/anomalies').status_code==200

            results=current_results()
            charts=app_module.chart_data(results)
            assert sum(charts['score_values'])==4
            assert charts['feature_stats']['Amount']
            assert charts['category_name']=='Category'
            assert charts['trend_name']=='Date'

def test_generic_constant_scores_are_not_all_flagged():
    class ConstantScoreModel:
        def fit(self, values):
            return self

        def decision_function(self, values):
            return pd.Series([0.25] * len(values))

    raw=pd.DataFrame({'Amount':[10,20,30]})
    with tempfile.TemporaryDirectory() as directory:
        output_dir=Path(directory)
        with patch.object(app_module,'OUTPUT_DIR',output_dir), \
             patch.object(app_module,'ANALYSIS_META_PATH',output_dir/'current_analysis.json'), \
             patch.object(app_module,'IsolationForest',return_value=ConstantScoreModel()):
            results,_=app_module.run_generic_inference(raw)
    assert results['Status'].eq('NORMAL').all()

def test_vercel_blob_storage_round_trips_analysis():
    raw_results = b'Amount,Anomaly Score,Status\n10,0.1,NORMAL\n100,0.9,ANOMALY\n'
    raw_metadata = json.dumps({'generic':True,'threshold':0.8}).encode()
    blob = Mock()
    blob.get_blob.side_effect = lambda path, **_: SimpleNamespace(
        body=raw_results if path == app_module.BLOB_RESULTS_PATH else raw_metadata
    )
    sample = pd.DataFrame({'Amount':[10,100],'Anomaly Score':[0.1,0.9],'Status':['NORMAL','ANOMALY']})
    metadata = {'generic':True,'threshold':0.8}
    with patch.dict('os.environ', {'VERCEL':'1'}), \
         patch.object(app_module, 'blob_client', return_value=blob):
        app_module.save_current_analysis(sample, metadata)
        results = current_results()
        loaded_metadata = app_module.current_analysis_metadata()
        download = app.test_client().get('/download')

    assert results['Status'].tolist() == ['NORMAL','ANOMALY']
    assert loaded_metadata == metadata
    assert download.status_code == 200 and download.data == raw_results
    assert blob.put_blob.call_count == 2
    assert all(call.kwargs['access'] == 'private' for call in blob.put_blob.call_args_list)

if __name__=='__main__':
    test_routes(); test_anomaly_pagination(); test_schema_and_results(); print('smoke tests passed')
