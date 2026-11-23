from pathlib import Path
import json
import io
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from sklearn.ensemble import IsolationForest
from werkzeug.utils import secure_filename
from feature_engineering import engineer_features, validate_columns, FEATURE_COLUMNS
from anomaly_detection import score_models, explain_record, recommend_action

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
app.secret_key = 'inventory-anomaly-demo-key'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
MODEL_DIR = ROOT / 'models'; OUTPUT_DIR = ROOT / 'outputs'; UPLOAD_DIR = ROOT / 'uploads'
for p in [OUTPUT_DIR, UPLOAD_DIR]: p.mkdir(exist_ok=True)
ANALYSIS_META_PATH = OUTPUT_DIR / 'current_analysis.json'
MODEL_OUTPUT_COLUMNS = {'Anomaly Score', 'anomaly_score', 'kde_score', 'gmm_score', 'kmeans_score', 'baseline_score', 'pca_1', 'pca_2', 'cluster', 'Status', 'Key Indicators', 'Recommended Action'}


def artifacts():
    required = [MODEL_DIR/'preprocessing.pkl', MODEL_DIR/'models.pkl', MODEL_DIR/'metadata.json']
    if not all(p.exists() for p in required): raise FileNotFoundError('Model artifacts are missing. Run python train_model.py first.')
    return joblib.load(required[0]), joblib.load(required[1]), json.loads(required[2].read_text())


def current_results():
    path = OUTPUT_DIR / 'current_results.csv'
    if not path.exists(): path = OUTPUT_DIR / 'anomaly_results.csv'
    if not path.exists(): return pd.DataFrame()
    return pd.read_csv(path)


def view_rows(df, limit=200):
    if df.empty: return []
    out = df.copy()
    if 'Date' in out:
        out['Date'] = pd.to_datetime(out['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
    return out.head(limit).replace({np.nan: None}).to_dict('records')


def numeric_feature_columns(df):
    return [col for col in df.select_dtypes(include=[np.number]).columns if col not in MODEL_OUTPUT_COLUMNS]


def current_analysis_metadata():
    if not ANALYSIS_META_PATH.exists(): return {}
    return json.loads(ANALYSIS_META_PATH.read_text())


def save_current_analysis(df, metadata):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR/'current_results.csv', index=False)
    ANALYSIS_META_PATH.write_text(json.dumps(metadata))


def summary(df):
    if df.empty: return {'total':0,'normal':0,'anomalies':0,'pct':0,'avg_stock':0,'avg_sales':0,'avg_turnover':0}
    status = df.get('Status', pd.Series('NORMAL', index=df.index)).fillna('NORMAL')
    anomaly = status.eq('ANOMALY')
    def average(col, digits):
        if col not in df: return 0
        value=pd.to_numeric(df[col], errors='coerce').mean()
        return round(float(value),digits) if pd.notna(value) else 0
    return {'total':len(df),'normal':int((~anomaly).sum()),'anomalies':int(anomaly.sum()),'pct':round(float(anomaly.mean()*100),2),'avg_stock':average('Inventory Level',1),'avg_sales':average('Units Sold',1),'avg_turnover':average('Inventory_Turnover',2)}


def chart_data(df):
    data = {'status_labels':['NORMAL','ANOMALY'],'status_values':[0,0],'score_labels':[],'score_values':[],'features':[],'feature_name':None,'feature_stats':{},'category_name':None,'category_labels':[],'category_values':[],'trend_name':None,'trend_labels':[],'trend_values':[],'threshold':None}
    if df.empty: return data

    statuses = df.get('Status', pd.Series('NORMAL', index=df.index)).fillna('NORMAL').astype(str)
    data['status_values'] = [int(statuses.eq(label).sum()) for label in data['status_labels']]
    scores = pd.to_numeric(df.get('Anomaly Score', pd.Series(0, index=df.index)), errors='coerce').fillna(0).clip(0, 1)
    bins = np.linspace(0, 1, 11)
    counts, edges = np.histogram(scores, bins=bins)
    data['score_labels'] = [f'{edges[i]:.1f}–{edges[i + 1]:.1f}' for i in range(len(edges) - 1)]
    data['score_values'] = counts.astype(int).tolist()

    features = numeric_feature_columns(df)
    data['features'] = features
    data['feature_name'] = features[0] if features else None
    for feature in features:
        values = pd.to_numeric(df[feature], errors='coerce')
        feature_stats = []
        for label in data['status_labels']:
            group = values[statuses.eq(label)].dropna()
            feature_stats.append(float(group.median()) if not group.empty else None)
        data['feature_stats'][feature] = feature_stats

    excluded = MODEL_OUTPUT_COLUMNS | set(features)
    categories = []
    for col in df.columns:
        if col in excluded or pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        values = df[col].fillna('Unknown').astype(str)
        unique_count = values.nunique()
        if 2 <= unique_count <= 12:
            priority = 0 if any(word in col.lower() for word in ('category','region','type','group')) else 1
            categories.append((priority, col, values))
    if categories:
        _, category_name, values = min(categories, key=lambda item: (item[0], item[1]))
        grouped = pd.DataFrame({'category':values,'anomaly':statuses.eq('ANOMALY')}).groupby('category')['anomaly'].agg(['mean','count'])
        grouped = grouped.sort_values('count', ascending=False).head(10)
        data['category_name'] = category_name
        data['category_labels'] = grouped.index.astype(str).tolist()
        data['category_values'] = (grouped['mean'] * 100).round(2).tolist()

    for col in df.columns:
        if col in MODEL_OUTPUT_COLUMNS or not ('date' in col.lower() or 'time' in col.lower() or pd.api.types.is_datetime64_any_dtype(df[col])):
            continue
        dates = pd.to_datetime(df[col], errors='coerce')
        valid = dates.notna()
        if valid.sum() < 2 or dates[valid].nunique() < 2: continue
        unique_dates = dates[valid].nunique()
        if unique_dates <= 60:
            periods = dates[valid].dt.strftime('%Y-%m-%d')
        elif unique_dates <= 180:
            periods = dates[valid].dt.to_period('W').astype(str)
        else:
            periods = dates[valid].dt.to_period('M').astype(str)
        grouped = pd.DataFrame({'period':periods,'anomaly':statuses[valid].eq('ANOMALY').to_numpy()}).groupby('period')['anomaly'].agg(['mean','count']).sort_index().tail(24)
        data['trend_name'] = col
        data['trend_labels'] = grouped.index.astype(str).tolist()
        data['trend_values'] = (grouped['mean'] * 100).round(2).tolist()
        break
    return data


def run_inference(raw, threshold=None):
    pre, models, metadata = artifacts()
    df = engineer_features(raw)
    X = pre.transform(df[FEATURE_COLUMNS])
    scores = score_models(models, X)
    threshold = float(threshold if threshold is not None else metadata['threshold'])
    for k,v in scores.items(): df[k] = v
    df['Anomaly Score'] = df['anomaly_score'].round(6)
    df['Status'] = np.where(df['anomaly_score'] >= threshold, 'ANOMALY', 'NORMAL')
    df['Key Indicators'] = df.apply(explain_record, axis=1)
    df['Recommended Action'] = df.apply(recommend_action, axis=1)
    # Derived turnover is a clear, project-specific KPI.
    df['Inventory_Turnover'] = (df['Units Sold'] / df['Inventory Level'].replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).fillna(0)
    save_current_analysis(df, {'generic':False,'threshold':threshold,'method':'Saved inventory ensemble','features':numeric_feature_columns(df)})
    return df, threshold


def run_generic_inference(raw):
    if len(raw) < 2:
        raise ValueError('Upload at least two data rows so the records can be compared.')
    df = raw.copy()
    df.columns = [str(column).strip() for column in df.columns]
    if df.columns.duplicated().any():
        raise ValueError('The CSV contains duplicate column names after trimming whitespace.')

    numeric = {}
    for column in df.columns:
        converted = pd.to_numeric(df[column], errors='coerce').replace([np.inf, -np.inf], np.nan)
        present = df[column].notna() & df[column].astype(str).str.strip().ne('')
        if not present.any() or converted[present].notna().mean() < .9:
            continue
        if converted.nunique(dropna=True) < 2:
            continue
        numeric[column] = converted
    if not numeric:
        raise ValueError('This CSV needs at least one numeric column with two or more different values.')

    feature_frame = pd.DataFrame(numeric, index=df.index)
    valid_rows = feature_frame.notna().any(axis=1)
    df = df.loc[valid_rows].reset_index(drop=True)
    feature_frame = feature_frame.loc[valid_rows].reset_index(drop=True)
    if len(df) < 2:
        raise ValueError('At least two rows need usable numeric values for analysis.')
    feature_frame = feature_frame.fillna(feature_frame.median())
    X = feature_frame.to_numpy(dtype=float)

    model = IsolationForest(n_estimators=200, contamination='auto', random_state=42, n_jobs=1)
    model.fit(X)
    raw_scores = -model.decision_function(X)
    raw_threshold = float(np.quantile(raw_scores, .95))
    minimum, maximum = float(np.min(raw_scores)), float(np.max(raw_scores))
    if maximum > minimum:
        scores = (raw_scores - minimum) / (maximum - minimum)
        threshold = float(np.clip((raw_threshold - minimum) / (maximum - minimum), 0, 1))
        status = np.where(raw_scores >= raw_threshold, 'ANOMALY', 'NORMAL')
    else:
        scores = np.zeros(len(df), dtype=float)
        threshold = 0.0
        status = np.full(len(df), 'NORMAL')

    for column in feature_frame.columns:
        df[column] = feature_frame[column]
    df['Anomaly Score'] = scores.round(6)
    df['Status'] = status
    feature_names = ', '.join(feature_frame.columns[:5])
    df['Key Indicators'] = f'Compared with other rows using numeric fields: {feature_names}.'
    df['Recommended Action'] = np.where(status == 'ANOMALY','Review this record and its source data.','No unusual pattern was detected.')
    save_current_analysis(df, {'generic':True,'threshold':threshold,'method':'Isolation Forest · 95th-percentile score cutoff','features':list(feature_frame.columns)})
    return df, threshold

@app.context_processor
def inject(): return {'project_name':'Inventory Intelligence','source_url':'https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset'}

@app.route('/')
def index():
    try: df=current_results(); meta=artifacts()[2]
    except FileNotFoundError: df=pd.DataFrame(); meta={}
    return render_template('index.html', summary=summary(df), meta=meta)

@app.route('/dashboard')
def dashboard():
    df=current_results()
    metadata=current_analysis_metadata()
    generic=bool(metadata.get('generic',not df.empty and bool(validate_columns(df))))
    threshold=metadata.get('threshold')
    if threshold is None and not generic:
        try: threshold=artifacts()[2].get('threshold')
        except FileNotFoundError: pass
    charts=chart_data(df)
    charts['threshold']=float(threshold) if threshold is not None else None
    return render_template('dashboard.html', summary=summary(df), charts=charts, generic=generic, features=charts['features'])

@app.route('/anomalies')
def anomalies():
    df=current_results()
    if not df.empty:
        for col in ['Product ID','Store ID','Category','Status']:
            val=request.args.get(col.lower().replace(' ','_'))
            if val and col in df: df=df[df[col].astype(str).eq(val)]
        try:
            minimum=float(request.args.get('min_score','0'))
            if 'Anomaly Score' in df: df=df[df['Anomaly Score']>=minimum]
        except ValueError: pass
    page_size=50
    total=len(df)
    total_pages=max(1,(total+page_size-1)//page_size)
    page=request.args.get('page',1,type=int) or 1
    page=min(max(1,page),total_pages)
    start=(page-1)*page_size
    page_df=df.iloc[start:start+page_size]
    rows=view_rows(page_df)
    for row,row_id in zip(rows,page_df.index):
        row['_row_id']=int(row_id)
        row['_row_number']=int(row_id)+1
    preferred=['Date','Product ID','Category','Store ID','Inventory Level','Units Sold','Inventory_Turnover','Anomaly Score','Status','Recommended Action']
    if validate_columns(df):
        source_columns=[col for col in df.columns if col not in MODEL_OUTPUT_COLUMNS][:6]
        columns=source_columns+['Anomaly Score','Status','Recommended Action']
    else:
        columns=[col for col in preferred if col in df.columns]
    products=sorted(df['Product ID'].dropna().astype(str).unique()) if 'Product ID' in df else []
    stores=sorted(df['Store ID'].dropna().astype(str).unique()) if 'Store ID' in df else []
    return render_template('anomalies.html', rows=rows, total=total, page=page, total_pages=total_pages, start_row=start+1 if total else 0, end_row=start+len(rows), columns=columns, filters={'products':products,'stores':stores})

@app.route('/details/<int:row_id>')
def details(row_id):
    df=current_results()
    if row_id<0 or row_id>=len(df): flash('That inventory record was not found.','warning'); return redirect(url_for('anomalies'))
    return render_template('details.html', row=df.iloc[row_id].to_dict(), row_id=row_id, columns=[col for col in df.columns if col not in {'pca_1','pca_2','kde_score','gmm_score','kmeans_score','baseline_score','cluster','anomaly_score'}])

@app.route('/upload', methods=['GET','POST'])
def upload():
    if request.method=='POST':
        file=request.files.get('file')
        if not file or not file.filename.lower().endswith('.csv'): flash('Please choose a valid CSV file.','danger'); return redirect(url_for('upload'))
        name=secure_filename(file.filename)
        path=UPLOAD_DIR/name; file.save(path)
        try:
            raw=pd.read_csv(path)
            if validate_columns(raw):
                df, threshold=run_generic_inference(raw)
                flash(f'Analysis complete: {len(df):,} records scored using numeric columns. Scores at or above the 95th-percentile cutoff are flagged.','success')
            else:
                df, threshold=run_inference(raw)
                flash(f'Analysis complete: {len(df):,} inventory records scored at the saved threshold ({threshold:.3f}).','success')
            return redirect(url_for('dashboard'))
        except Exception as exc: flash('The CSV could not be analyzed safely: ' + str(exc),'danger'); return redirect(url_for('upload'))
    return render_template('upload.html')

@app.route('/analyze')
def analyze():
    try: df, threshold=run_inference(pd.read_csv(ROOT/'data'/'retail_store_inventory.csv'))
    except Exception as exc: flash('Analysis failed: '+str(exc),'danger'); return redirect(url_for('index'))
    flash(f'Refreshed analysis for {len(df):,} observations using threshold {threshold:.3f}.','success'); return redirect(url_for('dashboard'))

@app.route('/results')
def results():
    df=current_results()
    if not df.empty and validate_columns(df): return redirect(url_for('dashboard'))
    return render_template('results.html', rows=view_rows(df, 100), summary=summary(df))

@app.route('/blog')
def blog(): return render_template('blog.html')

@app.route('/download')
def download():
    path=OUTPUT_DIR/'current_results.csv'
    if not path.exists(): path=OUTPUT_DIR/'anomaly_results.csv'
    return send_file(path, as_attachment=True, download_name='inventory_anomaly_results.csv', mimetype='text/csv')

@app.errorhandler(413)
def too_large(_): flash('File is too large. Please upload a CSV under 16 MB.','danger'); return redirect(url_for('upload'))

if __name__ == '__main__': app.run(host='0.0.0.0', port=5000, debug=False)
