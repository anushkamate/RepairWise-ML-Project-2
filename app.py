from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "repairwise_dataset.csv"
TARGET = "recommended_action"

app = FastAPI(title="RepairWise API", version="1.0.0")

def build_model():
    if not DATA_PATH.exists():
        raise RuntimeError(f"Dataset not found: {DATA_PATH}")
    data = pd.read_csv(DATA_PATH)
    if len(data) != 5000:
        raise RuntimeError(f"Expected 5,000 dataset rows; found {len(data)}")
    X = data.drop(columns=[TARGET])
    y = data[TARGET]
    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    numeric = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical),
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), numeric)
    ])

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", DecisionTreeClassifier(
            max_depth=5, random_state=42, class_weight="balanced"
        ))
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(
            y_test, y_pred, labels=pipeline.named_steps["classifier"].classes_
        ).tolist(),
        "classes": pipeline.named_steps["classifier"].classes_.tolist(),
        "dataset_rows": int(len(data)),
        "label_counts": data[TARGET].value_counts().to_dict(),
    }
    return pipeline, X.columns.tolist(), metrics

MODEL, FEATURE_COLUMNS, METRICS = build_model()

class DeviceInput(BaseModel):
    device_type: Literal["Smartphone", "Laptop", "Tablet", "TV", "Washing Machine", "Refrigerator"]
    fault_type: Literal["Battery", "Screen", "Charging", "Software", "Motherboard", "Motor", "Cooling", "Other"]
    device_age_years: int = Field(ge=0, le=15)
    repair_cost: float = Field(ge=0, le=500000)
    replacement_cost: float = Field(gt=0, le=1000000)
    fault_severity: Literal["Minor", "Moderate", "Severe"]
    previous_repairs: int = Field(ge=0, le=10)
    parts_available: Literal["Yes", "No"]

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML

@app.get("/api/health")
def health():
    return {"status": "ok", "dataset_rows": METRICS["dataset_rows"]}

@app.get("/api/metrics")
def metrics():
    return {
        "accuracy": METRICS["accuracy"],
        "report": METRICS["report"],
        "confusion_matrix": METRICS["confusion_matrix"],
        "classes": METRICS["classes"],
        "dataset_rows": METRICS["dataset_rows"],
        "label_counts": METRICS["label_counts"],
    }

@app.post("/api/assess")
def assess(payload: DeviceInput):
    try:
        row = pd.DataFrame([payload.model_dump()]).reindex(columns=FEATURE_COLUMNS)
        prediction = str(MODEL.predict(row)[0])
        probabilities = MODEL.predict_proba(row)[0]
        scores = {
            str(label): round(float(prob), 4)
            for label, prob in zip(MODEL.named_steps["classifier"].classes_, probabilities)
        }
        ratio = payload.repair_cost / payload.replacement_cost
        return {
            "prediction": prediction,
            "model_scores": scores,
            "repair_cost_ratio": round(ratio, 4),
            "dataset_rows": METRICS["dataset_rows"],
            "disclaimer": "Synthetic-data demonstration only; not validated repair advice."
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not assess device: {exc}") from exc

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>RepairWise — Repair or Replace?</title>
<style>
:root{--bg:#f5f7fb;--card:#fff;--ink:#172033;--muted:#667085;--line:#e5e9f2;--accent:#275efe;--accent2:#eaf0ff;--good:#087443;--warn:#9a6700}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif}
header{background:#101828;color:#fff;padding:28px 20px}header .wrap,main{max-width:1100px;margin:auto}
.brand{font-weight:800;letter-spacing:-.04em;font-size:28px}.brand span{color:#8fb0ff}
header p{color:#d0d5dd;margin:8px 0 0;max-width:700px;line-height:1.5}
main{padding:24px 20px 50px}.notice{background:#fff7e6;border:1px solid #f4d58d;color:#754d00;padding:12px 14px;border-radius:12px;font-size:13px;line-height:1.5;margin-bottom:20px}
.layout{display:grid;grid-template-columns:1.15fr .85fr;gap:20px}.card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:22px;box-shadow:0 8px 24px rgba(16,24,40,.04)}
h2{font-size:18px;margin:0 0 6px} .sub{font-size:13px;color:var(--muted);margin:0 0 20px;line-height:1.5}
.fields{display:grid;grid-template-columns:1fr 1fr;gap:15px}.field{display:flex;flex-direction:column;gap:7px}.field.full{grid-column:1/-1}
label{font-size:13px;font-weight:650;color:#344054}input,select{width:100%;border:1px solid #d0d5dd;border-radius:10px;padding:11px 12px;background:white;color:var(--ink);font:inherit;font-size:14px;outline:none}
input:focus,select:focus{border-color:var(--accent);box-shadow:0 0 0 3px #275efe20}
button{border:0;border-radius:11px;background:var(--accent);color:#fff;font-weight:700;padding:13px 18px;font-size:14px;cursor:pointer;width:100%;margin-top:18px}button:disabled{opacity:.6;cursor:wait}
.result{margin-top:18px;border-radius:13px;background:#f8fafc;border:1px solid var(--line);padding:16px;display:none}.result h3{margin:0 0 8px;font-size:17px}.result p{font-size:13px;color:var(--muted);line-height:1.5;margin:6px 0}
.metric{background:#f8faff;border:1px solid #e0e8ff;border-radius:13px;padding:16px;margin-top:12px}.metric .big{font-size:27px;font-weight:800;letter-spacing:-.04em}.metric .label{font-size:12px;color:var(--muted);margin-top:4px}
.bar{height:8px;background:#edf0f5;border-radius:20px;overflow:hidden;margin:6px 0 12px}.bar i{display:block;height:100%;background:var(--accent);border-radius:20px}
.classrow{font-size:12px;display:flex;justify-content:space-between;gap:10px}.small{font-size:12px;color:var(--muted);line-height:1.6}
.pills{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px}.pill{font-size:12px;border-radius:30px;padding:7px 10px;background:#f2f4f7;color:#475467}
footer{max-width:1100px;margin:auto;padding:0 20px 30px;color:var(--muted);font-size:12px}
@media(max-width:780px){.layout{grid-template-columns:1fr}.fields{grid-template-columns:1fr}.field.full{grid-column:auto}header{padding:24px 18px}main{padding:18px 14px 36px}.card{padding:18px}}
</style>
</head>
<body>
<header><div class="wrap"><div class="brand">🔧 Repair<span>Wise</span></div><p>ML-assisted repair-versus-replace decision support for everyday electronic devices.</p></div></header>
<main>
<div class="notice"><strong>Academic prototype:</strong> The 5,000 records and recommendation labels are synthetic and rule-generated. This demo is not validated repair advice. Confirm device safety, warranty, diagnosis, parts availability, and actual quotes with a qualified technician.</div>
<div class="layout">
<section class="card">
<h2>Assess a device</h2><p class="sub">Enter the device and fault details. The Decision Tree model will return a suggested category.</p>
<form id="deviceForm">
<div class="fields">
<div class="field"><label for="device_type">Device type</label><select id="device_type"><option>Smartphone</option><option>Laptop</option><option>Tablet</option><option>TV</option><option>Washing Machine</option><option>Refrigerator</option></select></div>
<div class="field"><label for="fault_type">Fault type</label><select id="fault_type"><option>Battery</option><option>Screen</option><option>Charging</option><option>Software</option><option>Motherboard</option><option>Motor</option><option>Cooling</option><option>Other</option></select></div>
<div class="field"><label for="device_age_years">Device age (years)</label><input id="device_age_years" type="number" min="0" max="15" value="3" required></div>
<div class="field"><label for="fault_severity">Fault severity</label><select id="fault_severity"><option>Minor</option><option selected>Moderate</option><option>Severe</option></select></div>
<div class="field"><label for="repair_cost">Estimated repair cost (₹)</label><input id="repair_cost" type="number" min="0" max="500000" step="100" value="3500" required></div>
<div class="field"><label for="replacement_cost">Replacement cost (₹)</label><input id="replacement_cost" type="number" min="1" max="1000000" step="500" value="18000" required></div>
<div class="field"><label for="previous_repairs">Previous repairs</label><input id="previous_repairs" type="number" min="0" max="10" value="1" required></div>
<div class="field"><label for="parts_available">Replacement parts available?</label><select id="parts_available"><option>Yes</option><option>No</option></select></div>
</div>
<button id="submitBtn" type="submit">Assess device</button>
</form>
<div class="result" id="result" aria-live="polite"></div>
</section>
<aside class="card">
<h2>Model overview</h2><p class="sub">This demo trains a Decision Tree using the included 5,000-row synthetic dataset.</p>
<div class="metric"><div class="big" id="datasetRows">5,000</div><div class="label">synthetic training records</div></div>
<div class="metric"><div class="big" id="accuracy">Loading…</div><div class="label">held-out test accuracy (agreement with synthetic labels only)</div></div>
<h2 style="margin-top:22px">Label distribution</h2><div id="labels" class="small">Loading dataset summary…</div>
<div class="pills"><span class="pill">75% train / 25% test</span><span class="pill">Preprocessing pipeline</span><span class="pill">Decision Tree</span></div>
</aside>
</div>
</main>
<footer>RepairWise • Educational ML prototype • Synthetic data does not establish real-world performance.</footer>
<script>
const form = document.getElementById('deviceForm');
const result = document.getElementById('result');
const btn = document.getElementById('submitBtn');
function moneyRatio(x){return (x*100).toFixed(1)+'%'}
form.addEventListener('submit', async (e)=>{
  e.preventDefault(); btn.disabled=true; btn.textContent='Assessing…'; result.style.display='block'; result.innerHTML='<p>Running the model…</p>';
  const payload={
    device_type:document.getElementById('device_type').value,
    fault_type:document.getElementById('fault_type').value,
    device_age_years:Number(document.getElementById('device_age_years').value),
    repair_cost:Number(document.getElementById('repair_cost').value),
    replacement_cost:Number(document.getElementById('replacement_cost').value),
    fault_severity:document.getElementById('fault_severity').value,
    previous_repairs:Number(document.getElementById('previous_repairs').value),
    parts_available:document.getElementById('parts_available').value
  };
  try{
    const res=await fetch('/api/assess',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const data=await res.json(); if(!res.ok) throw new Error(data.detail||'Request failed');
    const scores=Object.entries(data.model_scores).sort((a,b)=>b[1]-a[1]);
    result.innerHTML=`<h3>Suggested category: ${data.prediction}</h3>
      <p>Model score is a model output, not a guarantee or real-world probability of repair success.</p>
      <div style="margin-top:12px"><strong style="font-size:12px">Model class scores</strong>${scores.map(([name,val])=>`<div style="margin-top:10px"><div class="classrow"><span>${name}</span><b>${moneyRatio(val)}</b></div><div class="bar"><i style="width:${Math.max(0,Math.min(100,val*100))}%"></i></div></div>`).join('')}</div>
      <p><strong>Repair / replacement cost ratio:</strong> ${moneyRatio(data.repair_cost_ratio)}</p>
      <p>Confirm the diagnosis, parts availability, safety, warranty, and actual quotes with a qualified technician.</p>`;
  }catch(err){result.innerHTML=`<h3>Could not assess device</h3><p>${err.message}</p>`}
  finally{btn.disabled=false;btn.textContent='Assess device'}
});
fetch('/api/metrics').then(r=>r.json()).then(data=>{
  document.getElementById('datasetRows').textContent=Number(data.dataset_rows).toLocaleString();
  document.getElementById('accuracy').textContent=(data.accuracy*100).toFixed(1)+'%';
  const labels=document.getElementById('labels');
  labels.innerHTML=Object.entries(data.label_counts).sort((a,b)=>b[1]-a[1]).map(([name,count])=>`<div style="margin-top:12px"><div class="classrow"><span>${name}</span><b>${Number(count).toLocaleString()}</b></div><div class="bar"><i style="width:${count/data.dataset_rows*100}%"></i></div></div>`).join('');
}).catch(()=>{document.getElementById('accuracy').textContent='Unavailable';document.getElementById('labels').textContent='Dataset summary could not be loaded.'});
</script>
</body></html>"""
