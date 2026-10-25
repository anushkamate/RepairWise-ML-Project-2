# RepairWise — Vercel-ready version

A Vercel-compatible FastAPI web application that uses the included 5,000-row synthetic dataset and a scikit-learn Decision Tree.

## Deploy on Vercel

1. Extract this ZIP.
2. Create a new GitHub repository, for example `repairwise-vercel`.
3. Upload/push all files in this folder to the repository root:
   - `app.py`
   - `requirements.txt`
   - `vercel.json`
   - `repairwise_dataset.csv`
4. Sign in at https://vercel.com and choose **Add New → Project**.
5. Import the GitHub repository.
6. Vercel should detect the Python FastAPI app. Keep the root directory as `./`.
7. Click **Deploy**. Open the resulting `https://<project>.vercel.app` URL.
8. Test the app and check `https://<project>.vercel.app/api/health`.

## Run locally (optional)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Then open http://127.0.0.1:8000

## Endpoints

- `/` — RepairWise web interface
- `/api/health` — health check
- `/api/metrics` — held-out test metrics and label distribution
- `/api/assess` — model prediction endpoint (POST JSON)

## Data limitation

The 5,000 dataset rows and target labels are synthetic and rule-generated. The displayed test score measures agreement with these generated labels; it is not verified real-world repair accuracy. This is an educational prototype, not professional repair advice.
