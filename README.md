# MicroFinance — Streamlit Deployment

This repository contains a small loan repayment risk predictor built with Streamlit.

Quick steps to deploy to Streamlit Community Cloud (GitHub-backed):

1. Ensure the repo contains the trained artifacts `loan_model.pkl` and `model_columns.pkl`.
   - Locally run the trainer to produce these files:

```bash
python train_model.py
```

2. Commit the generated `loan_model.pkl` and `model_columns.pkl` to the repository and push to GitHub:

```bash
git add loan_model.pkl model_columns.pkl
git commit -m "Add trained model artifacts"
git push origin main
```

3. On Streamlit Cloud (https://share.streamlit.io):
   - Sign in with GitHub, click "New app" → choose this repo and branch.
   - For the app file, select `app.py` and deploy.

4. If you prefer to run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Notes:

- If you do not want to commit large binaries to GitHub, upload the `loan_model.pkl` and `model_columns.pkl` to a cloud storage (e.g., S3, Google Drive) and modify `app.py` to download them at runtime before loading.
- The minimal dependencies are in `requirements.txt`.

Files of interest:

- [app.py](app.py)
- [train_model.py](train_model.py)
- [.streamlit/config.toml](.streamlit/config.toml)
- [requirements.txt](requirements.txt)
