# Loupe dashboard

Streamlit dashboard for the Bina.az bargain finder. It reads `predictions.json`
(the output of `model/train.py`) and lets a visitor:

- check any listing by link or number, with listed price vs model price on one strip;
- browse the best deals, filtered by area, rooms and price;
- see how listings are spread around the model's price and how the model behaves;
- switch the interface between Azerbaijani and English.

No database, token or secret is needed, only `predictions.json`.

## Run locally

    python -m venv venv
    venv\Scripts\activate.bat        (Mac/Linux: source venv/bin/activate)
    pip install -r requirements.txt
    streamlit run app.py

## Deploy on Streamlit Community Cloud

1. Put these files at the root of a GitHub repository you own:
   `app.py`, `requirements.txt`, `predictions.json`, `.streamlit/config.toml`.
2. Go to https://share.streamlit.io, sign in with GitHub, choose **Create app**.
3. Pick the repository, branch `main`, main file `app.py`, then **Deploy**.

Community Cloud runs `streamlit run` from the repository root and only reads
`.streamlit/config.toml` from the root. If the dashboard lives in a subfolder
of a larger repo, use `dashboard/app.py` as the main file. The app still works,
but the theme file is ignored unless it also sits at the repo root.

## About the numbers

- Discount = (model price - listed price) / model price x 100. Alert levels:
  15% or more is very cheap, 10-15% is cheap.
- `model/train.py` trains on 80% of the listings and then predicts all of them.
  The dashboard marks the 304 listings held out by its split
  (`train_test_split`, `test_size=0.2`, `random_state=42`) as "not seen by the
  model". Re-running `train.py` on `cleaned_data/sale.csv` reproduces
  `predictions.json` exactly, so this marking matches the trained model.
  The marking is only applied while the file has 1,517 rows; if the dataset is
  regenerated with a different size, update `SPLIT_N` in `app.py`.
- On the unseen listings the mean relative error is 13.0% (median 10.0%),
  about the same size as the alert thresholds. Read a result as "worth a closer look".
