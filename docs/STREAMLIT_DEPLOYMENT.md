# Streamlit Community Cloud deployment

1. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in
   with the GitHub account that owns `0xSuleman/rala-formula-lab`.
2. Choose **New app**, repository `0xSuleman/rala-formula-lab`, branch `main`,
   and main file path `app.py`.
3. Deploy. Community Cloud reads the root `requirements.txt` and
   `.streamlit/config.toml`; no dataset download is needed for the default
   Sample mode.

The [official deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
and [file-organization guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization)
cover the same flow. Visitors can turn off Sample mode when they want to run a
local or hosted experiment; training is intentionally opt-in and may exceed
Community Cloud resource limits.
