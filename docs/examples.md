# End-to-end examples

## Downloading data, training a model, and uploading predictions

This example demonstrates how to use the `ChallengeClient` to download data, train a model, and upload predictions.

/// marimo-embed
    height: 750px
    mode: edit
    app_width: full

```python
@app.cell
def _():
    import marimo as mo
    import crowdcent_challenge as cc
    
    api_key = mo.ui.text(
        value="WTwooX7w.KyfcmBuUyUQUEb7l9CdhHajAXFM11Dck",
        label="API Key (Optional)",
        kind="password",
    )

    api_key
    return api_key, mo

@app.cell
def _(api_key, cc):
    client = cc.ChallengeClient(
        challenge_slug="main-challenge", 
        api_key=api_key.value
    )
    return (client,)

```

///

## Submission automation

### Scheduling a Kaggle notebook
If you're just starting out, we recommend using Kaggle Notebooks to schedule your submissions.

1. **Settings (⚙) → Schedule a notebook run → On**  
2. Choose **Frequency** (daily / weekly / monthly), **Start date**, **Runs ≤ 10** → **Save**  
3. A clock icon appears; each run writes a new **Version** with full logs & outputs  
4. Limits: **CPU-only • ≤ 9 h per run • 1 private / 5 public schedules active**  
5. Pause or delete the job anytime from the same Settings card  

<sub>Need GPUs? Trigger notebook commits with the Kaggle API from cron/GitHub Actions.</sub>

### Scheduling a Google Colab (Vertex AI) notebook

1. **Create a Google Cloud account** if you don't have one already
2. **Go to [Google Colab Notebooks in Vertex AI](https://console.cloud.google.com/vertex-ai/colab/notebooks)**
3. **Set up a schedule:**
   - Open your notebook in Colab
   - Click **Runtime → Manage sessions**
   - Select **Recurring** and configure your schedule
   - Set frequency (daily/weekly/monthly) and duration
   - Click **Save**
4. **Authentication options:**
   - Use service account keys stored securely
   - Set up environment variables in the Vertex AI console
   - Use Google Cloud's Secret Manager for API keys

<sub>Note: Scheduled Colab notebooks run on Google Cloud and may incur charges based on your usage.</sub>

