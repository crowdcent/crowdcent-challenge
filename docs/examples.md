# Interactive Examples

## Fetching Challenges and Inference Data

This example demonstrates how to use the `ChallengeClient` to fetch available challenges and get details about the current inference data period. You can optionally provide your API key below, or leave it blank if you have the `CROWDCENT_API_KEY` environment variable or a `.env` file configured.

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


@app.cell
def _(client):
    client.list_training_datasets()
    return

```

///
