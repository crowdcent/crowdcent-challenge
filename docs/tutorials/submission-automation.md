# Automate prediction submissions

Use **Cloud Runs** to generate predictions and submit them on a schedule. Your
project keeps the code, model files, run reports, and schedule together.
CrowdCent Cloud is open to anyone with a Challenge submission.

## Set up a schedule on the website

1. Open **Cloud** and fork the **Hyperliquid Ranking** Cookbook recipe,
   or open your own project.
2. Enable **Challenge access** so the code can download data and submit
   predictions with a temporary Challenge key.
3. Save your code, open **Run & schedule**, and select the file you want to automate.
4. Choose **On release**, or a daily, weekly, or monthly time and
   timezone. An earlier Cloud Run is optional.
5. Check the rule, hardware, and credit balance, then select **Set schedule**. Cloud Runs consume credits, and a
   scheduled run needs enough available credit to start.

<figure class="doc-screenshot" markdown>
[![Schedule form with the inference-release trigger, challenge selector, hardware, and Set schedule button](../assets/images/screenshots/cloud-schedule.png){ loading=lazy width="1280" height="355" }](https://crowdcent.com/cloud/){ target="_blank" rel="noopener" title="Open this page on CrowdCent" }
<figcaption>In the project’s <strong>Run &amp; schedule</strong> drawer, choose <strong>On release</strong> and the target challenge, then <strong>Set schedule</strong>.</figcaption>
</figure>

The schedule pins saved code and execution settings. Creating it starts no
compute and reserves no credits. Saving an edit does not change an armed
schedule; update the schedule explicitly to use the new version.

You can also start a Cloud Run first, inspect its report and the Challenge
submissions page, then schedule that successful run's exact settings through
Python or MCP.

## Use Python or an AI assistant

The Python client and MCP tools use the same project and scheduling API. Enable
**Allow Cloud** on your API key, then follow the
[Python quickstart](../crowdcent-cloud.md#python-quickstart).

An MCP-connected assistant can handle the workflow:

> Create a Cloud project from the Hyperliquid Ranking recipe, run it, and check
> whether the predictions were accepted. If they were, schedule it on each
> inference release.

Cloud Runs continue independently of your browser or local computer. Browser
and Cloud Sessions are for interactive development; opening a session is not
required to run a schedule.

## Separate training from prediction

Keep `optimize.py` and `predict.py` in the same project. The optimizer can save
`models/best.joblib`; the prediction script reads it from that path. Schedule
optimization weekly and prediction daily, or start prediction after a successful
optimization. Each file has its own pinned code and schedule. Neither needs to
have run before you set this up:

```python
client.schedule_cloud_project(
    project_id,
    entrypoint="optimize.py",
    envelope="m",
    time_limit_minutes=90,
    trigger="weekly",
    weekday=0,
    daily_at="02:00",
)
client.schedule_cloud_project(
    project_id,
    entrypoint="predict.py",
    trigger="after",
    after="optimize.py",
)
```

Each Cloud Run reads the output snapshot current when the run is created.
An after-success trigger starts a new run against the current snapshot; it does
not lock the folder to the triggering run's output if something newer has been
published. See [safe updates and concurrency](../crowdcent-cloud.md#safe-updates-and-concurrency)
for versioning, model retention, and concurrent editing.
