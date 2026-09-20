# Automate prediction submissions

Use **Cloud Runs** to generate predictions and submit them on a schedule. Your
project keeps the code, model files, run reports, and schedule together.
CrowdCent Cloud is in public preview for Challenger+ members (100+ CC Points).

## Set up a schedule on the website

1. Open **Tools → Cloud** and fork the **Hyperliquid Ranking** Cookbook recipe,
   or open your own project.
2. Enable **Challenge access** so the code can download data and submit
   predictions with a temporary Challenge key.
3. Start a **Cloud Run**. Check its report and the Challenge submissions page
   to confirm that it produced an accepted submission.
4. Schedule that successful run **on inference release**, or choose a daily,
   weekly, or monthly time and timezone.
5. Check the schedule and credit balance. Cloud Runs consume credits, and a
   scheduled run needs enough available credit to start.

The schedule pins the code, hardware, and parameters from the run you tested.
Saving an edit does not change an armed schedule. Test the new code with another
Cloud Run, then update the schedule to use it.

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
optimization. Each file has its own tested code and schedule.

Each Cloud Run reads the output snapshot current when the run is created.
An after-success trigger starts a new run against the current snapshot; it does
not lock the folder to the triggering run's output if something newer has been
published. See [safe updates and concurrency](../crowdcent-cloud.md#safe-updates-and-concurrency)
for versioning, model retention, and concurrent editing.
