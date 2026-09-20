# CrowdCent Cloud

CrowdCent Cloud brings your code, saved models, and automated workflows into one project. Develop marimo notebooks and Python scripts in the Browser or a **Cloud Session**, then use **Cloud Runs** to execute saved code on demand or on a schedule.

CrowdCent Cloud is in **public preview for Challenger+ members (100+ CC Points)**. Enable **Allow Cloud** on an API key to use the Python client or MCP tools.

## Overview

CrowdCent Cloud supports the standard competition workflow: develop a notebook, test execution on dedicated compute, and schedule automated runs on a daily cadence or whenever a challenge releases new inference data.

Key features include:

- **Python projects.** Use marimo notebooks saved as `.py` files, ordinary Python scripts, and supporting project files. Declare dependencies inside the file with PEP 723 script metadata.
- **Choose how to work.** Edit interactively in the free Browser runtime or on hosted hardware with Cloud Sessions. Cloud Runs execute saved code unattended.
- **Test before scheduling.** A schedule pins the code and settings from a successful Cloud Run. Later code edits take effect only when you test and select a new run.
- **Sandboxed isolation.** Runs execute with default-deny network rules. Temporary, non-trading API credentials allow data downloads and prediction submissions without exposing account secrets.

## Core concepts

The website and API / Python / MCP share the saved-project and Cloud Run workflow. Use the website for live editing:

| Workflow | Website | API / Python / MCP |
|---|:---:|:---:|
| Projects, saved files, and versions | ✓ | ✓ |
| Cloud Runs and schedules | ✓ | ✓ |
| Live editing | ✓ | — |

| Concept | What it means |
|---|---|
| **Project** | Your code, generated files, version history, Cloud Runs, and per-file schedules in one folder. |
| **Version** | An immutable copy of saved project code and dependency declarations. A changed save creates the next version. |
| **Output snapshot** | A recorded set of generated files, such as models and predictions. Output history is independent of code versions and subject to storage retention. |
| **Cloud Session** | A live Python kernel on hosted hardware, opened from the website. The free Browser runtime runs in your tab instead. |
| **Cloud Run** | An unattended execution of a saved version and named file on selected hardware. |
| **Schedule** | Clock, inference-release, or after-success triggers for a file's tested Cloud Run configuration. |
| **Recipe** | A starter notebook from the [CrowdCent Cookbook](https://github.com/crowdcent/crowdcent-cookbook). Forking a recipe creates your own project. |

## Browser and Cloud Sessions

Open a project on crowdcent.com to edit and execute its marimo notebook. Launch support covers marimo `.py` notebooks and Python scripts; Jupyter notebooks and JupyterLab are outside the launch scope.

When working locally or using an AI assistant via MCP, use your local development environment, then save files, start Cloud Runs, and configure schedules through the project API.

Use the menu beside **Edit code** to choose Browser or Cloud Session, switch hardware, or end your session.

### Browser runtime

The Browser runtime executes Python directly in your browser tab using WebAssembly via Pyodide and marimo.

- **Free compute.** Runs on your device without consuming Cloud compute credits.
- **Dependency installation.** Browser-compatible packages declared in your notebook install in the tab. Packages that require native Python may need a Cloud Session or Cloud Run.
- **Challenge data access.** Enabling the **Challenge access** toggle grants the session a temporary, short-lived token to read challenge datasets without exposing account credentials.

!!! warning "Browser memory and compute limits"
    The Browser runtime uses your device's CPU and browser memory. Large datasets and model training can exceed those limits. Use a **Cloud Session** or **Cloud Run** for workloads that need more memory or native Python packages.

### Cloud Sessions

Cloud Sessions run native Python on hosted CrowdCent hardware with the marimo editor.

- **Dedicated compute.** Suitable for heavier data transformations and local model training.
- **Usage billing.** Billed by the started minute at the selected size's hourly rate. The rate is displayed before you start.
- **Runtime switching.** Your project files move with you; the kernel restarts, so rerun cells to recreate in-memory variables. Files beyond the Browser transfer limits stay saved in the project.

| Size | vCPU | Memory | Accelerator |
|---|---|---|---|
| S | 2 | 5 GiB | |
| M | 4 | 16 GiB | |
| L | 8 | 48 GiB | |
| GPU | 3 | 11 GiB | one NVIDIA L4 24 GB |

### Session lifecycle and recovery

Use **Save version** to preserve code and generated files. Cloud Sessions also attempt to save when you click **End** or reach their idle or lifetime limit. For Challenger tier and above, the defaults are 60 minutes idle and 12 hours total; the workspace shows its current limits. Conflicting saves are retained for recovery without replacing newer saved work. Save regularly: recovery depends on the session still being reachable.

With **Challenge access** enabled, Browser and Cloud Sessions can download Challenge data and submit predictions using a temporary key. Those keys cannot trade or start other Cloud resources. Use Cloud Runs for unattended submissions and scheduled workflows.

## Cloud Runs

Runs execute an immutable snapshot of your project in an isolated container. You can trigger a run manually from the workspace or programmatically via the API and MCP tools. Once queued, you can monitor execution status until completion.

| Size | vCPU | Memory | Accelerator |
|---|---|---|---|
| S | 2 | 8 GiB | |
| M | 4 | 16 GiB | |
| L | 8 | 32 GiB | |
| GPU | 4 | 16 GiB | one NVIDIA L4 24 GB |

Every size may run for up to a day; you can choose a shorter deadline. Startup time includes preparing hardware and installing declared dependencies, and varies with the workload.

### Notebook buttons and forms

Every manual or scheduled Cloud Run receives `CROWDCENT_RUN_ID` in its environment, but interactive notebook sessions do not. So, a recipe intended to submit unattended during a run can use this to pass its button gate during a run:

```python
import os

# In the submission cell; submit is a mo.ui.run_button from an earlier cell.
mo.stop(not os.environ.get("CROWDCENT_RUN_ID") and not submit.value)
```

The Cookbook's `hyperliquid_ranking` recipe uses this pattern to submit automatically to slot 1 during Cloud Runs. Cloud executes marimo notebooks with `marimo export html`; in the current runtime, `mo.running_in_notebook()` returns `True` and `mo.app_meta().mode` is `"edit"` even during a run. Use the run ID to distinguish unattended Cloud execution.

### Network and access control

Runs execute with default-deny outbound networking. Projects created from Cookbook recipes include pre-approved public endpoints, such as third-party market data feeds.

When the **Challenge access** option is enabled, the run receives a scoped, short-lived Challenge API key valid only for the duration of the run. This allows the notebook to fetch new inference data and post predictions to the challenge.

### Dependency management

Dependencies are declared at the top of your script or notebook using [PEP 723 inline script metadata](https://packaging.python.org/en/latest/specifications/inline-script-metadata/):

```python
# /// script
# dependencies = [
#     "polars>=1.0.0",
#     "lightgbm>=4.0.0",
#     "crowdcent-challenge",
# ]
# ///
```

Declare the packages your notebook imports so its environment can be recreated. Cloud Runs and Cloud Sessions read this metadata when preparing the notebook environment. Browser execution also requires those packages to be available for its browser-based Python runtime.

### A folder of scripts

A project is a tree of files, not one notebook. Pass the other files beside the entrypoint (`files={"download_data.py": ..., "submit.py": ...}` in the client, or drop them into the workspace); every top-level `.py` becomes a job of its own that you run by name (`entrypoint="submit.py"`) and chain on the pipeline. A repository that says "run these three scripts in order" is one project, three jobs, and a daily chain.

### Hardware and time limits

Choose `s`, `m`, `l`, or `gpu_s`; the tables above show the resources for Cloud Runs and Cloud Sessions separately. Each size has the same hourly rate across both, with usage billed by the started minute.

A Cloud Run can run for up to 24 hours. You can set a shorter deadline. Starting requires enough available credits for an hour at the selected rate, or the chosen deadline if shorter; more credit is reserved as the run continues. Runs stop at their deadline or last funded minute. A stopped run can preserve its latest valid output checkpoint; check its report to see what was retained.

A GPU notebook declares its framework like any other dependency, `jax[cuda12]` for Keras on JAX (the `centimators` sequence models) or `torch`; the `gpu_s` machine carries the NVIDIA driver, and its own `xgboost` trains on the GPU with `device="cuda"` with nothing declared.

### Run reports

When a Cloud Run finishes, it produces an execution report containing:

- **State.** Current execution phase (`queued`, `running`, `done`, or `failed`).
- **Facts.** Recorded metadata including project version, hardware environment, and API permissions used.
- **Logs.** A bounded tail of stdout and stderr output.
- **Diagnostics.** Error details and stack traces if execution failed.
- **Artifacts.** Metadata for files generated during the run.

A status of `done` confirms that the script finished successfully within its time budget. To verify that predictions were accepted, check the challenge submissions dashboard or call `list_submissions` in the Python client.

### Parameters

Most runs need none. Parameters are for a notebook you launch several times with different settings, such as a tuning study.

Parameters are named values a run is told when it starts: `trials=100 lr_max=0.2`. Set them under **Advanced** on the Run form, or pass `parameters` to `run_cloud_project`. The run's code receives them as command-line arguments, `--trials=100 --lr_max=0.2`, which a marimo notebook reads with `mo.cli_args()` and a plain script with `sys.argv`. Numbers, `true`/`false`, and text are typed the way marimo types them. A run takes up to 32 parameters, named with lowercase letters, digits, and underscores.

For a form with complete defaults, a Cloud Run can use those defaults without parameters. Define `DEFAULTS` in the notebook and use it to initialize the form, then read the selected settings in a later cell:

```python
import os

answered = dict(mo.cli_args()) or search_form.value
mo.stop(
    not answered and not os.environ.get("CROWDCENT_RUN_ID"),
    mo.md("Choose a search and press **Tune**."),
)
search = {**DEFAULTS, **(answered or {})}
```

A Cloud Run with no parameters uses the defaults; a run told `trials=100` overrides that setting. An interactive notebook without arguments waits for the form. Runs use defaults saved in the code, not widget values changed in an interactive session. A run's parameters show on its report and beside it under History, scheduling the run keeps them, and `get_cloud_run` returns them as `parameters`.

```python
for trials, lr_max in [(60, 0.1), (60, 0.3), (120, 0.1)]:
    client.run_cloud_project(
        project["id"],
        envelope="l",
        time_limit_minutes=90,
        parameters={"trials": trials, "lr_max": lr_max},
    )
```

The executed notebook appears on the run report, and generated project files appear under History for later runs to read. The `optuna_tuning` recipe in the [CrowdCent Cookbook](https://github.com/crowdcent/crowdcent-cookbook) is a complete example. It runs up to two Optuna trials concurrently. Choose parallelism in your code to suit the selected hardware.

## Automated schedules

You can automate recurring executions for any Cloud Run that completed successfully (`state: done`).

When you configure a schedule, it pins the exact version, environment settings, hardware size, and parameters used by that successful run. Editing or saving new code in your project does not change what the active schedule runs. To deploy new code to an existing schedule, test it with a fresh Cloud Run and update the schedule to that new run.

### Supported triggers

- **Daily.** Runs every day at a specified `HH:MM` time in your chosen IANA timezone.
- **Weekly.** Runs once a week on a weekday (`0`=Mon … `6`=Sun) at `HH:MM`.
- **Monthly.** Runs once a month on day `1`–`28` at `HH:MM`.
- **On inference release.** Runs automatically whenever the target challenge publishes a new inference dataset.
- **After.** Runs once another job of the same project has succeeded — how a folder of scripts becomes a chain.

Schedules can be paused and resumed at any time without needing to recreate the configuration.

## Credits and billing

Cloud Sessions and Cloud Runs consume Cloud credits. Centaur also uses this
balance if you enable paid asks after its included allowance.

- **Unified credit balance.** Accounts maintain a balance measured in integer cents (`available_cents`).
- **Included tier allowance.** Accounts receive a monthly credit allowance based on their tier. A reservation uses included credits when they cover it; otherwise it uses purchased credits.
- **One rate per machine.** Each size has the same hourly rate for Cloud Runs and Cloud Sessions, billed by the started minute. Credits are reserved up to an hour ahead of use; unused reservations are released when compute stops.
- **Low balance handling.** Insufficient credits prevent new compute. The API returns HTTP `402 CREDITS_REQUIRED` with a direct URL to add credits.

| Tier | CC Points | Included credits per month |
|---|---:|---:|
| Challenger | 100+ | $10 |
| Contender | 500+ | $25 |
| Centurion and Sovereign | 1,500+ | $50 |

Included credits reset on the first of each month at 00:00 UTC and do not roll
over. The month's allowance is set when you first reserve included credits that
month; later tier changes apply to the next month's allowance. Your current
allowance and remaining balance are shown under **Billing**. Purchased credits
are tracked separately from the monthly allowance.

```python
billing = client.get_cloud_billing()
billing["available_cents"]              # Total available balance (included + purchased)
billing["included"]["remaining_cents"]  # Monthly allowance remaining
billing["prices"]                       # Per size: hourly_cents, run_max_seconds
```

## Creating and managing projects

In the web interface, open **Tools → Cloud** to manage your projects.

- **New project.** Create a blank project in the Browser or Cloud runtime, import a repository from GitHub, fork a recipe from the Cookbook, or generate a starting point using Centaur.
- **Cookbook recipes.** Browse reviewed templates (such as `hyperliquid-ranking` or `numerai-dashboard`) to preview code or fork into your account.
- **Archiving projects.** Archiving hides a project from your active list, pauses its schedules, and ends its Cloud Sessions. Restoring it leaves schedules paused until you resume them.
- **Deleting projects.** Projects with recorded run history or billed session time must be archived to preserve their records. A project with no retained execution evidence can be deleted if no other project uses its output store.

## Python client and MCP tools

The Python client and MCP tools manage **project files, Cloud Runs, and schedules**. Create projects, read and edit saved files, download models, execute saved code, inspect results, and automate the next run.

Open live Browser and Cloud Sessions on the website. Centaur can work in that live workspace; an external API or MCP agent works with saved project files and Cloud Runs.

Complete method documentation is available in the [Cloud API reference](api-reference/cloud.md) and the [AI Agents (MCP) guide](ai-agents-mcp.md). REST endpoints are documented under the `cloud` tag in the [OpenAPI specification](https://crowdcent.com/api/swagger-ui/#/cloud).

### Python quickstart

```python
import time
from crowdcent_challenge import ChallengeClient

client = ChallengeClient("hyperliquid-ranking")

# Create a project from a Cookbook recipe
project = client.create_cloud_project(
    "Daily submitter",
    recipe="hyperliquid-ranking",
    challenge_access=True,
)

# Start a Cloud Run and wait for completion
run = client.run_cloud_project(project["id"])
while True:
    run = client.get_cloud_run(run["id"])
    if not run["live"]:
        break
    time.sleep(30)

print(f"Run completed with status: {run['state']} - {run['detail']}")
if run["state"] != "done":
    raise RuntimeError("Inspect the failed run before scheduling it.")

# Schedule the verified run on every new inference release
client.schedule_cloud_project(
    project["id"],
    run["id"],
    trigger="on_inference_release",
    challenge="hyperliquid-ranking",
)

# Or on a clock: daily, weekly (weekday 0=Mon..6=Sun), or monthly (day 1..28)
# client.schedule_cloud_project(
#     project["id"], run["id"], trigger="daily", daily_at="13:30", timezone="UTC",
# )
# client.schedule_cloud_project(
#     project["id"], run["id"],
#     trigger="weekly", daily_at="02:00", weekday=0, timezone="UTC",
# )
# client.schedule_cloud_project(
#     project["id"], run["id"],
#     trigger="monthly", daily_at="07:30", day=15, timezone="UTC",
# )
```

### Safe updates and concurrency

Saving file edits via `update_cloud_project` requires specifying `base_version`, the version number your edit was based on. If another session or agent has saved a newer version, the request returns HTTP `409 VERSION_CONFLICT`. This prevents unintended overwrites.

```python
project = client.get_cloud_project(project["id"])

client.update_cloud_project(
    project["id"],
    files={project["filename"]: new_source, "helpers/features.py": helper_source},
    base_version=project["latest_version"],
)
```

Read files with `get_cloud_project_files(project_id, path="helpers/features.py")`.
`files` updates only the named paths; use `None` to delete. Other files, including
binary inputs, remain unchanged. A rename is a deletion and addition in one edit.
Use `filename` when changing which file is the primary notebook.

Code versions and generated-output snapshots are independent. A saved schedule
pins its tested code; each occurrence reads the output snapshot current when the
run is created. An optimizer can write `models/best.joblib`, and a later prediction
script can load it from the same path. Clock and `after` triggers are alternatives:
either can start the job. They do not mean “wait for both.” An `after` run reads the
current output snapshot, which may include a newer publication than its triggering run.

File listings return immutable `version` or `snapshot` selectors. Supply one when
reading or downloading a model to reproduce the listed bytes:

```python
files = client.get_cloud_project_files(project["id"])
model = next(item for item in files["files"] if item["path"] == "models/best.joblib")
client.download_cloud_project_file(
    project["id"], model["path"], "best.joblib", snapshot=model["snapshot"],
)
```

Cloud Sessions and Cloud Runs support output files up to 4 GiB and output snapshots up
to 8 GiB, subject to the account's retained-storage allowance. The Browser runtime
has a smaller transfer budget and identifies files it cannot carry. Current output
paths remain available; superseded snapshots follow retention limits, so history
is not permanent model retention. Keep separately named model files when both
models must remain in the current folder.

An open Browser or Cloud workspace retains its own files when the API saves a
new version. Its saved status offers **Save my workspace**, **Load saved version**,
and **Review differences**. Loading preserves the local code in history first;
overwriting still checks that the reviewed remote version has not changed again.
Unchanged local model files never republish over newer remote models. Conflicting
output edits are retained in a snapshot without moving the current output folder.

### Idempotent requests

Calls to `create_cloud_project` and `run_cloud_project` include an auto-generated idempotency key so network retries cannot accidentally create duplicate projects or launch unintended runs. You can also supply a custom `idempotency_key` argument when implementing custom retry loops.

### Using AI assistants

AI tools connected via the CrowdCent MCP server can manage the Cloud workflow through matching tool names (`create_cloud_project`, `run_cloud_project`, `get_cloud_run`, etc.).

The hosted MCP server reads project text and file metadata. To download a binary
model to your computer, use Python, REST, or the local MCP server's
`download_cloud_project_file` tool.

```
"Create a Cloud project from the hyperliquid-ranking recipe, run it, and verify the output. If the run succeeds, schedule it on each inference release."
```

### Centaur in Cloud

Centaur helps you edit notebooks and project files, run code, inspect results,
and manage Cloud Runs and schedules. In an open notebook, it works in your live
workspace; **Save version** preserves those edits.

| | |
|---|---|
| **On its own** | reads your projects, runs, data and trading; edits notebooks and files; creates and archives projects; configures schedules when asked |
| **Asks first** | launches paid Cloud Runs · submissions (predictions, your Challenge key) · trading (real money, always asks) |
| **Never** | buy credits or change your login or API keys |

Cloud Run launches and submissions have approval switches in the panel. When enabled, that action can proceed without another card; otherwise Centaur waits for your response. Unanswered cards expire without executing the action. Submissions are enabled by default; trading always requires confirmation. The panel also shows capabilities that are unavailable, such as Challenge access disabled for the project.

Scheduled Cloud Runs consume credits when they execute; configuring a schedule does not show a separate approval card. Code Centaur executes in an attached workspace has that workspace's filesystem and permitted network access.

## Security and permissions

Python/REST API and MCP access requires an API key with **Allow Cloud** enabled in [profile settings](https://crowdcent.com/profile/settings/). On the website, sign in with an eligible account.

- **Privilege separation.** Temporary keys issued to Cloud Runs and sessions cannot place trades. Keep your own development and automation API keys separate from trading-enabled keys.
- **Recursive prevention.** Scoped tokens provisioned for runs and interactive sessions cannot spawn secondary Cloud resources or modify billing settings.
- **Authentication check.** Call `client.check_auth()` to inspect the capabilities enabled on your current API key.
