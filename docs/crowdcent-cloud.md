# CrowdCent Cloud

CrowdCent Cloud is a hosted development and scheduled execution workspace for the CrowdCent Challenge. You can edit notebooks directly in your browser or on cloud instances, save versioned snapshots, run batch jobs in sandboxed compute environments, and schedule recurring submissions without maintaining your own cron infrastructure.

CrowdCent Cloud is in pilot for members with a submission on the board: if you have submitted to a challenge, you can enable Cloud on an API key and bring the code you already run.

## Overview

CrowdCent Cloud supports the standard competition workflow: develop a notebook, test execution on dedicated compute, and schedule automated runs on a daily cadence or whenever a challenge releases new inference data.

Key features include:

- **Self-contained notebooks.** Projects use marimo notebooks by default, with native support for Jupyter notebooks and Python scripts. Dependencies are declared directly inside the file using standard PEP 723 script metadata.
- **Interactive sessions and runs.** Develop interactively using free in-browser WebAssembly kernels or on-demand CPython cloud instances. Batch executions run separately in isolated sandboxes.
- **Deterministic scheduling.** Schedules link directly to a specific successful batch run rather than whatever code is currently in your editor. Updating code will not alter an active schedule until you test the new version and choose to reschedule.
- **Sandboxed isolation.** Runs execute with default-deny network rules. Temporary, non-trading API credentials allow data downloads and prediction submissions without exposing account secrets.

## Core concepts

| Term | Description | Access |
|---|---|---|
| **Project** | The permanent container for your notebook, saved versions, run history, and schedule. | Web, API, MCP |
| **Version** | An immutable snapshot of your project's files and dependencies. Each save creates a new incremented version. | Web, API, MCP |
| **Interactive session** | An active editing environment with a live kernel in your browser (WASM) or on a cloud instance (CPython). | Web only |
| **Run** | An isolated, unattended execution of a specific saved version on dedicated hardware. | Web, API, MCP |
| **File** | Every top-level `.py` file in a project runs by name (`entrypoint`) and can be chained after another; a folder of scripts is one project with several runnable files. | Web, API, MCP |
| **Schedule** | An automated trigger (daily, weekly, monthly, on inference data release, or after another job) that executes a verified run. | Web, API, MCP |
| **Recipe** | A reviewed starter notebook from the [CrowdCent Cookbook](https://github.com/crowdcent/crowdcent-cookbook). | Web, API, MCP |
| **Fork** | Creates a new project from a Cookbook recipe or GitHub repository. | Web, API, MCP |

## Interactive sessions

Interactive sessions are available exclusively through the web interface at crowdcent.com. They provide a live development environment where you can write and test code interactively in your browser.

When working locally or using an AI assistant via MCP, you can iterate in your local environment, then use the Python client or MCP tools to push code, trigger remote  runs, and configure automated schedules.

When you open a project on the web, you enter the editor. The **Runtime** menu in the workspace lets you choose where the kernel executes, switch hardware, or end your session.

### Browser runtime

The Browser runtime executes Python directly in your browser tab using WebAssembly via Pyodide and marimo.

- **Instant and free.** Starts immediately without consuming cloud credits.
- **Dependency installation.** Packages declared in your notebook install directly in the browser environment.
- **Challenge data access.** Enabling the **Challenge access** toggle grants the session a temporary, short-lived token to read challenge datasets without exposing account credentials.

!!! warning "Browser memory and compute limits"
    Because the Browser runtime runs inside your browser tab using WebAssembly, it is constrained by browser memory limits (typically 2 to 4 GB) and single-threaded execution. Downloading full datasets, heavy data transformations, and model training will quickly exhaust tab memory or crash the kernel. Use a **Cloud session** or trigger a **Batch run** for workloads that need dedicated memory and CPython performance.

### Cloud session runtime

The Cloud session runtime runs full CPython on hosted CrowdCent hardware with native support for both marimo and JupyterLab.

- **Dedicated compute.** Suitable for heavier data transformations and local model training.
- **Hourly billing.** Billed per started hour based on instance size. The rate is displayed before you start the session.
- **Seamless switching.** Switching between Browser and Cloud runtimes transfers your code and restarts the kernel on the target instance.

| Size | vCPU | Memory | Accelerator |
|---|---|---|---|
| S | 2 | 5 GiB | |
| M | 4 | 16 GiB | |
| L | 8 | 48 GiB | |
| GPU | 3 | 11 GiB | one NVIDIA L4 24 GB |

### Session lifecycle and recovery

Sessions automatically save your work when you click **End**, or when an idle timeout triggers (30 minutes idle or 2 hours total on the Starter tier; 60 minutes idle or 12 hours total on Challenger tier and above). If a session closes due to inactivity, a recovery snapshot is preserved so you can restore your progress when you reopen the project.

Interactive sessions do not have access to live trading credentials and cannot submit predictions directly. Submissions and scheduled automations run through runs.

## Runs

Runs execute an immutable snapshot of your project in an isolated container. You can trigger a run manually from the workspace or programmatically via the API and MCP tools. Once queued, you can monitor execution status until completion.

| Size | vCPU | Memory | Accelerator |
|---|---|---|---|
| S | 2 | 8 GiB | |
| M | 4 | 16 GiB | |
| L | 8 | 32 GiB | |
| GPU | 4 | 16 GiB | one NVIDIA L4 24 GB |

Every size may run for up to a day; a time limit is a deadline you choose under that (see Hardware and time limits). A GPU run reaches its first line of code about three minutes after you press Run: the machine boots and installs its driver, then your declared packages install.

### Notebook buttons and forms

Every manual or scheduled Cloud *run* receives `CROWDCENT_RUN_ID` in its environment, but interactive notebook sessions do not. So, a recipe intended to submit unattended during a run can use this to pass its button gate during a run:

```python
import os

# In the submission cell; submit is a mo.ui.run_button from an earlier cell.
mo.stop(not os.environ.get("CROWDCENT_RUN_ID") and not submit.value)
```

The Cookbook's `hyperliquid_ranking` recipe uses this pattern to submit automatically to slot 1 during Cloud runs. Cloud executes marimo notebooks with `marimo export html`; in the current runtime, `mo.running_in_notebook()` returns `True` and `mo.app_meta().mode` is `"edit"` even during a run. Use the run ID to distinguish unattended Cloud execution.

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

Both interactive sessions and batch runners read this block to install pinned wheels in the environment. The machine already carries the common stack — numpy, pandas, polars, pyarrow, scipy, scikit-learn, lightgbm, xgboost, optuna, joblib, altair, python-dotenv, and `crowdcent-challenge` — so a script that needs only those declares nothing and installs nothing.

### A folder of scripts

A project is a tree of files, not one notebook. Pass the other files beside the entrypoint (`files={"download_data.py": ..., "submit.py": ...}` in the client, or drop them into the workspace); every top-level `.py` becomes a job of its own that you run by name (`entrypoint="submit.py"`) and chain on the pipeline. A repository that says "run these three scripts in order" is one project, three jobs, and a daily chain.

### Hardware and time limits

Every run names a hardware size: `s` (2 vCPU, 8 GB), `m` (4 vCPU, 16 GB), `l` (8 vCPU, 32 GB), or `gpu_s` (4 vCPU, 16 GB, one NVIDIA L4 24 GB). A Cloud session offers the same four sizes at the same rates. Every run may live a day; a time limit is a deadline you choose under that, never a price paid up front, and without one the day applies. Starting a run needs an hour at the size's rate available (or the whole time limit, when shorter); the run then covers itself an hour at a time and pays only for the minutes it used. A run stopped at its limit, stopped by you, or stopped because your credits ran out keeps what it had written to the project folder; a run that crashes keeps nothing.

A GPU notebook declares its framework like any other dependency, `jax[cuda12]` for Keras on JAX (the `centimators` sequence models) or `torch`; the `gpu_s` machine carries the NVIDIA driver, and its own `xgboost` trains on the GPU with `device="cuda"` with nothing declared.

### Run reports

When a batch run finishes, it produces an execution report containing:

- **State.** Current execution phase (`queued`, `running`, `done`, or `failed`).
- **Facts.** Recorded metadata including project version, hardware environment, and API permissions used.
- **Logs.** A bounded tail of stdout and stderr output.
- **Diagnostics.** Error details and stack traces if execution failed.
- **Artifacts.** Metadata for files generated during the run.

A status of `done` confirms that the script finished successfully within its time budget. To verify that predictions were accepted, check the challenge submissions dashboard or call `list_submissions` in the Python client.

### Parameters

Most runs need none. Parameters are for a notebook you launch several times with different settings, such as a tuning study.

Parameters are named values a run is told when it starts: `trials=100 lr_max=0.2`. Set them under **Advanced** on the Run form, or pass `parameters` to `run_cloud_project`. The run's code receives them as command-line arguments, `--trials=100 --lr_max=0.2`, which a marimo notebook reads with `mo.cli_args()` and a plain script with `sys.argv`. Numbers, `true`/`false`, and text are typed the way marimo types them. A run takes up to 32 parameters, named with lowercase letters, digits, and underscores.

For a form with complete defaults, a Cloud run can use those defaults without parameters. Define `DEFAULTS` in the notebook and use it to initialize the form, then read the selected settings in a later cell:

```python
import os

answered = dict(mo.cli_args()) or search_form.value
mo.stop(
    not answered and not os.environ.get("CROWDCENT_RUN_ID"),
    mo.md("Choose a search and press **Tune**."),
)
search = {**DEFAULTS, **(answered or {})}
```

A Cloud run with no parameters uses the defaults; a run told `trials=100` overrides that setting. An interactive notebook without arguments waits for the form. Runs use defaults saved in the code, not widget values changed in an interactive session. A run's parameters show on its report and beside it under History, scheduling the run keeps them, and `get_cloud_run` returns them as `parameters`.

```python
for trials, lr_max in [(60, 0.1), (60, 0.3), (120, 0.1)]:
    client.run_cloud_project(
        project["id"],
        envelope="l",
        time_limit_minutes=90,
        parameters={"trials": trials, "lr_max": lr_max},
    )
```

What a run finds is ordinary output: the executed notebook on its report shows it, and files it writes to the project folder appear under History for the next run to read. The `optuna_tuning` recipe in the [CrowdCent Cookbook](https://github.com/crowdcent/crowdcent-cookbook) is a complete example. One Optuna study is one run: its trials run in parallel across the machine's cores, so a bigger machine runs more of them at once.

## Automated schedules

You can automate recurring executions for any batch run that completed successfully (`state: done`).

When you configure a schedule, it pins the exact version, environment settings, hardware size, and parameters used by that successful run. Editing or saving new code in your project does not change what the active schedule runs. To deploy new code to an existing schedule, test it with a fresh batch run and update the schedule to that new run.

### Supported triggers

- **Daily.** Runs every day at a specified `HH:MM` time in your chosen IANA timezone.
- **Weekly.** Runs once a week on a weekday (`0`=Mon … `6`=Sun) at `HH:MM`.
- **Monthly.** Runs once a month on day `1`–`28` at `HH:MM`.
- **On inference release.** Runs automatically whenever the target challenge publishes a new inference dataset.
- **After.** Runs once another job of the same project has succeeded — how a folder of scripts becomes a chain.

Schedules can be paused and resumed at any time without needing to recreate the configuration.

## Credits and billing

Interactive Cloud sessions and runs consume Cloud credits.

- **Unified credit balance.** Accounts maintain a balance measured in integer cents (`available_cents`).
- **Included tier allowance.** Accounts receive a monthly credit allowance based on their tier. Included credits are consumed before any purchased prepaid balance.
- **One rate per machine.** Each shape has an hourly rate, the same whether it runs a notebook for you or hosts a live session, billed by the started minute. Runs and sessions both bill as they go, an hour at a time, and stop at the last funded minute if the balance runs out.
- **Low balance handling.** If an account lacks sufficient credits to start a run or session, the request returns HTTP `402 CREDITS_REQUIRED` with a direct URL to add credits.

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
- **Archiving projects.** Archiving a project hides it from your active list and automatically pauses any associated schedules and active sessions. Archived projects can be restored at any time.
- **Deleting projects.** Projects without billed compute history can be permanently deleted. Projects that have executed paid runs or sessions can be archived to preserve historical accounting records.

## Python client and MCP tools

The Python client and MCP tools are designed for **remote batch execution and automated scheduling**. They allow you to programmatically create projects, push code versions, trigger remote runs on dedicated hardware, monitor execution, and configure recurring schedules.

The API and MCP tools do not open or manage interactive browser/cloud sessions. Interactive sessions are meant for manual experimentation on the web, whereas AI agents can already handle interactive development locally.

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
)

# Start a batch run and wait for completion
run = client.run_cloud_project(project["id"])
while True:
    run = client.get_cloud_run(run["id"])
    if not run["live"]:
        break
    time.sleep(30)

print(f"Run completed with status: {run['state']} - {run['detail']}")

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

Saving updates via `update_cloud_project` requires specifying `base_version`, the version number your edit was based on. If another user, session, or automated process has saved a newer version, the request returns HTTP `409 VERSION_CONFLICT`. This prevents unintended overwrites.

```python
project = client.get_cloud_project(project["id"])

client.update_cloud_project(
    project["id"],
    source=new_source,
    base_version=project["source"]["version"],
)
```

### Idempotent requests

Calls to `create_cloud_project` and `run_cloud_project` include an auto-generated idempotency key so network retries cannot accidentally create duplicate projects or launch unintended runs. You can also supply a custom `idempotency_key` argument when implementing custom retry loops.

### Using AI assistants

AI tools connected via the CrowdCent MCP server can manage the Cloud workflow through matching tool names (`create_cloud_project`, `run_cloud_project`, `get_cloud_run`, etc.).

```
"Create a Cloud project from the hyperliquid-ranking recipe, run it, and verify the output. If the run succeeds, schedule it on each inference release."
```

### Centaur in Cloud

Centaur is the assistant inside Cloud. It works through the same API your key does, as you, and it can do nothing you could not do yourself. The panel's empty state says what it can do where you are:

| | |
|---|---|
| **On its own** | reads your projects, runs, data and trading; edits notebooks and files; creates and archives projects |
| **Asks first** | paid runs (runs, schedules) · submissions (predictions, your Challenge key) · trading (real money, always asks) |
| **Never** | buy credits, change your login or keys, reach the internet |

Each kind under *Asks first* is a switch in the panel. Lit, it runs without asking; unlit, it shows a card in the conversation and Centaur waits. Unanswered for 90 seconds, a card is skipped and nothing runs. A card can also be told "don't ask again for this kind". Submissions start lit, since the Challenge key a project gets is short-lived and only downloads data and submits; trading always asks. The panel also says what is off in the current place, such as a project whose Challenge access is disabled or an account with no live trading account.

## Security and permissions

Access to Cloud features requires an API key with the **Allow Cloud** permission enabled in [profile settings](https://crowdcent.com/profile/settings/).

- **Privilege separation.** Cloud API credentials and execution tokens cannot place trades or access wallet funds. Keep live trading keys separate from development and automation keys.
- **Recursive prevention.** Scoped tokens provisioned for runs and interactive sessions cannot spawn secondary Cloud resources or modify billing settings.
- **Authentication check.** Call `client.check_auth()` to inspect the capabilities enabled on your current API key.
