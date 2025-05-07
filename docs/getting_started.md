| Level | Description | Key Components |
|-------|-------------|----------------|
| **Challenge** | Top-level competition | Unique slug, name, description, dates |
| **Training Dataset** | Versioned data for building models | Features + ground truth labels |
| **Inference Data** | Periodic releases for predictions | Features only (no labels) or just a universe |
| **Submissions** | Your predictions | ID + n time-horizon predictions |


First, find a challenge you're interested in.

Then, install the `crowdcent-challenge` package if you want to interact programmatically.
If you're just looking to submit predictions, you can use the web interface. Like this: