# Contributing
Contributions are welcome! The `crowdcent-challenge` API client and documentation are open-source projects and contributions can be as simple as a fact check or as complex as a new feature.

Here's a breakdown of the standard GitHub workflow:

1. Fork the [repository](https://github.com/crowdcent/crowdcent-challenge)
2. Clone your fork:
   ```bash
   git clone https://github.com/<your-username>/crowdcent-challenge.git
   cd crowdcent-challenge
   uv pip install -e .[dev]
   ```
3. Branch off:
   ```bash
   git checkout -b my-feature
   ```
4. Make changes & test
5. Commit with clear messages:
   ```bash
   git commit -m "feat: Add new feature"
   ```
6. Push & open a PR from your fork
    ```bash
    git push origin my-feature
    ```

Include tests where appropriate. Keep PRs focused - one feature/fix per PR.