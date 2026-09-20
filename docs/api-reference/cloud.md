# CrowdCent Cloud API

This page provides auto-generated documentation for the CrowdCent Cloud methods in `ChallengeClient`. These methods manage projects, versions, runs, schedules, and billing in [CrowdCent Cloud](../crowdcent-cloud.md), matching the corresponding [MCP tools](../ai-agents-mcp.md#crowdcent-cloud). The underlying REST endpoints are listed under the `cloud` tag in the [OpenAPI documentation](https://crowdcent.com/api/swagger-ui/#/cloud).

`schedule_cloud_project` schedules saved code when `run_id` is omitted, or reuses
a successful run's exact settings when it is supplied. Both use the existing
project schedule endpoint. Creating a schedule does not start a run.

::: crowdcent_challenge.client.cloud.CloudAPI
    options:
      show_root_heading: true
      show_source: true
      members_order: source
      heading_level: 2
