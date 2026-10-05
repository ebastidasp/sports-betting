# Worlds/MSI data-source choice

## Decision

Use **PandaScore as the one schedule and match-score provider**. Keep the existing Oracle's Elixir cache as the historical source for model training and team features. Do not combine three live feeds in the first version.

The app uses PandaScore's REST endpoints for upcoming and running matches. Its `results` field supplies the current series score. The app deliberately does not consume live in-game frames or calculate in-play win probabilities.

## Candidate comparison

| Candidate | Fit for this app | Access / tradeoff | Decision |
| --- | --- | --- | --- |
| **PandaScore** | Documents League of Legends upcoming and running match endpoints, event/tournament metadata, series results, and a separate live-data product. | API token required. Its published plan reference lists fixture/schedule endpoints for all plans; LoL live frames require a real-time data plan. Confirm Worlds/MSI coverage and current account entitlements after issuing a token. | **Select for V1**: clearest documented REST fit for schedule and match-level score/status. |
| **GRID** | Advertises official publisher/organizer data and League of Legends live data. | Access is by request/contact and data use is licensed within an agreement; this is more involved than a first personal prototype. | Strong future option if official low-latency data or commercial licensing becomes important. |
| **Riot developer APIs** | Official Riot APIs are useful for game/player APIs and developer-run tournaments. | The documented Tournament API is for managing a developer's own tournament; Live Client Data is served from a local game client. The public developer documentation does not describe a general professional Worlds/MSI schedule endpoint. | Not selected as the public schedule source. |

## References

- PandaScore [getting started](https://developers.pandascore.co/docs/getting-started.md): REST schedule/running endpoints, match status lifecycle, and authentication.
- PandaScore [plan reference](https://developers.pandascore.co/docs/plan-reference.md): LoL endpoint availability and live-frame plan gates.
- PandaScore [rate and connection limits](https://developers.pandascore.co/docs/rate-and-connections-limits.md): schedule-plan request limits and WebSocket connections.
- PandaScore [LoL live data sample](https://developers.pandascore.co/docs/data-sample-league-of-legends.md): in-game frame/event fields.
- GRID [official data platform](https://www.grid.gg/): League of Legends coverage, access request, and licensed-data terms.
- Riot [League of Legends developer documentation](https://developer.riotgames.com/docs/lol): documented Tournament and local Live Client APIs.

PandaScore documents match endpoints and schemas, but event coverage is account- and competition-dependent. Once a token is available, verify that upcoming and running Worlds/MSI fixtures appear before relying on it for a live deployment.

## Refresh behavior

- Upcoming schedule: cached for 15 minutes.
- Running series and scores: cached for 30 seconds, then refreshed from the running-match endpoint.
- Requests are server-side; the token is read from `PANDASCORE_API_TOKEN` and is never sent to the browser.
- On a provider error, the dashboard keeps the last successful in-memory snapshot and labels it stale.
