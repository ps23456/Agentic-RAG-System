# RAG Slack bridge

Slack does not call your Python RAG server directly. This small **Node + Bolt (Socket Mode)** process holds the three Slack secrets, listens for mentions, slash commands, and shared files, and calls your existing **HTTP RAG API** (`RAG_BASE_URL` with `X-API-Key`).

## Why `/raglist` said the app did not respond

That means **no Bolt process was connected** to Slack. Putting tokens in the main repo `.env` only configures the Python app if something reads them there; the Slack integration **must** run this bridge (or equivalent) with those variables.

## Setup

1. **Run the RAG backend** so `RAG_BASE_URL` is reachable (e.g. `http://127.0.0.1:8000`).
2. In this directory:

   ```bash
   cp .env.example .env
   ```

   Fill in:

   - `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN`, `SLACK_SIGNING_SECRET`
   - `RAG_BASE_URL`, `RAG_API_KEY`, `RAG_CUSTOMER_ID` (API key needs scopes such as `docs:read`, `docs:write`, `chat:run`, `index:run` as used by upload, list, chat stream, and job polling)

3. Install and start:

   ```bash
   npm install
   npm run dev
   ```

   Leave this process running. In Slack, use `/raglist` or `@YourBot …`.

4. **Slack app configuration** (high level):

   - **Socket Mode** enabled; App-Level Token with `connections:write` → `SLACK_APP_TOKEN`
   - **Bot Token Scopes** (examples): `app_mentions:read`, `chat:write`, `commands`, `files:read`
   - **Event Subscriptions** (Bolt): `app_mention`, `file_shared`
   - **Slash command**: create `/raglist` pointing at your app (Bolt handles it over the socket)
   - Install the app to the workspace; **invite the bot** to channels where you want mentions

5. Optional **`SLACK_TEAM_ID`**: set to your workspace team id so only that workspace is accepted (useful if you reuse code across workspaces).

## Swagger vs this bot

`/docs` may highlight **`POST /api/query`** under a “rag” tag. This bridge does **not** use that route. It calls:

- `GET /api/documents` — `/raglist`
- `POST /api/chat/stream` — `@mentions`
- `POST /api/upload` and `GET /api/index/jobs/{id}` — shared files

If those paths are missing from OpenAPI on the server, the deployed build may differ from this repo; the bot still needs them (or the code must be pointed at whatever replaces them).

## Troubleshooting

- **`invalid_auth` after `unset`:** If `rag-bot/.env` still has **placeholder** lines from `.env.example` (`xoxb-your-bot-token`), they used to overwrite a good repo-root `.env`. Load order is fixed (repo root wins). Best practice: **delete Slack lines from `rag-bot/.env`** if you keep secrets only in the root `.env`, or replace placeholders with real tokens in `rag-bot/.env`.

## Security

Do not commit `rag-bot/.env`. If a signing secret or token was pasted in chat, **rotate** it in the Slack app settings.
