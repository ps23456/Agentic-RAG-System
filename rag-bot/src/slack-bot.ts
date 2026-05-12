import dotenv from "dotenv";
import { resolve } from "node:path";
import { WebClient } from "@slack/web-api";
// @slack/bolt is CJS; Node ESM may not surface `import { App }` (e.g. Node 22+). Use default interop.
import slackBolt from "@slack/bolt";

const { App } = slackBolt as typeof import("@slack/bolt");
import * as rag from "./rag-client.js";
import { resolveRagForSlackTeam } from "./tenant-map.js";

// Load cwd `.env` first, then repo root — last wins so real tokens in `../.env` are NOT overwritten
// by placeholder lines still sitting in `rag-bot/.env` from `.env.example`.
dotenv.config({ path: resolve(process.cwd(), ".env"), override: true });
dotenv.config({ path: resolve(process.cwd(), "..", ".env"), override: true });

const ALLOW_UPLOAD_EXT = new Set([
  ".pdf",
  ".md",
  ".txt",
  ".json",
  ".png",
  ".jpg",
  ".jpeg",
  ".tiff",
  ".tif",
  ".bmp",
  ".gif",
  ".webp",
]);

function teamFrom(body: { team_id?: string; team?: { id?: string } }): string {
  return body.team_id || body.team?.id || "";
}

function extOf(name: string): string {
  const i = name.lastIndexOf(".");
  return i >= 0 ? name.slice(i).toLowerCase() : "";
}

/** Trim, strip quotes, CR, and invisible chars from .env / shell pastes. */
function normalizeSecret(raw: string | undefined): string {
  if (!raw) return "";
  let s = raw
    .trim()
    .replace(/\r/g, "")
    .replace(/[\u200b-\u200d\ufeff]/g, "");
  if (
    (s.startsWith('"') && s.endsWith('"')) ||
    (s.startsWith("'") && s.endsWith("'"))
  ) {
    s = s.slice(1, -1).trim();
  }
  return s;
}

async function sayLong(
  say: (a: { text: string; thread_ts?: string }) => Promise<unknown>,
  text: string,
  thread_ts?: string,
) {
  const max = 3500;
  for (let i = 0; i < text.length; i += max) {
    await say({ text: text.slice(i, i + max), thread_ts });
  }
}

const botToken = normalizeSecret(process.env.SLACK_BOT_TOKEN);
const appToken = normalizeSecret(process.env.SLACK_APP_TOKEN);
const signingSecret = normalizeSecret(process.env.SLACK_SIGNING_SECRET);

if (!botToken || !appToken || !signingSecret) {
  console.error(
    "Missing SLACK_BOT_TOKEN, SLACK_APP_TOKEN, or SLACK_SIGNING_SECRET. Copy rag-bot/.env.example to rag-bot/.env and fill values.",
  );
  process.exit(1);
}

if (!botToken.startsWith("xoxb-")) {
  console.error(
    "SLACK_BOT_TOKEN must be the Bot User OAuth Token and start with xoxb- (Slack app → OAuth & Permissions → install / reinstall app).",
  );
  process.exit(1);
}
if (!appToken.startsWith("xapp-")) {
  console.error(
    "SLACK_APP_TOKEN must be an App-Level Token starting with xapp- (Basic Information → App-Level Tokens → create with connections:write). Do not put the bot token here.",
  );
  process.exit(1);
}

if (process.env.RAG_BOT_DEBUG === "1") {
  console.error(
    `[rag-bot] token lengths: BOT=${botToken.length} APP=${appToken.length} SIGNING=${signingSecret.length}`,
  );
}

const app = new App({
  token: botToken,
  signingSecret,
  socketMode: true,
  appToken,
});

app.command("/raglist", async ({ ack, respond, body }) => {
  await ack();
  const team = teamFrom(body);
  const cfg = resolveRagForSlackTeam(team);
  if (!cfg) {
    await respond("This Slack workspace is not mapped to RAG (check SLACK_TEAM_ID / RAG_* in rag-bot/.env).");
    return;
  }
  try {
    const files = await rag.listDocs(cfg, cfg.customerId);
    if (!files.length) {
      await respond("No documents for this tenant/customer.");
      return;
    }
    const lines = files.slice(0, 80).map((f) => `• ${f.name} [${f.index_status}]`);
    await respond(lines.join("\n"));
  } catch (e) {
    await respond(`RAG error: ${e instanceof Error ? e.message : String(e)}`);
  }
});

app.event("app_mention", async ({ event, say }) => {
  const team = "team_id" in event && typeof event.team_id === "string" ? event.team_id : "";
  const cfg = resolveRagForSlackTeam(team);
  if (!cfg) {
    await say({ text: "Workspace not configured for RAG.", thread_ts: "thread_ts" in event ? event.thread_ts : undefined });
    return;
  }
  const raw = "text" in event && typeof event.text === "string" ? event.text : "";
  const query = raw.replace(/<@[^>]+>/g, "").trim();
  if (!query) {
    await say({
      text: "Mention me with a question, e.g. `@bot what is in the lease?`",
      thread_ts: "thread_ts" in event ? event.thread_ts : undefined,
    });
    return;
  }
  let out = "";
  try {
    for await (const ev of rag.streamChat(cfg, query, { customer_id: cfg.customerId })) {
      if (ev.kind === "token") out += ev.text;
      else if (ev.kind === "error") {
        out = `Error: ${ev.message}`;
        break;
      }
    }
    await sayLong(say, out || "(empty reply)", "thread_ts" in event ? event.thread_ts : undefined);
  } catch (e) {
    await say({
      text: `RAG error: ${e instanceof Error ? e.message : String(e)}`,
      thread_ts: "thread_ts" in event ? event.thread_ts : undefined,
    });
  }
});

app.event("file_shared", async ({ event, client }) => {
  const team =
    "team_id" in event && typeof event.team_id === "string"
      ? event.team_id
      : "user_team_id" in event && typeof event.user_team_id === "string"
        ? event.user_team_id
        : "";
  const cfg = resolveRagForSlackTeam(team);
  if (!cfg) return;

  const fileId =
    "file_id" in event && typeof event.file_id === "string"
      ? event.file_id
      : "file" in event && event.file && typeof event.file === "object" && "id" in event.file
        ? String((event.file as { id?: string }).id)
        : "";
  if (!fileId) return;

  const channelId =
    "channel_id" in event && typeof event.channel_id === "string" ? event.channel_id : "";

  try {
    const info = await client.files.info({ file: fileId });
    const file = info.file;
    if (!file?.name) return;
    const ext = extOf(file.name);
    if (!ALLOW_UPLOAD_EXT.has(ext)) return;

    const url = file.url_private_download || file.url_private;
    if (!url) return;

    const bin = await fetch(url, { headers: { Authorization: `Bearer ${botToken}` } });
    if (!bin.ok) {
      console.error("Slack file download failed", bin.status, await bin.text());
      return;
    }
    const buf = Buffer.from(await bin.arrayBuffer());
    const up = await rag.uploadFile(cfg, buf, file.name, cfg.customerId);
    if (up.index_job_id) {
      const job = await rag.pollUntilIndexed(cfg, up.index_job_id);
      if (job.status === "failed") {
        if (channelId) {
          await client.chat.postMessage({
            channel: channelId,
            text: `Upload received for \`${file.name}\` but indexing failed: ${job.error_message || "unknown"}`,
          });
        }
        return;
      }
    }
    if (channelId) {
      await client.chat.postMessage({
        channel: channelId,
        text: `Indexed \`${file.name}\` (${cfg.customerId}).`,
      });
    }
  } catch (e) {
    console.error("file_shared handler", e);
    if (channelId) {
      await client.chat.postMessage({
        channel: channelId,
        text: `RAG upload error: ${e instanceof Error ? e.message : String(e)}`,
      });
    }
  }
});

(async () => {
  try {
    const probe = new WebClient(botToken);
    await probe.auth.test();
  } catch (e) {
    console.error(
      "\nSlack rejected SLACK_BOT_TOKEN (auth.test). Wrong token, revoked install, or not the Bot User OAuth Token from this app.\n",
    );
    console.error(
      "Also run: unset SLACK_BOT_TOKEN SLACK_APP_TOKEN SLACK_SIGNING_SECRET  (then npm run dev) if you ever exported old tokens in this shell.\n",
    );
    console.error(e);
    process.exit(1);
  }

  try {
    await app.start();
    console.log(
      "RAG Slack bridge is running (Socket Mode). Try /raglist or @mention the bot.",
    );
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    if (msg.includes("invalid_auth")) {
      console.error(`
Slack returned invalid_auth while starting Socket Mode — usually SLACK_APP_TOKEN (xapp- + connections:write), not the bot token.

Fix:
  • Regenerate App-Level Token on this Slack app, scope connections:write, paste into SLACK_APP_TOKEN.
  • SLACK_SIGNING_SECRET = Basic Information → Signing Secret (not xoxb / not xapp).
`);
    }
    console.error(e);
    process.exit(1);
  }
})();
