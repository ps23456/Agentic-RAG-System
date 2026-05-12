/** Map a Slack workspace (team) to RAG HTTP credentials. */

export interface RagTenantConfig {
  baseUrl: string;
  apiKey: string;
  customerId: string;
}

/** Resolve RAG config for `slackTeamId` (e.g. T0ABC123). */
export function resolveRagForSlackTeam(slackTeamId: string): RagTenantConfig | null {
  const baseUrl = (process.env.RAG_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
  const apiKey = (process.env.RAG_API_KEY || "").trim();
  const customerId = (process.env.RAG_CUSTOMER_ID || "default").trim() || "default";
  const allowOnly = (process.env.SLACK_TEAM_ID || "").trim();
  if (allowOnly && allowOnly !== slackTeamId) return null;
  if (!apiKey) return null;
  return { baseUrl, apiKey, customerId };
}
