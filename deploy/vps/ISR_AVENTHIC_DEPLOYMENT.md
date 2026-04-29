# ISR Deployment Runbook (Hostinger VPS + Private GitHub + Docker Backend + Nginx)

Target setup:

- Domain: `isr.aventhic.com` (A record already points to VPS)
- Frontend: served by Nginx from static Vite build
- Backend: Docker container on `127.0.0.1:7000`
- Reverse proxy flow: `frontend -> Nginx -> backend`
- HTTPS: Let's Encrypt via Certbot

---

## 0) Pre-checks (do this on your laptop first)

1. Confirm DNS is pointing to your VPS:

```bash
nslookup isr.aventhic.com
```

It should return your VPS public IP.

2. Make sure your private GitHub repo is ready and has latest code.

---

## 1) SSH into VPS

```bash
ssh root@<YOUR_VPS_IP>
```

If you use another user, replace `root`.

---

## 2) Basic server setup

```bash
apt update && apt -y upgrade
apt -y install git curl nginx ufw ca-certificates gnupg lsb-release rsync
ufw allow OpenSSH
ufw allow 80
ufw allow 443
ufw --force enable
```

---

## 3) Install Docker + Compose

```bash
curl -fsSL https://get.docker.com | sh
docker --version
docker compose version
```

---

## 4) Create app directory

```bash
mkdir -p /opt/agentic-rag
cd /opt/agentic-rag
```

---

## 5) Clone private GitHub repo (token auth)

Use a GitHub Personal Access Token (PAT) with repo read access.

### Option A (recommended): one-time prompt (no token in shell history)

```bash
git clone https://github.com/<ORG_OR_USER>/<REPO>.git .
```

When prompted:

- Username: your GitHub username
- Password: your GitHub PAT (not GitHub account password)

### Option B (not recommended): token in URL

```bash
git clone https://<GITHUB_USERNAME>:<GITHUB_PAT>@github.com/<ORG_OR_USER>/<REPO>.git .
```

After cloning, immediately sanitize remote URL:

```bash
git remote set-url origin https://github.com/<ORG_OR_USER>/<REPO>.git
```

---

## 6) Configure environment variables

Copy template:

```bash
cp /opt/agentic-rag/deploy/vps/.env.vps.example /opt/agentic-rag/.env
```

Edit:

```bash
nano /opt/agentic-rag/.env
```

Set real values for:

- `BACKEND_API_KEY`
- `GROQ_API_KEY`
- `OPENAI_API_KEY` (optional)
- `MISTRAL_OCR_API_KEY` (if used)
- `TENANT_SETTINGS_MASTER_KEY`
- `LOG_LEVEL=INFO`
- `CORS_ALLOWED_ORIGINS=https://isr.aventhic.com`

Keep these path values:

- `CLAIM_SEARCH_DATA=/app/data`
- `CHROMA_PERSIST_DIR=/app/data/chroma`
- `TENANT_DB_PATH=/app/storage/tenant_registry.db`

---

## 7) Build frontend on VPS

Install Node 20 (if Node is missing or old):

```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt -y install nodejs
node -v
npm -v
```

Build frontend:

```bash
cd /opt/agentic-rag/frontend
npm install
npm run build
```

Publish static files for Nginx:

```bash
mkdir -p /var/www/agentic-rag/frontend/dist
rsync -a --delete /opt/agentic-rag/frontend/dist/ /var/www/agentic-rag/frontend/dist/
```

---

## 8) Start backend Docker container

```bash
cd /opt/agentic-rag/deploy/vps
docker compose -f docker-compose.backend.yml up -d --build
docker compose -f docker-compose.backend.yml ps
docker compose -f docker-compose.backend.yml logs -f
```

Stop logs with `Ctrl+C` after you confirm container is up.

---

## 9) Configure Nginx for domain

Copy config:

```bash
cp /opt/agentic-rag/deploy/vps/nginx-agentic-rag.conf /etc/nginx/sites-available/isr.aventhic.com.conf
```

Edit server name:

```bash
nano /etc/nginx/sites-available/isr.aventhic.com.conf
```

Change this line:

```nginx
server_name _;
```

to:

```nginx
server_name isr.aventhic.com;
```

Enable site:

```bash
ln -sf /etc/nginx/sites-available/isr.aventhic.com.conf /etc/nginx/sites-enabled/isr.aventhic.com.conf
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl restart nginx
```

---

## 10) Verify HTTP before SSL

```bash
curl http://isr.aventhic.com/api/health
```

Expected response:

```json
{"status":"ok"}
```

---

## 11) Enable HTTPS with Certbot

```bash
apt -y install certbot python3-certbot-nginx
certbot --nginx -d isr.aventhic.com
```

Choose redirect to HTTPS when prompted.

Verify:

```bash
curl https://isr.aventhic.com/api/health
```

---

## 12) Functional smoke tests

### Health

```bash
curl https://isr.aventhic.com/api/health
```

### Query (authenticated)

```bash
curl -X POST "https://isr.aventhic.com/query" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <BACKEND_API_KEY>" \
  -d '{"question":"Say hello in one line","customer_id":"demo1","stream":false}'
```

### Upload (authenticated)

```bash
curl -X POST "https://isr.aventhic.com/api/upload" \
  -H "X-API-Key: <BACKEND_API_KEY>" \
  -F "customer_id=demo1" \
  -F "files=@/path/to/sample.pdf"
```

---

## 13) Generate tenant key (optional but recommended)

```bash
cd /opt/agentic-rag
python3 scripts/create_tenant_key.py \
  --tenant-slug demo-tenant \
  --tenant-name "Demo Tenant" \
  --user-email admin@demo.local \
  --user-name "Demo Admin" \
  --label "demo-key" \
  --role owner \
  --scopes "docs:write,docs:read,query:run,chat:run,index:run,admin:read,admin:write"
```

Use returned `api_key` as tenant key in client integrations.

---

## 14) Update/deploy new code later

```bash
cd /opt/agentic-rag
git pull

cd /opt/agentic-rag/frontend
npm install
npm run build
rsync -a --delete /opt/agentic-rag/frontend/dist/ /var/www/agentic-rag/frontend/dist/

cd /opt/agentic-rag/deploy/vps
docker compose -f docker-compose.backend.yml up -d --build
```

---

## 15) Token security best practices

1. Prefer Option A clone (prompt input), not token in URL.
2. Never store PAT in `.env` or repo files.
3. If token was pasted in shell accidentally, rotate token in GitHub immediately.
4. Keep server `.env` private:

```bash
chmod 600 /opt/agentic-rag/.env
```

---

## 16) Quick troubleshooting

### Nginx config error

```bash
nginx -t
journalctl -u nginx -n 100 --no-pager
```

### Backend container not healthy

```bash
cd /opt/agentic-rag/deploy/vps
docker compose -f docker-compose.backend.yml logs -f
docker ps
```

### 502 from Nginx

- Usually backend container is down or not listening on `7000`.
- Check container logs and ensure compose service is running.

