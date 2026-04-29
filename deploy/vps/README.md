# Hostinger VPS Deployment (Frontend -> Nginx -> Backend Docker on port 7000)

This setup runs:

- `frontend` as static files served by host Nginx
- `backend` in Docker on `127.0.0.1:7000`
- Nginx reverse-proxy routes:
  - `/api/*` -> backend
  - `/query` -> backend
  - all other routes -> frontend SPA (`index.html`)

## 1) VPS prerequisites (Ubuntu)

```bash
sudo apt update && sudo apt -y upgrade
sudo apt -y install git curl nginx ufw
sudo ufw allow OpenSSH
sudo ufw allow 80
sudo ufw allow 443
sudo ufw --force enable
```

## 2) Install Docker + Compose plugin

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker
docker --version
docker compose version
```

## 3) Clone project on VPS

```bash
sudo mkdir -p /opt/agentic-rag
sudo chown -R $USER:$USER /opt/agentic-rag
cd /opt/agentic-rag
git clone <YOUR_REPO_URL> .
```

## 4) Configure backend environment

```bash
cp deploy/vps/.env.vps.example .env
```

Edit `.env` and set real values (`BACKEND_API_KEY`, `GROQ_API_KEY`, `TENANT_SETTINGS_MASTER_KEY`, etc).

For public-IP testing, set:

```env
CORS_ALLOWED_ORIGINS=http://<YOUR_VPS_PUBLIC_IP>
```

## 5) Build frontend static files

```bash
cd /opt/agentic-rag/frontend
npm install
npm run build
```

## 6) Publish frontend build for Nginx

```bash
sudo mkdir -p /var/www/agentic-rag/frontend
sudo rsync -a --delete /opt/agentic-rag/frontend/dist/ /var/www/agentic-rag/frontend/dist/
```

## 7) Start backend container on port 7000

```bash
cd /opt/agentic-rag/deploy/vps
docker compose -f docker-compose.backend.yml up -d --build
docker compose -f docker-compose.backend.yml ps
docker compose -f docker-compose.backend.yml logs -f
```

## 8) Install Nginx site config

```bash
sudo cp /opt/agentic-rag/deploy/vps/nginx-agentic-rag.conf /etc/nginx/sites-available/agentic-rag.conf
sudo ln -sf /etc/nginx/sites-available/agentic-rag.conf /etc/nginx/sites-enabled/agentic-rag.conf
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
```

## 9) Verify

From your laptop:

```bash
curl "http://<YOUR_VPS_PUBLIC_IP>/api/health"
```

Optional protected route check:

```bash
curl -X POST "http://<YOUR_VPS_PUBLIC_IP>/query" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <BACKEND_API_KEY>" \
  -d '{"question":"Say hello in one line","customer_id":"demo1","stream":false}'
```

## 10) Update workflow

```bash
cd /opt/agentic-rag
git pull

cd /opt/agentic-rag/frontend
npm install
npm run build
sudo rsync -a --delete /opt/agentic-rag/frontend/dist/ /var/www/agentic-rag/frontend/dist/

cd /opt/agentic-rag/deploy/vps
docker compose -f docker-compose.backend.yml up -d --build
```

## 11) Optional HTTPS later (domain required)

If you attach a domain, you can add SSL:

```bash
sudo apt -y install certbot python3-certbot-nginx
sudo certbot --nginx -d api.yourdomain.com
```

