# Deployment Guide

This document describes how the application is configured to serve static files from the FastAPI backend, following the Databricks GenAI app template pattern.

## Architecture Overview

### Development Mode
```
┌─────────────────┐         ┌──────────────────┐
│   Vite Dev      │  Proxy  │   FastAPI        │
│   Server        │────────▶│   Backend        │
│   Port 3000     │  /api/* │   Port 8000      │
└─────────────────┘         └──────────────────┘
```

- Frontend: Vite dev server with hot reload on port 3000
- Backend: FastAPI on port 8000
- API calls: Proxied from `/api` → `http://localhost:8000/api`
- CORS: Enabled for localhost:3000

### Production Mode
```
┌────────────────────────────────┐
│     FastAPI Server             │
│        Port 8000               │
├────────────────────────────────┤
│  Static Files (/)              │
│  - index.html                  │
│  - JS/CSS bundles              │
│  - Assets                      │
├────────────────────────────────┤
│  API Endpoints (/api/*)        │
│  - /api/predict                │
│  - /api/feedback               │
│  - /api/health                 │
└────────────────────────────────┘
```

- Single server deployment
- Static files served from `static/`
- API endpoints at `/api/*`
- CORS disabled (same origin)

## Key Configuration Files

### 1. Backend: `main.py`

**Static File Serving:**
```python
# Static file serving for production
static_path = Path(__file__).parent / "static"
if static_path.exists():
    logger.info(f"Serving static files from {static_path}")
    app.mount("/", StaticFiles(directory=str(static_path), html=True), name="static")
else:
    logger.info("Static files directory not found. Run 'npm run build' in frontend directory.")
```

**Environment-based CORS:**
```python
env = os.getenv("ENV", "development")
allowed_origins = ["http://localhost:3000", "http://localhost:5173"] if env == "development" else []

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    ...
)
```

**API Routes:** All routes prefixed with `/api`
- `/api` - Health check
- `/api/health` - Server health
- `/api/predict` - Image generation
- `/api/feedback` - User feedback

### 2. Frontend: `frontend/vite.config.js`

**Development Proxy:**
```javascript
server: {
  port: 3000,
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
    },
  },
}
```

**Build Output:**
```javascript
build: {
  outDir: '../static',
  emptyOutDir: true,
}
```

### 3. Frontend: `frontend/src/App.jsx`

**Environment-aware API URL:**
```javascript
const API_URL = import.meta.env.DEV ? 'http://localhost:8000' : ''
```

This ensures:
- Development: Calls `http://localhost:8000/api/*`
- Production: Calls `/api/*` (same origin)

## Build Process

### Automated Build Script

Run the provided build script:
```bash
./build.sh
```

This will:
1. Install frontend dependencies
2. Build frontend → `static/`
3. Create Python virtual environment
4. Install backend dependencies

### Manual Build Steps

1. Build frontend:
```bash
cd frontend
npm install
npm run build
```

2. Verify output:
```bash
ls -la static/
# Should contain: index.html, assets/, etc.
```

3. Install backend:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the Application

### Development Mode

**Terminal 1 - Backend:**
```bash
source venv/bin/activate
python main.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

Access at: `http://localhost:3000`

### Production Mode

**Single Command:**
```bash
source venv/bin/activate
export ENV=production
python main.py
```

Access at: `http://localhost:8000`

**With Gunicorn (Recommended):**
```bash
source venv/bin/activate
export ENV=production
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000
```

## How It Works

### 1. Route Matching Order (Production)

FastAPI processes routes in order:
1. API routes (`/api/*`) - Handled by FastAPI endpoints
2. Static files (`/*`) - Handled by StaticFiles middleware

The `html=True` parameter enables SPA routing by serving `index.html` for unmatched paths.

### 2. Development Proxy

Vite's proxy configuration intercepts `/api` requests and forwards them to `http://localhost:8000/api`.

### 3. CORS Handling

- **Development:** Backend allows requests from localhost:3000
- **Production:** No CORS headers needed (same origin)

## Deployment Considerations

### Single Server Deployment

**Advantages:**
- Simplified deployment
- No CORS issues
- Single domain/port
- Easier SSL setup

**Configuration:**
1. Build frontend: `npm run build`
2. Set `ENV=production`
3. Deploy backend with static files

### Docker Deployment

Example `Dockerfile`:
```dockerfile
# Build stage
FROM node:18 AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Runtime stage
FROM python:3.10
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY main.py ./
COPY --from=frontend-build /app/static ./static
ENV ENV=production
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Cloud Deployment

**AWS / GCP / Azure:**
1. Build frontend locally or in CI/CD
2. Include `static/` in deployment package
3. Set environment variable `ENV=production`
4. Deploy backend with reverse proxy (nginx/ALB)

**Databricks:**
- Follow Databricks GenAI app template pattern
- Deploy as Databricks App
- Static files served alongside API

## Troubleshooting

### Static Files Not Served

**Symptom:** 404 errors in production

**Solution:**
1. Verify `static/` exists
2. Check `index.html` is present
3. Ensure `StaticFiles` is mounted with `html=True`
4. Check backend logs for static path message

### CORS Errors in Development

**Symptom:** API calls blocked by CORS

**Solution:**
1. Verify backend is running on port 8000
2. Check CORS middleware allows localhost:3000
3. Ensure Vite proxy is configured
4. Restart both frontend and backend

### API 404 in Production

**Symptom:** API calls return 404

**Solution:**
1. Verify endpoints are prefixed with `/api`
2. Check `API_URL` in `App.jsx`
3. Ensure frontend build is recent
4. Clear browser cache

### Page Refresh 404

**Symptom:** Direct navigation to routes fails

**Solution:**
- Verify `html=True` in `StaticFiles` mount
- This enables fallback to `index.html` for SPA routing

## Reference

This implementation follows the pattern from:
https://github.com/databricks-solutions/databricks-genai-app-template

Key differences from standard deployments:
- API routes prefixed with `/api`
- Environment-based CORS configuration
- Static files mounted at root with `html=True`
- Frontend build output to `static/`
