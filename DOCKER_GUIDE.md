# 🐳 Docker Deployment Guide

Complete guide to containerize and run your Diabetes Prediction API with Docker.

---

## 📋 Prerequisites

### 1. Install Docker

**macOS:**
```bash
# Install Docker Desktop
brew install --cask docker
# Or download from: https://www.docker.com/products/docker-desktop
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install docker.io docker-compose -y
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER  # Add user to docker group
# Log out and back in for group changes
```

**Windows:**
- Download Docker Desktop from https://www.docker.com/products/docker-desktop
- Follow installation wizard

### 2. Verify Installation

```bash
docker --version
# Output: Docker version 24.x.x

docker-compose --version
# Output: Docker Compose version 2.x.x
```

---

## 🚀 Quick Start (3 Steps)

### Step 1: Ensure Models Are Trained

```bash
# Must have trained models in artifacts/models/
ls -la artifacts/models/lightgbm/
ls -la artifacts/models/xgboost/
ls -la artifacts/models/catboost/

# If empty, train first:
python src/main.py --ingest --train
```

### Step 2: Build Docker Image

```bash
docker-compose build
```

Expected output:
```
Building api
Step 1/10 : FROM python:3.10-slim
...
Successfully built abc123def456
Successfully tagged diabetes-prediction:latest
```

⏱️ First build takes 3-5 minutes (downloads dependencies)

### Step 3: Start the Application

```bash
docker-compose up -d
```

Expected output:
```
Creating diabetes-prediction-api ... done
```

✅ **API is now running at http://localhost:8000**

---

## 🧪 Testing Your Docker Container

### 1. Check Container Status

```bash
docker-compose ps
```

Expected output:
```
NAME                       STATUS         PORTS
diabetes-prediction-api    Up (healthy)   0.0.0.0:8000->8000/tcp
```

### 2. View Logs

```bash
docker-compose logs -f api
```

You should see:
```
🚀 Starting Diabetes Prediction API...
🔄 Loading models from disk...
✅ Successfully loaded 15 models
✅ API ready to serve predictions!
```

### 3. Test Health Endpoint

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "models_loaded": true,
  "total_models": 15
}
```

### 4. Test Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 52.0,
    "gender": 1.0,
    "bmi": 29.3,
    "waist_to_hip_ratio": 0.94,
    "systolic_bp": 138.0,
    "diastolic_bp": 88.0,
    "heart_rate": 75.0,
    "cholesterol": 215.0,
    "ldl": 135.0,
    "hdl": 45.0,
    "triglycerides": 175.0,
    "physical_activity": 2.5,
    "screen_time": 5.0,
    "sleep_duration": 6.5,
    "hypertension_history": 1.0,
    "cardiovascular_history": 0.0,
    "family_history": 1.0
  }'
```

### 5. Open Web Interface

Browser: http://localhost:8000/

---

## 🎛️ Docker Commands Reference

### Container Management

```bash
# Start containers
docker-compose up -d

# Stop containers
docker-compose down

# Restart containers
docker-compose restart

# View running containers
docker-compose ps

# View all containers (including stopped)
docker ps -a
```

### Logs & Debugging

```bash
# View logs (follow mode)
docker-compose logs -f

# View logs for specific service
docker-compose logs -f api

# Last 100 lines
docker-compose logs --tail=100 api

# Access container shell
docker-compose exec api bash

# Run command inside container
docker-compose exec api python --version
```

### Images & Builds

```bash
# Build/rebuild image
docker-compose build

# Build without cache (clean build)
docker-compose build --no-cache

# Pull latest base images
docker-compose pull

# List images
docker images

# Remove image
docker rmi diabetes-prediction:latest
```

### Cleanup

```bash
# Stop and remove containers
docker-compose down

# Remove containers + volumes
docker-compose down -v

# Remove unused images
docker image prune -a

# Remove everything (containers, images, volumes, networks)
docker system prune -a --volumes
```

---

## 🔧 Advanced Configuration

### Running with Different Ports

Edit [docker-compose.yml](docker-compose.yml):
```yaml
ports:
  - "8080:8000"  # Change 8080 to your desired port
```

Or override when running:
```bash
docker run -p 8080:8000 diabetes-prediction
```

### Running Multiple Workers (Production)

Edit [Dockerfile](Dockerfile):
```dockerfile
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Mounting Volumes (Update Models Without Rebuild)

Already configured in [docker-compose.yml](docker-compose.yml):
```yaml
volumes:
  - ./artifacts:/app/artifacts  # Models sync automatically
```

Update models locally:
```bash
python src/main.py --train      # Train new models
docker-compose restart          # Restart to load new models
```

### Environment Variables

Add to [docker-compose.yml](docker-compose.yml):
```yaml
environment:
  - DEFAULT_THRESHOLD=0.60
  - CV_FOLDS=5
  - API_WORKERS=2
```

---

## 📊 Monitoring & Maintenance

### Check Resource Usage

```bash
# Real-time stats
docker stats diabetes-prediction-api

# One-time check
docker-compose stats --no-stream
```

### Health Checks

Docker automatically monitors `/health` endpoint every 30 seconds.

Check health status:
```bash
docker inspect diabetes-prediction-api | grep -A 10 Health
```

### View Container Details

```bash
docker inspect diabetes-prediction-api
```

---

## 🐛 Troubleshooting

### Problem: Container Exits Immediately

**Check logs:**
```bash
docker-compose logs api
```

**Common causes:**
- Models not trained → Run `python src/main.py --train`
- Port already in use → Change port in docker-compose.yml
- Syntax error in code → Check logs for stack trace

### Problem: "Models not loaded" Error

**Solution:**
```bash
# Ensure models exist
ls -la artifacts/models/*/

# If empty, train models
python src/main.py --train

# Rebuild image
docker-compose build

# Restart
docker-compose up -d
```

### Problem: Port 8000 Already in Use

**Find what's using the port:**
```bash
lsof -i :8000
# Or on Linux:
netstat -tuln | grep 8000
```

**Solutions:**
1. Stop the conflicting process
2. Use a different port in docker-compose.yml
3. Change API port in Dockerfile

### Problem: Permission Denied

**On Linux:**
```bash
sudo usermod -aG docker $USER
# Log out and back in
```

### Problem: Out of Disk Space

**Clean up:**
```bash
# Remove stopped containers
docker container prune

# Remove unused images
docker image prune -a

# Remove everything unused
docker system prune -a --volumes
```

### Problem: Slow Build Time

**Solutions:**
1. Use `.dockerignore` (already configured)
2. Build with cache: `docker-compose build`
3. Check internet connection (dependencies download)

---

## 📦 Docker Image Details

### What's Inside the Container?

```
/app/
├── src/              # ML pipeline code
├── app/              # FastAPI application
├── static/           # HTML/CSS/JS frontend
├── artifacts/        # Models & outputs
│   └── models/       # 15 trained models
└── requirements.txt  # Dependencies
```

### Image Size

- **Unoptimized**: ~1.2GB
- **Optimized** (with .dockerignore): ~400MB
- **Base Python**: ~150MB
- **Dependencies**: ~250MB

### Layers

```
Layer 1: Python 3.10 base image
Layer 2: System dependencies
Layer 3: Python packages (cached)
Layer 4: Application code
Layer 5: User & permissions
```

---

## 🎯 Production Checklist

Before deploying to production:

- [ ] Models trained and verified
- [ ] Health checks working
- [ ] Logs configured (max size, retention)
- [ ] Resource limits set (CPU, memory)
- [ ] Secrets externalized (not in image)
- [ ] HTTPS configured (nginx + Let's Encrypt)
- [ ] Monitoring setup (CloudWatch, Datadog, etc.)
- [ ] Backup strategy for models
- [ ] CI/CD pipeline configured
- [ ] Load testing completed

---

## 🌐 Next Steps

1. ✅ **Docker Setup** (YOU ARE HERE!)
2. ⬜ **AWS EC2 Deployment** - Deploy to cloud
3. ⬜ **Nginx Reverse Proxy** - Add HTTPS
4. ⬜ **CI/CD Pipeline** - Automate deployments

Ready for AWS? See [AWS_DEPLOYMENT_GUIDE.md](AWS_DEPLOYMENT_GUIDE.md)

---

## 📚 Additional Resources

- **Docker Docs**: https://docs.docker.com/
- **Docker Compose Docs**: https://docs.docker.com/compose/
- **FastAPI Docker**: https://fastapi.tiangolo.com/deployment/docker/
- **Best Practices**: https://docs.docker.com/develop/dev-best-practices/

---

## 🎉 Success!

If you can access http://localhost:8000 and make predictions, your Docker setup is complete!

**Your containerized ML API is now:**
- ✅ Portable (runs anywhere Docker runs)
- ✅ Reproducible (same environment every time)
- ✅ Production-ready (health checks, logging, resource limits)
- ✅ Easy to deploy (one command to AWS EC2!)
