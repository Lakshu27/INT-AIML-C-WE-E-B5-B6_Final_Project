# Deployment

## A. Hugging Face Spaces (Docker) - required "deployed demo link"
1. Create a Space -> SDK **Docker** -> blank template.
2. Push this repository to the Space (or connect GitHub):
   ```bash
   git remote add space https://huggingface.co/spaces/<user>/coverwise-ai
   git push space main
   ```
3. Space -> Settings -> **Variables and secrets**: add `GROQ_API_KEY` (secret) and `LLM_PROVIDER=groq`.
   Optional: `USE_EMBEDDINGS=0` for a faster, lighter build.
4. The `Dockerfile` installs Tesseract, trains the model at build time and serves Streamlit on port 7860.
5. Use only the fictional sample PDFs or public policy wordings in the demo.

## B. AWS EC2 (optional extension)
```bash
# Ubuntu 22.04/24.04, t3.small or larger, security group: allow 22 (your IP) and 8501
sudo apt update && sudo apt install -y python3-venv tesseract-ocr git
git clone https://github.com/<you>/CoverWise_AI.git && cd CoverWise_AI
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env && nano .env            # or read keys from AWS Secrets Manager / SSM
python scripts/train_model.py
nohup streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 > app.log 2>&1 &
```
Production touches: Nginx reverse proxy + HTTPS (certbot), a `systemd` service instead of `nohup`, keys in
AWS Secrets Manager / SSM Parameter Store, and stop the instance when not in use to avoid charges.

## C. Docker anywhere
```bash
docker build -t coverwise .
docker run -p 7860:7860 --env-file .env coverwise
```
