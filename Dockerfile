# Works for Hugging Face Spaces (Docker SDK) and AWS EC2 / any Docker host.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
COPY --chown=user . .
RUN python scripts/train_model.py
EXPOSE 7860
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]
