FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --root-user-action=ignore -r requirements.txt \
    && groupadd --system app \
    && useradd --system --gid app --home-dir /app --no-create-home app \
    && chown app:app /app

COPY --chown=app:app . .
USER app
CMD ["python", "validate.py"]
