# Unit 2.15 -- backend container image.
#
# Deliberately simple, single-stage build: this project has no compiled
# extensions or heavy build step beyond `pip install`, so a multi-stage
# build would add complexity without a measurable image-size or build-time
# benefit. Documented as a considered trade-off, not an oversight.
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (separate layer) so code changes don't
# invalidate the (slow) dependency-install layer on every rebuild.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
