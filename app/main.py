from fastapi import FastAPI

app = FastAPI(
    title="End-to-End MLOps Pipeline",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "project": "End-to-End MLOps Pipeline",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }
