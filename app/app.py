from uvicorn import run
from fastapi import FastAPI
from controllers import veg_class_router

def create_app() -> FastAPI:
    app = FastAPI(
        title="Vegetables inference service",
        description="Returns the class of a vegetable",
    )

    @app.get("/")
    def root():
        return {"message": "app is live"}
    
    app.include_router(veg_class_router)

    return app


def run_server():
    run(
        "app:create_app",
        host="0.0.0.0",
        port=8084,
        factory=True,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    print("Trying run")
    try:
        run_server()
    except Exception as e:
        print(e)