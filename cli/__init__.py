from .app import app
from .predict import predict_app

app.add_typer(predict_app)

__all__ = ["app"]
