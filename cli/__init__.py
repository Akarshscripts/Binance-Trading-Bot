from .app import app
from .predict import predict_app
from .backtest import backtest_app
from .research import quant_research_app

app.add_typer(predict_app)
app.add_typer(backtest_app)
app.add_typer(quant_research_app)

__all__ = ["app"]
