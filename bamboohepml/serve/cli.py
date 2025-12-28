"""
服务 CLI

提供命令行接口启动服务。
"""
import typer
from typing import Optional
from typing_extensions import Annotated

from .fastapi_server import serve_fastapi
from .ray_serve import serve_ray

app = typer.Typer(help="BambooHepMl 服务命令")


@app.command()
def fastapi(
    model_path: Annotated[str, typer.Option("-m", "--model", help="模型文件路径")],
    pipeline_config: Annotated[Optional[str], typer.Option("-c", "--config", help="Pipeline 配置文件路径")] = None,
    host: Annotated[str, typer.Option("--host", help="主机地址")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port", help="端口号")] = 8000,
) -> None:
    """启动 FastAPI 服务。"""
    serve_fastapi(
        model_path=model_path,
        pipeline_config_path=pipeline_config,
        host=host,
        port=port,
    )


def ray(
    model_path: Annotated[Optional[str], typer.Option("-m", "--model", help="模型文件路径")] = None,
    pipeline_config: Annotated[Optional[str], typer.Option("-c", "--config", help="Pipeline 配置文件路径")] = None,
    run_id: Annotated[Optional[str], typer.Option("--run-id", help="MLflow run ID")] = None,
) -> None:
    """启动 Ray Serve 服务。"""
    serve_ray(
        model_path=model_path,
        pipeline_config_path=pipeline_config,
        run_id=run_id,
    )


# 这些函数会被主 CLI 导入使用

