import os
import sys
from pathlib import Path
from loguru import logger

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

def setup_logger(log_file=None, rotation="500 MB"):
    """
    设置loguru日志记录器
    
    Args:
        log_file: 日志文件路径，如果为None则只输出到控制台
        rotation: 日志文件轮转设置，默认500MB
    """
    # 清除默认配置
    logger.remove()
    
    # 添加标准输出
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=LOG_LEVEL,
        colorize=True
    )
    
    # 如果指定了日志文件，添加文件输出
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
            level=LOG_LEVEL,
            rotation=rotation,
            compression="zip"
        )
    
    return logger

# 默认设置
logger = setup_logger()

# 导出logger供其他模块使用
__all__ = ["logger", "setup_logger"] 