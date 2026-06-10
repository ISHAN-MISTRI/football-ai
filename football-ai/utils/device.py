import torch
from loguru import logger


def resolve_device(preferred: str | None = None) -> str:
    if preferred and preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        device = "cuda"
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"Using CUDA: {name} ({vram_gb:.1f} GB VRAM)")
        return device
    logger.warning("CUDA unavailable — falling back to CPU (slower inference)")
    return "cpu"


def clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
