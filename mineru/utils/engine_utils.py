# Copyright (c) Opendatalab. All rights reserved.
import os

from loguru import logger

from mineru.utils.check_sys_env import (
    is_linux_environment,
    is_mac_environment,
    is_mac_os_version_supported,
    is_windows_environment,
)


def get_vlm_engine(inference_engine: str, is_async: bool = False) -> str:
    """Select or validate the VLM inference engine."""
    if inference_engine == 'auto':
        env_engine = os.getenv("MINERU_VLM_ENGINE", "").strip().lower()
        if env_engine:
            inference_engine = env_engine
            logger.info(f"Using VLM engine from MINERU_VLM_ENGINE={env_engine}.")
        elif is_windows_environment():
            inference_engine = _select_windows_engine()
        elif is_linux_environment():
            inference_engine = _select_linux_engine(is_async)
        elif is_mac_environment():
            inference_engine = _select_mac_engine()
        else:
            logger.warning("Unknown operating system, falling back to transformers")
            inference_engine = 'transformers'

    formatted_engine = _format_engine_name(inference_engine)
    logger.info(f"Using {formatted_engine} as the inference engine for VLM.")
    return formatted_engine


def _select_windows_engine() -> str:
    """Select the default VLM engine on Windows."""
    try:
        import lmdeploy  # noqa: F401
        return 'lmdeploy'
    except ImportError:
        return 'transformers'


def _select_linux_engine(is_async: bool) -> str:
    """Select the default VLM engine on Linux."""
    try:
        import vllm  # noqa: F401
        return 'vllm-async' if is_async else 'vllm'
    except ImportError:
        try:
            import lmdeploy  # noqa: F401
            return 'lmdeploy'
        except ImportError:
            return 'transformers'


def _select_mac_engine() -> str:
    """Select the default VLM engine on macOS."""
    try:
        from mlx_vlm import load as mlx_load  # noqa: F401
        if is_mac_os_version_supported():
            return 'mlx'
        return 'transformers'
    except ImportError:
        return 'transformers'


def _format_engine_name(engine: str) -> str:
    """Normalize engine names used by MinerU's backend dispatch."""
    if engine != 'transformers':
        return f"{engine}-engine"
    return engine
