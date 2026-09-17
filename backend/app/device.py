import logging
from typing import Tuple

log = logging.getLogger('gate.device')

_CACHED_DEVICE: str | None = None
_CACHED_HAS_CUDA: bool | None = None


def resolve_optimal_device(preference: str | bool | None = 'auto') -> Tuple[str, bool]:
    """
    Dynamically probes hardware capabilities to find the best working execution device.
    
    Returns:
        tuple of (device_str, is_cuda_enabled)
        e.g. ('cuda:0', True) or ('cpu', False)
    """
    global _CACHED_DEVICE, _CACHED_HAS_CUDA
    pref_str = str(preference).strip().lower() if preference is not None else 'auto'

    if pref_str in ('false', 'cpu'):
        return 'cpu', False

    if _CACHED_DEVICE is not None and _CACHED_HAS_CUDA is not None:
        if pref_str in ('auto', 'true', 'cuda', 'gpu', '1', '0'):
            return _CACHED_DEVICE, _CACHED_HAS_CUDA

    try:
        import torch
        if not torch.cuda.is_available():
            _CACHED_DEVICE = 'cpu'
            _CACHED_HAS_CUDA = False
            return 'cpu', False

        device_name = torch.cuda.get_device_name(0)
        # Probe test: actively verify the current PyTorch install has compatible
        # CUDA kernels for this specific GPU architecture (e.g. catches sm_120 mismatches)
        try:
            probe = torch.zeros(1, device='cuda:0')
            del probe
            log.info(f'Hardware acceleration verified on {device_name} (cuda:0).')
            _CACHED_DEVICE = 'cuda:0'
            _CACHED_HAS_CUDA = True
            return 'cuda:0', True
        except Exception as exc:
            log.warning(
                f'CUDA is detected ({device_name}) but tensor execution failed ({exc}). '
                'Auto-falling back to CPU safely without crashing.'
            )
            _CACHED_DEVICE = 'cpu'
            _CACHED_HAS_CUDA = False
            return 'cpu', False

    except Exception as exc:
        log.warning(f'Error probing hardware device: {exc}. Defaulting to CPU.')
        _CACHED_DEVICE = 'cpu'
        _CACHED_HAS_CUDA = False
        return 'cpu', False
