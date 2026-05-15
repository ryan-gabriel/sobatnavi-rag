import asyncio
import logging
from functools import wraps

logger = logging.getLogger(__name__)

def retry_with_backoff(retries=3, backoff_in_seconds=1):
    """Decorator untuk mencoba ulang (retry) fungsi jika terjadi error teknis."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < retries:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    logger.warning(f"Gagal memanggil {func.__name__} (Percobaan {attempt}/{retries}): {e}")
                    if attempt == retries:
                        raise e  # Lempar error jika sudah batas maksimal
                    await asyncio.sleep(backoff_in_seconds * (2 ** (attempt - 1)))
        return wrapper
    return decorator