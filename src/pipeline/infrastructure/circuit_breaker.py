"""Circuit breaker pattern for fault tolerance."""

import logging
import time
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker for external service calls."""
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        expected_exception: type = Exception
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        if self._state == CircuitState.OPEN:
            # Check if we should try half-open
            if self._last_failure_time and \
               time.time() - self._last_failure_time >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
                logger.info(f"Circuit {self.name} entering HALF_OPEN state")
        
        return self._state
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            raise CircuitBreakerOpenError(
                f"Circuit {self.name} is OPEN - service unavailable"
            )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Call async function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            raise CircuitBreakerOpenError(
                f"Circuit {self.name} is OPEN - service unavailable"
            )
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= 3:  # Need 3 successes to close
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                logger.info(f"Circuit {self.name} CLOSED - service recovered")
        else:
            self._failure_count = max(0, self._failure_count - 1)
    
    def _on_failure(self):
        """Handle failed call."""
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.error(
                f"Circuit {self.name} OPENED after {self._failure_count} failures"
            )
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator for circuit breaker."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


# Registry of circuit breakers
_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str, **kwargs) -> CircuitBreaker:
    """Get or create circuit breaker."""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, **kwargs)
    return _circuit_breakers[name]
