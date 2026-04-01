"""Strategy registry — 전략을 이름으로 조회/생성."""

from src.strategies.base import Strategy

_REGISTRY: dict[str, type[Strategy]] = {}


def register(cls: type[Strategy]) -> type[Strategy]:
    """전략 클래스를 레지스트리에 등록하는 데코레이터."""
    instance = cls()
    _REGISTRY[instance.name] = cls
    return cls


def get_strategy(name: str) -> Strategy:
    """이름으로 전략 인스턴스 생성."""
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(_REGISTRY.keys())}")
    return cls()


def list_strategies() -> list[dict]:
    """대시보드 표시용 전략 목록."""
    result = []
    for name, cls in _REGISTRY.items():
        instance = cls()
        result.append({
            "name": instance.name,
            "label": instance.label,
            "description": instance.description,
            "mode": instance.mode.value,
        })
    return result


def _register_all():
    """모든 전략 모듈을 임포트하여 등록."""
    from src.strategies import scalper  # noqa: F401
    from src.strategies import adaptive_scalper  # noqa: F401
    from src.strategies import smart_scalper  # noqa: F401
    from src.strategies import aggressive_scalper  # noqa: F401
    from src.strategies import contrarian_scalper  # noqa: F401
    from src.strategies import data_driven_scalper  # noqa: F401
    from src.strategies import pattern_scalper  # noqa: F401


_register_all()
