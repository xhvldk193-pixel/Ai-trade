# =============================================
# Multi-Agent Debate, Judge, Memory & Reflection Layer
# =============================================
# TradingAgents(https://github.com/TauricResearch/TradingAgents) 의
# 애널리스트 팀 패턴을 BTC 선물 맥락으로 포팅한 레이어.
#
# 구조:
#   - Phase 1: Bull / Bear Researcher (debate.py, prompts.py)
#   - Phase 2: Investment Judge (judge.py)
#   - Phase 3: Risk Triad + Pipeline (risk_triad.py, risk_prompts.py, pipeline.py)
#   - Phase 4: BM25 Memory + AgentMemories + Reflection (memory.py, reflection.py)
#   - Phase 5: Signal Processing (signal_processing.py)
#
# 철학:
#   - LangGraph/LangChain 의존성 없이 순수 anthropic SDK 로 동작
#   - 각 에이전트는 역할별 독립 메모리를 갖고 재귀적으로 개선됨
#   - 최종 종합은 analyzer.analyze_with_claude() 가 담당
# =============================================

import logging as _logging

_log = _logging.getLogger(__name__)


# ── 단일 Anthropic 클라이언트 ────────────────────────
# 분석 1회당 5개 에이전트가 각자 client = anthropic.Anthropic(...) 으로 생성하면
# HTTP connection pool 이 매번 새로 만들어지고 5개 동시 생성됨.
# 모듈 전역 단일 인스턴스로 통일 — connection pool 재사용.
_anthropic_client = None
_anthropic_client_lock = None  # threading.Lock - 지연 생성

def get_anthropic_client():
    """전역 anthropic.Anthropic 인스턴스 반환. 첫 호출 시 lazy 생성."""
    global _anthropic_client, _anthropic_client_lock
    if _anthropic_client is not None:
        return _anthropic_client
    if _anthropic_client_lock is None:
        import threading
        _anthropic_client_lock = threading.Lock()
    with _anthropic_client_lock:
        if _anthropic_client is None:
            import anthropic as _a
            from config import CLAUDE_API_KEY as _key
            _anthropic_client = _a.Anthropic(api_key=_key)
    return _anthropic_client


# ── Phase 1 ─────────────────────────────────────────
from .debate import run_bull_bear_debate, format_debate_block, DebateResult

# ── Phase 2 ─────────────────────────────────────────
try:
    from .judge import run_judge, format_judge_block, JudgeResult
    _JUDGE_OK = True
except Exception as _judge_exc:
    run_judge = None           # type: ignore
    format_judge_block = None  # type: ignore
    JudgeResult = None         # type: ignore
    _JUDGE_OK = False
    _log.warning("agents.judge 로드 실패 — %s: %s", type(_judge_exc).__name__, _judge_exc)

# ── Phase 3 ─────────────────────────────────────────
from .risk_triad import run_risk_triad, format_risk_block, RiskTriadResult
from .pipeline import run_pipeline, PipelineResult

# ── Phase 4 (메모리/리플렉션 — rank_bm25 의존성) ─────
try:
    from .memory import (
        FinancialSituationMemory,
        AgentMemories,
        format_memory_block,
        get_memory,
        get_agent_memories,
    )
    _MEMORY_OK = True
except Exception as _memory_exc:
    FinancialSituationMemory = None   # type: ignore
    AgentMemories = None              # type: ignore
    format_memory_block = None        # type: ignore
    get_memory = None                 # type: ignore
    get_agent_memories = None         # type: ignore
    _MEMORY_OK = False
    _log.warning(
        "agents.memory 로드 실패 — %s: %s (rank_bm25 미설치 여부 확인)",
        type(_memory_exc).__name__, _memory_exc,
    )

try:
    from .reflection import reflect_on_record, reflect_for_role, ReflectionResult
    _REFLECTION_OK = True
except Exception as _reflect_exc:
    reflect_on_record = None   # type: ignore
    reflect_for_role = None    # type: ignore
    ReflectionResult = None    # type: ignore
    _REFLECTION_OK = False
    _log.warning(
        "agents.reflection 로드 실패 — %s: %s",
        type(_reflect_exc).__name__, _reflect_exc,
    )

# ── Phase 5 (Signal Processing) ──────────────────────
try:
    from .signal_processing import extract_trading_signal, TradingSignal
    _SIGNAL_OK = True
except Exception as _signal_exc:
    extract_trading_signal = None  # type: ignore
    TradingSignal = None           # type: ignore
    _SIGNAL_OK = False
    _log.warning(
        "agents.signal_processing 로드 실패 — %s: %s",
        type(_signal_exc).__name__, _signal_exc,
    )


__all__ = [
    # Phase 1
    "run_bull_bear_debate",
    "format_debate_block",
    "DebateResult",
    # Phase 2
    "run_judge",
    "format_judge_block",
    "JudgeResult",
    # Phase 3
    "run_risk_triad",
    "format_risk_block",
    "RiskTriadResult",
    "run_pipeline",
    "PipelineResult",
    # Phase 4
    "FinancialSituationMemory",
    "AgentMemories",
    "format_memory_block",
    "get_memory",
    "get_agent_memories",
    "reflect_on_record",
    "reflect_for_role",
    "ReflectionResult",
    # Phase 5
    "extract_trading_signal",
    "TradingSignal",
]
