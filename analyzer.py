# =============================================
# Claude API 연동 - 매매 시그널 분석
# =============================================
import re
import time
import anthropic
from typing import Optional
from config import CLAUDE_API_KEY, CLAUDE_MODEL, DEFAULT_SYMBOL, symbol_to_pair
from indicators import summarize_indicators, fibonacci_swing_levels, fib_window_for_tf
from account_context import fetch_account_context, format_account_context
from market_context import fetch_market_context, format_market_context
from macro_fetcher import fetch_macro_context, format_macro_context
from time_utils import now_kst
from agents import (
    run_bull_bear_debate,
    format_debate_block,
    DebateResult,
    run_pipeline,
    PipelineResult,
)
try:
    from agents.memory import get_memory  # may be None if rank_bm25 missing
except Exception:
    get_memory = None  # type: ignore
try:
    from agents.memory import get_agent_memories  # AgentMemories 싱글턴 팩토리
except Exception:
    get_agent_memories = None  # type: ignore
try:
    from agents.situation_digest import summarize_situation_tags
except Exception:
    summarize_situation_tags = None  # type: ignore
try:
    from agents.signal_processing import extract_trading_signal, TradingSignal
except Exception:
    extract_trading_signal = None  # type: ignore
    TradingSignal = None           # type: ignore

import os as _os
import logging as _logging

_memory_logger = _logging.getLogger(__name__)
# 메모리 쓰기 전역 스위치 — 스테이징/백테스트 환경에서 기록 방지용
MEMORY_WRITE_ENABLED = _os.getenv("MEMORY_WRITE_ENABLED", "1").lower() not in ("0", "false", "no")

PAIR_LABEL = symbol_to_pair(DEFAULT_SYMBOL)

SYSTEM_PROMPT = (
    f"당신은 10년 경력의 {PAIR_LABEL} 선물 시장 애널리스트입니다.\n"
    "역할: 정량 데이터와 시장 심리를 엮어 현재 구조를 해석하고 명확한 매매 관점을 제시하는 인간형 리서치 애널리스트.\n"
    f"전문 영역: {PAIR_LABEL} 선물 단기 분석 (수십 분~수 시간, 주로 15m·1h 기준 모멘텀 및 단기 추세 추종).\n"
    "리스크 성향: 근거 기반 결정주의. 근거가 충분하면 명확한 방향, 근거가 약하면 정직하게 중립을 선택.\n"
    "  중립은 회피가 아니라 정직한 판단입니다. 단, 약한 근거로 무리하게 방향성을 잡지 마세요.\n"
    "분석 철학: 단일 지표 신호보다 멀티 타임프레임 정렬, 가격 구조, 파생상품 심리, 거시 레짐의 정합성을 중시합니다.\n"
    "제공되는 데이터는 1d·4h·1h·15m·5m 캔들 + 거시·파생상품 심리 지표 + 계좌/포지션 제약 정보 + 최근 12시간·72시간·7일 계좌 운영 맥락입니다.\n"
    "\n"
    "근거 인용 규칙 (환각 방지):\n"
    "- 구체적 수치(가격, RSI, 펀딩비, OI, ATR 등)는 반드시 입력 데이터에 있는 값만 인용하세요.\n"
    "- 입력에 없는 수치는 절대 지어내지 마세요. 데이터에 없으면 '데이터 미제공'으로 명시.\n"
    "- 추세·심리·구조 같은 정성적 해석은 자유롭되, 그 해석이 어떤 입력 데이터 라인에서 나왔는지 명확히.\n"
    "\n"
    "애널리스트 원칙:\n"
    "1. 먼저 확정된 사실을 말하고, 그 다음 해석을 제시하세요.\n"
    "2. 주도 시나리오를 명확하게 밀되, 반대 시나리오와 관점이 약해지는 조건도 함께 적으세요.\n"
    "3. 방향성 신호가 있으면 지금 당장 취할 행동을 구체적으로 제시하세요.\n"
    "4. 5m는 진입 타이밍 힌트일 뿐, 큰 방향의 핵심 근거로 과대평가하지 마세요.\n"
    "5. 계좌/포지션 정보와 최근 운영 맥락은 시장 방향의 근거가 아니라 실행 제약입니다.\n"
    "   오픈 포지션이 있을 경우 신규 진입 대신 현재 포지션 관리를 우선적으로 분석하세요.\n"
    "   ★ 포지션 보유 중에는 [매매 파라미터]를 현재 포지션 관리 레벨로 작성하세요.\n"
    "   ★ 포지션 보유 중 [매매 파라미터] 형식 절대 엄수:\n"
    "      • 진입가: $[현재 보유 포지션 진입가 그대로]\n"
    "      • 손절가: $[권고 SL 숫자 하나 — 서술·조건문 금지. 자동매매가 이 값을 거래소에 즉시 반영]\n"
    "      • 목표가: $[권고 TP 숫자 하나 — 서술·조건문 금지. 자동매매가 이 값을 거래소에 즉시 반영]\n"
    "      ※ 공격적/보수적 서술은 [대응] 섹션에만. [매매 파라미터]는 반드시 숫자 하나만.\n"
    "   ★ [대응] 섹션에 추가 진입 조언은 절대 작성하지 마세요. 포지션 관리(SL이동/TP조정/청산조건)만 작성하세요.\n"
    "6. 최근 계좌 운영 맥락이 보이면 수익 보호 모드인지, 손실 복구 시도인지 읽되 관측된 사실에 기대어 표현하세요.\n"
    "7. 박스권(레인지) 레짐 특별 규칙:\n"
    "   - 상단 저항 ±0.3% 영역: 숏만 허용\n"
    "   - 하단 지지 ±0.3% 영역: 롱만 허용\n"
    "   - 박스 중간 50% 구간: 진입 금지\n"
    "   - 박스 돌파 후 거래량이 평균의 1.5배 미달이면 추세 추종 금지\n"
    "   - 박스권 SL/TP: 피보나치 X, 직전 고점/저점 기준 O\n"
    "\n"
    "확신도 산정 원칙 (이 시나리오대로 진행될 확률 = 승률 추정):\n"
    "  90~100: 모든 TF/지표/거시 정합 + 명백한 구조적 트리거 발동\n"
    "  75~89:  주요 TF 정합 + 분명한 우위, 일부 약한 신호 존재\n"
    "  60~74:  방향성 보이나 한두 개 반대 신호, 진입 시 작은 사이즈\n"
    "  50~59:  근거 약함, 관망 권장 (자동매매에서 진입 차단됨)\n"
    "  50미만: 방향성 부재 또는 반대 신호 우세 → 홀드\n"
    "★ 50~95 전 구간을 활용하세요. 모든 분석을 65~75 구간에 몰아넣는 것은 자기검열 오류.\n"
    "★ 강한 정합 신호일 때 80~90을 자신 있게 출력하세요. 약한 근거에 65를 쓰지 마세요.\n"
)

USER_PROMPT_TEMPLATE = """분석 기준 시각: {now_kst} (KST)
⚠️ 각 타임프레임의 마지막 캔들은 현재 형성 중인 미완성봉입니다. 확정된 신호로 해석하지 마세요.

{debate_block_separator}{debate_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[현재 시장 데이터] {pair_label}

{context_blob}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[분석 지침]
당신은 위 토론(과거 체크리스트 + Bull/Bear + Judge + Risk)을 바탕으로 최종 결론을 내리는 종합 애널리스트입니다.
⚑ 과거 체크리스트가 있다면 현재 데이터로 각 항목을 먼저 확인하고 본문에 반영하세요.
⚠ 반복 실수 금지 항목이 있다면 이번 판단에서 해당 패턴을 피했는지 명시하세요.
- Judge 판정과 다른 결론을 내려도 됩니다 (그 경우 이유를 본문에 명시).
- Risk 3자 권고(공격/보수/중도) 중 어느 입장에 가까운지 본문에서 명확히 하세요.
- 지금 {pair_label}의 시장 관점을 애널리스트처럼 정리해주세요.
- 근거가 충분하면 명확한 방향성을, 근거가 약하면 정직하게 중립을 제시하세요.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
응답 작성 시 절대 준수 (위반 시 응답 무효):
1. 구체 수치(가격·RSI·펀딩비·OI·ATR 등)는 입력 데이터에 있는 값만 인용. 환각 절대 금지.
2. 매매 파라미터(진입가/손절가/목표가)는 단일 숫자 하나만. 범위·N/A·조건부 표기 금지.
3. 목표가는 R:R 1.5 이상을 만족하도록 설정 (|목표가-진입가| ≥ 1.5 × |진입가-손절가|).
4. 손절가/목표가는 [ATR & 구조적 매매 레벨] 섹션의 SL/TP 후보를 우선 사용하세요.
   구조적 레벨이 R:R 1.5 미달 시 ATR × 1.5~2.0으로 TP를 확장하세요.
   SL은 ATR × 1.0 미만 절대 금지.

이제 아래 형식으로 응답하세요. 제목과 순서를 바꾸지 마세요:

📊 관점: [상방 우위 / 하방 우위 / 중립]
💯 확신도: [숫자]%  ← 이 시나리오대로 진행될 확률(승률 추정). 50~95 전 구간 활용. 강한 정합 시 80~90 자신 있게.
🧭 시장 레짐: [주 레짐] + [보조 레짐]
   ← 주: 상승 추세/하락 추세/박스/이벤트 대기 중 (4개 중 1개)
   ← 보조: 변동성 확장/변동성 축소 (2개 중 1개)
   ← 예: "상승 추세 + 변동성 확장" / "박스 + 변동성 축소"

🤖 매매 파라미터  ← 자동매매가 직접 파싱. 아래 4줄 형식 절대 엄수. 누락·변형 금지.
• 진입가: $[숫자 하나]   ← 반드시 이 레이블과 $ 형식 유지. 조건문·텍스트 혼합 금지.
  ← 의도에 맞춰 가격 설정:
    - 즉시 진입 의도: 현재가 그대로
    - 되돌림 대기: 매수면 현재가보다 낮게, 매도면 현재가보다 높게
    - 돌파 후 진입: 돌파 목표가 + 0.1% (예: 돌파 목표 $82,829 → 진입가: $82,912)
• 손절가: $[숫자 하나]   ← [ATR & 구조적 매매 레벨]의 SL 후보 우선 사용. ATR × 1.0 미만 금지.
• 목표가: $[숫자 하나]   ← [ATR & 구조적 매매 레벨]의 TP 후보 우선 사용. R:R 1.5 이상 필수.
• 권장 레버리지: [숫자]배 (1~10)

📌 먼저 보이는 사실
• [확정된 사실 — 입력 데이터에서 직접 인용]
• [확정된 사실 — 입력 데이터에서 직접 인용]

🧠 해석
• [왜 이런 관점이 나오는지]
• [멀티 타임프레임 / 파생 / 거시 연결]

🔄 반대 시나리오
• [내 관점과 반대되는 해석]
• [무엇이 나오면 반대 시나리오가 우세해지는지]

📍 관심 레벨
• 1차 저항: $[숫자 또는 N/A]
• 1차 지지: $[숫자 또는 N/A]
• 상방 돌파 트리거: $[숫자 또는 N/A]
• 하방 이탈 트리거: $[숫자 또는 N/A]

📝 대응{position_management_note}
• 공격적: [지금 즉시 취할 구체적 행동 — 진입 방향·레벨·조건]
• 보수적: [관망이 필요하다면 그 조건, 불필요한 관망은 쓰지 마세요]

⚠️ 관점이 약해지는 조건: [1줄]

💬 한줄 요약: [텔레그램 알림용 — 50자 이내. 방향성 + 핵심 트리거 + 액션]
   예: "BTC 71500 돌파 시 매수, SL 70400, R:R 2.5"

기타 형식 요구사항:
- [시장 레짐]은 "주 + 보조" 형식 준수. 예: "박스 + 변동성 축소".
- [관심 레벨]의 4개 항목은 각 줄마다 숫자 또는 N/A만. 이유·조건·괄호 설명 금지.
- 2차 저항/지지 같은 추가 항목을 만들지 마세요.
- [권장 레버리지]는 반드시 정수(예: 5배)로만. 범위·슬래시 표기 금지.
"""


def _tf_alignment_summary(multi_tf_data: dict) -> str:
    """타임프레임 간 추세 정렬 상태를 한눈에 비교 (세부 지표는 아래 각 TF 섹션 참조)"""
    lines = [
        "[타임프레임 추세 정렬 스냅샷]",
        "  형식: 가격 | SMA200 대비(▲상위/▼하위)  ← 방향 요약만. SMA200 절대값은 아래 각 TF 섹션 [지표 현재값] 참조",
        "  ※ 세부 지표(RSI·MACD·거래량)도 아래 각 TF 섹션 참조",
    ]
    tf_order = ["1d", "4h", "1h", "15m", "5m"]
    ordered = {tf: multi_tf_data[tf] for tf in tf_order if tf in multi_tf_data}

    for tf, df in ordered.items():
        last   = df.iloc[-1]
        price  = last["close"]
        sma200 = last["sma_200"]
        trend  = "▲" if price > sma200 else "▼"
        lines.append(
            f"  {tf:>3s}: ${price:,.0f} | SMA200 {trend}${sma200:,.0f}"
        )

    return "\n".join(lines)


def _build_context_blob(
    multi_tf_data: dict,
    macro_snapshot: Optional[dict] = None,
    return_raw: bool = False,
):
    """
    Bull/Bear/최종 애널리스트가 공통으로 보는 데이터 블록을 조립한다.
    (기존 build_prompt 의 데이터 수집부를 추출 — debate 에도 재사용하기 위함)

    Parameters
    ----------
    return_raw : bool
        True 이면 (context_blob, raw_ctx_dict) 튜플을 반환.
        raw_ctx_dict 는 situation digest 생성 등 정규화 태그용.
    """
    # 타임프레임 정렬 요약
    tf_alignment = _tf_alignment_summary(multi_tf_data)

    # 거시경제 지표 수집
    macro_context_str = "[거시경제 지표]\n  데이터 수집 실패"
    macro_payload = macro_snapshot
    if macro_payload is None:
        try:
            macro_payload = fetch_macro_context()
        except Exception as exc:
            macro_context_str = f"[거시경제 지표]\n  데이터 수집 실패 — {exc}"
    if macro_payload is not None:
        try:
            macro_context_str = format_macro_context(macro_payload)
        except Exception as exc:
            macro_context_str = f"[거시경제 지표]\n  데이터 가공 실패 — {exc}"

    # 시장 데이터 수집
    market_context_str  = "[시장 심리 & 파생상품 데이터]\n  데이터 수집 실패 — 기술적 지표만으로 판단"
    market_ctx: Optional[dict] = None
    try:
        market_ctx = fetch_market_context()
        market_context_str = format_market_context(market_ctx)
    except Exception as exc:
        market_context_str = f"[시장 심리 & 파생상품 데이터]\n  데이터 수집 실패 — {exc}"

    # 계좌 / 리스크 제약 수집
    account_context_str = "[계좌 / 리스크 제약]\n  데이터 수집 실패 — 계좌 제약 없이 시장 데이터만으로 판단"
    account_ctx: Optional[dict] = None
    try:
        account_ctx = fetch_account_context()
        account_context_str = format_account_context(account_ctx)
    except Exception as exc:
        account_context_str = f"[계좌 / 리스크 제약]\n  데이터 수집 실패 — {exc}"

    # 비트겟 설정 + 포지션 + 리스크 제약 (단일 호출로 통합)
    try:
        from bitget_trader import BitgetAutoTrader as _BAT
        # config 경유 — \r \n 등 정제됨, env 직접 읽기 시 서명 실패 위험
        from config import (
            BITGET_API_KEY as _api_key,
            BITGET_SECRET_KEY as _secret_key,
            BITGET_PASSPHRASE as _passphrase,
            AUTO_TRADE_LEVERAGE as _lev,
            AUTO_TRADE_USDT as _alloc_pct,
        )

        if _api_key and _secret_key and _passphrase:
            _bt        = _BAT(_api_key, _secret_key, _passphrase)
            _acct      = _bt.get_account()
            _positions = _bt.get_positions()
            _equity    = float(_acct.get("equity", 0) or 0)
            _avail     = float(_acct.get("available", 0) or 0)
            _today_pnl = float(_acct.get("todayProfitLoss", 0) or 0)

            # 배분 계산 (1~100이면 비율, 그 외면 고정 USDT)
            if 1 <= _alloc_pct <= 100:
                _trade_margin = _equity * (_alloc_pct / 100) * 0.95
                _alloc_label  = f"잔고의 {_alloc_pct:.0f}% = ${_trade_margin:,.2f} 증거금"
            else:
                _trade_margin = _alloc_pct
                _alloc_label  = f"고정 ${_trade_margin:,.2f} 증거금"

            _pos_size = _trade_margin * _lev

            bitget_ctx = (
                f"\n[비트겟 자동매매 설정]"
                f"\n  총 잔고: ${_equity:,.2f} USDT  |  가용: ${_avail:,.2f} USDT"
                f"\n  오늘 실현손익: ${_today_pnl:+,.2f} USDT"
                f"\n  이번 거래 배분: {_alloc_label}"
                f"\n  ⚠️ 현재 설정 레버리지: {_lev}배 (권장 레버리지는 이 값 기준으로 제시)"
                f"\n  예상 포지션 규모: ${_pos_size:,.2f} USDT"
            )

            if _positions:
                bitget_ctx += "\n[현재 오픈 포지션]"
                for p in _positions:
                    _side     = p.get("holdSide", "").upper()
                    _qty      = p.get("total", 0)
                    _entry    = float(p.get("averageOpenPrice", 0) or 0)
                    _upnl     = float(p.get("unrealizedPL", 0) or 0)
                    _upnl_r   = float(p.get("unrealizedPLR", 0) or 0)
                    _p_lev    = p.get("leverage", _lev)
                    bitget_ctx += (
                        f"\n  {_side} {_qty}계약  진입가 ${_entry:,.2f}"
                        f"  미실현 ${_upnl:+,.2f} ({_upnl_r*100:+.2f}%)  레버리지 {_p_lev}x"
                    )
                # 포지션 있으면 리스크 경고 + 신규 진입 금지 명시
                _total_upnl = sum(float(p.get("unrealizedPL", 0) or 0) for p in _positions)
                if _equity > 0:
                    _risk_pct = abs(_total_upnl) / _equity * 100
                    bitget_ctx += f"\n  ⚠️ 현재 미실현 리스크: 잔고의 {_risk_pct:.1f}%"
                bitget_ctx += "\n  🚫 포지션 보유 중 — 신규 진입 금지. 기존 포지션 관리(SL/TP 조정·청산)만 분석하세요."
            else:
                bitget_ctx += "\n  현재 포지션: 없음 (신규 진입 가능)"

            account_context_str += bitget_ctx
    except Exception as _bg_exc:
        account_context_str += f"\n[비트겟 데이터]\n  수집 실패 — {_bg_exc}"

    # SL/TP 후보 계산 (데이트레이딩 기준)
    # SL = 피보나치 스윙 저점/고점 (구조적 근거 우선)
    #      ATR×0.5 = 최소 거리 하한 (노이즈 손절 방지용)
    # TP = 피보나치 연장선 (FibStats 도달률 우선순위)
    # ATR = 저변동성 진입 필터용으로만 활용
    _chosen_fib = {"long": None, "short": None}   # try 밖에서 미리 초기화 — 예외 시에도 안전
    _r1h   = None   # try 밖 초기화 — 예외 시 None으로 안전 폴백
    _r4h   = None
    _base_r = None
    try:
        MIN_RR = 1.5
        ATR_MIN_FLOOR = 0.5   # SL 최소 거리: ATR×0.5 하한

        # ── FibStats 로드 (TP 연장선 우선순위) ──────────────────────────
        try:
            from agents.fib_stats import get_fib_stats as _gfs
            _fib_stats       = _gfs()
            _long_ext_order  = _fib_stats.preferred_extensions("long")
            _short_ext_order = _fib_stats.preferred_extensions("short")
            _fib_stats_note  = _fib_stats.log_summary()
        except Exception:
            _fib_stats       = None
            _long_ext_order  = (1.272, 1.618, 2.0)
            _short_ext_order = (1.272, 1.618, 2.0)
            _fib_stats_note  = ""

        def _calc_tpsl_block(df, tf_label):
            """
            SL: 스윙 저점/고점 기준 (피보나치 구조)
                ATR×0.5 보다 가까우면 ATR 하한 적용
            TP: FibStats 도달률 우선순위 + R:R 1.5 보장
            """
            atr_col = df["atr"] if "atr" in df.columns else None
            if atr_col is None or len(atr_col) == 0:
                return None
            atr_v = float(atr_col.iloc[-1])
            cur   = float(df.iloc[-1]["close"])
            if atr_v <= 0:
                return None

            # 저변동성 감지 (ATR < 현재가×0.2%) → 진입 경고 플래그
            _atr_pct = atr_v / cur * 100
            _low_vol = _atr_pct < 0.2

            sw = fibonacci_swing_levels(df, window=fib_window_for_tf(tf_label))
            if not sw:
                return {"atr": atr_v, "cur": cur, "sw": None, "low_vol": _low_vol}

            sw_low  = sw["swing_low"]
            sw_high = sw["swing_high"]
            sw_diff = sw_high - sw_low

            # SL: 스윙 저점/고점 기준, ATR×0.5 최소 거리 보장
            _sl_long_swing  = round(sw_low  * 0.999, 1)
            _sl_short_swing = round(sw_high * 1.001, 1)
            _atr_floor_long  = round(cur - atr_v * ATR_MIN_FLOOR, 1)
            _atr_floor_short = round(cur + atr_v * ATR_MIN_FLOOR, 1)
            sl_long  = min(_sl_long_swing,  _atr_floor_long)
            sl_short = max(_sl_short_swing, _atr_floor_short)
            sl_long_dist  = cur - sl_long
            sl_short_dist = sl_short - cur

            # TP: FibStats 도달률 우선 + R:R 1.5 보장
            def pick_tp_long(sl_d):
                for ext in _long_ext_order:
                    tp = round(sw_low + sw_diff * ext, 1)
                    if sl_d > 0 and (tp - cur) / sl_d >= MIN_RR:
                        return tp, ext
                return round(cur + sl_d * 2.5, 1), None

            def pick_tp_short(sl_d):
                for ext in _short_ext_order:
                    tp = round(sw_high - sw_diff * ext, 1)
                    if sl_d > 0 and (cur - tp) / sl_d >= MIN_RR:
                        return tp, ext
                return round(cur - sl_d * 2.5, 1), None

            tp_long,  ext_long  = pick_tp_long(sl_long_dist)
            tp_short, ext_short = pick_tp_short(sl_short_dist)
            rr_long  = round((tp_long  - cur) / sl_long_dist,  2) if sl_long_dist  > 0 else 0
            rr_short = round((cur - tp_short) / sl_short_dist, 2) if sl_short_dist > 0 else 0

            return {
                "atr": atr_v, "cur": cur, "sw": sw, "low_vol": _low_vol,
                "sw_low": sw_low, "sw_high": sw_high,
                "sl_long": sl_long,  "sl_short": sl_short,
                "sl_long_swing": _sl_long_swing, "sl_short_swing": _sl_short_swing,
                "tp_long": tp_long,  "ext_long": ext_long,  "rr_long": rr_long,
                "tp_short": tp_short, "ext_short": ext_short, "rr_short": rr_short,
            }

        # ── 1h: 주 기준 (데이트레이딩) ──────────────────────────────────
        _r1h = None
        if "1h" in multi_tf_data:
            try:
                _r1h = _calc_tpsl_block(multi_tf_data["1h"], "1h")
            except Exception:
                pass

        # ── 4h: 상위 구조 경고 참고용 ───────────────────────────────────
        _r4h = None
        if "4h" in multi_tf_data:
            try:

                _r4h = _calc_tpsl_block(multi_tf_data["4h"], "4h")
            except Exception:
                pass

        # 선택된 fib_ext를 외부로 노출 (analyze_with_claude 반환값에 포함)
        _chosen_fib = {"long": None, "short": None}
        _base_r = _r1h if (_r1h and _r1h.get("sw")) else (_r4h if (_r4h and _r4h.get("sw")) else None)
        if _base_r:
            _chosen_fib["long"]  = _base_r.get("ext_long")
            _chosen_fib["short"] = _base_r.get("ext_short")

        # ── 프롬프트 조립 ────────────────────────────────────────────────
        def _fmt_tpsl_block(r, tf_label):
            sw = r["sw"]
            ext_l = f"Fib {r['ext_long']}"  if r["ext_long"]  else "SL×2.5 폴백"
            ext_s = f"Fib {r['ext_short']}" if r["ext_short"] else "SL×2.5 폴백"
            vol_warn = "\n  ⚠️ 저변동성 구간 — 진입 보류 권고 (ATR < 0.2%)" if r.get("low_vol") else ""
            # SL 근거 표시: 스윙 기준인지 ATR 하한인지
            _sl_l_src = "스윙저점" if r["sl_long"]  == r.get("sl_long_swing")  else f"ATR×{ATR_MIN_FLOOR} 하한"
            _sl_s_src = "스윙고점" if r["sl_short"] == r.get("sl_short_swing") else f"ATR×{ATR_MIN_FLOOR} 하한"
            return (
                f"\n[구조적 매매 레벨] ← {tf_label} 스윙 기준{vol_warn}"
                f"\n  {tf_label} ATR: ${r['atr']:,.2f}"
                f"\n  스윙 저점: ${r['sw_low']:,.1f} ({sw['swing_low_ago']}봉 전)"
                f"  /  스윙 고점: ${r['sw_high']:,.1f} ({sw['swing_high_ago']}봉 전)"
                f"\n  ─ 롱 기준 ─"
                f"\n    SL: ${r['sl_long']:,.1f} ({_sl_l_src})  TP: ${r['tp_long']:,.1f} ({ext_l})  R:R {r['rr_long']:.2f}"
                f"\n  ─ 숏 기준 ─"
                f"\n    SL: ${r['sl_short']:,.1f} ({_sl_s_src})  TP: ${r['tp_short']:,.1f} ({ext_s})  R:R {r['rr_short']:.2f}"
                f"\n  ※ SL = 스윙 저점/고점 기준 (ATR×{ATR_MIN_FLOOR} 최소 거리 보장)"
                f"  TP = R:R {MIN_RR} 이상 피보나치 연장선 자동 선택."
            )

        if _r1h and _r1h.get("sw"):
            account_context_str += _fmt_tpsl_block(_r1h, "1h")
            if _r4h and _r4h.get("sw"):
                r4 = _r4h
                account_context_str += (
                    f"\n  [4h 구조 참고] 스윙저점 ${r4['sw_low']:,.1f} / 스윙고점 ${r4['sw_high']:,.1f}"
                    f" — 이 레벨 인근에서 1h 신호와 충돌 시 진입 재고."
                )
        elif _r4h and _r4h.get("sw"):
            account_context_str += _fmt_tpsl_block(_r4h, "4h")
        elif _r1h:
            # 스윙 없음, ATR만
            account_context_str += (
                f"\n[ATR]\n  1h ATR: ${_r1h['atr']:,.2f} | "
                f"SL권고(1.0배): ${_r1h['atr']*1.0:,.2f} | "
                f"SL권고(1.5배): ${_r1h['atr']*1.5:,.2f}"
            )
        elif _r4h:
            account_context_str += (
                f"\n[ATR]\n  4h ATR: ${_r4h['atr']:,.2f} | "
                f"SL권고(1.0배): ${_r4h['atr']*1.0:,.2f} | "
                f"SL권고(1.5배): ${_r4h['atr']*1.5:,.2f}"
            )
    except Exception:
        pass

# 각 타임프레임 상세 지표
    tf_order = ["1d", "4h", "1h", "15m", "5m"]
    ordered  = {tf: multi_tf_data[tf] for tf in tf_order if tf in multi_tf_data}
    parts    = [summarize_indicators(tf, df) for tf, df in ordered.items()]

    def fib_format(df, result, overlap_note: str = "") -> str:
        if result is None:
            return "유효한 스윙 포인트를 찾을 수 없음"

        current_price = df.iloc[-1]["close"]
        sw_low  = result["swing_low"]
        sw_high = result["swing_high"]
        if result["direction"] == "up":
            header = (
                f"상승 스윙: 저점 ${sw_low:,.0f} ({result['swing_low_ago']}봉 전) → "
                f"고점 ${sw_high:,.0f} ({result['swing_high_ago']}봉 전)"
            )
        else:
            header = (
                f"하락 스윙: 고점 ${sw_high:,.0f} ({result['swing_high_ago']}봉 전) → "
                f"저점 ${sw_low:,.0f} ({result['swing_low_ago']}봉 전)"
            )
        if overlap_note:
            header += f"\n  {overlap_note}"

        # 현재가가 피보 범위 밖에 있으면 레벨이 지지/저항으로 기능하지 않음
        if current_price < sw_low:
            return (
                f"{header}\n"
                f"  ⚠️ 현재가(${current_price:,.0f})가 스윙저점 아래 — 해당 레벨 유효하지 않음"
            )
        if current_price > sw_high:
            return (
                f"{header}\n"
                f"  ⚠️ 현재가(${current_price:,.0f})가 스윙고점 위 — 해당 레벨 유효하지 않음"
            )

        levels_str = "  ".join(f"Fib{r}=${p:,.0f}" for r, p in result["levels"].items())
        return f"{header}\n  {levels_str}"

    # 스윙 계산 (1h·4h 각각 한 번만 호출)
    _res_1h = fibonacci_swing_levels(multi_tf_data["1h"], window=fib_window_for_tf("1h")) if "1h" in multi_tf_data else None
    _res_4h = fibonacci_swing_levels(multi_tf_data["4h"], window=fib_window_for_tf("4h")) if "4h" in multi_tf_data else None

    # 1h·4h 스윙 구간 중복 탐지 — 두 TF가 동일 스윙을 잡으면 독립 확인 아님
    def _pct_close(a, b, tol=1.0):
        return abs(a - b) / max(abs(b), 1e-9) * 100 < tol

    _overlap_note = ""
    if _res_1h and _res_4h:
        if (_pct_close(_res_1h["swing_low"],  _res_4h["swing_low"])
                and _pct_close(_res_1h["swing_high"], _res_4h["swing_high"])):
            _overlap_note = "※ 1h·4h 동일 스윙 구간 탐지 — 두 레벨은 독립 확인 아님"

    fib_1h = fib_format(multi_tf_data["1h"], _res_1h, _overlap_note) if "1h" in multi_tf_data else "N/A"
    fib_4h = fib_format(multi_tf_data["4h"], _res_4h, _overlap_note) if "4h" in multi_tf_data else "N/A"

    # 최종 블록 조립 (기존 USER_PROMPT_TEMPLATE 의 내부 데이터 섹션과 동일 순서)
    indicators_summary = "\n\n".join(parts)
    context_blob = (
        f"{tf_alignment}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{macro_context_str}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{market_context_str}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{account_context_str}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"[피보나치 레벨]\n1h 기준: {fib_1h}\n4h 기준: {fib_4h}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{indicators_summary}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f""
    )
    if return_raw:
        return context_blob, {
            "macro": macro_payload,
            "market": market_ctx,
            "account": account_ctx,
            "chosen_fib": _chosen_fib,
            "tpsl_levels": _base_r,   # 코드 계산 SL/TP — AI 출력 덮어쓰기용
        }
    return context_blob


def build_prompt(
    multi_tf_data: dict,
    macro_snapshot: Optional[dict] = None,
    debate_block: str = "",
) -> str:
    """
    최종 애널리스트 Claude 호출용 user prompt 를 조립한다.

    Parameters
    ----------
    multi_tf_data : dict
        {tf: DataFrame} 형태의 멀티 TF 인디케이터 데이터.
    macro_snapshot : dict, optional
        미리 수집된 거시 스냅샷. None 이면 fetch_macro_context() 수행.
    debate_block : str, optional
        agents.format_debate_block() 의 출력. 빈 문자열이면 토론 섹션 생략.
    """
    context_blob = _build_context_blob(multi_tf_data, macro_snapshot)
    now_kst_label = now_kst().strftime("%Y-%m-%d %H:%M")

    # debate_block 앞에 ━━━ 구분선을 붙여 시각적 경계를 만든다. 비어 있으면 통째로 생략.
    if debate_block:
        debate_separator = "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    else:
        debate_separator = ""

    # 포지션 보유 중이면 [대응] 섹션에 신규 진입 금지 명시
    _has_open_position = "신규 진입 금지" in context_blob
    if _has_open_position:
        _pos_note = (
            " ← 🚫 포지션 보유 중. 추가 진입 조언 금지.\n"
            "  공격적/보수적 모두 현재 포지션 관리(SL이동·TP조정·청산조건)만 작성하세요."
        )
    else:
        _pos_note = ""

    return USER_PROMPT_TEMPLATE.format(
        now_kst=now_kst_label,
        pair_label=PAIR_LABEL,
        context_blob=context_blob,
        debate_block_separator=debate_separator,
        debate_block=debate_block,
        position_management_note=_pos_note,
    )


VIEW_TO_SIGNAL = {
    "상방 우위": "매수",
    "하방 우위": "매도",
    "중립": "홀드",
}

REPORT_SECTION_LABELS = {
    "view": "관점",
    "regime": "시장 레짐",
    "facts": "먼저 보이는 사실",
    "interpretation": "해석",
    "counter_scenario": "반대 시나리오",
    "response": "대응",
    "invalidation": "관점이 약해지는 조건",
    "summary": "한줄 요약",
}


def parse_report_sections(text: str) -> dict:
    """애널리스트 리포트의 핵심 섹션을 구조적으로 파싱."""
    sections = {
        "view": None,
        "regime": None,
        "facts": [],
        "interpretation": [],
        "counter_scenario": [],
        "response": [],
        "invalidation": None,
        "summary": None,
    }

    current_block = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m = re.match(r'📊\s*관점\s*[:：]\s*(.+)$', line)
        if m:
            sections["view"] = m.group(1).strip()
            current_block = None
            continue

        if re.match(r'💯\s*(?:확신도|신뢰도)\s*[:：]', line):
            current_block = None
            continue

        m = re.match(r'🧭\s*시장\s*레짐\s*[:：]\s*(.+)$', line)
        if m:
            sections["regime"] = m.group(1).strip()
            current_block = None
            continue

        if re.match(r'📌\s*먼저\s*보이는\s*사실', line):
            current_block = "facts"
            continue

        if re.match(r'🧠\s*해석', line):
            current_block = "interpretation"
            continue

        if re.match(r'🔄\s*반대\s*시나리오', line):
            current_block = "counter_scenario"
            continue

        if re.match(r'📍\s*관심\s*레벨', line):
            current_block = None
            continue

        if re.match(r'📝\s*대응', line):
            current_block = "response"
            continue

        m = re.match(r'⚠️\s*관점이\s*약해지는\s*조건\s*[:：]\s*(.+)$', line)
        if m:
            sections["invalidation"] = m.group(1).strip()
            current_block = None
            continue

        m = re.match(r'💬\s*한줄\s*요약\s*[:：]\s*(.+)$', line)
        if m:
            sections["summary"] = m.group(1).strip()
            current_block = None
            continue

        if current_block in ("facts", "interpretation", "counter_scenario", "response"):
            item = re.sub(r'^[•\-]\s*', '', line).strip()
            if item:
                sections[current_block].append(item)

    required_keys = (
        "view",
        "regime",
        "facts",
        "interpretation",
        "counter_scenario",
        "response",
        "invalidation",
        "summary",
    )
    missing_sections = []
    for key in required_keys:
        value = sections[key]
        if isinstance(value, list):
            if not value:
                missing_sections.append(REPORT_SECTION_LABELS[key])
        elif not value:
            missing_sections.append(REPORT_SECTION_LABELS[key])

    return {
        "sections": sections,
        "missing_sections": missing_sections,
        "format_ok": not missing_sections,
    }


def parse_signal(text: str) -> tuple[str, int]:
    # ── 관점/시그널 파싱: 새 포맷(관점) 우선, 구 포맷(시그널) 폴백 ──
    # 정규식을 라인 시작에만 매칭하도록 강화 — 본문에 '관점' 단어가 평문으로
    # 들어와도 잘못 매칭되지 않도록.
    signal = "홀드"
    sig_match = re.search(
        r'(?:^|\n)\s*(?:📊\s*)?(?:관점|시그널)\s*[:：]\s*'
        r'(상방 우위|하방 우위|중립|매수|매도|홀드)',
        text,
    )
    if sig_match:
        raw_signal = sig_match.group(1)
        signal = VIEW_TO_SIGNAL.get(raw_signal, raw_signal)
    else:
        front = text[:300]
        keys = ("상방 우위", "하방 우위", "중립", "매수", "매도", "홀드")
        positions = {kw: front.find(kw) for kw in keys if kw in front}
        if positions:
            raw_signal = min(positions, key=positions.get)
            signal = VIEW_TO_SIGNAL.get(raw_signal, raw_signal)

    # ── 확신도/신뢰도 파싱 ──
    # [버그 수정] 기존 [^:\n] 패턴은 콜론을 제외해 "신뢰도: 72%" 형식에서 매칭 실패
    # \D*? 로 변경 — 숫자가 아닌 모든 문자(콜론·공백 포함)를 lazily 건너뜀
    confidence = 50
    conf_match = re.search(r'(?:확신도|신뢰도)\D*?(\d{1,3})', text)
    if conf_match:
        confidence = min(int(conf_match.group(1)), 100)

    return signal, confidence


def parse_leverage(text: str) -> Optional[int]:
    """
    Claude 분석 텍스트에서 권장 레버리지를 파싱.
    '권장 레버리지' 필드 우선, 없으면 자유 텍스트에서 탐색.
    반환: 1~20 범위 정수 or None
    """
    # 구조화 필드 우선 (매매 파라미터 섹션)
    m = re.search(r'권장\s*레버리지\s*[:：]\s*(\d+)\s*배', text)
    if m:
        val = int(m.group(1))
        if 1 <= val <= 20:
            return val

    # 자유 텍스트 폴백 패턴들
    patterns = [
        r'레버리지\s*[:：]\s*(\d+)\s*배',
        r'(\d+)\s*배\s*레버리지',
        r'leverage\s*[:：]?\s*(\d+)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 20:
                return val
    return None


def parse_trade_levels(text: str) -> dict:
    """관심 레벨 파싱. 구 포맷(진입/손절/목표/손익비)도 폴백 지원."""
    def _price_from_line(label: str):
        m = re.search(rf'^\s*[•\-]?\s*{label}\s*[:：]\s*(.+)$', text, re.MULTILINE)
        if not m:
            return None

        value_text = m.group(1).strip()
        if re.match(r'^N/?A\b', value_text, re.IGNORECASE):
            return None

        dollar_match = re.search(r'\$([\d,]+(?:\.\d+)?)', value_text)
        if dollar_match:
            val = dollar_match.group(1).replace(',', '').strip()
            try:
                return float(val)
            except ValueError:
                return None

        # 범위 표기 폴백: "79,400~79,500" or "$79,400~79,500" → 낮은 값 사용
        range_match = re.search(r'\$?([\d,]+(?:\.\d+)?)\s*[~–\-]\s*\$?([\d,]+(?:\.\d+)?)', value_text)
        if range_match:
            try:
                v1 = float(range_match.group(1).replace(",", ""))
                v2 = float(range_match.group(2).replace(",", ""))
                return min(v1, v2)
            except ValueError:
                pass

        numeric_only_match = re.fullmatch(r'([\d,]+(?:\.\d+)?)', value_text)
        if not numeric_only_match:
            return None

        val = numeric_only_match.group(1).replace(',', '').strip()
        try:
            return float(val)
        except ValueError:
            return None

    resistance   = _price_from_line(r'1차\s*저항')
    support      = _price_from_line(r'1차\s*지지')
    bull_trigger = _price_from_line(r'상방\s*돌파\s*트리거')
    bear_trigger = _price_from_line(r'하방\s*이탈\s*트리거')

    entry  = _price_from_line(r'진입가')
    stop   = _price_from_line(r'손절가')
    target = _price_from_line(r'목표가')

    rr = None
    rr_m = re.search(r'손익비\s*[:：]\s*([\d.]+)\s*[:：]\s*1', text)
    if rr_m:
        try:
            rr = float(rr_m.group(1))
        except ValueError:
            pass

    return {
        "resistance": resistance if resistance is not None else target,
        "support": support if support is not None else stop,
        "bull_trigger": bull_trigger if bull_trigger is not None else entry,
        "bear_trigger": bear_trigger,
        "entry": entry,
        "stop": stop,
        "target": target,
        "rr": rr,
    }


def analyze_with_claude(
    multi_tf_data: dict,
    macro_snapshot: Optional[dict] = None,
    debate: Optional[DebateResult] = None,
    pipeline: Optional[PipelineResult] = None,
    raw_ctx: Optional[dict] = None,   # _build_context_blob 결과 — TPSL override용
) -> dict:
    """
    최종 애널리스트 Claude 호출.

    토론/리스크/메모리 컨텍스트 주입 우선순위:
      1) pipeline (Phase 2+3 통합 블록, combined_block)
      2) debate   (Phase 1 단독 블록, 하위 호환)
      3) 없음
    """
    # 분석 1회당 여러 에이전트가 같은 클라이언트를 공유 — connection pool 재사용.
    from agents import get_anthropic_client
    client = get_anthropic_client()

    if pipeline is not None and pipeline.combined_block:
        debate_block = pipeline.combined_block
    elif debate is not None:
        debate_block = format_debate_block(debate)
    else:
        debate_block = ""

    prompt = build_prompt(
        multi_tf_data,
        macro_snapshot=macro_snapshot,
        debate_block=debate_block,
    )
    # 실제 출력 구조: 약 10개 섹션 × 3~5줄 ≈ 600~1000 tokens.
    # 12000은 과도하며 디버깅용 대형 마진. ANALYST_MAX_TOKENS으로 조절 가능.
    # 기본값 4000: 충분한 여유 + 비용·속도 개선.
    _analyst_max_tokens = int(_os.getenv("ANALYST_MAX_TOKENS", "4000"))
    # SYSTEM_PROMPT 는 매 호출마다 동일하므로 prompt caching (ephemeral, ~5분 TTL).
    # 4시간 주기 + debate→judge→risk→final 짧은 시간 안에 여러 번 호출되므로
    # 캐시 hit 시 입력 비용 90% 절감. 첫 호출은 25% 추가 비용이지만 곧 회수됨.
    request_kwargs = {
        "model": CLAUDE_MODEL,
        "max_tokens": _analyst_max_tokens,
        "system": [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [{"role": "user", "content": prompt}],
    }

    # thinking 완전 비활성화 — 최종 분석은 이미 debate/judge/risk 블록이 reasoning을 제공하므로
    # adaptive thinking은 수만 토큰을 소모해 비용을 크게 높임. 구조화 출력에는 불필요.

    # 529/429 과부하 대비 지수 백오프 재시도 (최대 4회: 10s → 20s → 40s → 80s)
    max_retries = 4
    wait = 10
    message = None
    for attempt in range(max_retries):
        try:
            message = client.messages.create(**request_kwargs)
            break

        except anthropic.APIStatusError as e:
            if e.status_code in (429, 529) and attempt < max_retries - 1:
                time.sleep(wait)
                wait *= 2  # 10 → 20 → 40 → 80초
            else:
                raise

    # 응답 타입 방어 검사 — SDK 버전이나 API 오류로 인해 예상 외 타입이 반환될 수 있음
    if message is None:
        raise RuntimeError("Anthropic API 응답 없음 (모든 재시도 소진)")
    if not hasattr(message, "content") or not isinstance(message.content, list):
        raise RuntimeError(
            f"Anthropic API 응답 형식 오류 — 타입: {type(message).__name__}, "
            f"content: {getattr(message, 'content', '(없음)')!r:.200}"
        )

    # 응답 블록에서 텍스트만 추출 (thinking 블록 제외)
    raw_text = next(
        (b.text for b in message.content if b.type == "text"), ""
    )
    signal, confidence = parse_signal(raw_text)
    trade_levels = parse_trade_levels(raw_text)

    report_meta = parse_report_sections(raw_text)
    claude_leverage = parse_leverage(raw_text)

    # Judge 결과 추출 (signal processing 에 활용)
    judge_result = (
        pipeline.judge if pipeline is not None else None
    )

    # Trading Signal 구조화 (extract_trading_signal 이 가용할 때만)
    trading_signal_dict = None
    if extract_trading_signal is not None:
        try:
            ts = extract_trading_signal(raw_text, judge_result=judge_result)
            trading_signal_dict = ts.to_dict()
        except Exception as _ts_exc:
            _memory_logger.warning("extract_trading_signal 실패 — %s", _ts_exc)

    # ── 코드 계산 SL/TP로 AI 출력값 강제 덮어쓰기 ──────────────────────
    # AI는 방향(매수/매도)과 진입가만 결정
    # SL = 1h 스윙 저점/고점 (구조적 근거), TP = 피보나치 연장선 (코드 계산)
    try:
        _sig_tmp, _ = parse_signal(raw_text)
        _tpsl_r = raw_ctx.get("tpsl_levels")   # _build_context_blob에서 넘어온 코드 계산값

        if _sig_tmp in ("매수", "매도") and _tpsl_r and _tpsl_r.get("sw"):
            _fib_dir_tmp = "long" if _sig_tmp == "매수" else "short"

            # SL: 스윙 저점/고점 (코드 계산) 강제 적용
            if _fib_dir_tmp == "long":
                _code_sl = _tpsl_r.get("sl_long")
                _code_tp = _tpsl_r.get("tp_long")
            else:
                _code_sl = _tpsl_r.get("sl_short")
                _code_tp = _tpsl_r.get("tp_short")

            # 진입가 파싱 — AI가 설정한 진입가 기준으로 SL/TP 방향 검증
            _entry_price = trade_levels.get("entry") or _tpsl_r.get("cur")

            if _code_sl is not None and _entry_price:
                # 진입가가 스윙 저점보다 낮은 경우 (박스 하단/지지선 터치 진입)
                # → SL이 진입가보다 위에 있어 무효
                # → ATR×1.0 사용 (0.5는 노이즈에 손절되므로 최소 1.0 필요)
                _atr_v = _tpsl_r.get("atr", 0)
                if _fib_dir_tmp == "long" and _code_sl >= _entry_price:
                    _code_sl = round(_entry_price - _atr_v * 1.0, 1)
                    _memory_logger.info("[TPSL override] 진입가<스윙저점(지지선 진입) → ATR×1.0 SL=$%.1f", _code_sl)
                elif _fib_dir_tmp == "short" and _code_sl <= _entry_price:
                    _code_sl = round(_entry_price + _atr_v * 1.0, 1)
                    _memory_logger.info("[TPSL override] 진입가>스윙고점(저항선 진입) → ATR×1.0 SL=$%.1f", _code_sl)

            if _code_sl is not None:
                _ai_sl = trade_levels.get("stop")
                trade_levels["stop"] = _code_sl
                _memory_logger.info(
                    "[TPSL override] SL: AI=$%s → 코드=$%s (스윙 저점 기준)",
                    _ai_sl, _code_sl
                )
            if _code_tp is not None:
                _ai_tp = trade_levels.get("target")
                trade_levels["target"] = _code_tp
                _memory_logger.info(
                    "[TPSL override] TP: AI=$%s → 코드=$%s (피보나치 연장선)",
                    _ai_tp, _code_tp
                )

            # FibStats 학습 메타 저장
            from agents.fib_stats import get_fib_stats as _gfs2
            _chosen = raw_ctx.get("chosen_fib", {})
            _actual_ext = _chosen.get(_fib_dir_tmp)
            trade_levels["fib_ext"]       = _actual_ext
            trade_levels["fib_direction"] = _fib_dir_tmp

        elif _sig_tmp in ("매수", "매도"):
            # 스윙 계산 실패 시 — AI 출력값 그대로 사용하되 로그 남김
            _memory_logger.warning(
                "[TPSL override] 스윙 계산 실패 → AI 출력 SL/TP 그대로 사용 (신뢰도 낮음)"
            )
            _fib_dir_tmp = "long" if _sig_tmp == "매수" else "short"
            trade_levels["fib_direction"] = _fib_dir_tmp

    except Exception as _ov_exc:
        _memory_logger.warning("[TPSL override] 실패: %s → AI 출력값 사용", _ov_exc)


    return {
        "signal":       signal,
        "confidence":   confidence,
        "raw_text":     raw_text,
        "trade_levels": trade_levels,
        "prompt_used":  prompt,
        "report_sections": report_meta["sections"],
        "report_format_ok": report_meta["format_ok"],
        "report_missing_sections": report_meta["missing_sections"],
        "trading_signal": trading_signal_dict,
        "claude_leverage": claude_leverage,
        "debate":       (
            pipeline.debate.to_payload() if pipeline is not None and pipeline.debate
            else (debate.to_payload() if debate is not None else None)
        ),
        "judge":        (
            pipeline.judge.to_payload() if pipeline is not None and pipeline.judge is not None
            else None
        ),
        "risk":         (
            pipeline.risk.to_payload() if pipeline is not None and pipeline.risk
            else None
        ),
        "memories":     (
            list(pipeline.memories) if pipeline is not None else []
        ),
    }


def run_full_analysis(
    multi_tf_data: dict,
    macro_snapshot: Optional[dict] = None,
    progress_cb=None,
) -> dict:
    """
    Bull/Bear 토론 + Risk Triad + 메모리 회상 + 최종 애널리스트 호출까지
    묶은 편의 함수. server.py 의 _run_job 에서 ThreadPoolExecutor 로 호출.

    Parameters
    ----------
    multi_tf_data : dict
        멀티 TF 캔들/지표 DataFrame.
    macro_snapshot : dict, optional
        이미 수집된 거시 스냅샷.
    progress_cb : callable, optional
        (phase, detail) -> None. 단계별 진행률 보고.
    """
    # 1) 공통 데이터 블록 + 원본 ctx 조립 (모든 에이전트가 이것을 본다)
    context_blob, raw_ctx = _build_context_blob(
        multi_tf_data, macro_snapshot, return_raw=True
    )

    # 1-a) BM25 매칭용 '정규화 태그' 생성 — 원본 blob 대신 이걸로 저장/검색
    situation_tags = ""
    if summarize_situation_tags is not None:
        try:
            situation_tags = summarize_situation_tags(
                multi_tf_data=multi_tf_data,
                macro_snapshot=raw_ctx.get("macro"),
                market_ctx=raw_ctx.get("market"),
                account_ctx=raw_ctx.get("account"),
            )
        except Exception as exc:
            _memory_logger.warning("situation_tags 생성 실패 — %s", exc)
            situation_tags = ""

    # 1-b) 분석 시점 현재가 추출 (reflection baseline)
    price_at_analysis: Optional[float] = None
    try:
        # 가장 짧은 TF 의 마지막 봉 close 를 분석 시점 현재가로 사용
        for tf in ("5m", "15m", "1h", "4h", "1d"):
            if tf in multi_tf_data and len(multi_tf_data[tf]) > 0:
                price_at_analysis = float(multi_tf_data[tf].iloc[-1]["close"])
                break
    except Exception as exc:
        _memory_logger.warning("price_at_analysis 추출 실패 — %s", exc)

    # 2) 메모리 객체 준비 (rank_bm25 미설치 시 None)
    memory_obj = None
    if get_memory is not None:
        try:
            memory_obj = get_memory("analyst")
        except Exception:
            memory_obj = None

    # 역할별 에이전트 메모리 (AgentMemories 싱글턴)
    agent_memories_obj = None
    if get_agent_memories is not None:
        try:
            agent_memories_obj = get_agent_memories()
            print(f"[AGENTMEM] OK: {agent_memories_obj}", flush=True)
        except Exception as exc:
            print(f"[AGENTMEM-ERR] {exc}", flush=True)
            _memory_logger.warning("get_agent_memories 실패 — %s", exc)

    # 쿼리로는 정규화 태그를 쓰고, 태그가 없으면 핵심 키워드 라인 추출
    if situation_tags:
        memory_query = situation_tags
    else:
        _kw_lines = []
        for _l in context_blob.splitlines():
            _s = _l.strip()
            if any(kw in _s for kw in ("RSI", "MACD", "펀딩", "추세", "정렬", "스큐", "OI", "포지션")):
                _kw_lines.append(_s)
            if len(_kw_lines) >= 8:
                break
        memory_query = " | ".join(_kw_lines) if _kw_lines else context_blob[:300]

    # 3) 파이프라인 실행: Bull/Bear → Judge → Risk Triad → Memory
    pipeline = run_pipeline(
        context_blob=context_blob,
        pair_label=PAIR_LABEL,
        memory=memory_obj,
        current_situation=memory_query,
        progress_cb=progress_cb,
        agent_memories=agent_memories_obj,
        price_at_analysis=price_at_analysis,   # ← reflection baseline
    )

    # 4) 최종 애널리스트 호출
    if progress_cb:
        progress_cb("final", "최종 애널리스트 종합 중")

    result = analyze_with_claude(
        multi_tf_data,
        macro_snapshot=macro_snapshot,
        pipeline=pipeline,
        raw_ctx=raw_ctx,   # TPSL override용
    )

    # 5) 메모리에 이번 상황-조언 페어 기록 (reflection 을 위한 씨앗)
    if memory_obj is not None and MEMORY_WRITE_ENABLED:
        try:
            # situation = 지표 태그 + 핵심 수치 요약
            # 리플렉션 프롬프트에서 당시 상황을 구체적으로 재현하기 위해
            # 태그(BM25 매칭용) + 가격/신호/확신도(리플렉션 문맥용) 를 함께 저장
            _sig  = result.get("signal", "")
            _conf = result.get("confidence", 0)
            _tl   = result.get("trade_levels") or {}
            _tp_s = f"TP ${_tl['target']:,.0f}" if _tl.get("target") else ""
            _sl_s = f"SL ${_tl['stop']:,.0f}"   if _tl.get("stop")   else ""
            _lvl_s = " | ".join(filter(None, [_tp_s, _sl_s]))
            _price_s = f"${price_at_analysis:,.2f}" if price_at_analysis else "N/A"

            if situation_tags:
                situation_for_memory = (
                    f"[지표태그] {situation_tags}\n"
                    f"[신호] {_sig} | 확신도 {_conf}% | 가격 {_price_s}"
                    + (f" | {_lvl_s}" if _lvl_s else "")
                )
            else:
                situation_for_memory = (
                    f"[신호] {_sig} | 확신도 {_conf}% | 가격 {_price_s}"
                    + (f" | {_lvl_s}" if _lvl_s else "") + "\n"
                    + context_blob[:400]
                )

            # judge 판정도 메타에 기록
            judge_meta = {}
            if pipeline is not None and pipeline.judge is not None and pipeline.judge.enabled:
                judge_meta = {
                    "judge_verdict": pipeline.judge.verdict,
                    "judge_bull_key": pipeline.judge.bull_key,
                    "judge_bear_key": pipeline.judge.bear_key,
                }
            stored = memory_obj.add_situation(
                situation=situation_for_memory,
                advice=result.get("raw_text", ""),
                outcome="",
                meta={
                    "signal": _sig,
                    "confidence": _conf,
                    "trade_levels": _tl,
                    "trading_signal": result.get("trading_signal"),
                    "pair": PAIR_LABEL,
                    "price_at_analysis": price_at_analysis,
                    "situation_tags": situation_tags,
                    **judge_meta,
                },
            )
            if stored is None:
                _memory_logger.info("memory.add_situation: 최근 기록과 유사 — dedup skip")
        except Exception as exc:
            # 메모리 쓰기 실패는 조용히 무시 (분석 결과는 이미 나왔다)
            _memory_logger.warning("memory.add_situation 실패 — %s", exc)

    return result


