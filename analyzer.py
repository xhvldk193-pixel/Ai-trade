# =============================================
# Claude API 연동 - 매매 시그널 분석 (인자 충돌 방어판)
# =============================================
import re
import os as _os
import logging as _logging
from typing import Optional, Callable, Any

# ... (상단 import 및 변수 설정은 이전과 동일)

# **수정 포인트**: progress_cb를 인자 리스트에서 아예 빼고 
# **kwargs를 통해 내부에서 추출하도록 변경했습니다. 
# 이렇게 하면 '중복 입력' 에러가 절대 발생하지 않습니다.
def run_full_analysis(multi_tf_data: dict, *args, **kwargs) -> dict:
    """
    고도화된 분석 루틴.
    :param multi_tf_data: 타임프레임별 데이터
    :param args: 기타 위치 인자 (중복 방지용)
    :param kwargs: progress_cb 및 기타 키워드 인자 흡수
    """
    
    # kwargs에서 progress_cb를 찾고, 없으면 args의 첫 번째 값을 시도, 둘 다 없으면 None
    progress_cb = kwargs.get('progress_cb')
    if progress_cb is None and len(args) > 0:
        progress_cb = args[0]

    def notify(msg: str):
        if progress_cb and callable(progress_cb): 
            try:
                progress_cb(msg)
            except:
                pass
        _logging.info(msg)

    notify("📊 분석 시작 및 메모리 점검 중...")
    
    # 1. 메모리 초기화
    mem = None
    if get_memory:
        try:
            mem = get_memory("analyst")
            if mem: 
                mem.cleanup_old_no_outcome_records(days_threshold=3)
        except Exception as e:
            _logging.warning(f"메모리 초기화 에러: {e}")

    # 2. 현재가 추출
    try:
        price_at_analysis = float(multi_tf_data["1h"].iloc[-1]["close"])
    except:
        price_at_analysis = 0.0
    
    # 3. 에이전트 토론 실행
    notify("🗣️ 에이전트 토론(Pipeline) 진행 중...")
    try:
        pipeline = run_pipeline(
            context_blob=_build_context_blob(multi_tf_data), 
            pair_label=PAIR_LABEL, 
            current_situation="전략적 포지션 분석", 
            price_at_analysis=price_at_analysis
        )
    except Exception as e:
        notify(f"⚠️ 토론 파이프라인 에러: {e}")
        pipeline = None
    
    # 4. Claude 최종 판단
    notify("🧠 Claude 최종 판단 도출 중...")
    result = analyze_with_claude(multi_tf_data, pipeline)
    
    # 5. 메모리 저장
    if MEMORY_WRITE_ENABLED and mem:
        try:
            notify("💾 분석 결과 메모리 저장 중...")
            debate_log = getattr(pipeline, "combined_block", "토론 생략") if pipeline else "토론 생략"
            full_reasoning = f"[DEBATE LOG]\n{debate_log}\n\n[FINAL RESULT]\n{result['raw_text']}"
            
            mem.add_situation(
                situation=result['prompt'][:1000], 
                advice=full_reasoning,
                meta={
                    "confidence": result['confidence'], 
                    "levels": result['levels'],
                    "price_at_analysis": price_at_analysis, 
                    "view": result['view']
                }
            )
        except:
            pass
    
    notify("✅ 분석 완료")
    return result
