#!/bin/bash
set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🚀 BTC AI 자동매매 (Bitget)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "$(dirname "$0")"

if ! command -v python3 &>/dev/null; then
  echo "❌ python3가 필요합니다."
  exit 1
fi

echo "✅ Python: $(python3 --version)"
echo "📦 패키지 설치 중..."
pip3 install -q -r requirements.txt
echo "✅ 패키지 설치 완료"

if [ ! -f ".env" ]; then
  echo "⚠️  .env 파일이 없습니다. .env.example 을 복사해 키를 입력하세요."
fi

echo ""
echo "🌐 http://localhost:8000"
echo "   (종료: Ctrl+C)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# main:app — 배포 환경과 동일한 진입점 사용
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
