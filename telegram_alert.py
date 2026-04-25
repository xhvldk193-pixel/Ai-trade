import logging,requests
log=logging.getLogger(__name__)
_T="";_C=""
def init(t,c):
 global _T,_C;_T=t;_C=c
def send(m):
 if not _T or not _C:return
 try:requests.post(f"https://api.telegram.org/bot{_T}/sendMessage",json={"chat_id":_C,"text":m,"parse_mode":"HTML"},timeout=5)
 except:pass
def alert_trade(r):
 a=r.get("action","")
 if a not in("long","short"):return
 e="🟢 롱" if a=="long" else "🔴 숏"
 send(f"{e} 진입\n확신도 {r.get('confidence')}%\n진입가 ${r.get('price',0):,.0f}\nTP ${r.get('tp') or 0:,.0f}\nSL ${r.get('sl') or 0:,.0f}")
def alert_analysis(a,p,min_confidence=65,**k):
 if a.get("confidence",0)<min_confidence:return
 s=a.get("signal","");e="📈" if s=="매수" else "📉" if s=="매도" else "⏸"
 send(f"{e} AI분석\n{s} 확신도{a.get('confidence')}%\n현재가 ${p:,.0f}")