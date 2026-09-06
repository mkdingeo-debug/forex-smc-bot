#!/usr/bin/env python3
"""
================================================================================
NATHANIEL RIDGE — RIDGECREST FX | ANALISTA SMC/ICT MULTI-PAR (Forex)
================================================================================
Asistente de análisis técnico (NO ejecuta órdenes, NO gestiona fondos) basado
en metodología Smart Money Concepts (SMC) + Price Action + ICT, aplicado a
pares de Forex. Es el mismo motor de análisis que el bot de criptomonedas
(ltc_smc_analyst.py) — la metodología SMC/ICT nació precisamente en el
trading de Forex — solo cambia la fuente de datos (Twelve Data en vez de
Bybit) y el formato de los símbolos.

Cubre:
1. Estructura de mercado: swing highs/lows, BOS y CHoCH.
2. Liquidez: equal highs / equal lows.
3. Order Blocks, Breaker Blocks y Fair Value Gaps (FVG).
4. Premium / Discount Pricing y niveles OTE (0.618 / 0.705 / 0.79).
5. Plan de trading hipotético: entrada (POI), invalidación (SL), objetivos
   (TP1/TP2/TP3) y ratio Riesgo:Beneficio, con filtro mínimo de calidad.
6. Objetivos de extensión Fibonacci (1.272 / 1.618 / 2.0) cuando el precio
   ya se movió sin retroceso.

------------------------------------------------------------------------------
AVISO IMPORTANTE / DISCLAIMER
------------------------------------------------------------------------------
- Herramienta EDUCATIVA e INFORMATIVA. No es asesoría financiera.
- No ejecuta órdenes ni tiene acceso a tus fondos.
- El "plan de trading" es un ESCENARIO HIPOTÉTICO calculado con reglas
  automatizadas y simplificadas de SMC/ICT. No sustituye el juicio de un
  analista humano ni garantiza resultados.
- El mercado Forex suele operarse con apalancamiento: el riesgo de pérdida
  puede superar el capital depositado. Valida siempre con tu propio análisis
  y gestión de riesgo antes de operar con dinero real.
------------------------------------------------------------------------------
Requisitos:
    pip install requests pandas numpy

Uso básico:
    python forex_smc_analyst.py --interval 1h --outputsize 300

Variables de entorno:
    export TWELVEDATA_API_KEY="tu-api-key-de-twelvedata"
    export SYMBOLS="EUR/USD,USD/JPY,GBP/USD,USD/CHF,AUD/USD,USD/CAD,NZD/USD,EUR/JPY,GBP/JPY,EUR/GBP"
    export SWING_ORDER=4
    export OB_LOOKBACK=8
    export LIQUIDITY_TOLERANCE_PCT=0.05
    export NEARBY_ZONE_PCT=0.03
    export RSI_PERIOD=14
    export VOLUME_LOOKBACK=20
    export VOLUME_ZSCORE_THRESHOLD=2.0

Envío por Telegram (para despliegue 24/7 en Render):
    export TELEGRAM_BOT_TOKEN="tu-token-de-botfather"
    export TELEGRAM_CHAT_ID="tu-chat-id"
    python forex_smc_analyst.py --interval 1h --watch 60 --telegram
================================================================================
"""
import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Literal

import numpy as np
import pandas as pd
import requests

# ==============================================================================
# CONFIGURACIÓN (todo ajustable por variables de entorno)
# ==============================================================================
TWELVEDATA_BASE_URL = "https://api.twelvedata.com"
TWELVEDATA_API_KEY = os.environ.get("TWELVEDATA_API_KEY", "").strip()

DEFAULT_SYMBOLS = "EUR/USD,USD/JPY,GBP/USD,USD/CHF,AUD/USD,USD/CAD,NZD/USD,EUR/JPY,GBP/JPY,EUR/GBP"
SYMBOLS = [s.strip().upper() for s in os.environ.get("SYMBOLS", DEFAULT_SYMBOLS).split(",") if s.strip()]

SWING_ORDER = int(os.environ.get("SWING_ORDER", "4"))
OB_LOOKBACK = int(os.environ.get("OB_LOOKBACK", "8"))
LIQUIDITY_TOLERANCE_PCT = float(os.environ.get("LIQUIDITY_TOLERANCE_PCT", "0.05"))
NEARBY_ZONE_PCT = float(os.environ.get("NEARBY_ZONE_PCT", "0.03"))
RSI_PERIOD = int(os.environ.get("RSI_PERIOD", "14"))
VOLUME_LOOKBACK = int(os.environ.get("VOLUME_LOOKBACK", "20"))
VOLUME_ZSCORE_THRESHOLD = float(os.environ.get("VOLUME_ZSCORE_THRESHOLD", "2.0"))

# Pausa entre llamadas a la API para no exceder el límite gratuito de Twelve
# Data (8 llamadas por minuto). 8 segundos de pausa = máx 7.5 llamadas/min,
# con margen de seguridad.
API_CALL_DELAY_SECONDS = float(os.environ.get("API_CALL_DELAY_SECONDS", "8"))

DEPLOY_LABEL = os.environ.get("DEPLOY_LABEL", "").strip()

OTE_LEVELS = (0.618, 0.705, 0.79)  # niveles institucionales de retroceso (ICT OTE)

ANALYST_PERSONA = """
Eres Nathaniel Ridge, analista principal de Ridgecrest FX, trader institucional y analista de finanzas con más de 20
años de experiencia en mercados financieros (Wall Street 1996-2013, Forex y
cripto 2013-presente). Eres especialista de alto nivel en Smart Money
Concepts (SMC), Price Action e ICT (Inner Circle Trader) — metodología que
nació precisamente en el trading de divisas institucional. Tu análisis es
riguroso, objetivo y directo: identificas la huella de las instituciones
(liquidez, order blocks, desequilibrios) y evitas términos imprecisos o
lenguaje de hype. Presentas siempre: estructura y sesgo, liquidez, zonas de
oferta y demanda, un plan de trading hipotético con entrada/invalidación/
objetivos/ratio riesgo:beneficio, y una nota de gestión de riesgo. Aclaras
siempre que esto es un escenario técnico automatizado, no asesoría
financiera personalizada, y que el apalancamiento típico de Forex amplifica
tanto ganancias como pérdidas.
"""


# ==============================================================================
# 1. DESCARGA DE DATOS (Twelve Data API)
# ==============================================================================
def fetch_klines(symbol: str, interval: str = "1h", outputsize: int = 300) -> pd.DataFrame:
    if not TWELVEDATA_API_KEY:
        raise RuntimeError(
            "Falta la variable de entorno TWELVEDATA_API_KEY. "
            "Consíguela gratis en https://twelvedata.com/register"
        )
    url = f"{TWELVEDATA_BASE_URL}/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVEDATA_API_KEY,
    }
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") == "error" or "values" not in data:
        raise RuntimeError(f"Twelve Data API error para {symbol}: {data.get('message', data)}")

    rows = data["values"]
    if not rows:
        raise RuntimeError(f"Twelve Data no devolvió datos para {symbol}. Revisa símbolo/intervalo.")

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["datetime"], utc=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    # Forex spot no siempre trae volumen real; si falta, se rellena con 0 y
    # la alerta de flujo de capital simplemente no dispara (comportamiento
    # seguro, ver detect_capital_flow_alert).
    df["volume"] = df["volume"].astype(float) if "volume" in df.columns else 0.0

    # Twelve Data devuelve normalmente lo más reciente primero; lo ordenamos
    # de forma ascendente (más antiguo -> más reciente) para el resto del
    # motor de análisis.
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


# ==============================================================================
# 2. ESTRUCTURA DE MERCADO (Swing points, BOS, CHoCH)
# ==============================================================================
@dataclass
class SwingPoint:
    index: int
    timestamp: pd.Timestamp
    price: float
    kind: Literal["high", "low"]


@dataclass
class StructureEvent:
    index: int
    timestamp: pd.Timestamp
    price: float
    kind: Literal["BOS_bullish", "BOS_bearish", "CHoCH_bullish", "CHoCH_bearish"]


def find_swing_points(df: pd.DataFrame, order: int = SWING_ORDER) -> List[SwingPoint]:
    swings: List[SwingPoint] = []
    highs, lows = df["high"].values, df["low"].values
    for i in range(order, len(df) - order):
        window_h = highs[i - order: i + order + 1]
        window_l = lows[i - order: i + order + 1]
        if highs[i] == window_h.max() and highs[i] == window_h[order]:
            swings.append(SwingPoint(i, df["timestamp"][i], highs[i], "high"))
        if lows[i] == window_l.min() and lows[i] == window_l[order]:
            swings.append(SwingPoint(i, df["timestamp"][i], lows[i], "low"))
    swings.sort(key=lambda s: s.index)
    return swings


def detect_structure_events(swings: List[SwingPoint]) -> List[StructureEvent]:
    events: List[StructureEvent] = []
    trend: Optional[str] = None
    last_high: Optional[SwingPoint] = None
    last_low: Optional[SwingPoint] = None
    for s in sorted(swings, key=lambda s: s.index):
        if s.kind == "high":
            if last_high is not None:
                if s.price > last_high.price:
                    if trend == "down":
                        events.append(StructureEvent(s.index, s.timestamp, s.price, "CHoCH_bullish"))
                    elif trend == "up":
                        events.append(StructureEvent(s.index, s.timestamp, s.price, "BOS_bullish"))
                    trend = "up"
            last_high = s
        else:
            if last_low is not None:
                if s.price < last_low.price:
                    if trend == "up":
                        events.append(StructureEvent(s.index, s.timestamp, s.price, "CHoCH_bearish"))
                    elif trend == "down":
                        events.append(StructureEvent(s.index, s.timestamp, s.price, "BOS_bearish"))
                    trend = "down"
            last_low = s
    return events


# ==============================================================================
# 3. ORDER BLOCKS, BREAKER BLOCKS, FVG Y LIQUIDEZ
# ==============================================================================
@dataclass
class Zone:
    kind: str
    top: float
    bottom: float
    index: int
    timestamp: pd.Timestamp
    note: str = ""
    mitigated: bool = False


def find_order_blocks(df: pd.DataFrame, events: List[StructureEvent], lookback: int = OB_LOOKBACK) -> List[Zone]:
    zones: List[Zone] = []
    for ev in events:
        start = max(0, ev.index - lookback)
        segment = df.iloc[start:ev.index]
        if segment.empty:
            continue
        bullish_event = "bullish" in ev.kind
        if bullish_event:
            bearish_candles = segment[segment["close"] < segment["open"]]
            if not bearish_candles.empty:
                c = bearish_candles.iloc[-1]
                zones.append(Zone(
                    "order_block_bullish", top=c["open"], bottom=c["low"],
                    index=int(c.name), timestamp=c["timestamp"],
                    note=f"OB alcista previo a {ev.kind} ({ev.timestamp:%Y-%m-%d %H:%M})",
                ))
        else:
            bullish_candles = segment[segment["close"] > segment["open"]]
            if not bullish_candles.empty:
                c = bullish_candles.iloc[-1]
                zones.append(Zone(
                    "order_block_bearish", top=c["high"], bottom=c["open"],
                    index=int(c.name), timestamp=c["timestamp"],
                    note=f"OB bajista previo a {ev.kind} ({ev.timestamp:%Y-%m-%d %H:%M})",
                ))
    return zones


def tag_breaker_blocks(df: pd.DataFrame, zones: List[Zone]) -> None:
    for z in zones:
        if z.kind not in ("order_block_bullish", "order_block_bearish"):
            continue
        after = df.iloc[z.index + 1:]
        if after.empty:
            continue
        if z.kind == "order_block_bullish":
            broken = after[after["close"] < z.bottom]
            if not broken.empty:
                z.mitigated = True
                z.note += " → posible BREAKER (bajista si el precio regresa aquí)"
        else:
            broken = after[after["close"] > z.top]
            if not broken.empty:
                z.mitigated = True
                z.note += " → posible BREAKER (alcista si el precio regresa aquí)"


def find_fair_value_gaps(df: pd.DataFrame) -> List[Zone]:
    zones: List[Zone] = []
    for i in range(2, len(df)):
        c1, c3 = df.iloc[i - 2], df.iloc[i]
        if c1["high"] < c3["low"]:
            zones.append(Zone(
                "fvg_bullish", top=c3["low"], bottom=c1["high"],
                index=i, timestamp=df.iloc[i - 1]["timestamp"],
                note="FVG alcista (desequilibrio / posible zona de reequilibrio)",
            ))
        elif c1["low"] > c3["high"]:
            zones.append(Zone(
                "fvg_bearish", top=c1["low"], bottom=c3["high"],
                index=i, timestamp=df.iloc[i - 1]["timestamp"],
                note="FVG bajista (desequilibrio / posible zona de reequilibrio)",
            ))
    return zones


def find_liquidity_pools(swings: List[SwingPoint], tolerance_pct: float = LIQUIDITY_TOLERANCE_PCT) -> List[Zone]:
    zones: List[Zone] = []
    for kind, pts in (("high", [s for s in swings if s.kind == "high"]),
                       ("low", [s for s in swings if s.kind == "low"])):
        pts = sorted(pts, key=lambda s: s.price)
        used = set()
        for i, a in enumerate(pts):
            if a.index in used:
                continue
            cluster = [a]
            for b in pts[i + 1:]:
                if abs(b.price - a.price) / a.price * 100 <= tolerance_pct:
                    cluster.append(b)
            if len(cluster) >= 2:
                for p in cluster:
                    used.add(p.index)
                avg_price = float(np.mean([p.price for p in cluster]))
                zones.append(Zone(
                    f"liquidity_{kind}", top=avg_price, bottom=avg_price,
                    index=cluster[-1].index, timestamp=cluster[-1].timestamp,
                    note=f"{len(cluster)} equal {kind}s agrupados cerca de {avg_price:.5f} "
                         f"(liquidez minorista — posible objetivo de barrido)",
                ))
    return zones


# ==============================================================================
# 4. PREMIUM / DISCOUNT PRICING + NIVELES OTE (ICT)
# ==============================================================================
@dataclass
class PremiumDiscount:
    range_high: float
    range_low: float
    pct_in_range: float
    zone: str
    impulse_up: bool
    ote_levels: Dict[float, float]


def compute_premium_discount(df: pd.DataFrame, swings: List[SwingPoint]) -> Optional[PremiumDiscount]:
    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]
    if not highs or not lows:
        return None
    last_high, last_low = highs[-1], lows[-1]
    range_high = max(last_high.price, last_low.price)
    range_low = min(last_high.price, last_low.price)
    rng = range_high - range_low
    if rng <= 0:
        return None
    price = float(df["close"].iloc[-1])
    pct = (price - range_low) / rng
    if pct > 1.0:
        extra = price - range_high
        zone = (f"EXTENSIÓN ALCISTA — {extra:.5f} por encima del último swing high "
                f"confirmado ({range_high:.5f}), equivalente al {(pct - 1) * 100:.0f}% del rango")
    elif pct < 0.0:
        extra = range_low - price
        zone = (f"EXTENSIÓN BAJISTA — {extra:.5f} por debajo del último swing low "
                f"confirmado ({range_low:.5f}), equivalente al {(-pct) * 100:.0f}% del rango")
    elif pct < 0.45:
        zone = "descuento (discount)"
    elif pct > 0.55:
        zone = "premium"
    else:
        zone = "equilibrio (50%)"
    impulse_up = last_high.index > last_low.index
    ote_levels = {}
    for fib in OTE_LEVELS:
        ote_levels[fib] = (range_high - rng * fib) if impulse_up else (range_low + rng * fib)
    return PremiumDiscount(range_high, range_low, pct, zone, impulse_up, ote_levels)


EXTENSION_RATIOS = (1.272, 1.618, 2.0)


def compute_extension_targets(pd_info: PremiumDiscount) -> Optional[Dict[float, float]]:
    rng = pd_info.range_high - pd_info.range_low
    if rng <= 0:
        return None
    targets: Dict[float, float] = {}
    if pd_info.pct_in_range > 1.0:
        for ratio in EXTENSION_RATIOS:
            targets[ratio] = pd_info.range_high + rng * (ratio - 1.0)
    elif pd_info.pct_in_range < 0.0:
        for ratio in EXTENSION_RATIOS:
            targets[ratio] = pd_info.range_low - rng * (ratio - 1.0)
    else:
        return None
    return targets


# ==============================================================================
# 5. INDICADOR DE CONFLUENCIA: RSI
# ==============================================================================
def calculate_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> float:
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty and not pd.isna(rsi.iloc[-1]) else 50.0


# ==============================================================================
# 6. ALERTA DE FLUJO DE CAPITAL (anomalía de volumen, si el bróker la reporta)
# ==============================================================================
def detect_capital_flow_alert(df: pd.DataFrame, lookback: int = VOLUME_LOOKBACK,
                               z_threshold: float = VOLUME_ZSCORE_THRESHOLD) -> Optional[str]:
    vol = df["volume"]
    if len(vol) < lookback + 1 or vol.sum() == 0:
        # Forex spot muchas veces no reporta volumen real; sin datos de
        # volumen, esta alerta simplemente no se genera (no es un error).
        return None
    recent = vol.iloc[-(lookback + 1):-1]
    mean, std = recent.mean(), recent.std()
    if std == 0 or pd.isna(std):
        return None
    current = df.iloc[-1]
    z = (current["volume"] - mean) / std
    if z >= z_threshold:
        direction = "ENTRADA fuerte de capital (compra)" if current["close"] > current["open"] else "SALIDA fuerte de capital (venta)"
        return f"⚡ {direction} detectada — volumen {z:.1f}σ sobre el promedio de {lookback} velas."
    return None


# ==============================================================================
# 7. PLAN DE TRADING HIPOTÉTICO
# ==============================================================================
@dataclass
class TradePlan:
    entry_low: float
    entry_high: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    rr1: float
    rr2: float
    rr3: float
    tp1_source: str


MAX_TP1_DISTANCE_MULT = 2.0


def compute_trade_plan(bias: str, price: float, pd_info: Optional[PremiumDiscount],
                        zones: List[Zone]) -> Optional[TradePlan]:
    if pd_info is None or bias not in ("alcista", "bajista"):
        return None
    if pd_info.pct_in_range < 0.0 or pd_info.pct_in_range > 1.0:
        return None
    if bias == "alcista" and not pd_info.impulse_up:
        return None
    if bias == "bajista" and pd_info.impulse_up:
        return None

    level_618 = pd_info.ote_levels[0.618]
    level_79 = pd_info.ote_levels[0.79]
    entry_low, entry_high = sorted([level_618, level_79])
    entry_mid = (entry_low + entry_high) / 2
    rng = pd_info.range_high - pd_info.range_low
    buffer = rng * 0.02
    max_tp1_distance = rng * MAX_TP1_DISTANCE_MULT
    tp1_source = "límite del rango"

    if bias == "alcista":
        sl = pd_info.range_low - buffer
        targets = sorted(
            [z for z in zones if z.kind == "liquidity_high" and z.top > entry_mid],
            key=lambda z: z.top,
        )
        if targets and (targets[0].top - entry_mid) <= max_tp1_distance:
            tp1 = targets[0].top
            tp1_source = f"liquidez en {tp1:.5f}"
        else:
            tp1 = pd_info.range_high
        tp2 = pd_info.range_high
        risk = entry_mid - sl
        tp3 = entry_mid + risk * 3
    else:
        sl = pd_info.range_high + buffer
        targets = sorted(
            [z for z in zones if z.kind == "liquidity_low" and z.bottom < entry_mid],
            key=lambda z: -z.bottom,
        )
        if targets and (entry_mid - targets[0].bottom) <= max_tp1_distance:
            tp1 = targets[0].bottom
            tp1_source = f"liquidez en {tp1:.5f}"
        else:
            tp1 = pd_info.range_low
        tp2 = pd_info.range_low
        risk = sl - entry_mid
        tp3 = entry_mid - risk * 3

    if risk <= 0:
        return None

    rr1 = abs(tp1 - entry_mid) / risk
    if rr1 < 0.8:
        return None
    rr2 = abs(tp2 - entry_mid) / risk
    rr3 = abs(tp3 - entry_mid) / risk

    return TradePlan(entry_low, entry_high, sl, tp1, tp2, tp3, rr1, rr2, rr3, tp1_source)


# ==============================================================================
# 8. GENERACIÓN DE SEÑAL / SESGO
# ==============================================================================
@dataclass
class SignalReport:
    symbol: str
    price: float
    bias: str
    confidence: str
    last_event: Optional[StructureEvent]
    liquidity_zones: List[Zone]
    ob_zones: List[Zone]
    fvg_zones: List[Zone]
    invalidation: Optional[float]
    invalidation_broken: bool
    rsi: float
    rsi_note: str
    premium_discount: Optional[PremiumDiscount]
    extension_targets: Optional[Dict[float, float]]
    trade_plan: Optional[TradePlan]
    capital_flow_alert: Optional[str]
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def find_structural_invalidation(swings: List[SwingPoint], last_event: Optional[StructureEvent],
                                  bias: str) -> Optional[float]:
    if last_event is None or bias not in ("alcista", "bajista"):
        return None
    if bias == "alcista":
        candidates = [s for s in swings if s.kind == "low" and s.index <= last_event.index]
    else:
        candidates = [s for s in swings if s.kind == "high" and s.index <= last_event.index]
    if not candidates:
        return last_event.price
    return candidates[-1].price


def generate_signal(symbol: str, df: pd.DataFrame, events: List[StructureEvent], zones: List[Zone]) -> SignalReport:
    price = float(df["close"].iloc[-1])
    last_event = events[-1] if events else None

    if last_event is None:
        bias, confidence = "neutral / rango", "baja"
    elif "bullish" in last_event.kind:
        bias, confidence = "alcista", "media-alta" if last_event.kind.startswith("BOS") else "media"
    else:
        bias, confidence = "bajista", "media-alta" if last_event.kind.startswith("BOS") else "media"

    swings_all = find_swing_points(df)
    invalidation = find_structural_invalidation(swings_all, last_event, bias)

    invalidation_broken = False
    if invalidation is not None:
        if bias == "bajista" and price > invalidation:
            invalidation_broken = True
        elif bias == "alcista" and price < invalidation:
            invalidation_broken = True
    if invalidation_broken:
        confidence = "muy baja — posible lectura obsoleta"

    rsi = calculate_rsi(df)
    rsi_note = f"{rsi:.1f} (neutral)"
    if bias == "alcista" and not invalidation_broken:
        if rsi < 35:
            confidence, rsi_note = "alta", f"{rsi:.1f} (sobreventa — refuerza el sesgo alcista)"
        elif rsi > 75:
            confidence, rsi_note = "baja", f"{rsi:.1f} (sobrecompra — precaución, posible agotamiento)"
    elif bias == "bajista" and not invalidation_broken:
        if rsi > 65:
            confidence, rsi_note = "alta", f"{rsi:.1f} (sobrecompra — refuerza el sesgo bajista)"
        elif rsi < 25:
            confidence, rsi_note = "baja", f"{rsi:.1f} (sobreventa — precaución, posible agotamiento)"

    def zone_mid(z: Zone) -> float:
        return (z.top + z.bottom) / 2

    liquidity_zones = sorted(
        [z for z in zones if z.kind.startswith("liquidity_")],
        key=lambda z: abs(zone_mid(z) - price),
    )
    ob_zones = sorted(
        [z for z in zones if "order_block" in z.kind and not z.mitigated],
        key=lambda z: abs(zone_mid(z) - price),
    )
    fvg_zones = sorted(
        [z for z in zones if z.kind.startswith("fvg_")],
        key=lambda z: abs(zone_mid(z) - price),
    )

    pd_info = compute_premium_discount(df, swings_all)
    extension_targets = compute_extension_targets(pd_info) if pd_info else None
    trade_plan = None if invalidation_broken else compute_trade_plan(bias, price, pd_info, zones)
    capital_flow_alert = detect_capital_flow_alert(df)

    return SignalReport(
        symbol=symbol, price=price, bias=bias, confidence=confidence,
        last_event=last_event, liquidity_zones=liquidity_zones[:4],
        ob_zones=ob_zones[:3],
        fvg_zones=fvg_zones[:3], invalidation=invalidation,
        invalidation_broken=invalidation_broken,
        rsi=rsi, rsi_note=rsi_note, premium_discount=pd_info,
        extension_targets=extension_targets,
        trade_plan=trade_plan, capital_flow_alert=capital_flow_alert,
    )


# ==============================================================================
# 9. NARRATIVA — plantilla determinista
# ==============================================================================
def render_template_report(report: SignalReport, interval: str) -> str:
    L = []
    label_prefix = f"[{DEPLOY_LABEL}] " if DEPLOY_LABEL else ""
    L.append(f"═══ {label_prefix}RIDGECREST FX | NATHANIEL RIDGE | {report.symbol} | TF {interval} ═══")
    L.append(f"{report.generated_at:%Y-%m-%d %H:%M UTC}")
    L.append("")

    if report.capital_flow_alert:
        L.append(report.capital_flow_alert)
        L.append("")

    if report.invalidation_broken:
        L.append(
            "🚨 ATENCIÓN: el precio ya cerró más allá del nivel de invalidación de "
            "esta lectura. La estructura probablemente ya está cambiando de nuevo — "
            "trata este sesgo como OBSOLETO hasta la próxima confirmación."
        )
        L.append("")

    L.append("1) ESTRUCTURA Y SESGO")
    L.append(f"   Precio: {report.price:.5f} | Sesgo: {report.bias.upper()} | Confianza: {report.confidence}")
    L.append(f"   RSI({RSI_PERIOD}): {report.rsi_note}")
    if report.last_event:
        L.append(f"   Último evento: {report.last_event.kind} en {report.last_event.price:.5f} "
                  f"({report.last_event.timestamp:%m-%d %H:%M})")
        L.append(f"   Invalidación de esta lectura: {report.invalidation:.5f}")
    else:
        L.append("   Sin eventos de estructura claros todavía.")

    if report.premium_discount:
        pd_i = report.premium_discount
        L.append("")
        L.append("2) PREMIUM / DISCOUNT (ICT)")
        L.append(f"   Rango activo (último swing): {pd_i.range_low:.5f} — {pd_i.range_high:.5f}")
        L.append(f"   Precio: {pd_i.zone}")
        levels_txt = " | ".join(f"{fib}: {p:.5f}" for fib, p in pd_i.ote_levels.items())
        L.append(f"   Niveles OTE: {levels_txt}")

    L.append("")
    L.append("3) LIQUIDEZ")
    if report.liquidity_zones:
        for z in report.liquidity_zones:
            L.append(f"   - {z.note}")
    else:
        L.append("   - Sin agrupaciones claras de liquidez cerca del rango analizado.")

    L.append("")
    L.append("4) OFERTA / DEMANDA (Order Blocks & FVG)")
    if report.ob_zones:
        for z in report.ob_zones:
            L.append(f"   - [{z.kind}] {z.bottom:.5f}-{z.top:.5f} | {z.note}")
    else:
        L.append("   - Sin Order Blocks no mitigados relevantes.")
    if report.fvg_zones:
        for z in report.fvg_zones:
            L.append(f"   - [{z.kind}] {z.bottom:.5f}-{z.top:.5f}")

    if report.trade_plan:
        tp = report.trade_plan
        L.append("")
        L.append("5) PLAN DE TRADING HIPOTÉTICO")
        L.append(f"   Entrada (POI): {tp.entry_low:.5f} — {tp.entry_high:.5f}")
        L.append(f"   Invalidación (SL): {tp.stop_loss:.5f}")
        L.append(f"   TP1: {tp.take_profit_1:.5f} (R:R {tp.rr1:.2f}) — fuente: {tp.tp1_source}")
        L.append(f"   TP2: {tp.take_profit_2:.5f} (R:R {tp.rr2:.2f})")
        L.append(f"   TP3: {tp.take_profit_3:.5f} (R:R {tp.rr3:.2f})")
    elif report.extension_targets:
        L.append("")
        L.append("5) OBJETIVOS DE EXTENSIÓN (movimiento fuerte, sin retroceso aún)")
        L.append("   No hay entrada recomendada aquí — el precio ya se movió sin retroceso, "
                  "entrar ahora implica peor R:R y mayor riesgo de reversión. Estos son "
                  "niveles de referencia por si el movimiento continúa:")
        for ratio, level in report.extension_targets.items():
            L.append(f"   Extensión {ratio}: {level:.5f}")
        L.append("   Si buscas entrar, lo prudente es esperar un retroceso hacia las zonas "
                  "de la sección 4, no perseguir el precio aquí.")
    else:
        L.append("")
        L.append("5) PLAN DE TRADING HIPOTÉTICO")
        L.append("   Sin plan accionable en este momento (precio en extensión fuera del "
                  "último rango confirmado, o el Riesgo:Beneficio disponible no es "
                  "favorable). Se recomienda esperar un retroceso o una nueva "
                  "confirmación de estructura.")

    L.append("")
    L.append(
        "⚠ Escenario técnico automatizado (SMC/ICT simplificado). No es asesoría "
        "financiera ni garantía de resultados. El apalancamiento en Forex amplifica "
        "ganancias y pérdidas. Define tu propia gestión de riesgo."
    )
    return "\n".join(L)


# ==============================================================================
# 10. TELEGRAM
# ==============================================================================
def send_telegram_message(text: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("[Aviso] Falta TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID. No se envió mensaje.")
        return
    if len(text) > 4000:
        text = text[:3990] + "\n...(recortado)"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        resp = requests.post(url, data={"chat_id": chat_id, "text": text}, timeout=15)
        resp.raise_for_status()
        result = resp.json()
        if not result.get("ok"):
            print(f"[Error Telegram] {result}")
    except Exception as exc:
        print(f"[Error enviando a Telegram] {exc}")


# ==============================================================================
# 11. CLI
# ==============================================================================
def analyze_symbol(symbol: str, interval: str, outputsize: int) -> str:
    df = fetch_klines(symbol=symbol, interval=interval, outputsize=outputsize)
    swings = find_swing_points(df)
    events = detect_structure_events(swings)
    zones = []
    zones += find_order_blocks(df, events)
    tag_breaker_blocks(df, zones)
    zones += find_fair_value_gaps(df)
    zones += find_liquidity_pools(swings)
    report = generate_signal(symbol, df, events, zones)
    return render_template_report(report, interval)


def run_once(symbols: List[str], interval: str, outputsize: int, use_telegram: bool) -> None:
    for i, symbol in enumerate(symbols):
        try:
            text = analyze_symbol(symbol, interval, outputsize)
            print(text)
            if use_telegram:
                send_telegram_message(text)
        except Exception as exc:
            print(f"[Error analizando {symbol}] {exc}")
        # Respeta el límite de 8 llamadas/min del plan gratuito de Twelve
        # Data, salvo en el último símbolo del lote.
        if i < len(symbols) - 1:
            time.sleep(API_CALL_DELAY_SECONDS)


def seconds_until_next_aligned_run(watch_minutes: int) -> float:
    now = datetime.now(timezone.utc)
    total_minutes_since_midnight = now.hour * 60 + now.minute
    next_slot = ((total_minutes_since_midnight // watch_minutes) + 1) * watch_minutes
    next_run = now.replace(hour=0, minute=0, second=0, microsecond=0)
    next_run += pd.Timedelta(minutes=next_slot)
    return max((next_run - now).total_seconds(), 0)


def main():
    parser = argparse.ArgumentParser(description="Analista SMC/ICT multi-par de Forex (Twelve Data)")
    parser.add_argument("--symbols", default=None,
                         help="Lista separada por comas, ej. 'EUR/USD,GBP/USD'. Si no se especifica, usa SYMBOLS.")
    parser.add_argument("--interval", default="1h",
                         help="Timeframe de Twelve Data: 1min,5min,15min,30min,45min,1h,2h,4h,1day,1week,1month")
    parser.add_argument("--outputsize", type=int, default=300, help="Número de velas a analizar (máx ~5000)")
    parser.add_argument("--telegram", action="store_true", help="Envía el reporte a Telegram")
    parser.add_argument("--watch", type=int, default=0, help="Repite cada N minutos (0 = una sola vez)")
    parser.add_argument("--no-align", action="store_true",
                         help="Desactiva la alineación al reloj.")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else SYMBOLS

    print(
        "⚠ Herramienta educativa. No ejecuta órdenes ni gestiona fondos. No es asesoría financiera.\n"
        f"Pares: {', '.join(symbols)}\n"
    )

    if args.watch <= 0:
        run_once(symbols, args.interval, args.outputsize, args.telegram)
        return

    while True:
        if not args.no_align:
            wait = seconds_until_next_aligned_run(args.watch)
            if wait > 1:
                mins = int(wait // 60)
                print(f"Esperando {mins} min para alinear el próximo reporte a un horario redondo...\n")
                time.sleep(wait)
        try:
            run_once(symbols, args.interval, args.outputsize, args.telegram)
        except KeyboardInterrupt:
            print("\nDetenido por el usuario.")
            sys.exit(0)
        except Exception as exc:
            print(f"[Error en el ciclo de análisis] {exc}")

        if args.no_align:
            print(f"\n(Próxima actualización en {args.watch} min)\n")
            time.sleep(args.watch * 60)
        else:
            time.sleep(5)


if __name__ == "__main__":
    main()
