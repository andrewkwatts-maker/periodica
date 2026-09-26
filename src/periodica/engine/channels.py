"""Engine channels: the physical quantities a renderer or simulation samples.

A data sheet is what a material *has*. A channel is what an engine *needs*.
They are not the same list, and the gap is why raw sheets are awkward to
consume: an engine asking for stiffness does not care whether the sheet
happens to store ``YoungsModulus_GPa``, or only ``ShearModulus_GPa`` and
``PoissonsRatio`` to derive it from, or nothing at all and the value has to
come from the constituents. Asking each caller to handle those three cases is
how the same fallback logic ends up copy-pasted into every consumer.

A channel closes the gap by being *defined as its fallback chain*::

    stiffness:
      1. canonical property `youngs_modulus`            confidence 1.0
      2. 2 * shear_modulus * (1 + poissons_ratio)       confidence 0.9
      3. 3 * bulk_modulus * (1 - 2 * poissons_ratio)    confidence 0.9
      4. volume-weighted mixture over the constituents  confidence 0.6

The chain lives in ``data/config/engine_channels.json``, so adding a channel
or a new way to derive one is a config edit. Resolution walks the chain and
returns the first hit *with its provenance*, so a consumer is never lied to:

    >>> resolve("Stainless_Steel_316L", "stiffness")
    ChannelValue(value=1.93e+11, unit='Pa', confidence=1.0,
                 method='property:youngs_modulus', ...)

    >>> resolve("Alumina_Al2O3", "thermal_diffusivity")
    ChannelValue(value=9.9e-06, unit='m2_s', confidence=0.95,
                 method='formula:thermal_conductivity / (density * specific_heat)', ...)

Every channel resolves to *something*: the last link in each chain is a
documented constant, so a higher app never has to handle None. Whether that
is acceptable is the app's call to make from ``confidence``, which is exactly
why confidence travels with the value.
"""
from __future__ import annotations

import ast
import json
import math
import operator
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

from periodica.get import Get, UnknownName
from periodica.properties import si_sheet, sheet as property_sheet

_CONFIG_PATH = Path(__file__).parent.parent / "data" / "config" / "engine_channels.json"

_config_cache: Optional[dict] = None
_config_lock = threading.Lock()


# ── Config ──────────────────────────────────────────────────────────────

def _strip_comments(text: str) -> str:
    return re.sub(r"//.*?(?=\n|$)", "", text)


def config() -> dict:
    """The parsed channel configuration (cached)."""
    global _config_cache
    if _config_cache is None:
        with _config_lock:
            if _config_cache is None:
                _config_cache = json.loads(
                    _strip_comments(_CONFIG_PATH.read_text(encoding="utf-8"))
                )
    return _config_cache


def reload_config() -> None:
    """Drop the cached channel config; next access re-reads the JSON."""
    global _config_cache
    with _config_lock:
        _config_cache = None


def list_channels() -> List[str]:
    """Every channel name, sorted."""
    return sorted(config()["channels"])


def channel_spec(name: str) -> dict:
    """The config block for one channel."""
    spec = config()["channels"].get(name)
    if spec is None:
        raise UnknownChannel(
            f"Unknown channel {name!r}. Available: {list_channels()}"
        )
    return spec


def channel_groups() -> Dict[str, List[str]]:
    """Named bundles of channels ("mechanical", "thermal", "optical", "bulk")."""
    return {k: list(v) for k, v in (config().get("channel_groups") or {}).items()}


def expand_channels(names: Union[str, Sequence[str], None]) -> List[str]:
    """Resolve channel names, group names and None into a concrete list.

    ``None`` means every channel; a group name expands to its members;
    duplicates collapse while keeping first-seen order.
    """
    if names is None:
        return list_channels()
    if isinstance(names, str):
        names = [names]
    groups = channel_groups()
    out: List[str] = []
    for name in names:
        for member in groups.get(name, [name]):
            if member not in out:
                out.append(member)
    return out


class UnknownChannel(KeyError):
    """A channel name that is not in the config."""


# ── Result type ─────────────────────────────────────────────────────────

class ChannelValue(NamedTuple):
    """A resolved channel, with where it came from.

    `confidence` is the config's own rating of the method used: 1.0 for a
    value read straight off the sheet, lower for each derivation step, lowest
    for a documented default. It is a provenance grade, not a measurement
    uncertainty -- treat it as "how far from the data is this".
    """

    channel: str
    value: Union[float, Tuple[float, float, float], None]
    unit: str
    confidence: float
    method: str
    note: str = ""

    @property
    def is_measured(self) -> bool:
        """True when the value was read from the sheet, not derived."""
        return self.method.startswith("property:")

    @property
    def is_default(self) -> bool:
        """True when nothing in the data supported the value."""
        return self.method.startswith("constant:")

    def as_dict(self) -> dict:
        return {
            "channel": self.channel,
            "value": list(self.value) if isinstance(self.value, tuple) else self.value,
            "unit": self.unit,
            "confidence": self.confidence,
            "method": self.method,
            "note": self.note,
        }


# ── Restricted expression evaluation ────────────────────────────────────

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}

#: The only callables a config expression may use.
_FUNCS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "abs": abs,
    "min": min,
    "max": max,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "atan": math.atan,
    "pow": math.pow,
}
_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}

_expr_cache: Dict[str, ast.Expression] = {}


def _parse_expr(expr: str) -> ast.Expression:
    cached = _expr_cache.get(expr)
    if cached is None:
        cached = ast.parse(expr, mode="eval")
        _validate_expr(cached)
        _expr_cache[expr] = cached
    return cached


def _validate_expr(tree: ast.Expression) -> None:
    """Reject anything beyond arithmetic over names and whitelisted calls.

    Config is package data, not user input, but a formula language that can
    reach attributes or imports is a liability regardless of who writes it.
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.Attribute, ast.Subscript, ast.Lambda, ast.Dict,
                             ast.List, ast.Set, ast.Tuple, ast.comprehension,
                             ast.ListComp, ast.DictComp, ast.SetComp,
                             ast.GeneratorExp, ast.Await, ast.IfExp,
                             ast.Starred, ast.NamedExpr, ast.JoinedStr)):
            raise ValueError(f"Disallowed syntax {type(node).__name__} in channel formula")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
                raise ValueError(
                    f"Channel formulas may only call {sorted(_FUNCS)}"
                )


def _safe_eval(expr: str, variables: Mapping[str, float]) -> float:
    """Evaluate a config arithmetic expression over `variables`."""
    tree = _parse_expr(expr)

    def ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return float(node.value)
            raise ValueError(f"Non-numeric constant {node.value!r} in formula")
        if isinstance(node, ast.Name):
            if node.id in variables:
                return float(variables[node.id])
            if node.id in _CONSTS:
                return _CONSTS[node.id]
            raise KeyError(node.id)
        if isinstance(node, ast.BinOp):
            op = _BIN_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"Disallowed operator {type(node.op).__name__}")
            return op(ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp):
            op = _UNARY_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"Disallowed unary {type(node.op).__name__}")
            return op(ev(node.operand))
        if isinstance(node, ast.Call):
            return _FUNCS[node.func.id](*(ev(a) for a in node.args))
        raise ValueError(f"Disallowed node {type(node).__name__} in formula")

    return float(ev(tree))


# ── Source handlers ─────────────────────────────────────────────────────

def _entry_of(name_or_entry: Union[str, Mapping[str, Any]]) -> dict:
    if isinstance(name_or_entry, str):
        return Get(name_or_entry)
    if isinstance(name_or_entry, Mapping):
        return dict(name_or_entry)
    raise TypeError(
        f"expected a registry name or entry dict, got {type(name_or_entry).__name__}"
    )


def _composition_of(entry: Mapping[str, Any]) -> Dict[str, float]:
    """Constituent symbol -> count, from whichever shape the entry uses.

    Curated material sheets use ``Components`` with percentage ranges;
    composed entries use ``Composition`` with counts. Both reduce to
    relative weights.
    """
    comp = entry.get("Composition")
    if isinstance(comp, Mapping) and comp:
        out: Dict[str, float] = {}
        for sym, count in comp.items():
            try:
                out[str(sym)] = float(count)
            except (TypeError, ValueError):
                continue
        if out:
            return out

    components = entry.get("Components")
    if isinstance(components, Sequence) and not isinstance(components, (str, bytes)):
        out = {}
        for item in components:
            if not isinstance(item, Mapping):
                continue
            sym = item.get("Element") or item.get("Symbol") or item.get("element")
            if not sym:
                continue
            lo = item.get("MinPercent")
            hi = item.get("MaxPercent")
            pct = item.get("Percent")
            if pct is None and lo is not None and hi is not None:
                try:
                    pct = (float(lo) + float(hi)) / 2.0
                except (TypeError, ValueError):
                    pct = None
            if pct is None:
                pct = lo if lo is not None else hi
            try:
                value = float(pct)
            except (TypeError, ValueError):
                continue
            out[str(sym)] = out.get(str(sym), 0.0) + value
        if out:
            return out
    return {}


def _mixture(entry: Mapping[str, Any], canonical: str, weighting: str) -> Optional[float]:
    """Rule-of-mixtures average of a canonical property over constituents.

    Volume weighting uses each constituent's count/percentage directly; mass
    weighting scales it by the constituent's density, which is the difference
    between a Voigt average and a mass-fraction average.
    """
    composition = _composition_of(entry)
    if not composition:
        return None
    total_value = 0.0
    total_weight = 0.0
    for sym, amount in composition.items():
        if amount <= 0:
            continue
        try:
            constituent = Get(sym)
        except (UnknownName, KeyError, TypeError):
            continue
        values = si_sheet(constituent)
        value = values.get(canonical)
        if value is None:
            continue
        weight = float(amount)
        if weighting == "mass":
            density = values.get("density")
            if density:
                weight *= float(density)
        total_value += float(value) * weight
        total_weight += weight
    if total_weight <= 0:
        return None
    return total_value / total_weight


_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{6}|[0-9a-fA-F]{3})$")


def _srgb_to_linear(component: float) -> float:
    """Undo the sRGB transfer function. Engines shade in linear light."""
    if component <= 0.04045:
        return component / 12.92
    return ((component + 0.055) / 1.055) ** 2.4


def _hex_to_linear_rgb(text: str) -> Optional[Tuple[float, float, float]]:
    match = _HEX_RE.match(str(text).strip())
    if not match:
        return None
    digits = match.group(1)
    if len(digits) == 3:
        digits = "".join(c * 2 for c in digits)
    srgb = [int(digits[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    return tuple(_srgb_to_linear(c) for c in srgb)  # type: ignore[return-value]


def _wavelength_to_linear_rgb(wavelength_m: float) -> Optional[Tuple[float, float, float]]:
    """Approximate linear-sRGB colour of a single wavelength.

    Delegates to the package's existing `utils.color_math.wavelength_to_rgb`
    rather than carrying a second copy of the CIE approximation, then
    linearises it.
    """
    try:
        from periodica.utils.color_math import wavelength_to_rgb
    except ImportError:  # pragma: no cover
        return None
    nm = float(wavelength_m) * 1e9
    if not (200.0 <= nm <= 2000.0):
        return None
    rgb = wavelength_to_rgb(nm)
    if not rgb:
        return None
    srgb = [max(0.0, min(1.0, float(c) / 255.0)) for c in tuple(rgb)[:3]]
    return tuple(_srgb_to_linear(c) for c in srgb)  # type: ignore[return-value]


# ── Resolution ──────────────────────────────────────────────────────────

def resolve(
    name_or_entry: Union[str, Mapping[str, Any]],
    channel: str,
    *,
    _stack: Optional[Tuple[str, ...]] = None,
) -> ChannelValue:
    """Resolve one channel for one entry, with provenance.

    Walks the channel's configured source chain and returns the first source
    that produces a value. Formula sources may reference other channels;
    cycles are detected and that source is skipped rather than recursing.
    """
    spec = channel_spec(channel)
    entry = _entry_of(name_or_entry)
    unit = str(spec.get("unit", ""))
    stack = _stack or ()

    if channel in stack:
        return ChannelValue(channel, None, unit, 0.0, "cycle",
                            f"cycle via {' -> '.join(stack + (channel,))}")

    flat = property_sheet(entry)
    values = si_sheet(entry)
    constants = config().get("constants") or {}

    for source in spec.get("sources") or ():
        kind = source.get("kind")
        confidence = float(source.get("confidence", 0.0))
        note = str(source.get("note", ""))

        if kind == "property":
            canonical = source.get("property")
            value = values.get(canonical)
            if value is not None:
                return ChannelValue(channel, float(value), unit, confidence,
                                    f"property:{canonical}", note)

        elif kind == "formula":
            expr = source.get("expr") or ""
            variables: Dict[str, float] = dict(constants)
            # Inputs this source explicitly accepts a default for. Some
            # constants are placeholders (we have no idea) and some are
            # genuine engineering conventions -- Poisson's ratio of 0.3 for a
            # metal is the second kind, and refusing to build on it would
            # throw away a far better estimate than the alternative. The
            # config has to say which, per input, and the confidence rating
            # still carries the weakness.
            allow_defaults = {str(n) for n in (source.get("allow_defaults") or ())}
            ok = True
            for input_name in source.get("inputs") or ():
                direct = values.get(input_name)
                if direct is not None:
                    variables[input_name] = float(direct)
                    continue
                # Not a property on this sheet -- try it as another channel.
                if input_name in config()["channels"]:
                    nested = resolve(entry, input_name, _stack=stack + (channel,))
                    # Constants are terminal, never ingredients. Feeding a
                    # defaulted value into a formula manufactures a precise-
                    # looking number out of nothing: Iron has no resistivity on
                    # its sheet, so a Wiedemann-Franz conductivity built on the
                    # insulating default came out 7e-18 W/mK. Skipping the
                    # source lets the chain fall through to this channel's own
                    # documented default instead, which is at least honest
                    # about being one.
                    usable = isinstance(nested.value, float) and (
                        not nested.is_default or input_name in allow_defaults
                    )
                    if usable:
                        variables[input_name] = nested.value  # type: ignore[assignment]
                        confidence = min(confidence, nested.confidence
                                         if not nested.is_default else confidence)
                        continue
                ok = False
                break
            if not ok:
                continue
            try:
                value = _safe_eval(expr, variables)
            except (KeyError, ValueError, ZeroDivisionError, OverflowError):
                continue
            if math.isfinite(value):
                return ChannelValue(channel, value, unit, confidence,
                                    f"formula:{expr}", note)

        elif kind == "mixture":
            canonical = source.get("property")
            value = _mixture(entry, str(canonical), str(source.get("weighting", "volume")))
            if value is not None and math.isfinite(value):
                return ChannelValue(
                    channel, value, unit, confidence,
                    f"mixture:{source.get('weighting', 'volume')}:{canonical}", note,
                )

        elif kind == "colour_hex":
            for field in source.get("fields") or ():
                raw = entry.get(field) or flat.get(field)
                if raw is None:
                    continue
                rgb = _hex_to_linear_rgb(raw)
                if rgb is not None:
                    return ChannelValue(channel, rgb, unit, confidence,
                                        f"colour_hex:{field}", note)

        elif kind == "colour_wavelength":
            wavelength = values.get(source.get("property"))
            if wavelength:
                rgb = _wavelength_to_linear_rgb(float(wavelength))
                if rgb is not None:
                    return ChannelValue(channel, rgb, unit, confidence,
                                        f"colour_wavelength:{source.get('property')}", note)

        elif kind == "colour_grey":
            nested = resolve(entry, str(source.get("channel")), _stack=stack + (channel,))
            if isinstance(nested.value, float):
                grey = max(0.0, min(1.0, nested.value))
                return ChannelValue(channel, (grey, grey, grey), unit,
                                    min(confidence, nested.confidence),
                                    f"colour_grey:{source.get('channel')}", note)

        elif kind == "constant":
            value = source.get("value")
            if value is not None:
                return ChannelValue(channel, float(value), unit, confidence,
                                    f"constant:{value}", note)

        else:
            raise ValueError(f"Unknown channel source kind {kind!r} for {channel!r}")

    return ChannelValue(channel, None, unit, 0.0, "unresolved",
                        "no source in the chain produced a value")


def resolve_all(
    name_or_entry: Union[str, Mapping[str, Any]],
    channels: Union[str, Sequence[str], None] = None,
) -> Dict[str, ChannelValue]:
    """Resolve several channels at once. Accepts channel or group names."""
    entry = _entry_of(name_or_entry)
    return {name: resolve(entry, name) for name in expand_channels(channels)}


def channel(
    name_or_entry: Union[str, Mapping[str, Any]],
    channel_name: str,
) -> Union[float, Tuple[float, float, float], None]:
    """Just the value of a channel, for callers that do not want provenance."""
    return resolve(name_or_entry, channel_name).value


__all__ = [
    "ChannelValue",
    "UnknownChannel",
    "channel",
    "channel_groups",
    "channel_spec",
    "config",
    "expand_channels",
    "list_channels",
    "reload_config",
    "resolve",
    "resolve_all",
]
