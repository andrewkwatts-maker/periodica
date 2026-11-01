"""The sheet contract a downstream app can actually build on.

An app consuming raw registry JSON has to answer four questions per property,
every time: is it there, what is it called here, what unit is it in, and can I
trust it. `EngineSheet` answers all four once, at the boundary::

    sheet = engine_sheet("Stainless_Steel_316L")

    sheet["density"]          # 7990.0  -- always kg/m3, always present
    sheet.confidence("density")   # 1.0
    sheet.method("density")       # 'property:density'
    sheet.weakest()               # the channels least supported by data

Three guarantees hold for every entry in the registry:

**Complete.** Every channel has a value. The chains in
``engine_channels.json`` end in a documented constant, so a consumer never
writes ``if value is None``. Completeness is not the same as knowing -- which
is what the next guarantee is for.

**Graded.** Each value carries the confidence of the method that produced it,
1.0 for read-from-sheet down to 0.1 for a bare default, and the method string
itself. An app that must not run on invented numbers sets a floor::

    engine_sheet("Iron", min_confidence=0.5)          # -> raises
    engine_sheet("Iron", min_confidence=0.5, strict=False).below_floor()

**Canonical.** One name and one SI unit per quantity, whatever the source JSON
called it. ``sheet["stiffness"]`` is pascals whether the sheet stored GPa, MPa,
or nothing at all and it came from the constituents.

The point of the split between `periodica.properties` and this module: that one
normalises *what the data says*, faithfully, keys and units intact. This one
normalises *what a consumer needs*, completely, and is allowed to derive and to
default -- as long as it says so.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

from periodica.engine.channels import (
    ChannelValue,
    expand_channels,
    list_channels,
    resolve_all,
)
from periodica.get import Get
from periodica.properties import si_sheet, validate_sheet


class SheetBelowFloor(ValueError):
    """A requested confidence floor was not met by every channel."""

    def __init__(self, name: str, floor: float, offenders: Sequence[ChannelValue]) -> None:
        detail = ", ".join(
            f"{cv.channel}={cv.confidence:.2f} ({cv.method})" for cv in offenders
        )
        super().__init__(
            f"{name}: {len(offenders)} channel(s) below confidence {floor:.2f}: {detail}"
        )
        self.name = name
        self.floor = floor
        self.offenders = list(offenders)


class EngineSheet(Mapping[str, Any]):
    """A complete, graded, canonical view of one entry for engine consumption.

    Reads as a mapping of channel name to value. Values are SI floats, except
    ``colour`` which is a linear-sRGB triple.
    """

    __slots__ = ("name", "entry", "_channels", "_floor")

    def __init__(
        self,
        name: str,
        entry: Mapping[str, Any],
        channels: Mapping[str, ChannelValue],
        *,
        min_confidence: float = 0.0,
    ) -> None:
        self.name = name
        self.entry = dict(entry)
        self._channels = dict(channels)
        self._floor = float(min_confidence)

    # -- Mapping --

    def __getitem__(self, channel: str) -> Any:
        return self._channels[channel].value

    def __iter__(self) -> Iterator[str]:
        return iter(self._channels)

    def __len__(self) -> int:
        return len(self._channels)

    def __repr__(self) -> str:
        return f"<EngineSheet {self.name!r}: {len(self._channels)} channels>"

    # -- Provenance --

    def resolved(self, channel: str) -> ChannelValue:
        """The full `ChannelValue`, with unit, confidence and method."""
        return self._channels[channel]

    def confidence(self, channel: str) -> float:
        """How far from the data this channel's value is. 1.0 = read directly."""
        return self._channels[channel].confidence

    def method(self, channel: str) -> str:
        """How the value was obtained, e.g. ``formula:...`` or ``constant:0.04``."""
        return self._channels[channel].method

    def unit(self, channel: str) -> str:
        return self._channels[channel].unit

    def measured(self) -> List[str]:
        """Channels read straight off the sheet."""
        return [c for c, v in self._channels.items() if v.is_measured]

    def defaulted(self) -> List[str]:
        """Channels that nothing in the data supported."""
        return [c for c, v in self._channels.items() if v.is_default]

    def below_floor(self, floor: Optional[float] = None) -> List[ChannelValue]:
        """Channels whose confidence is under `floor` (default: the sheet's)."""
        limit = self._floor if floor is None else float(floor)
        return [v for v in self._channels.values() if v.confidence < limit]

    def weakest(self, count: int = 5) -> List[ChannelValue]:
        """The `count` least-supported channels, weakest first."""
        return sorted(self._channels.values(), key=lambda v: v.confidence)[:count]

    @property
    def mean_confidence(self) -> float:
        """Average confidence across channels -- a one-number sheet grade."""
        if not self._channels:
            return 0.0
        return sum(v.confidence for v in self._channels.values()) / len(self._channels)

    # -- Export --

    def as_dict(self, *, provenance: bool = True) -> dict:
        """A JSON-serialisable form, with or without the provenance detail."""
        if not provenance:
            return {
                "name": self.name,
                "channels": {
                    c: (list(v.value) if isinstance(v.value, tuple) else v.value)
                    for c, v in self._channels.items()
                },
            }
        return {
            "name": self.name,
            "mean_confidence": self.mean_confidence,
            "channels": {c: v.as_dict() for c, v in self._channels.items()},
        }

    def to_json(self, *, provenance: bool = True, indent: int = 2) -> str:
        return json.dumps(self.as_dict(provenance=provenance), indent=indent)

    def uniforms(self) -> Dict[str, Union[float, Tuple[float, float, float]]]:
        """Flat name -> value map for a shader uniform block or constant buffer.

        Names are prefixed so they cannot collide with an engine's own
        uniforms, and colour expands to a float3.
        """
        out: Dict[str, Union[float, Tuple[float, float, float]]] = {}
        for name, value in self._channels.items():
            key = f"pd_{name}"
            out[key] = value.value if value.value is not None else 0.0
        return out


def engine_sheet(
    name_or_entry: Union[str, Mapping[str, Any]],
    channels: Union[str, Sequence[str], None] = None,
    *,
    min_confidence: float = 0.0,
    strict: bool = True,
) -> EngineSheet:
    """Build the engine-facing sheet for one entry.

    `channels` takes channel names, group names ("mechanical", "thermal",
    "optical", "bulk") or None for all of them.

    `min_confidence` records the floor the caller needs. With `strict` (the
    default) a sheet that cannot meet it raises `SheetBelowFloor` naming every
    offending channel and how it was obtained, which is far more use than a
    silent low-confidence value that reaches a physics solver. With
    ``strict=False`` the sheet is returned and `below_floor()` lists them.
    """
    if isinstance(name_or_entry, str):
        name = name_or_entry
        entry = Get(name_or_entry)
    else:
        entry = dict(name_or_entry)
        name = str(entry.get("Name") or entry.get("name") or "<entry>")

    resolved = resolve_all(entry, channels)
    sheet = EngineSheet(name, entry, resolved, min_confidence=min_confidence)

    if strict and min_confidence > 0:
        offenders = sheet.below_floor()
        if offenders:
            raise SheetBelowFloor(name, min_confidence, offenders)
    return sheet


def validate(
    name_or_entry: Union[str, Mapping[str, Any]],
    *,
    min_confidence: float = 0.0,
) -> dict:
    """Report on one entry's fitness for engine consumption.

    Combines the schema range checks from `periodica.properties` with channel
    coverage, so one call answers "can I ship this material":

        {"name", "ok", "errors", "warnings", "channels_total",
         "channels_measured", "channels_defaulted", "below_floor",
         "mean_confidence"}

    `ok` is False when the entry has a physically impossible property value or
    any channel under the floor.
    """
    if isinstance(name_or_entry, str):
        name = name_or_entry
        entry = Get(name_or_entry)
    else:
        entry = dict(name_or_entry)
        name = str(entry.get("Name") or entry.get("name") or "<entry>")

    issues = validate_sheet(entry, name=name)
    errors = [str(i) for i in issues if i.severity == "error"]
    warnings = [str(i) for i in issues if i.severity == "warning"]

    sheet = engine_sheet(entry, strict=False, min_confidence=min_confidence)
    below = [cv.channel for cv in sheet.below_floor()]

    return {
        "name": name,
        "ok": not errors and not below,
        "errors": errors,
        "warnings": warnings,
        "properties_recognised": len(si_sheet(entry)),
        "channels_total": len(sheet),
        "channels_measured": len(sheet.measured()),
        "channels_defaulted": len(sheet.defaulted()),
        "below_floor": below,
        "mean_confidence": sheet.mean_confidence,
    }


__all__ = [
    "EngineSheet",
    "SheetBelowFloor",
    "engine_sheet",
    "list_channels",
    "validate",
]
