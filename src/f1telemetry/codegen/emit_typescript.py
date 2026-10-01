"""Emit the TypeScript channel types for the dashboard (P0-T5).

``web/src/generated/channels.ts`` is the only file under ``web/src`` this repository
owns at P0. It is generated, committed, and never hand-edited; the app imports from it
rather than restating any of it.
"""

from __future__ import annotations

from f1telemetry.codegen._emit import TS_TYPES, render_ts, ts_str
from f1telemetry.contracts.channels import CORNERS, FAULT_TYPES, Channel, ChannelContract

__all__ = ["emit_typescript_channels"]

_INDENT = "  "


def _channel_literal(channel: Channel) -> str:
    quant = channel.quantisation
    quant_literal = (
        "null"
        if quant is None
        else f"{{ bits: {quant.bits}, fullScale: {quant.full_scale!r}, step: {quant.step!r} }}"
    )
    return (
        "{\n"
        f"{_INDENT * 3}name: {ts_str(channel.name)},\n"
        f"{_INDENT * 3}baseName: {ts_str(channel.base_name)},\n"
        f"{_INDENT * 3}group: {ts_str(channel.group)},\n"
        f"{_INDENT * 3}unit: {ts_str(channel.unit)},\n"
        f"{_INDENT * 3}dtype: {ts_str(channel.dtype)},\n"
        f"{_INDENT * 3}rateHz: {_ts_number(channel.rate_hz)},\n"
        f"{_INDENT * 3}event: {_ts_bool(channel.event)},\n"
        f"{_INDENT * 3}range: [{channel.range_min!r}, {channel.range_max!r}],\n"
        f"{_INDENT * 3}noiseModel: {ts_str(channel.noise_model)},\n"
        f"{_INDENT * 3}sigma: {channel.sigma!r},\n"
        f"{_INDENT * 3}quantise: {quant_literal},\n"
        f"{_INDENT * 3}faultEligible: ["
        + ", ".join(ts_str(f) for f in channel.fault_eligible)
        + "],\n"
        f"{_INDENT * 3}corners: [" + ", ".join(ts_str(c) for c in channel.corners) + "],\n"
        f"{_INDENT * 3}corner: {_ts_null_str(channel.corner)},\n"
        f"{_INDENT * 3}description: {ts_str(channel.description)},\n"
        f"{_INDENT * 3}source: {ts_str(channel.source)},\n"
        f"{_INDENT * 2}}}"
    )


def _ts_number(value: float | None) -> str:
    return "null" if value is None else repr(float(value))


def _ts_bool(value: bool) -> str:
    return "true" if value else "false"


def _ts_null_str(value: str | None) -> str:
    return "null" if value is None else ts_str(value)


def emit_typescript_channels(contract: ChannelContract) -> str:
    parts: list[str] = []
    parts.append(
        "export const CONTRACT_VERSION = " + ts_str(contract.version) + " as const\n\n"
        "export const CORNERS = [" + ", ".join(f"'{c}'" for c in CORNERS) + "] as const\n"
        "export type Corner = (typeof CORNERS)[number]\n\n"
        "export const GROUPS = [" + ", ".join(f"'{g}'" for g in contract.groups) + "] as const\n"
        "export type ChannelGroup = (typeof GROUPS)[number]\n\n"
        "export const FAULT_TYPES = [\n"
        + "".join(f"  '{f}',\n" for f in FAULT_TYPES)
        + "] as const\n"
        "export type FaultType = (typeof FAULT_TYPES)[number]\n"
    )

    parts.append(
        "export type NoiseModel = 'gaussian' | 'uniform' | 'none'\n"
        "export type Dtype =\n" + "".join(f"  | '{d}'\n" for d in TS_TYPES) + "\n"
        "export interface Quantisation {\n"
        "  bits: number\n"
        "  fullScale: number\n"
        "  step: number\n"
        "}\n\n"
        "export interface ChannelSpec {\n"
        "  name: ChannelName\n"
        "  baseName: string\n"
        "  group: ChannelGroup\n"
        "  unit: string\n"
        "  dtype: Dtype\n"
        "  rateHz: number | null\n"
        "  event: boolean\n"
        "  range: [number, number]\n"
        "  noiseModel: NoiseModel\n"
        "  sigma: number\n"
        "  quantise: Quantisation | null\n"
        "  faultEligible: readonly FaultType[]\n"
        "  corners: readonly Corner[]\n"
        "  corner: Corner | null\n"
        "  description: string\n"
        "  source: string\n"
        "}\n"
    )

    channel_names = " | ".join(f"'{n}'" for n in contract.names)
    parts.append(f"export type ChannelName = {channel_names}\n")
    corner_names = " | ".join(f"'{c.name}'" for c in contract.channels if c.is_corner_channel)
    parts.append(f"export type CornerChannelName = {corner_names}\n")
    sample_type = "number"

    parts.append(
        "export const CHANNELS: { readonly [K in ChannelName]: ChannelSpec } = {\n"
        + "".join(f"{_INDENT}{c.name}: {_channel_literal(c)},\n" for c in contract.channels)
        + "}\n"
    )

    parts.append(
        "export const CHANNEL_NAMES = [\n"
        + "".join(f"  '{c.name}',\n" for c in contract.channels)
        + "] as const satisfies readonly ChannelName[]\n"
    )

    parts.append(
        "export type GroupChannels = { readonly [K in ChannelGroup]: readonly ChannelName[] }\n"
        "export const CHANNELS_BY_GROUP: GroupChannels = {\n"
        + "".join(
            f"{_INDENT}{group}: ["
            + ", ".join(f"'{c.name}'" for c in contract.group_members(group))
            + "],\n"
            for group in contract.groups
        )
        + "}\n"
    )

    for group in contract.groups:
        members = contract.group_members(group)
        parts.append(
            f"export interface {''.join(p.capitalize() for p in group.split('_'))}Channels {{\n"
            + "".join(f"  {c.name}: {sample_type}\n" for c in members)
            + "}\n"
        )

    parts.append(
        "export function channelsForGroup(group: ChannelGroup): readonly ChannelSpec[] {\n"
        "  return CHANNELS_BY_GROUP[group].map((name) => CHANNELS[name])\n"
        "}\n\n"
        "export function cornerChannels(baseName: string): readonly ChannelSpec[] {\n"
        "  return CHANNEL_NAMES.map((name) => CHANNELS[name]).filter(\n"
        "    (channel) => channel.baseName === baseName,\n"
        "  )\n"
        "}\n\n"
        "export function channelRateHz(name: ChannelName): number | null {\n"
        "  return CHANNELS[name].rateHz\n"
        "}\n"
    )

    return render_ts("\n".join(parts).rstrip() + "\n")
