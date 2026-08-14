"""Deterministic source-to-canonical column mapping with conflict reporting."""

from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Sequence

from pharmacy_reconciliation.ingestion.normalization import normalize_header
from pharmacy_reconciliation.ingestion.profiles import BILLING_ALIASES, ORDERING_ALIASES
from pharmacy_reconciliation.ingestion.schemas import (
    BILLING_COLUMNS,
    BILLING_REQUIRED_COLUMNS,
    ORDERING_COLUMNS,
    ORDERING_REQUIRED_COLUMNS,
)


class MappingStatus(str, Enum):
    EXACT = "EXACT"
    ALIAS = "ALIAS"
    MANUAL = "MANUAL"
    AMBIGUOUS = "AMBIGUOUS"
    UNMAPPED = "UNMAPPED"


class ManualMappingError(ValueError):
    """Raised when a human-supplied mapping is structurally invalid."""


@dataclass(frozen=True)
class ColumnMapping:
    source_column: str
    normalized_source_column: str
    canonical_field: str | None
    status: MappingStatus
    candidates: tuple[str, ...] = ()
    note: str | None = None


@dataclass(frozen=True)
class MappingConflict:
    code: str
    source_columns: tuple[str, ...]
    canonical_fields: tuple[str, ...]
    message: str


@dataclass(frozen=True)
class MappingResult:
    dataset_type: str
    columns: tuple[ColumnMapping, ...]
    required_fields_missing: tuple[str, ...]
    conflicts: tuple[MappingConflict, ...]

    @property
    def is_unambiguous(self) -> bool:
        return not self.conflicts and not self.required_fields_missing

    @property
    def mapped_count(self) -> int:
        return sum(item.status in {MappingStatus.EXACT, MappingStatus.ALIAS, MappingStatus.MANUAL} for item in self.columns)

    @property
    def ambiguous_count(self) -> int:
        return sum(item.status == MappingStatus.AMBIGUOUS for item in self.columns)

    @property
    def unmapped_count(self) -> int:
        return sum(item.status == MappingStatus.UNMAPPED for item in self.columns)

    @property
    def resolved_mapping(self) -> dict[str, str]:
        return {
            item.source_column: item.canonical_field
            for item in self.columns
            if item.canonical_field is not None
            and item.status in {MappingStatus.EXACT, MappingStatus.ALIAS, MappingStatus.MANUAL}
        }


@dataclass(frozen=True)
class MappingProfile:
    dataset_type: str
    canonical_fields: tuple[str, ...]
    required_fields: tuple[str, ...]
    aliases: Mapping[str, tuple[str, ...]]


PROFILES = MappingProxyType({
    "billing": MappingProfile("billing", BILLING_COLUMNS, BILLING_REQUIRED_COLUMNS, BILLING_ALIASES),
    "ordering": MappingProfile("ordering", ORDERING_COLUMNS, ORDERING_REQUIRED_COLUMNS, ORDERING_ALIASES),
})


def _profile(dataset_type: str) -> MappingProfile:
    try:
        return PROFILES[dataset_type]
    except KeyError as exc:
        raise ValueError("dataset_type must be `billing` or `ordering`.") from exc


def _candidate_registry(profile: MappingProfile) -> dict[str, list[tuple[str, MappingStatus]]]:
    registry: dict[str, list[tuple[str, MappingStatus]]] = {}
    for canonical in profile.canonical_fields:
        registry.setdefault(normalize_header(canonical), []).append((canonical, MappingStatus.EXACT))
        for alias in profile.aliases.get(canonical, ()):
            registry.setdefault(normalize_header(alias), []).append((canonical, MappingStatus.ALIAS))
    return registry


def _resolve_manual_sources(
    source_columns: tuple[str, ...],
    manual_mapping: Mapping[str, str],
    profile: MappingProfile,
) -> dict[int, str]:
    resolved: dict[int, str] = {}
    normalized = [normalize_header(column) for column in source_columns]
    for requested_source, destination in manual_mapping.items():
        if destination not in profile.canonical_fields:
            raise ManualMappingError(
                f"Manual destination `{destination}` is not a canonical {profile.dataset_type} field."
            )
        exact_positions = [index for index, column in enumerate(source_columns) if column == requested_source]
        positions = exact_positions or [
            index for index, value in enumerate(normalized)
            if value == normalize_header(requested_source)
        ]
        if not positions:
            raise ManualMappingError(f"Manual source column `{requested_source}` does not exist.")
        if len(positions) > 1:
            raise ManualMappingError(
                f"Manual source `{requested_source}` matches multiple source columns; use unique headers first."
            )
        index = positions[0]
        if index in resolved and resolved[index] != destination:
            raise ManualMappingError(f"Source column `{source_columns[index]}` has multiple manual destinations.")
        resolved[index] = destination
    return resolved


def map_columns(
    source_columns: Sequence[object],
    dataset_type: str,
    manual_mapping: Mapping[str, str] | None = None,
) -> MappingResult:
    """Map headers without fuzzy matching or silent conflict resolution."""
    profile = _profile(dataset_type)
    sources = tuple(str(column) for column in source_columns)
    normalized = tuple(normalize_header(column) for column in sources)
    manual = _resolve_manual_sources(sources, manual_mapping or {}, profile)
    registry = _candidate_registry(profile)
    mappings: list[ColumnMapping] = []
    conflicts: list[MappingConflict] = []

    duplicate_keys = {key for key in normalized if normalized.count(key) > 1}
    for key in sorted(duplicate_keys):
        duplicate_sources = tuple(source for source, value in zip(sources, normalized) if value == key)
        conflicts.append(MappingConflict(
            "duplicate_normalized_header", duplicate_sources, (),
            f"Multiple source columns normalize to `{key}`.",
        ))

    manually_claimed = set(manual.values())
    for index, (source, normalized_source) in enumerate(zip(sources, normalized)):
        if normalized_source in duplicate_keys:
            mappings.append(ColumnMapping(
                source, normalized_source, None, MappingStatus.AMBIGUOUS, (),
                "Duplicate normalized source header.",
            ))
            continue
        if index in manual:
            destination = manual[index]
            mappings.append(ColumnMapping(
                source, normalized_source, destination, MappingStatus.MANUAL, (destination,),
                "Human-confirmed override.",
            ))
            continue

        candidates = registry.get(normalized_source, [])
        unique_destinations = tuple(dict.fromkeys(candidate[0] for candidate in candidates))
        if len(unique_destinations) > 1:
            mappings.append(ColumnMapping(
                source, normalized_source, None, MappingStatus.AMBIGUOUS,
                tuple(sorted(unique_destinations)), "Source alias matches multiple canonical fields.",
            ))
            conflicts.append(MappingConflict(
                "source_maps_to_multiple_fields", (source,), tuple(sorted(unique_destinations)),
                f"Source column `{source}` could map to multiple canonical fields.",
            ))
        elif len(unique_destinations) == 1:
            destination = unique_destinations[0]
            if destination in manually_claimed:
                mappings.append(ColumnMapping(
                    source, normalized_source, None, MappingStatus.UNMAPPED, (destination,),
                    f"Ignored because a manual mapping claims `{destination}`.",
                ))
            else:
                status = next(candidate[1] for candidate in candidates if candidate[0] == destination)
                mappings.append(ColumnMapping(source, normalized_source, destination, status, (destination,)))
        else:
            mappings.append(ColumnMapping(source, normalized_source, None, MappingStatus.UNMAPPED))

    destination_sources: dict[str, list[int]] = {}
    for index, item in enumerate(mappings):
        if item.canonical_field is not None:
            destination_sources.setdefault(item.canonical_field, []).append(index)
    for destination, positions in destination_sources.items():
        if len(positions) <= 1:
            continue
        source_names = tuple(mappings[position].source_column for position in positions)
        conflicts.append(MappingConflict(
            "multiple_sources_for_field", source_names, (destination,),
            f"Multiple source columns map to canonical field `{destination}`.",
        ))
        for position in positions:
            mappings[position] = replace(
                mappings[position], canonical_field=None, status=MappingStatus.AMBIGUOUS,
                candidates=(destination,), note="Multiple sources target the same canonical field.",
            )

    mapped_destinations = {
        item.canonical_field for item in mappings if item.canonical_field is not None
    }
    missing = tuple(field for field in profile.required_fields if field not in mapped_destinations)
    return MappingResult(profile.dataset_type, tuple(mappings), missing, tuple(conflicts))

