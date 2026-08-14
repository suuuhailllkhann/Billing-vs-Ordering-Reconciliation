"""Canonical post-ingestion schemas used by reconciliation and analytics."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldDefinition:
    name: str
    kind: str
    required: bool
    description: str


BILLING_FIELDS = (
    FieldDefinition("billing_id", "identifier", True, "Unique source billing transaction identifier."),
    FieldDefinition("patient_id", "identifier", True, "Internal patient identifier; never an ML feature by default."),
    FieldDefinition("patient_name", "identity/display", False, "Fictional in repository examples; not an ML feature by default."),
    FieldDefinition("date_of_birth", "date/identity", False, "Identity/display date; not an ML feature by default."),
    FieldDefinition("prescription_id", "identifier", True, "Prescription or fill identifier supplied by the source."),
    FieldDefinition("ndc", "identifier/string", True, "Medication code preserved as supplied after whitespace trimming."),
    FieldDefinition("drug_name", "categorical/string", True, "Medication display name; exact matching only."),
    FieldDefinition("billing_date", "date", True, "Date used for inclusive period filtering."),
    FieldDefinition("insurance_name", "categorical/string", False, "Billing insurer; missing values become UNKNOWN in insurance analytics."),
    FieldDefinition("bin_number", "categorical/string", False, "Billing BIN; missing values become UNKNOWN in insurance analytics."),
    FieldDefinition("quantity_billed", "numeric", True, "Non-negative billed quantity."),
    FieldDefinition("days_supply", "numeric/integer", False, "Optional non-negative whole-number supply duration."),
    FieldDefinition("refills_remaining", "numeric/integer", False, "Optional non-negative whole-number refill count."),
)

ORDERING_FIELDS = (
    FieldDefinition("order_id", "identifier", True, "Unique source order transaction identifier."),
    FieldDefinition("ndc", "identifier/string", True, "Medication code preserved as supplied after whitespace trimming."),
    FieldDefinition("drug_name", "categorical/string", True, "Medication display name; exact matching only."),
    FieldDefinition("ordered_date", "date", True, "Date used for inclusive period filtering."),
    FieldDefinition("quantity_ordered", "numeric", True, "Non-negative ordered quantity."),
)

BILLING_COLUMNS = tuple(field.name for field in BILLING_FIELDS)
ORDERING_COLUMNS = tuple(field.name for field in ORDERING_FIELDS)
BILLING_REQUIRED_COLUMNS = tuple(field.name for field in BILLING_FIELDS if field.required)
ORDERING_REQUIRED_COLUMNS = tuple(field.name for field in ORDERING_FIELDS if field.required)

