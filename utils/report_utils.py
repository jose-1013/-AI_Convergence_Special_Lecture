from __future__ import annotations

from typing import Any


def _format_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_fixed_width_table(
    *,
    rows: list[dict[str, Any]],
    columns: tuple[str, ...],
) -> str:
    string_rows = [
        {column: _format_cell(row.get(column)) for column in columns}
        for row in rows
    ]
    widths = {
        column: max([len(column), *(len(row[column]) for row in string_rows)])
        for column in columns
    }
    header = " | ".join(column.ljust(widths[column]) for column in columns)
    separator = "-+-".join("-" * widths[column] for column in columns)
    body = [
        " | ".join(row[column].ljust(widths[column]) for column in columns)
        for row in string_rows
    ]
    return "\n".join([header, separator, *body]) if body else "\n".join([header, separator])


__all__ = ["build_fixed_width_table"]
