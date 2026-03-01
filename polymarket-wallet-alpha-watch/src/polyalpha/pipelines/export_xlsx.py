from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font
from openpyxl.worksheet.worksheet import Worksheet

from polyalpha.pipelines.compute_pnl import WalletPerformance


def run_export_xlsx(
    *,
    wallet_rows: list[WalletPerformance],
    out_path: Path,
    evidence_rows: list[dict[str, object]] | None = None,
    memo_rows: list[dict[str, object]] | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ordered_rows = sorted(
        wallet_rows,
        key=lambda row: (-row.total_pnl, -row.realized_pnl, -row.total_invested, row.wallet_address),
    )
    table_rows = [
        {
            "Rank": i + 1,
            "Name": row.name,
            "Wallet Address": row.wallet_address,
            "Tags": ", ".join(row.tags),
            "Wins": row.wins,
            "Losses": row.losses,
            "Win Rate": row.win_rate,
            "Total P&L ($)": row.total_pnl,
            "Total Invested ($)": row.total_invested,
        }
        for i, row in enumerate(ordered_rows)
    ]
    primary_df = pd.DataFrame(
        table_rows,
        columns=[
            "Rank",
            "Name",
            "Wallet Address",
            "Tags",
            "Wins",
            "Losses",
            "Win Rate",
            "Total P&L ($)",
            "Total Invested ($)",
        ],
    )

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        primary_df.to_excel(writer, sheet_name="HighSignalWallets", index=False)
        summary_sheet = writer.book["HighSignalWallets"]
        _format_primary_sheet(summary_sheet)

        if evidence_rows:
            evidence_df = pd.DataFrame(evidence_rows)
            evidence_df.to_excel(writer, sheet_name="Evidence", index=False)
            _format_evidence_sheet(writer.book["Evidence"])

        if memo_rows:
            memo_df = pd.DataFrame(memo_rows)
            memo_df.to_excel(writer, sheet_name="Memos", index=False)
            _format_memo_sheet(writer.book["Memos"])


def build_evidence_rows(wallet_rows: list[WalletPerformance]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for wallet in wallet_rows:
        if not wallet.top_markets:
            rows.append(
                {
                    "Wallet Address": wallet.wallet_address,
                    "Name": wallet.name,
                    "Market Condition ID": "",
                    "Market Question": "",
                    "Invested ($)": wallet.total_invested,
                    "Realized P&L ($)": wallet.realized_pnl,
                    "Unrealized P&L ($)": wallet.unrealized_pnl,
                    "Total P&L ($)": wallet.total_pnl,
                    "Trade Count": wallet.trade_count,
                }
            )
            continue

        for market in wallet.top_markets:
            rows.append(
                {
                    "Wallet Address": wallet.wallet_address,
                    "Name": wallet.name,
                    "Market Condition ID": market.condition_id,
                    "Market Question": market.question or "",
                    "Invested ($)": market.invested,
                    "Realized P&L ($)": market.realized_pnl,
                    "Unrealized P&L ($)": market.unrealized_pnl,
                    "Total P&L ($)": market.total_pnl,
                    "Trade Count": wallet.trade_count,
                }
            )
    return rows


def _format_primary_sheet(sheet: Worksheet) -> None:
    sheet.freeze_panes = "A2"
    _apply_header_style(sheet)
    _set_column_widths(
        sheet,
        {
            "A": 8,
            "B": 24,
            "C": 46,
            "D": 28,
            "E": 10,
            "F": 10,
            "G": 12,
            "H": 16,
            "I": 18,
        },
    )
    _format_columns(sheet, percent_cols={"G"}, currency_cols={"H", "I"})


def _format_evidence_sheet(sheet: Worksheet) -> None:
    sheet.freeze_panes = "A2"
    _apply_header_style(sheet)
    _set_column_widths(
        sheet,
        {
            "A": 46,
            "B": 24,
            "C": 34,
            "D": 60,
            "E": 14,
            "F": 16,
            "G": 17,
            "H": 14,
            "I": 12,
        },
    )
    _format_columns(sheet, percent_cols=set(), currency_cols={"E", "F", "G", "H"})


def _format_memo_sheet(sheet: Worksheet) -> None:
    sheet.freeze_panes = "A2"
    _apply_header_style(sheet)
    _auto_fit_columns(sheet, min_width=18, max_width=70)
    for column in ("C", "D", "E", "F"):
        for cell in sheet[column]:
            cell.alignment = Alignment(wrap_text=True, vertical="top")


def _apply_header_style(sheet: Worksheet) -> None:
    for cell in sheet[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center")


def _set_column_widths(sheet: Worksheet, widths: dict[str, int]) -> None:
    for column, width in widths.items():
        sheet.column_dimensions[column].width = width


def _auto_fit_columns(sheet: Worksheet, *, min_width: int, max_width: int) -> None:
    for column_cells in sheet.columns:
        column = column_cells[0].column_letter
        best = max((len(str(cell.value)) if cell.value is not None else 0) for cell in column_cells)
        sheet.column_dimensions[column].width = min(max(best + 2, min_width), max_width)


def _format_columns(sheet: Worksheet, *, percent_cols: set[str], currency_cols: set[str]) -> None:
    for column in percent_cols:
        for cell in sheet[column][1:]:
            if isinstance(cell.value, (int, float)):
                cell.number_format = "0.00%"
    for column in currency_cols:
        for cell in sheet[column][1:]:
            if isinstance(cell.value, (int, float)):
                cell.number_format = '"$"#,##0.00'
