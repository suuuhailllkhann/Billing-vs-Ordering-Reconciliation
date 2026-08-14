"""PySide6 desktop client for the authoritative reconciliation backend."""

from html import escape
from pathlib import Path

import pandas as pd
from PySide6.QtCore import QDate, Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QPageSize, QStandardItem, QStandardItemModel, QTextDocument
from PySide6.QtPrintSupport import QPrinter
from PySide6.QtWidgets import (
    QComboBox,
    QDateEdit,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from pharmacy_reconciliation.application.controller import ReconciliationController, WorkflowError
from pharmacy_reconciliation.ingestion.loaders import UnsupportedFileTypeError
from pharmacy_reconciliation.ingestion.mapping import ManualMappingError, MappingStatus
from pharmacy_reconciliation.ingestion.schemas import BILLING_COLUMNS, ORDERING_COLUMNS


class PandasTableModel(QStandardItemModel):
    def __init__(self, frame: pd.DataFrame):
        super().__init__()
        self.setColumnCount(len(frame.columns))
        self.setHorizontalHeaderLabels(frame.columns.tolist())
        for row in frame.itertuples(index=False):
            items = []
            for column_index, value in enumerate(row):
                display = "" if pd.isna(value) else str(value)
                item = QStandardItem(display)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if frame.columns[column_index] == "status":
                    colors = {"MATCHED": "#c6f6d5", "SHORT": "#feb2b2", "EXTRA": "#fef3c7"}
                    if value in colors:
                        item.setBackground(QColor(colors[value]))
                items.append(item)
            self.appendRow(items)


class ManualMappingDialog(QDialog):
    """Human-confirmation dialog; all mapping decisions remain in the backend."""

    def __init__(self, ingestion, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Resolve {ingestion.dataset_type.title()} Column Mapping")
        self.resize(760, 420)
        layout = QVBoxLayout(self)
        explanation = QLabel(
            "Select a canonical destination only when you recognize the source column. "
            "Leave unrelated columns as Ignore. Ambiguous fields are never selected automatically."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        unresolved = [item for item in ingestion.mapping.columns if item.status in {
            MappingStatus.AMBIGUOUS, MappingStatus.UNMAPPED,
        }]
        canonical = BILLING_COLUMNS if ingestion.dataset_type == "billing" else ORDERING_COLUMNS
        self.table = QTableWidget(len(unresolved), 4)
        self.table.setHorizontalHeaderLabels(["Source column", "Normalized", "Status", "Map to"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._selectors: list[tuple[str, QComboBox]] = []
        for row, item in enumerate(unresolved):
            self.table.setItem(row, 0, QTableWidgetItem(item.source_column))
            self.table.setItem(row, 1, QTableWidgetItem(item.normalized_source_column))
            self.table.setItem(row, 2, QTableWidgetItem(item.status.value))
            selector = QComboBox()
            selector.addItem("Ignore", None)
            for field in canonical:
                selector.addItem(field, field)
            if item.candidates:
                selector.setToolTip("Possible destination(s): " + ", ".join(item.candidates))
            self.table.setCellWidget(row, 3, selector)
            self._selectors.append((item.source_column, selector))
        layout.addWidget(self.table)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def manual_mapping(self) -> dict[str, str]:
        return {
            source: selector.currentData()
            for source, selector in self._selectors
            if selector.currentData() is not None
        }


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Billing vs Ordering Dashboard")
        self.resize(1350, 850)
        self.controller = ReconciliationController()
        self.billing_path: str | None = None
        self.ordering_path: str | None = None
        self.filtered_inventory = pd.DataFrame()
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.addWidget(self._build_file_section())
        root.addWidget(self._build_period_section())
        root.addWidget(self._build_results_section(), stretch=1)
        self.statusBar().showMessage("Load billing and ordering exports to begin.")

    def _build_file_section(self) -> QWidget:
        group = QGroupBox("1. Load and validate exports")
        layout = QGridLayout(group)
        self.load_billing_btn = QPushButton("Load Billing CSV/XLSX")
        self.load_orders_btn = QPushButton("Load Ordering CSV/XLSX")
        self.resolve_billing_btn = QPushButton("Resolve Billing Mapping")
        self.resolve_orders_btn = QPushButton("Resolve Ordering Mapping")
        self.resolve_billing_btn.setEnabled(False)
        self.resolve_orders_btn.setEnabled(False)
        self.load_billing_btn.clicked.connect(lambda: self._choose_file("billing"))
        self.load_orders_btn.clicked.connect(lambda: self._choose_file("ordering"))
        self.resolve_billing_btn.clicked.connect(lambda: self._resolve_mapping("billing"))
        self.resolve_orders_btn.clicked.connect(lambda: self._resolve_mapping("ordering"))
        self.billing_file_label = QLabel("No billing file selected")
        self.ordering_file_label = QLabel("No ordering file selected")
        self.billing_quality = QLabel("Not loaded")
        self.ordering_quality = QLabel("Not loaded")
        for label in (self.billing_file_label, self.ordering_file_label, self.billing_quality, self.ordering_quality):
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.load_billing_btn, 0, 0)
        layout.addWidget(self.billing_file_label, 0, 1)
        layout.addWidget(self.resolve_billing_btn, 0, 2)
        layout.addWidget(self.billing_quality, 1, 0, 1, 3)
        layout.addWidget(self.load_orders_btn, 2, 0)
        layout.addWidget(self.ordering_file_label, 2, 1)
        layout.addWidget(self.resolve_orders_btn, 2, 2)
        layout.addWidget(self.ordering_quality, 3, 0, 1, 3)
        return group

    def _build_period_section(self) -> QWidget:
        group = QGroupBox("2. Select reconciliation period")
        layout = QHBoxLayout(group)
        self.start_date = QDateEdit(QDate.currentDate().addMonths(-1))
        self.end_date = QDateEdit(QDate.currentDate())
        for widget in (self.start_date, self.end_date):
            widget.setCalendarPopup(True)
            widget.setDisplayFormat("yyyy-MM-dd")
        self.run_btn = QPushButton("Run Reconciliation")
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._run_reconciliation)
        self.export_btn = QPushButton("Export Current Inventory PDF")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_to_pdf)
        layout.addWidget(QLabel("Start (inclusive)"))
        layout.addWidget(self.start_date)
        layout.addWidget(QLabel("End (inclusive)"))
        layout.addWidget(self.end_date)
        layout.addWidget(self.run_btn)
        layout.addWidget(self.export_btn)
        layout.addStretch()
        return group

    def _new_table(self) -> QTableView:
        table = QTableView()
        table.setSortingEnabled(True)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        table.verticalHeader().setDefaultSectionSize(28)
        return table

    def _build_results_section(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Medication"))
        self.medication_filter = QComboBox()
        self.medication_filter.setMinimumWidth(320)
        self.medication_filter.addItem("All medications", None)
        self.medication_filter.currentIndexChanged.connect(self._apply_medication_filter)
        self.patient_count_label = QLabel("Unique patients: —")
        filter_row.addWidget(self.medication_filter)
        filter_row.addWidget(self.patient_count_label)
        filter_row.addStretch()
        layout.addLayout(filter_row)
        self.tabs = QTabWidget()
        self.inventory_table = self._new_table()
        self.insurance_table = self._new_table()
        self.patient_summary_table = self._new_table()
        self.patient_details_table = self._new_table()
        self.tabs.addTab(self.inventory_table, "Inventory Reconciliation")
        self.tabs.addTab(self.insurance_table, "Billing by Insurance / BIN")
        patient_splitter = QSplitter(Qt.Orientation.Vertical)
        patient_splitter.addWidget(self.patient_summary_table)
        patient_splitter.addWidget(self.patient_details_table)
        self.tabs.addTab(patient_splitter, "Patient Billing Details")
        layout.addWidget(self.tabs)
        return container

    def _choose_file(self, dataset_type: str) -> None:
        title = "Select Billing Export" if dataset_type == "billing" else "Select Ordering Export"
        path, _ = QFileDialog.getOpenFileName(
            self, title, "", "Supported Exports (*.csv *.xlsx);;CSV Files (*.csv);;Excel Files (*.xlsx)"
        )
        if path:
            self._load_file(dataset_type, path)

    def _load_file(self, dataset_type: str, path: str, manual_mapping=None) -> None:
        try:
            if dataset_type == "billing":
                result = self.controller.load_billing(path, manual_mapping)
                self.billing_path = path
                self.billing_file_label.setText(Path(path).name)
                self.billing_quality.setText(self._quality_text(result))
            else:
                result = self.controller.load_ordering(path, manual_mapping)
                self.ordering_path = path
                self.ordering_file_label.setText(Path(path).name)
                self.ordering_quality.setText(self._quality_text(result))
            self._update_mapping_buttons()
            self._update_ready_state()
        except (UnsupportedFileTypeError, ManualMappingError, OSError, ValueError) as exc:
            QMessageBox.warning(self, "File could not be processed", str(exc))
        except Exception:
            QMessageBox.critical(
                self, "File could not be processed",
                "The export could not be read. Confirm that it is a valid CSV/XLSX file and is not open or damaged.",
            )

    def _quality_text(self, result) -> str:
        report = result.report
        ready = "READY" if result.ready_for_reconciliation else "NOT READY"
        lines = [
            f"{ready} • Rows: {report['rows_read']} • Valid: {report['valid_rows']} • "
            f"Invalid: {report['invalid_rows']} • Warnings: {report['warning_count']}",
            f"Columns: {report['columns_mapped']} mapped, {report['columns_ambiguous']} ambiguous, "
            f"{report['columns_unmapped']} unmapped",
        ]
        if result.row_count == 0:
            lines.append("The file contains headers but no data rows.")
        if result.mapping.required_fields_missing:
            lines.append("Missing required fields: " + ", ".join(result.mapping.required_fields_missing))
        if result.mapping.conflicts:
            lines.append("Mapping conflicts: " + " | ".join(conflict.message for conflict in result.mapping.conflicts))
        if result.validation:
            errors = [issue.message for issue in result.validation.issues if issue.severity == "error"]
            warnings = [issue.message for issue in result.validation.issues if issue.severity == "warning"]
            if errors:
                lines.append("Data errors: " + " | ".join(errors))
            if warnings:
                lines.append("Warnings: " + " | ".join(warnings))
        return "\n".join(lines)

    def _update_mapping_buttons(self) -> None:
        billing = self.controller.billing_ingestion
        ordering = self.controller.ordering_ingestion
        self.resolve_billing_btn.setEnabled(bool(
            billing and (billing.mapping.conflicts or billing.mapping.required_fields_missing)
        ))
        self.resolve_orders_btn.setEnabled(bool(
            ordering and (ordering.mapping.conflicts or ordering.mapping.required_fields_missing)
        ))

    def _resolve_mapping(self, dataset_type: str) -> None:
        ingestion = self.controller.billing_ingestion if dataset_type == "billing" else self.controller.ordering_ingestion
        path = self.billing_path if dataset_type == "billing" else self.ordering_path
        if not ingestion or not path:
            return
        dialog = ManualMappingDialog(ingestion, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selections = dialog.manual_mapping()
            if not selections:
                QMessageBox.information(self, "No mappings selected", "No manual mappings were selected.")
                return
            self._load_file(dataset_type, path, selections)

    def _update_ready_state(self) -> None:
        ready = self.controller.inputs_ready
        self.run_btn.setEnabled(ready)
        if ready:
            bounds = self.controller.available_date_bounds()
            if bounds:
                start, end = bounds
                self.start_date.setDate(QDate(start.year, start.month, start.day))
                self.end_date.setDate(QDate(end.year, end.month, end.day))
            self.statusBar().showMessage("Both exports are ready. Select dates and run reconciliation.")
        else:
            self.statusBar().showMessage("Review file readiness before reconciliation.")

    def _run_reconciliation(self) -> None:
        start = self.start_date.date().toString("yyyy-MM-dd")
        end = self.end_date.date().toString("yyyy-MM-dd")
        if self.start_date.date() > self.end_date.date():
            QMessageBox.warning(self, "Invalid date range", "Start date must be on or before end date.")
            return
        try:
            result = self.controller.reconcile(start, end)
        except (WorkflowError, ValueError) as exc:
            QMessageBox.warning(self, "Cannot reconcile", str(exc))
            return
        except Exception:
            QMessageBox.critical(
                self, "Cannot reconcile", "Reconciliation could not be completed. Review both data-quality summaries."
            )
            return
        if result.inventory.empty:
            self._clear_results()
            QMessageBox.information(
                self, "No records in period", "No billing or ordering records fall inside the selected date range."
            )
            return
        self._populate_medications(result.inventory)
        self._apply_medication_filter()
        self.export_btn.setEnabled(True)
        self.statusBar().showMessage(
            f"Reconciliation complete: {len(result.inventory)} medication(s), {start} through {end}."
        )

    def _populate_medications(self, inventory: pd.DataFrame) -> None:
        self.medication_filter.blockSignals(True)
        self.medication_filter.clear()
        self.medication_filter.addItem("All medications", None)
        for ndc, drug_name in inventory[["ndc", "drug_name"]].itertuples(index=False, name=None):
            self.medication_filter.addItem(f"{drug_name} ({ndc})", (ndc, drug_name))
        self.medication_filter.blockSignals(False)

    def _filtered(self, frame: pd.DataFrame, medication) -> pd.DataFrame:
        if medication is None or frame.empty:
            return frame.copy()
        ndc, drug_name = medication
        return frame.loc[(frame["ndc"] == ndc) & (frame["drug_name"] == drug_name)].copy()

    def _apply_medication_filter(self) -> None:
        result = self.controller.dashboard_result
        if not result:
            return
        medication = self.medication_filter.currentData()
        self.filtered_inventory = self._filtered(result.inventory, medication)
        insurance = self._filtered(result.insurance, medication)
        patient_summary = self._filtered(result.patient_summary, medication)
        patient_details = self._filtered(result.patient_details, medication)
        self._set_table(self.inventory_table, self.filtered_inventory)
        self._set_table(self.insurance_table, insurance)
        self._set_table(self.patient_summary_table, patient_summary)
        self._set_table(self.patient_details_table, patient_details)
        unique_count = int(patient_details["patient_id"].nunique()) if not patient_details.empty else 0
        self.patient_count_label.setText(f"Unique patients: {unique_count}")

    def _set_table(self, table: QTableView, frame: pd.DataFrame) -> None:
        if frame.empty:
            table.setModel(None)
            return
        table.setModel(PandasTableModel(frame))
        table.resizeColumnsToContents()

    def _clear_results(self) -> None:
        for table in (self.inventory_table, self.insurance_table, self.patient_summary_table, self.patient_details_table):
            table.setModel(None)
        self.filtered_inventory = pd.DataFrame()
        self.medication_filter.clear()
        self.medication_filter.addItem("All medications", None)
        self.patient_count_label.setText("Unique patients: —")
        self.export_btn.setEnabled(False)

    def _export_to_pdf(self) -> None:
        if self.filtered_inventory.empty:
            QMessageBox.information(self, "Nothing to export", "Run reconciliation before exporting a report.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Inventory PDF", "billing_vs_orders.pdf", "PDF Files (*.pdf)"
        )
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        frame = self.filtered_inventory
        headers = "".join(f"<th>{escape(str(column))}</th>" for column in frame.columns)
        rows = []
        for row in frame.itertuples(index=False):
            cells = "".join(f"<td>{escape('' if pd.isna(value) else str(value))}</td>" for value in row)
            rows.append(f"<tr>{cells}</tr>")
        html = (
            "<html><head><style>body{font-family:Arial;font-size:9pt}"
            "table{border-collapse:collapse;width:100%}th,td{border:1px solid #444;padding:4px;text-align:center}"
            "th{background:#f0f0f0}</style></head><body><h2>Medication Inventory Reconciliation</h2>"
            f"<table><tr>{headers}</tr>{''.join(rows)}</table></body></html>"
        )
        document = QTextDocument()
        document.setHtml(html)
        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
        printer.setOutputFileName(path)
        printer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
        document.print_(printer)
        if not Path(path).exists():
            QMessageBox.critical(self, "PDF error", "The PDF could not be created.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))
