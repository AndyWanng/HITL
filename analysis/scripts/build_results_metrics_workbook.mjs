import fs from "node:fs/promises";
import path from "node:path";
import { Workbook, SpreadsheetFile } from "@oai/artifact-tool";

const PROJECT_ROOT = path.resolve(".");
const OUT_DIR = path.join(PROJECT_ROOT, "analysis", "results", "metrics_full");
const OUTPUT_XLSX = path.join(OUT_DIR, "hemorrhage_results_metrics_full.xlsx");

const CSV_SHEETS = [
  ["Summary Overall", "summary_overall.csv"],
  ["Summary Timepoint", "summary_by_timepoint.csv"],
  ["Summary Pattern", "summary_by_timepoint_pattern.csv"],
  ["Round Delta", "round_delta_case_metrics.csv"],
  ["Prediction Agreement", "summary_prediction_agreement.csv"],
  ["Agreement Details", "prediction_round_agreement_metrics.csv"],
  ["Label Agreement", "summary_label_agreement.csv"],
  ["Label Agreement Details", "epibios_label_round_agreement_metrics.csv"],
  ["Static Reference", "summary_static_reference.csv"],
  ["Static Ref Details", "epibios_static_reference_metrics.csv"],
  ["Case Metrics", "mask_case_metrics.csv"],
  ["Manifest", "prediction_manifest.csv"],
];

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let inQuotes = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    const next = text[i + 1];
    if (inQuotes) {
      if (ch === '"' && next === '"') {
        cell += '"';
        i += 1;
      } else if (ch === '"') {
        inQuotes = false;
      } else {
        cell += ch;
      }
    } else {
      if (ch === '"') {
        inQuotes = true;
      } else if (ch === ",") {
        row.push(cell);
        cell = "";
      } else if (ch === "\n") {
        row.push(cell);
        rows.push(row);
        row = [];
        cell = "";
      } else if (ch === "\r") {
        // ignore
      } else {
        cell += ch;
      }
    }
  }
  if (cell.length > 0 || row.length > 0) {
    row.push(cell);
    rows.push(row);
  }
  return rows;
}

function toCellValue(value) {
  if (value === "") {
    return null;
  }
  if (/^-?\d+(?:\.\d+)?(?:e[+-]?\d+)?$/i.test(value)) {
    const n = Number(value);
    if (Number.isFinite(n)) return n;
  }
  return value;
}

function colName(index) {
  let n = index + 1;
  let out = "";
  while (n > 0) {
    const rem = (n - 1) % 26;
    out = String.fromCharCode(65 + rem) + out;
    n = Math.floor((n - 1) / 26);
  }
  return out;
}

function preferredWidthPx(header) {
  const text = String(header || "");
  if (text.includes("path")) return 360;
  if (text === "case_id") return 300;
  if (text.includes("timepoint")) return 180;
  if (text.includes("analysis") || text.includes("evaluation") || text.includes("postprocess")) return 145;
  if (text.includes("animal")) return 125;
  if (text.includes("dice") || text.includes("hd95") || text.includes("assd") || text.includes("rve") || text.includes("kappa")) return 118;
  if (text.includes("volume") || text.includes("voxels")) return 135;
  return Math.max(88, Math.min(170, 8 * text.length + 26));
}

function applyTableFormatting(sheet, rows, cols, headers) {
  if (rows <= 0 || cols <= 0) return;
  const lastCol = colName(cols - 1);
  try {
    sheet.freezePanes.freezeRows(1);
  } catch {}
  try {
    sheet.getRange(`A1:${lastCol}1`).format = {
      fill: "#1F2937",
      font: { bold: true, color: "#FFFFFF" },
      wrapText: true,
    };
  } catch {}
  for (let i = 0; i < cols; i += 1) {
    const col = colName(i);
    try {
      sheet.getRange(`${col}:${col}`).format.columnWidthPx = preferredWidthPx(headers[i]);
    } catch {}
  }
  for (let i = 0; i < cols; i += 1) {
    const h = String(headers[i] || "");
    if (/dice|jaccard|hd95|assd|rve|similarity|kappa|precision|sensitivity|specificity|agreement|volume|voxels/i.test(h)) {
      const col = colName(i);
      try {
        sheet.getRange(`${col}2:${col}${rows}`).format.numberFormat = "0.0000";
      } catch {}
    }
  }
}

async function addCsvSheet(workbook, sheetName, csvName) {
  const sheet = workbook.worksheets.getOrAdd(sheetName, {
    renameFirstIfOnlyNewSpreadsheet: true,
  });
  const csvText = await fs.readFile(path.join(OUT_DIR, csvName), "utf8");
  const rows = parseCsv(csvText).map((row) => row.map(toCellValue));
  const maxCols = rows.reduce((acc, row) => Math.max(acc, row.length), 0);
  const padded = rows.map((row) => {
    const copy = row.slice();
    while (copy.length < maxCols) copy.push(null);
    return copy;
  });
  if (padded.length > 0 && maxCols > 0) {
    const range = `A1:${colName(maxCols - 1)}${padded.length}`;
    sheet.getRange(range).values = padded;
    applyTableFormatting(sheet, padded.length, maxCols, padded[0]);
  }
  return { sheetName, rows: padded.length, cols: maxCols, csvName };
}

function addReadme(workbook, manifest) {
  const sheet = workbook.worksheets.getOrAdd("README");
  const rows = [
    ["Hemorrhage segmentation metrics workbook", ""],
    ["Generated from", "analysis/results masks plus local ARAMRA external predictions"],
    ["Primary outputs", "CSV files in analysis/results/metrics_full and this workbook"],
    ["Metric definitions", ""],
    ["Dice", "Overlap score between model mask and reference mask; higher is better."],
    ["HD95 (mm)", "95th percentile symmetric surface distance; lower is better."],
    ["ASSD (mm)", "Average symmetric surface distance; lower is better."],
    ["RVE (%)", "Relative volume error: 100 * (prediction volume - reference volume) / reference volume."],
    ["Volume similarity", "Agreement of predicted and reference volumes; higher is better."],
    ["Surface Dice 1 mm", "Boundary agreement within 1 mm; higher is better."],
    ["Cohen kappa", "Voxel-level agreement corrected for chance; less dominated by background than raw voxel agreement."],
    ["Included prediction groups", ""],
    ["Animal-level EpiBios", "analysis/results/animal-level/epibios oof"],
    ["Animal-level ARAMRA", "analysis/results/animal-level/aramra predictions"],
    ["Case-level EpiBios", "analysis/results/case-level"],
    ["Case-level ARAMRA", "analysis/workspace_v0_full_external_analysis/predictions/aramra"],
    ["Sheet inventory", ""],
    ...manifest.map((item) => [item.sheetName, `${item.rows} rows x ${item.cols} columns from ${item.csvName}`]),
  ];
  sheet.getRange(`A1:B${rows.length}`).values = rows;
  try {
    sheet.getRange("A:A").format.columnWidthPx = 220;
    sheet.getRange("B:B").format.columnWidthPx = 760;
    sheet.getRange("A1:B1").format = { fill: "#1F2937", font: { bold: true, color: "#FFFFFF" } };
    sheet.getRange(`A1:B${rows.length}`).format.wrapText = true;
  } catch {}
}

async function main() {
  const workbook = Workbook.create();
  const manifest = [];
  for (const [sheetName, csvName] of CSV_SHEETS) {
    manifest.push(await addCsvSheet(workbook, sheetName, csvName));
  }
  addReadme(workbook, manifest);

  await fs.mkdir(OUT_DIR, { recursive: true });
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(OUTPUT_XLSX);
  const stat = await fs.stat(OUTPUT_XLSX);
  console.log(JSON.stringify({ output: OUTPUT_XLSX, bytes: stat.size, sheets: manifest.length + 1 }, null, 2));
}

main().catch((error) => {
  console.error(error.stack || error.message || String(error));
  process.exit(1);
});
