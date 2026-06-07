import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const PROJECT_ROOT = path.resolve(".");
const OUT_DIR = path.join(PROJECT_ROOT, "analysis", "results", "metrics_full");
const INPUT_XLSX = path.join(OUT_DIR, "hemorrhage_results_metrics_full.xlsx");
const PREVIEW_DIR = path.join(OUT_DIR, "workbook_preview");

async function saveBlob(blob, filePath) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, Buffer.from(await blob.arrayBuffer()));
}

async function main() {
  const input = await FileBlob.load(INPUT_XLSX);
  const workbook = await SpreadsheetFile.importXlsx(input);

  const readme = await workbook.inspect({
    kind: "table",
    range: "README!A1:B25",
    include: "values,formulas",
    tableMaxRows: 25,
    tableMaxCols: 2,
  });
  const summary = await workbook.inspect({
    kind: "table",
    range: "Summary Overall!A1:N12",
    include: "values,formulas",
    tableMaxRows: 12,
    tableMaxCols: 14,
  });
  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 300 },
    summary: "final formula error scan",
  });

  await fs.mkdir(PREVIEW_DIR, { recursive: true });
  const rendered = [];
  for (const sheetName of ["README", "Summary Overall", "Prediction Agreement", "Case Metrics"]) {
    const blob = await workbook.render({ sheetName, range: "A1:N30", scale: 1 });
    const filePath = path.join(PREVIEW_DIR, `${sheetName.replaceAll(" ", "_")}.png`);
    await saveBlob(blob, filePath);
    rendered.push(filePath);
  }

  console.log(
    JSON.stringify(
      {
        workbook: INPUT_XLSX,
        readmePreview: readme.ndjson.split("\n").slice(0, 4),
        summaryPreview: summary.ndjson.split("\n").slice(0, 4),
        errorScan: errors.ndjson,
        rendered,
      },
      null,
      2,
    ),
  );
}

main().catch((error) => {
  console.error(error.stack || error.message || String(error));
  process.exit(1);
});

