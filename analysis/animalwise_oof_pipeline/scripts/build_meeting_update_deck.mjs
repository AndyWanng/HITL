import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const ARTIFACT_TOOL =
  process.env.ARTIFACT_TOOL_MJS ||
  "C:/Users/22396/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const { Presentation, PresentationFile } = await import(pathToFileURL(ARTIFACT_TOOL).href);

const W = 1280;
const H = 720;
const RUN_DIR = path.resolve("analysis/animalwise_oof_pipeline/runs/20260524_223407_server_animalwise_oof");
const OUT = path.join(RUN_DIR, "hemorrhage_hitl_update_20260525.pptx");
const PREVIEW_DIR = path.join(RUN_DIR, "ppt_preview");

const C = {
  bg: "#F7F8FA",
  ink: "#17202A",
  text: "#243041",
  muted: "#667085",
  faint: "#D7DCE3",
  line: "#B9C2CF",
  blue: "#205493",
  blue2: "#4F7FB9",
  green: "#2E7D32",
  orange: "#B8651D",
  red: "#A33A3A",
  panel: "#FFFFFF",
  dark: "#1F2A3A",
};

function addShape(slide, x, y, w, h, fill = C.panel, lineFill = "transparent", lineWidth = 0, name = undefined) {
  return slide.shapes.add({
    geometry: "rect",
    name,
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: "solid", fill: lineFill, width: lineWidth },
  });
}

function addText(slide, text, x, y, w, h, opts = {}) {
  const shape = addShape(slide, x, y, w, h, opts.fill || "transparent", opts.lineFill || "transparent", opts.lineWidth || 0, opts.name);
  shape.text = text;
  shape.text.fontSize = opts.size || 24;
  shape.text.color = opts.color || C.text;
  shape.text.bold = Boolean(opts.bold);
  shape.text.typeface = opts.face || "Aptos";
  shape.text.alignment = opts.align || "left";
  shape.text.verticalAlignment = opts.valign || "top";
  shape.text.insets = opts.insets || { left: 0, right: 0, top: 0, bottom: 0 };
  return shape;
}

function addLine(slide, x, y, w, h, color = C.line, width = 1) {
  return addShape(slide, x, y, w, h, color, color, 0);
}

function baseSlide(presentation, section, title, subtitle, slideNo) {
  const slide = presentation.slides.add();
  addShape(slide, 0, 0, W, H, C.bg);
  addText(slide, section.toUpperCase(), 58, 32, 530, 24, { size: 13, color: C.blue, bold: true });
  addText(slide, title, 58, 60, 1060, 82, { size: 31, color: C.ink, bold: true, face: "Aptos Display" });
  if (subtitle) {
    addText(slide, subtitle, 58, 150, 1035, 42, { size: 15, color: C.muted });
  }
  addLine(slide, 58, 670, 1068, 1, C.faint, 1);
  addText(slide, "Source: run 20260524_223407_server_animalwise_oof", 58, 682, 650, 20, { size: 10, color: C.muted });
  addText(slide, String(slideNo).padStart(2, "0"), 1110, 680, 60, 20, { size: 11, color: C.muted, align: "right" });
  return slide;
}

function metricBox(slide, label, value, note, x, y, w, h, color = C.blue) {
  addShape(slide, x, y, w, h, C.panel, C.faint, 1);
  addText(slide, value, x + 18, y + 16, w - 36, 38, { size: 31, color, bold: true, face: "Aptos Display" });
  addText(slide, label, x + 18, y + 58, w - 36, 26, { size: 14, color: C.ink, bold: true });
  addText(slide, note, x + 18, y + 85, w - 36, h - 98, { size: 11, color: C.muted });
}

function bullet(slide, text, x, y, w, color = C.text) {
  addShape(slide, x, y + 8, 6, 6, C.blue, C.blue, 0);
  addText(slide, text, x + 18, y, w - 18, 42, { size: 16, color });
}

function smallHeader(slide, text, x, y, w) {
  addText(slide, text.toUpperCase(), x, y, w, 22, { size: 12, color: C.blue, bold: true });
  addLine(slide, x, y + 28, w, 1, C.faint, 1);
}

function table(slide, x, y, colW, rowH, header, rows, opts = {}) {
  const totalW = colW.reduce((a, b) => a + b, 0);
  addShape(slide, x, y, totalW, rowH, opts.headerFill || C.dark);
  let xx = x;
  header.forEach((h, i) => {
    addText(slide, h, xx + 8, y + 8, colW[i] - 16, rowH - 12, { size: opts.headerSize || 12, color: "#FFFFFF", bold: true });
    xx += colW[i];
  });
  rows.forEach((row, r) => {
    const yy = y + rowH * (r + 1);
    addShape(slide, x, yy, totalW, rowH, r % 2 === 0 ? "#FFFFFF" : "#F2F4F7", C.faint, 0.5);
    let cx = x;
    row.forEach((cell, i) => {
      addText(slide, String(cell), cx + 8, yy + 7, colW[i] - 16, rowH - 10, {
        size: opts.size || 12,
        color: i === 0 ? C.ink : C.text,
        bold: i === 0 && opts.boldFirst !== false,
        align: opts.numeric?.includes(i) ? "right" : "left",
      });
      cx += colW[i];
    });
  });
}

function barChart(slide, x, y, w, h, series, opts = {}) {
  const maxVal = opts.maxVal || Math.max(...series.map((d) => d.value));
  const minVal = opts.minVal || 0;
  const plotW = w - 140;
  const barH = opts.barH || 24;
  const gap = opts.gap || 16;
  series.forEach((d, i) => {
    const yy = y + i * (barH + gap);
    addText(slide, d.label, x, yy + 2, 112, 24, { size: 13, color: C.text });
    addShape(slide, x + 120, yy, plotW, barH, "#E7EBF0");
    const ratio = Math.max(0, Math.min(1, (d.value - minVal) / (maxVal - minVal)));
    addShape(slide, x + 120, yy, plotW * ratio, barH, d.color || C.blue);
    addText(slide, d.display || d.value.toFixed(4), x + 130 + plotW * ratio, yy - 1, 88, 24, {
      size: 12,
      color: C.ink,
      bold: true,
    });
  });
}

function twoBarChart(slide, x, y, w, h, items, opts = {}) {
  const maxVal = opts.maxVal || 0.75;
  const minVal = opts.minVal || 0.5;
  const groupGap = 60;
  const barW = 34;
  const plotH = h - 70;
  items.forEach((item, i) => {
    const gx = x + i * groupGap;
    const r0H = ((item.r0 - minVal) / (maxVal - minVal)) * plotH;
    const r1H = ((item.r1 - minVal) / (maxVal - minVal)) * plotH;
    addShape(slide, gx, y + plotH - r0H, barW, r0H, C.blue2);
    addShape(slide, gx + barW + 5, y + plotH - r1H, barW, r1H, C.green);
    addText(slide, item.label, gx - 3, y + plotH + 12, groupGap + 8, 32, { size: 11, color: C.text, align: "center" });
    addText(slide, item.r0.toFixed(3), gx - 2, y + plotH - r0H - 19, barW + 2, 16, { size: 9, color: C.blue2, align: "center", bold: true });
    addText(slide, item.r1.toFixed(3), gx + barW + 3, y + plotH - r1H - 19, barW + 4, 16, { size: 9, color: C.green, align: "center", bold: true });
  });
  addText(slide, "R0", x + w - 86, y + 4, 28, 16, { size: 10, color: C.blue2, bold: true });
  addShape(slide, x + w - 108, y + 8, 16, 8, C.blue2);
  addText(slide, "R1", x + w - 38, y + 4, 28, 16, { size: 10, color: C.green, bold: true });
  addShape(slide, x + w - 60, y + 8, 16, 8, C.green);
}

function flowNode(slide, title, body, x, y, w, h, color = C.blue) {
  addShape(slide, x, y, w, h, C.panel, C.faint, 1);
  addShape(slide, x, y, 7, h, color, color, 0);
  addText(slide, title, x + 18, y + 14, w - 34, 24, { size: 15, color: C.ink, bold: true });
  addText(slide, body, x + 18, y + 44, w - 34, h - 54, { size: 12, color: C.muted });
}

async function saveBlob(blob, filePath) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, Buffer.from(await blob.arrayBuffer()));
}

async function build() {
  const p = Presentation.create({ slideSize: { width: W, height: H } });

  let s = baseSlide(
    p,
    "Meeting update",
    "HITL hemorrhage segmentation: revised status after animal-wise and OOD analysis",
    "This update separates the original finetune result from the newer longitudinal data audit, animal-wise OOF retraining, and ARAMRA external evaluation.",
    1,
  );
  metricBox(s, "EpiBios source cohort", "127 / 33", "cases / strict animals; longitudinal repeated scans", 58, 215, 265, 150, C.blue);
  metricBox(s, "ARAMRA OOD cohort", "171 / 96", "cases / strict animals; independent 9.4T target cohort", 356, 215, 265, 150, C.orange);
  metricBox(s, "New split unit", "animal", "all timepoints from the same rat stay in one fold", 654, 215, 265, 150, C.green);
  metricBox(s, "Completed run", "R0 + R1", "animal-wise OOF plus ARAMRA ensemble evaluation", 952, 215, 265, 150, C.dark);
  addText(s, "Main message", 58, 445, 200, 28, { size: 15, color: C.blue, bold: true });
  addText(
    s,
    "R1/revised labels improve source-cohort animal-wise performance, but this gain only weakly transfers to the independent ARAMRA cohort; the D9/M5 timepoint gap remains.",
    58,
    477,
    1020,
    70,
    { size: 25, color: C.ink, bold: true, face: "Aptos Display" },
  );

  s = baseSlide(
    p,
    "Context",
    "What changed since the last meeting",
    "The previous update focused on the original scan-level HITL finetune metrics. The new work clarified data structure and re-ran the key evaluation under stricter assumptions.",
    2,
  );
  smallHeader(s, "Previously reported", 62, 210, 510);
  bullet(s, "Original workspace_v0 R0/R1 models and finetune result were the main numbers.", 70, 258, 500);
  bullet(s, "Reported improvement was framed around scan-level OOF and revised-label finetune.", 70, 324, 500);
  bullet(s, "ARAMRA was not yet integrated into the main evaluation story.", 70, 390, 500);
  smallHeader(s, "What was added", 650, 210, 510);
  bullet(s, "Recognized repeated scans from the same animal across longitudinal timepoints.", 658, 258, 500);
  bullet(s, "Standardized and evaluated the independent ARAMRA002 cohort.", 658, 324, 500);
  bullet(s, "Built a separate animal-wise OOF pipeline for R0/R1 source evaluation.", 658, 390, 500);
  bullet(s, "Added static-reference and bootstrap analyses to avoid overclaiming.", 658, 456, 500);

  s = baseSlide(
    p,
    "Data structure",
    "The datasets are longitudinal rather than independent single scans",
    "The evaluation unit matters because one animal can contribute multiple sessions/timepoints.",
    3,
  );
  table(
    s,
    68,
    206,
    [160, 110, 120, 235, 390],
    44,
    ["Cohort", "Cases", "Animals", "Main timepoints", "Notes"],
    [
      ["EpiBios", "127", "33", "D02/D09/D28/M05, etc.", "source cohort with 3-4 labeled timepoints for most animals"],
      ["ARAMRA002", "171", "96", "D9 and M5", "independent cohort, mostly D9/M5 pairs plus irregular animals"],
      ["Old OOF", "127", "33", "mixed", "scan-level folds; same-animal siblings can appear in train"],
      ["New OOF", "127", "33", "mixed", "animal-wise folds; all timepoints from one animal held out together"],
    ],
    { size: 12 },
  );
  addText(s, "Why this matters", 70, 440, 230, 24, { size: 15, color: C.blue, bold: true });
  addText(
    s,
    "A scan-level fold can test a new session from an animal that is already represented in training. The new animal-wise split asks a stricter question: performance on unseen animals.",
    70,
    475,
    1040,
    58,
    { size: 22, color: C.ink, bold: true, face: "Aptos Display" },
  );

  s = baseSlide(
    p,
    "Evaluation design",
    "A separate lightweight pipeline now answers four narrower questions",
    "The goal was not to replace the old HITL workflow, but to isolate source animal generalization and ARAMRA OOD behavior.",
    4,
  );
  flowNode(s, "Stage 0", "Build metadata, standardize ARAMRA manifest, and create animal-wise folds with no animal leakage.", 70, 220, 240, 165, C.blue);
  flowNode(s, "Stage 1: EpiAW R0", "Train 5 animal-wise fold models using round_0 binary labels and evaluate OOF.", 348, 220, 240, 165, C.blue2);
  flowNode(s, "Stage 2: EpiAW R1", "Finetune corresponding folds with round_1 labels and animal-wise R0 OOF soft target.", 626, 220, 240, 165, C.green);
  flowNode(s, "Stage 3: ARAMRA OOD", "Use 5-fold R0/R1 ensembles to predict ARAMRA; ARAMRA is evaluation-only.", 904, 220, 240, 165, C.orange);
  addText(s, "Follow-up analyses", 70, 458, 300, 24, { size: 15, color: C.blue, bold: true });
  table(
    s,
    70,
    492,
    [320, 720],
    36,
    ["Analysis", "Purpose"],
    [
      ["Static-reference matrix", "Separate prediction-side improvement from label-reference shift."],
      ["Animal bootstrap on ARAMRA", "Quantify uncertainty using animal_id_strict as the resampling unit."],
      ["Baseline comparison", "Compare old scan-level OOF, new animal-wise OOF, and ARAMRA OOD."],
    ],
    { size: 12 },
  );

  s = baseSlide(
    p,
    "Run status",
    "The animal-wise R0/R1 + ARAMRA OOD run completed",
    "Run directory: analysis/animalwise_oof_pipeline/runs/20260524_223407_server_animalwise_oof",
    5,
  );
  metricBox(s, "Split integrity", "PASS", "0 animal overlap between folds; ARAMRA not used for training", 70, 210, 255, 135, C.green);
  metricBox(s, "Training stages", "5 + 5", "R0 fold models and R1 finetune fold models completed", 355, 210, 255, 135, C.blue);
  metricBox(s, "External eval", "171", "ARAMRA cases predicted by R0/R1 5-fold ensembles", 640, 210, 255, 135, C.orange);
  metricBox(s, "Follow-up", "3", "static-reference, bootstrap CI, baseline comparison", 925, 210, 255, 135, C.dark);
  table(
    s,
    80,
    430,
    [260, 305, 435],
    38,
    ["Output", "Location", "Status"],
    [
      ["OOF summaries", "metrics/r0_oof_summary.json, r1_oof_summary.json", "completed"],
      ["ARAMRA metrics", "metrics/aramra_summary.json, group/pair CSVs", "completed"],
      ["Static-reference", "metrics/animalwise_static_reference_summary.json", "completed"],
      ["Bootstrap CI", "metrics/aramra_bootstrap_ci.csv", "completed, 5000 animal bootstraps"],
    ],
    { size: 11 },
  );

  s = baseSlide(
    p,
    "Source cohort",
    "EpiBios animal-wise OOF: R1 is consistently better than R0",
    "This is a stricter source-cohort evaluation because the held-out fold contains unseen animals, not just unseen sessions.",
    6,
  );
  twoBarChart(
    s,
    85,
    238,
    470,
    280,
    [
      { label: "Macro", r0: 0.6682, r1: 0.6975 },
      { label: "Animal", r0: 0.6689, r1: 0.6974 },
      { label: "Micro", r0: 0.6959, r1: 0.7256 },
      { label: "Lesion-F1", r0: 0.5323, r1: 0.5509 },
    ],
    { minVal: 0.50, maxVal: 0.75 },
  );
  table(
    s,
    650,
    218,
    [76, 118, 118, 118, 118],
    34,
    ["Fold", "R0 macro", "R1 macro", "Delta", "R1 micro"],
    [
      ["1", "0.6458", "0.6749", "+0.0291", "0.7128"],
      ["2", "0.6963", "0.7112", "+0.0149", "0.7260"],
      ["3", "0.6628", "0.6901", "+0.0273", "0.7129"],
      ["4", "0.6533", "0.7076", "+0.0543", "0.7291"],
      ["5", "0.6883", "0.7084", "+0.0201", "0.7453"],
    ],
    { size: 11, numeric: [1, 2, 3, 4] },
  );
  addText(s, "Readout", 650, 472, 160, 24, { size: 15, color: C.blue, bold: true });
  addText(
    s,
    "R1 improves source animal-wise macro Dice by about +0.029 and every fold moves in the same direction.",
    650,
    505,
    485,
    48,
    { size: 18, color: C.ink, bold: true },
  );

  s = baseSlide(
    p,
    "Static reference",
    "The R1 gain contains both prediction-side improvement and label-reference shift",
    "This matrix prevents overinterpreting the R1-vs-R1 result as pure model improvement.",
    7,
  );
  table(
    s,
    78,
    205,
    [160, 160, 140, 155, 145],
    42,
    ["Prediction", "Reference", "Macro Dice", "Animal Dice", "Micro Dice"],
    [
      ["R0", "R0", "0.6682", "0.6689", "0.6959"],
      ["R0", "R1", "0.6866", "0.6866", "0.7197"],
      ["R1", "R0", "0.6784", "0.6790", "0.7006"],
      ["R1", "R1", "0.6975", "0.6974", "0.7256"],
    ],
    { size: 13, numeric: [2, 3, 4] },
  );
  barChart(
    s,
    860,
    230,
    300,
    190,
    [
      { label: "Observed", value: 0.0293, display: "+0.0293", color: C.green },
      { label: "Pred-side", value: 0.0109, display: "+0.0109", color: C.blue },
      { label: "Ref shift", value: 0.0184, display: "+0.0184", color: C.orange },
    ],
    { maxVal: 0.032, barH: 26 },
  );
  addText(s, "Interpretation", 78, 475, 180, 24, { size: 15, color: C.blue, bold: true });
  addText(
    s,
    "The animal-wise source improvement is real under the R1 label system, but the decomposition suggests the total gain should be described as label-system improvement, not solely model improvement.",
    78,
    508,
    1030,
    60,
    { size: 20, color: C.ink, bold: true, face: "Aptos Display" },
  );

  s = baseSlide(
    p,
    "External cohort",
    "ARAMRA OOD: R1 is slightly higher, but the absolute gain is small",
    "ARAMRA remains evaluation-only in this run. The current source-only animal-wise models do not close the external gap.",
    8,
  );
  table(
    s,
    70,
    205,
    [225, 110, 120, 120, 120, 120],
    38,
    ["Model", "Variant", "Macro", "Animal", "Micro", "Lesion-F1"],
    [
      ["Old workspace R0", "raw", "0.5718", "0.5735", "0.5745", "0.3682"],
      ["Old workspace R1", "raw", "0.5732", "0.5750", "0.5762", "0.3749"],
      ["Animal-wise R0", "raw", "0.5548", "0.5572", "0.5571", "0.3746"],
      ["Animal-wise R1", "raw", "0.5582", "0.5604", "0.5613", "0.3779"],
    ],
    { size: 12, numeric: [2, 3, 4, 5] },
  );
  smallHeader(s, "Animal-bootstrap R1 minus R0 on ARAMRA", 900, 212, 285);
  table(
    s,
    900,
    255,
    [100, 70, 135],
    34,
    ["Metric", "Delta", "95% CI"],
    [
      ["Macro Dice", "+0.0035", "[0.0020, 0.0050]"],
      ["Animal Dice", "+0.0032", "[0.0021, 0.0042]"],
      ["Micro Dice", "+0.0042", "[0.0026, 0.0058]"],
      ["Lesion-F1", "+0.0033", "[-0.0006, 0.0071]"],
    ],
    { size: 11 },
  );
  addText(
    s,
    "Cautious readout: the Dice delta is directionally stable under animal bootstrap, but the magnitude is too small to claim that source-only R1 solves OOD generalization.",
    90,
    505,
    1035,
    58,
    { size: 20, color: C.ink, bold: true, face: "Aptos Display" },
  );

  s = baseSlide(
    p,
    "Longitudinal target shift",
    "The D9/M5 gap persists after animal-wise retraining",
    "The external error is not only cohort-level; it has a timepoint/stage component.",
    9,
  );
  twoBarChart(
    s,
    95,
    230,
    280,
    270,
    [
      { label: "D9", r0: 0.5748, r1: 0.5782 },
      { label: "M5", r0: 0.5265, r1: 0.5302 },
    ],
    { minVal: 0.50, maxVal: 0.60 },
  );
  metricBox(s, "R1 D9 - M5 macro gap", "0.0480", "D9 0.5782 vs M5 0.5302, raw macro Dice", 490, 235, 250, 125, C.orange);
  metricBox(s, "D9/M5 paired delta", "-0.0514", "mean M5-minus-D9 Dice across 71 pairs under R1", 775, 235, 250, 125, C.red);
  metricBox(s, "M5 lesion-F1", "0.3990", "M5 detection is not uniformly worse; overlap/extent is the main issue", 490, 390, 535, 125, C.blue);
  addText(
    s,
    "Interpretation: M5 likely represents a late-stage / smaller-residual-lesion failure mode. The source-side R1 refinement does not reduce this timepoint gap.",
    80,
    560,
    1040,
    48,
    { size: 19, color: C.ink, bold: true },
  );

  s = baseSlide(
    p,
    "Claims and caveats",
    "What the current evidence supports, and what it does not support yet",
    "This framing is important for a thesis/paper discussion because the same numbers can be overclaimed if the evaluation unit is ignored.",
    10,
  );
  smallHeader(s, "Supported by current artifacts", 78, 205, 500);
  bullet(s, "Old scan-level OOF is not the cleanest unseen-animal evaluation.", 86, 250, 500);
  bullet(s, "R1 label system improves EpiBios animal-wise OOF across all five folds.", 86, 318, 500);
  bullet(s, "ARAMRA is an independent target cohort and shows limited source-only transfer.", 86, 386, 500);
  bullet(s, "D9/M5 timepoint gap is stable under the new animal-wise models.", 86, 454, 500);
  smallHeader(s, "Not supported yet", 682, 205, 500);
  bullet(s, "Do not claim +0.03 is pure model improvement; label-reference shift contributes.", 690, 250, 500, C.text);
  bullet(s, "Do not claim R1 meaningfully solves external ARAMRA generalization.", 690, 318, 500, C.text);
  bullet(s, "Do not treat ARAMRA as training data in this phase; it was held out for OOD eval.", 690, 386, 500, C.text);
  bullet(s, "Do not claim M5 is worse on every metric; lesion-F1 is metric-dependent.", 690, 454, 500, C.text);

  s = baseSlide(
    p,
    "Paper direction",
    "The story has shifted from 'one HITL finetune improves Dice' to a generalization problem",
    "The stronger thesis is about converting expert label revision into robust animal-level and target-cohort generalization.",
    11,
  );
  addText(s, "Working thesis", 78, 215, 200, 24, { size: 15, color: C.blue, bold: true });
  addText(
    s,
    "One-round expert label refinement improves source-cohort animal-level performance, but source-only refinement is insufficient for independent longitudinal target-cohort generalization.",
    78,
    250,
    1030,
    75,
    { size: 25, color: C.ink, bold: true, face: "Aptos Display" },
  );
  table(
    s,
    90,
    390,
    [265, 365, 380],
    42,
    ["Evidence block", "Current result", "Implication"],
    [
      ["Animal-wise OOF", "R1 macro Dice 0.6975 vs R0 0.6682", "R1 label system helps source unseen animals"],
      ["Static reference", "prediction gain and reference shift both present", "need cautious causal language"],
      ["ARAMRA OOD", "R1-R0 macro Dice only +0.0035", "source-only refinement does not externalize enough"],
      ["D9/M5 analysis", "R1 paired M5-D9 delta about -0.051", "timepoint-aware adaptation is justified"],
    ],
    { size: 11 },
  );

  s = baseSlide(
    p,
    "Next steps",
    "Recommended next experiments before making a publication-level claim",
    "The immediate objective is to move from source-only analysis to target-aware experiments with a locked ARAMRA test protocol.",
    12,
  );
  flowNode(s, "1. Lock the target protocol", "Keep a held-out ARAMRA animal-level test set. Use ARAMRA train/val only for adaptation and model selection.", 78, 215, 310, 170, C.blue);
  flowNode(s, "2. Run simple target baselines", "ARAMRA-only, EpiBios+ARAMRA naive pooled, and animal-balanced pooled training.", 432, 215, 310, 170, C.blue2);
  flowNode(s, "3. Add label-trust/timepoint logic", "Trust-weighted clean/noisy labels and D9/M5-balanced sampling before heavier model changes.", 786, 215, 310, 170, C.green);
  addText(s, "Near-term meeting decision", 80, 460, 260, 24, { size: 15, color: C.blue, bold: true });
  addText(
    s,
    "The completed animal-wise/OOD analysis is sufficient as the diagnostic foundation. The next GPU work should test whether target adaptation, not another source-only revise loop, improves ARAMRA and especially M5.",
    80,
    495,
    1010,
    74,
    { size: 22, color: C.ink, bold: true, face: "Aptos Display" },
  );

  await fs.mkdir(RUN_DIR, { recursive: true });
  await fs.mkdir(PREVIEW_DIR, { recursive: true });
  for (let i = 0; i < p.slides.count; i += 1) {
    const slide = p.slides.getItem(i);
    const blob = await p.export({ slide, format: "png", scale: 1 });
    await saveBlob(blob, path.join(PREVIEW_DIR, `slide-${String(i + 1).padStart(2, "0")}.png`));
  }
  const pptx = await PresentationFile.exportPptx(p);
  await pptx.save(OUT);
  const stat = await fs.stat(OUT);
  console.log(JSON.stringify({ out: OUT, bytes: stat.size, slideCount: p.slides.count, previewDir: PREVIEW_DIR }, null, 2));
}

build().catch((error) => {
  console.error(error.stack || error.message || String(error));
  process.exit(1);
});
