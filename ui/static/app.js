const templateStatus = document.getElementById("template-status");
const processStatus = document.getElementById("process-status");
const parseStatus = document.getElementById("parse-status");

const templateLinks = document.getElementById("template-links");
const templatePdf = document.getElementById("template-pdf");
const templateJson = document.getElementById("template-json");

const numQsInput = document.getElementById("num-qs");
const templateJsonUrlInput = document.getElementById("template-json-url");
const warpedUrlInput = document.getElementById("warped-url");

const origPreview = document.getElementById("orig-preview");
const warpedPreview = document.getElementById("warped-preview");
const debugPreview = document.getElementById("debug-preview");

const examCodeEl = document.getElementById("exam-code");
const studentIdEl = document.getElementById("student-id");
const questionCountEl = document.getElementById("question-count");
const answersBody = document.getElementById("answers-body");

let lastTemplateJsonUrl = "";
let lastWarpedUrl = "";

const setStatus = (el, message, type = "") => {
  el.textContent = message;
  el.className = `status ${type}`.trim();
};

const updateSummary = (examCode, studentId, questions) => {
  examCodeEl.textContent = examCode || "-";
  studentIdEl.textContent = studentId || "-";
  questionCountEl.textContent = Object.keys(questions || {}).length.toString();
};

const fillAnswersTable = (questions) => {
  answersBody.innerHTML = "";
  const entries = Object.entries(questions || {})
    .map(([key, value]) => [Number(key), value])
    .sort((a, b) => a[0] - b[0]);

  if (!entries.length) {
    const row = document.createElement("tr");
    row.innerHTML = "<td colspan=\"2\">No answers parsed.</td>";
    answersBody.appendChild(row);
    return;
  }

  entries.forEach(([qNum, ans]) => {
    const row = document.createElement("tr");
    row.innerHTML = `<td>${qNum}</td><td>${ans || ""}</td>`;
    answersBody.appendChild(row);
  });
};

const generateTemplate = async () => {
  const numQs = Number.parseInt(numQsInput.value, 10) || 45;
  setStatus(templateStatus, "Generating template...", "");

  try {
    const response = await fetch("/api/generate-template", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ num_qs: numQs }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Failed to generate template");
    }

    lastTemplateJsonUrl = data.json_url;
    templatePdf.href = data.pdf_url;
    templateJson.href = data.json_url;
    templateLinks.classList.remove("hidden");
    templateJsonUrlInput.value = data.json_url;

    setStatus(templateStatus, "Template generated.", "good");
  } catch (error) {
    setStatus(templateStatus, error.message, "bad");
  }
};

const processImage = async () => {
  const fileInput = document.getElementById("image-file");
  if (!fileInput.files.length) {
    setStatus(processStatus, "Please select an image file.", "bad");
    return;
  }

  setStatus(processStatus, "Processing image...", "");

  const formData = new FormData();
  formData.append("image", fileInput.files[0]);

  try {
    const response = await fetch("/api/process-image", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Failed to process image");
    }

    lastWarpedUrl = data.warped_url;
    origPreview.src = data.original_url;
    warpedPreview.src = data.warped_url;
    warpedUrlInput.value = data.warped_url;

    setStatus(processStatus, "Warped image ready.", "good");
  } catch (error) {
    setStatus(processStatus, error.message, "bad");
  }
};

const parseAnswers = async () => {
  const warpedUrl = warpedUrlInput.value.trim() || lastWarpedUrl;
  const templateJsonUrl = templateJsonUrlInput.value.trim() || lastTemplateJsonUrl;

  if (!warpedUrl || !templateJsonUrl) {
    setStatus(parseStatus, "Provide both a template JSON and warped image URL.", "bad");
    return;
  }

  setStatus(parseStatus, "Parsing answers...", "");

  try {
    const response = await fetch("/api/parse-answers", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ warped_url: warpedUrl, template_json_url: templateJsonUrl }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Failed to parse answers");
    }

    updateSummary(data.exam_code, data.student_id, data.questions);
    fillAnswersTable(data.questions);

    if (data.debug_url) {
      debugPreview.src = data.debug_url;
    }

    setStatus(parseStatus, "Answers parsed.", "good");
  } catch (error) {
    setStatus(parseStatus, error.message, "bad");
  }
};

document.getElementById("btn-generate").addEventListener("click", generateTemplate);
document.getElementById("btn-process").addEventListener("click", processImage);
document.getElementById("btn-parse").addEventListener("click", parseAnswers);
