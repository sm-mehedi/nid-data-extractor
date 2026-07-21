(() => {
  const form = document.getElementById("upload-form");
  const submitBtn = document.getElementById("submit-btn");
  const loadingEl = document.getElementById("loading");
  const loadingText = document.getElementById("loading-text");
  const resultsEl = document.getElementById("results");
  const bannerEl = document.getElementById("banner");
  const warningsEl = document.getElementById("warnings");
  const fieldRowsEl = document.getElementById("field-rows");
  const rawJsonEl = document.getElementById("raw-json");
  const errorPanelEl = document.getElementById("error-panel");
  const errorListEl = document.getElementById("error-list");

  const FIELD_LABELS = {
    name: "Full Name",
    fatherName: "Father's Name",
    motherName: "Mother's Name",
    dateOfBirth: "Date of Birth",
    nidNumber: "NID Number",
    presentAddress: "Present Address",
    permanentAddress: "Permanent Address",
  };

  const LOADING_MESSAGES = [
    "Uploading images…",
    "Running OCR on both sides…",
    "Validating the machine-readable zone…",
    "Translating and structuring fields…",
    "Almost done…",
  ];

  // The form's submit handler is attached FIRST and unconditionally, before
  // any other setup runs. This is deliberate: if some later, non-essential
  // setup step throws (e.g. a null element lookup), it must never be able to
  // prevent event.preventDefault() from being wired up — otherwise the
  // browser falls back to a native GET form submission (file inputs then
  // serialize as their filename as plain text in the URL query string,
  // e.g. "?front_image=photo.jpg", and no actual image bytes are ever sent).
  if (form) {
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const frontInput = document.getElementById("front_image");
      const backInput = document.getElementById("back_image");
      const frontFile = frontInput && frontInput.files[0];
      const backFile = backInput && backInput.files[0];
      if (!frontFile || !backFile) return;

      const formData = new FormData();
      formData.append("front_image", frontFile);
      formData.append("back_image", backFile);

      startLoading();
      try {
        const response = await fetch("/api/v1/nid/extract", {
          method: "POST",
          body: formData,
        });
        const payload = await response.json();
        stopLoading();
        if (payload.success) {
          renderResults(payload);
        } else {
          renderErrors(payload);
        }
      } catch (err) {
        stopLoading();
        renderErrors({ errors: ["Network error: could not reach the server."] });
      }
    });
  } else {
    console.error("upload-form element not found; submit handler not attached.");
  }

  function setupPreview(inputId, previewId, zoneId) {
    const input = document.getElementById(inputId);
    const preview = document.getElementById(previewId);
    const zone = document.getElementById(zoneId);
    if (!input || !preview || !zone) {
      console.error(`setupPreview: missing element(s) for ${inputId}`);
      return;
    }
    input.addEventListener("change", () => {
      const file = input.files && input.files[0];
      if (!file) return;
      const url = URL.createObjectURL(file);
      preview.innerHTML = "";
      const img = document.createElement("img");
      img.src = url;
      img.alt = "";
      preview.appendChild(img);
      zone.classList.add("has-file");
    });
  }

  try {
    setupPreview("front_image", "preview-front", "zone-front");
    setupPreview("back_image", "preview-back", "zone-back");
  } catch (err) {
    console.error("Preview setup failed (non-fatal):", err);
  }

  let loadingInterval = null;

  function startLoading() {
    resultsEl.classList.add("hidden");
    errorPanelEl.classList.add("hidden");
    loadingEl.classList.remove("hidden");
    submitBtn.disabled = true;
    let i = 0;
    loadingText.textContent = LOADING_MESSAGES[0];
    loadingInterval = setInterval(() => {
      i = (i + 1) % LOADING_MESSAGES.length;
      loadingText.textContent = LOADING_MESSAGES[i];
    }, 1800);
  }

  function stopLoading() {
    loadingEl.classList.add("hidden");
    submitBtn.disabled = false;
    if (loadingInterval) clearInterval(loadingInterval);
  }

  function clearChildren(el) {
    while (el.firstChild) el.removeChild(el.firstChild);
  }

  function renderResults(payload) {
    clearChildren(fieldRowsEl);
    clearChildren(warningsEl);
    bannerEl.classList.add("hidden");

    const lowConfidenceFields = new Set();
    for (const w of payload.warnings || []) {
      const match = /^Low confidence on field: (.+)$/.exec(w);
      if (match) lowConfidenceFields.add(match[1]);
    }

    const data = payload.data || {};
    for (const [key, label] of Object.entries(FIELD_LABELS)) {
      const row = document.createElement("div");
      row.className = "field-row";

      const labelEl = document.createElement("span");
      labelEl.className = "field-label";
      labelEl.textContent = label;

      const valueEl = document.createElement("span");
      valueEl.className = "field-value";
      const value = data[key];
      valueEl.textContent = value ? value : "—";

      if (!value || lowConfidenceFields.has(key)) {
        const badge = document.createElement("span");
        badge.className = "badge-low-confidence";
        badge.textContent = "LOW CONFIDENCE";
        valueEl.appendChild(badge);
      }

      row.appendChild(labelEl);
      row.appendChild(valueEl);
      fieldRowsEl.appendChild(row);
    }

    const otherWarnings = (payload.warnings || []).filter((w) => !/^Low confidence on field: /.test(w));
    if (otherWarnings.length > 0) {
      warningsEl.classList.remove("hidden");
      const title = document.createElement("strong");
      title.textContent = "Warnings";
      const list = document.createElement("ul");
      for (const w of otherWarnings) {
        const li = document.createElement("li");
        li.textContent = w;
        list.appendChild(li);
      }
      warningsEl.appendChild(title);
      warningsEl.appendChild(list);
    }

    rawJsonEl.textContent = JSON.stringify(payload, null, 2);
    resultsEl.classList.remove("hidden");
  }

  function renderErrors(payload) {
    clearChildren(errorListEl);
    const errors = (payload && payload.errors) || ["Unknown error."];
    for (const err of errors) {
      const li = document.createElement("li");
      li.textContent = err;
      errorListEl.appendChild(li);
    }
    errorPanelEl.classList.remove("hidden");
  }
})();
