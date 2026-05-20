const fmtPct = (value) => value === null || value === undefined ? "--" : `${(value * 100).toFixed(1)}%`;
const fmtNum = (value, digits = 3) => Number(value || 0).toFixed(digits);
const byId = (id) => document.getElementById(id);

let state = null;

function fillSelect(select, values, fallback) {
  select.innerHTML = "";
  const all = [...new Set(values.filter(Boolean))];
  if (!all.includes(fallback)) all.unshift(fallback);
  all.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.appendChild(option);
  });
}

function currentBuildPayload() {
  return {
    base: byId("baseSelect").value,
    release1: byId("release1Select").value,
    release2: byId("release2Select").value,
    speed: byId("speedSelect").value,
    player_3pt: Number(byId("ratingInput").value),
  };
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Request failed");
  return data;
}

function renderSummary(data) {
  const summary = data.summary;
  byId("latestPatch").textContent = summary.latest_patch || "--";
  byId("sessionCount").textContent = summary.sessions;
  byId("attemptCount").textContent = summary.total_attempts.toLocaleString();
  byId("makeRate").textContent = fmtPct(summary.make_pct);
  byId("topScore").textContent = fmtNum(summary.top_score);
}

function renderLeaderboard(rows) {
  byId("leaderboardRows").innerHTML = rows.slice(0, 10).map((row) => `
    <tr>
      <td><strong>${row.base}</strong> / ${row.release1} / ${row.release2}</td>
      <td>${row.speed}</td>
      <td>${fmtNum(row.exploit_score)}</td>
      <td>${row.total_attempts}</td>
      <td>${fmtPct(row.green_pct)}</td>
    </tr>
  `).join("");
}

function renderPatchRows(rows) {
  const sorted = [...rows]
    .sort((a, b) => String(b.patch).localeCompare(String(a.patch)))
    .slice(0, 8);
  byId("patchRows").innerHTML = sorted.map((row) => `
    <div class="patch-row">
      <strong>${row.patch}</strong>
      <span>${row.base} / ${row.release1} / ${row.release2}</span>
      <strong>${fmtPct(row.mean_make_pct)}</strong>
    </div>
  `).join("");
}

function renderScore(data) {
  const badge = byId("confidenceBadge");
  badge.className = data.confidence;
  badge.textContent = data.confidence;
  byId("recommendation").textContent = data.recommendation;
  byId("exploitScore").textContent = fmtNum(data.exploit_score);
  byId("edge").textContent = `${data.edge >= 0 ? "+" : ""}${fmtPct(data.edge)}`;
  byId("expectedMake").textContent = fmtPct(data.expected_make_pct);
  byId("actualMake").textContent = fmtPct(data.actual_make_pct);
  byId("recentSessions").innerHTML = data.recent_sessions.length ? data.recent_sessions.map((row) => `
    <tr>
      <td>${row.date}</td>
      <td>${row.patch}</td>
      <td>${row.attempts}</td>
      <td>${row.makes}</td>
      <td>${row.greens}</td>
    </tr>
  `).join("") : `<tr><td colspan="5">No matching test sessions yet</td></tr>`;
}

async function scoreCurrentBuild() {
  const data = await api("/api/jumpshot/score", {
    method: "POST",
    body: JSON.stringify(currentBuildPayload()),
  });
  renderScore(data);
}

async function boot() {
  state = await api("/api/summary");
  renderSummary(state);
  renderLeaderboard(state.leaderboard);
  renderPatchRows(state.patch_impact);

  fillSelect(byId("baseSelect"), state.options.bases, "Curry");
  fillSelect(byId("release1Select"), state.options.release1, "Kobe");
  fillSelect(byId("release2Select"), state.options.release2, "Gay");
  fillSelect(byId("speedSelect"), state.options.speeds, "max");
  byId("ratingOutput").textContent = byId("ratingInput").value;
  await scoreCurrentBuild();
}

byId("ratingInput").addEventListener("input", (event) => {
  byId("ratingOutput").textContent = event.target.value;
});

byId("creatorForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  await scoreCurrentBuild();
});

byId("sessionForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  const payload = {
    ...currentBuildPayload(),
    date: form.get("date"),
    patch: form.get("patch"),
    attempts: Number(form.get("attempts")),
    makes: Number(form.get("makes")),
    greens: form.get("greens"),
  };
  try {
    const data = await api("/api/sessions", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    byId("sessionMessage").textContent = "Session saved.";
    renderScore(data.score);
    state = await api("/api/summary");
    renderSummary(state);
    renderLeaderboard(state.leaderboard);
    renderPatchRows(state.patch_impact);
  } catch (error) {
    byId("sessionMessage").textContent = error.message;
  }
});

boot().catch((error) => {
  byId("recommendation").textContent = error.message;
});

