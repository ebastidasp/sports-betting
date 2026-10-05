(() => {
  const formatDate = (value, options = {}) => {
    if (!value) return "Time not listed";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    return new Intl.DateTimeFormat(undefined, options).format(date);
  };

  const localizeTimes = (root = document) => {
    root.querySelectorAll("time[data-utc]").forEach((element) => {
      const value = element.dataset.utc;
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) return;
      element.textContent = formatDate(value, {
        dateStyle: "medium",
        timeStyle: "short",
      });
      element.title = `${date.toISOString()} (UTC)`;
    });
  };

  const text = (tag, className, value) => {
    const element = document.createElement(tag);
    if (className) element.className = className;
    element.textContent = value ?? "";
    return element;
  };

  const safeHttpsUrl = (value) => {
    try {
      const url = new URL(value);
      return url.protocol === "https:" ? url.href : "";
    } catch {
      return "";
    }
  };

  const renderTeam = (team, side, score, showScore) => {
    const block = text("div", `team-block team-${side}`, "");
    const imageUrl = safeHttpsUrl(team.image_url);
    let logo;
    if (imageUrl) {
      logo = document.createElement("img");
      logo.className = "team-logo";
      logo.alt = "";
      logo.loading = "lazy";
      logo.src = imageUrl;
    } else {
      logo = text("span", "team-logo logo-fallback", (team.acronym || team.name || "?").slice(0, 2).toUpperCase());
    }
    const name = text("span", "team-name", team.acronym || team.name);
    if (side === "a") {
      block.append(logo, name);
      if (showScore) block.append(text("strong", "team-score", score));
    } else {
      if (showScore) block.append(text("strong", "team-score", score));
      block.append(name, logo);
    }
    return block;
  };

  const renderCard = (match, state) => {
    const article = text("article", "match-card", "");
    article.dataset.matchId = match.id;

    const top = text("div", "match-card-top", "");
    top.append(text("span", `event-tag event-${String(match.event || "").toLowerCase()}`, match.event));
    if (state === "running") {
      const live = text("span", "live-tag", "LIVE");
      live.prepend(text("span", "live-pulse", ""));
      top.append(live);
    } else {
      top.append(text("span", "status-label", (match.status || "unknown").replaceAll("_", " ")));
    }
    article.append(top);
    article.append(text("div", "match-stage", match.stage || match.series_name || match.event_name));

    const scoreRow = text("div", "teams-score-row", "");
    const haveScore = state === "running" && match.score_a !== null && match.score_b !== null;
    scoreRow.append(renderTeam(match.team_a, "a", match.score_a, haveScore));
    scoreRow.append(text("span", "versus", haveScore ? "—" : "VS"));
    scoreRow.append(renderTeam(match.team_b, "b", match.score_b, haveScore));
    article.append(scoreRow);

    const bottom = text("div", "match-card-bottom", "");
    const format = state === "running" && match.current_game
      ? `Game ${match.current_game} in progress`
      : match.best_of ? `Best of ${match.best_of}` : "Format not listed";
    bottom.append(text("span", "", format));
    if (match.scheduled_at) {
      const time = text("time", "match-time", formatDate(match.scheduled_at, { dateStyle: "medium", timeStyle: "short" }));
      time.dateTime = match.scheduled_at;
      time.dataset.utc = match.scheduled_at;
      bottom.append(time);
    }
    article.append(bottom);

    const actions = text("div", "match-card-actions", "");
    if (match.rescheduled) actions.append(text("span", "rescheduled-label", "Rescheduled"));
    const query = new URLSearchParams({
      team_a: match.team_a.name || "",
      team_b: match.team_b.name || "",
      best_of: String(match.best_of || 5),
    });
    const forecast = text("a", "text-link", "Pre-match forecast ↗");
    forecast.href = `/predictor?${query.toString()}`;
    actions.append(forecast);
    const streamUrl = safeHttpsUrl(match.stream_url);
    if (streamUrl) {
      const stream = text("a", "text-link stream-link", "Stream ↗");
      stream.href = streamUrl;
      stream.target = "_blank";
      stream.rel = "noopener noreferrer";
      actions.append(stream);
    }
    article.append(actions);
    return article;
  };

  const updateFeedNote = (state, data) => {
    const noteId = state === "running" ? "live-feed-note" : "upcoming-feed-note";
    const updatedId = state === "running" ? "live-updated" : "upcoming-updated";
    let note = document.getElementById(noteId);
    if (data.error) {
      if (!note) {
        note = text("div", "notice", "");
        note.id = noteId;
        document.getElementById(state === "running" ? "live-matches" : "upcoming-matches").before(note);
      }
      note.className = `notice ${data.stale ? "notice-warning" : "notice-info"}`;
      note.textContent = data.error + (data.stale && data.updated_at ? ` Last successful update: ${formatDate(data.updated_at, { dateStyle: "medium", timeStyle: "short" })}.` : "");
    } else if (note) {
      note.remove();
    }
    const updated = document.getElementById(updatedId);
    if (updated) updated.textContent = data.updated_at
      ? `Updated ${formatDate(data.updated_at, { dateStyle: "medium", timeStyle: "short" })}`
      : "Waiting for feed";
  };

  const refreshState = async (state) => {
    try {
      const response = await fetch(`/api/matches?state=${state}`, { cache: "no-store" });
      if (!response.ok) return;
      const data = await response.json();
      const container = document.getElementById(state === "running" ? "live-matches" : "upcoming-matches");
      if (!container) return;
      container.replaceChildren();
      if (data.matches.length) {
        data.matches.forEach((match) => container.append(renderCard(match, state)));
      } else {
        const empty = text("div", "empty-state", "");
        empty.append(text("span", "empty-icon", state === "running" ? "◉" : "⌁"));
        empty.append(text("strong", "", state === "running" ? "No Worlds/MSI series live" : "No Worlds/MSI matches found"));
        empty.append(text("p", "", state === "running"
          ? "When a supported series starts, its current match score will appear here."
          : "Schedules appear here when the provider lists a match in either international event."));
        container.append(empty);
      }
      updateFeedNote(state, data);
      localizeTimes(container);
    } catch {
      // Keep the last rendered snapshot visible during a temporary disconnect.
    }
  };

  localizeTimes();
  if (document.getElementById("live-matches")) {
    window.setInterval(() => {
      if (document.visibilityState === "visible") refreshState("running");
    }, 30_000);
    window.setInterval(() => {
      if (document.visibilityState === "visible") refreshState("upcoming");
    }, 5 * 60_000);
  }
})();
