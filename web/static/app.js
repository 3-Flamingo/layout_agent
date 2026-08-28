(() => {
  let cases = [];
  let currentId = null;
  let currentPayload = null;
  let manifestMeta = null;
  let viewMode = "anno";
  let selectedId = null;
  let hitRegions = [];
  let stepEvents = [];
  let stepFilter = "all";
  let cam = { zoom: 1, x: 0, y: 0, caseId: null, suppressClick: false };
  let contentSize = { w: 0, h: 0 };

  const el = (id) => document.getElementById(id);
  const canvas = el("draw-canvas");
  const ctx = canvas.getContext("2d");

  async function fetchJson(url, opts) {
    const res = await fetch(url, opts);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  }

  function hex(c) {
    if (!c) return "#cccccc";
    return c.startsWith("#") ? c : `#${c}`;
  }

  function clearStepCards(placeholder) {
    stepEvents = [];
    const box = el("step-cards");
    if (!box) return;
    box.innerHTML = `<div class="step-empty">${placeholder || "点击左侧「运行 Agent」后，步骤会以可展开列表显示（默认收起，点击查看详情）。"}</div>`;
    const sub = el("steps-subtitle");
    if (sub) sub.textContent = "尚未运行";
  }

  function applyStepFilter() {
    const box = el("step-cards");
    if (!box) return;
    box.querySelectorAll(".step-row").forEach((row) => {
      const phase = row.dataset.phase || "";
      const show =
        stepFilter === "all" ||
        phase === stepFilter ||
        (stepFilter === "decide" && ["decide", "accept"].includes(phase)) ||
        (stepFilter === "schedule" && phase === "schedule") ||
        (stepFilter === "evaluate" && phase === "evaluate") ||
        (stepFilter === "execute" && phase === "execute");
      row.classList.toggle("hidden-by-filter", !show);
    });
  }

  function phaseIcon(phase) {
    const map = {
      perceive: "P",
      evaluate: "E",
      decide: "D",
      execute: "X",
      schedule: "B",
      accept: "✓",
      finalize: "F",
      done: "✓",
      error: "!",
      timeout: "T",
    };
    return map[phase] || "·";
  }

  function stepHeadline(evt) {
    const phase = evt.phase || "step";
    const labels = {
      perceive: "Perceive",
      evaluate: "Evaluate",
      decide: "Decide",
      execute: "Execute",
      schedule: "Brain",
      accept: "Accept",
      finalize: "Finalize",
      done: "Done",
      error: "Error",
      timeout: "Timeout",
    };
    const name = labels[phase] || phase;
    let dim = "";
    if (evt.loop_step) dim = `step ${evt.loop_step}`;
    if (phase === "decide" && evt.use_llm) dim = (dim ? dim + " · " : "") + "LLM";
    if (phase === "decide" && evt.use_llm === false) dim = (dim ? dim + " · " : "") + "rules";
    if (phase === "schedule" && evt.schedule?.decision) {
      dim = (dim ? dim + " · " : "") + evt.schedule.decision;
    }
    if (evt.metrics?.P != null) dim = (dim ? dim + " · " : "") + `P=${Number(evt.metrics.P).toFixed(1)}`;
    return { name, dim, full: evt.message || name };
  }

  function appendStepCard(evt, opts = {}) {
    const box = el("step-cards");
    if (!box) return;
    const empty = box.querySelector(".step-empty");
    if (empty) empty.remove();
    box.querySelectorAll(".step-row.is-live").forEach((n) => n.classList.remove("is-live"));

    const idx = stepEvents.length + 1;
    stepEvents.push(evt);
    const phase = evt.phase || "step";
    const { name, dim, full } = stepHeadline(evt);

    const chips = [];
    if (evt.loop_step) chips.push(`<span class="chip">loop #${evt.loop_step}</span>`);
    if (evt.use_llm != null) chips.push(`<span class="chip">${evt.use_llm ? "LLM" : "规则"}</span>`);
    if (evt.model) chips.push(`<span class="chip">${evt.model}</span>`);
    if (evt.metrics?.P != null) {
      const bad = Number(evt.metrics.P) > 0;
      chips.push(`<span class="chip ${bad ? "bad" : "ok"}">P=${Number(evt.metrics.P).toFixed(2)}</span>`);
    }
    if (evt.metrics?.IF_total != null) {
      chips.push(`<span class="chip">IF=${Number(evt.metrics.IF_total).toFixed(0)}</span>`);
    }
    if (evt.percent != null) chips.push(`<span class="chip">${evt.percent}%</span>`);

    let body = "";
    if (evt.schedule) body = JSON.stringify(evt.schedule, null, 2);
    else if (evt.actions?.length) body = JSON.stringify(evt.actions, null, 2);
    else if (evt.result) {
      body = JSON.stringify(
        {
          improved: evt.result.improved,
          steps: evt.result.steps,
          reason: evt.result.reason,
          time_ms: evt.result.time_ms,
          P_before: evt.result.report_before?.metrics?.P,
          P_after: evt.result.report_after?.metrics?.P,
          schedule_verdicts: evt.result.schedule_verdicts,
        },
        null,
        2
      );
    } else if (evt.metrics) body = JSON.stringify(evt.metrics, null, 2);
    else if (evt.exec_note) body = String(evt.exec_note);
    else body = full;

    const row = document.createElement("details");
    row.className = `step-row phase-${phase}`;
    row.dataset.phase = phase;
    if (opts.live !== false && !opts.silent) row.classList.add("is-live");
    row.open = !!opts.open;

    const rightBits = [];
    if (evt.percent != null) rightBits.push(`${evt.percent}%`);
    rightBits.push(`#${idx}`);

    row.innerHTML = `
      <summary class="step-summary">
        <span class="step-chevron">▸</span>
        <span class="step-icon ${phase}">${phaseIcon(phase)}</span>
        <span class="step-summary-title">${name}${dim ? `<span class="dim">${dim}</span>` : ""}</span>
        <span class="step-summary-right">${rightBits.join(" · ")}</span>
      </summary>
      <div class="step-detail">
        <div class="step-detail-meta">${chips.join("")}</div>
        <div class="step-detail-msg muted" style="margin-bottom:8px;font-size:0.8rem;color:var(--muted)">${String(full).replace(/</g, "&lt;")}</div>
        <pre class="step-body">${String(body).replace(/</g, "&lt;")}</pre>
      </div>
    `;
    box.appendChild(row);
    if (!opts.silent) row.scrollIntoView({ behavior: "smooth", block: "nearest" });
    const sub = el("steps-subtitle");
    if (sub) sub.textContent = `${stepEvents.length} 步 · 点击展开详情`;
    applyStepFilter();
  }

  function hydrateStepsFromRun(run) {
    if (!run) {
      clearStepCards();
      return;
    }
    clearStepCards();
    const empty = el("step-cards")?.querySelector(".step-empty");
    if (empty) empty.remove();

    appendStepCard(
      {
        phase: "evaluate",
        message: `历史结果 · 优化前 P=${Number(run.report_before?.metrics?.P || 0).toFixed(2)}`,
        metrics: run.report_before?.metrics,
        percent: 0,
      },
      { silent: true }
    );

    (run.actions_applied || []).forEach((actions, i) => {
      appendStepCard(
        {
          phase: "decide",
          message: `历史步骤 ${i + 1}：决策动作`,
          actions,
          loop_step: i + 1,
        },
        { silent: true }
      );
      const sv = (run.schedule_verdicts || [])[i];
      if (sv) {
        appendStepCard(
          {
            phase: "schedule",
            message: `历史大脑裁决 → ${sv.decision}`,
            schedule: sv,
            loop_step: i + 1,
          },
          { silent: true }
        );
      }
    });

    if (run.report_after) {
      appendStepCard(
        {
          phase: "done",
          message: `历史结果 · improved=${run.improved} · ${run.reason}`,
          result: run,
          metrics: run.report_after.metrics,
          percent: 100,
        },
        { silent: true }
      );
    }
    const sub = el("steps-subtitle");
    if (sub) sub.textContent = `已加载历史运行 · ${stepEvents.length} 条`;
  }

  function activeState() {
    const p = currentPayload;
    if (!p) return null;
    if ((viewMode === "after" || viewMode === "diff") && p.best_state) return p.best_state;
    return p.state;
  }

  function dist2(a, b) {
    if (!a || !b) return 0;
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  function computeDiff(before, after) {
    const changes = [];
    if (!before || !after) return changes;
    const mapLin = Object.fromEntries((before.linears || []).map((x) => [x.id, x]));
    for (const a of after.linears || []) {
      const b = mapLin[a.id];
      if (!b) continue;
      const d = dist2(a.point, b.point);
      if (d > 0.8) {
        changes.push({
          id: a.id,
          kind: "linear",
          dist: d,
          before: b.point,
          after: a.point,
          label: a.text?.content || String(a.value),
          color: a.color,
        });
      }
    }
    const mapRad = Object.fromEntries(
      [...(before.diameters || []), ...(before.radials || [])].map((x) => [x.id, x])
    );
    for (const a of [...(after.diameters || []), ...(after.radials || [])]) {
      const b = mapRad[a.id];
      if (!b) continue;
      const ae = a.path?.[a.path.length - 1] || a.anchor;
      const be = b.path?.[b.path.length - 1] || b.anchor;
      const d = dist2(ae, be);
      if (d > 0.8) {
        changes.push({
          id: a.id,
          kind: a.category || "radial",
          dist: d,
          before: be,
          after: ae,
          label: a.text?.content || a.id,
          color: a.color,
        });
      }
    }
    changes.sort((x, y) => y.dist - x.dist);
    return changes;
  }

  function renderDiffPanel(changes, run) {
    const banner = el("diff-banner");
    const list = el("diff-list");
    if (!banner || !list) return;
    if (viewMode !== "diff") {
      banner.hidden = true;
      list.hidden = true;
      return;
    }
    if (!currentPayload?.best_state) {
      banner.hidden = false;
      banner.innerHTML = "尚未运行 Agent，没有可对比的优化结果。";
      list.hidden = true;
      return;
    }
    const pb = run?.report_before?.metrics?.P;
    const pa = run?.report_after?.metrics?.P;
    const delta = pb != null && pa != null ? (pb - pa).toFixed(2) : "?";
    banner.hidden = false;
    banner.innerHTML = `
      <strong>优化对比</strong>：共移动 <strong>${changes.length}</strong> 处标注
      ${pb != null ? `· P ${Number(pb).toFixed(1)} → ${Number(pa).toFixed(1)}（Δ ${delta}）` : ""}
      <span class="diff-legend">
        <span><i style="background:#6b7280"></i>优化前（虚线）</span>
        <span><i style="background:#3ecf8e"></i>优化后 / 已移动</span>
        <span><i style="background:#3d9bfd"></i>位移箭头</span>
      </span>
    `;
    list.hidden = false;
    if (!changes.length) {
      list.innerHTML = `<div style="padding:12px;color:var(--muted);font-size:0.82rem">几何位置几乎未变（位移 &lt; 0.8）。分值变化可能来自评价口径/像素重叠统计，而非肉眼可见的位置调整。</div>`;
      return;
    }
    list.innerHTML = `
      <table>
        <thead><tr><th>ID</th><th>类型</th><th>标注</th><th>位移</th><th>前 → 后</th></tr></thead>
        <tbody>
          ${changes
            .map(
              (c) => `<tr data-id="${c.id}">
              <td>${c.id}</td>
              <td>${c.kind}</td>
              <td class="moved">${c.label || ""}</td>
              <td class="moved">${c.dist.toFixed(2)}</td>
              <td>(${c.before.x.toFixed(1)},${c.before.y.toFixed(1)}) → (${c.after.x.toFixed(1)},${c.after.y.toFixed(1)})</td>
            </tr>`
            )
            .join("")}
        </tbody>
      </table>`;
    list.querySelectorAll("tr[data-id]").forEach((tr) => {
      tr.onclick = () => selectAnno(tr.dataset.id);
    });
  }

  function applyCamera() {
    const stage = el("canvas-stage");
    if (!stage) return;
    stage.style.transform = `translate(${cam.x}px, ${cam.y}px) scale(${cam.zoom})`;
    stage.style.transformOrigin = "0 0";
    const btn = el("btn-zoom-reset");
    if (btn) btn.textContent = `${Math.round(cam.zoom * 100)}%`;
  }

  function fitCamera(cssW, cssH, force = false) {
    contentSize = { w: cssW, h: cssH };
    const wrap = el("canvas-wrap");
    if (!wrap || !cssW || !cssH) return;
    const needFit = force || cam.caseId !== currentId || !cam.caseId;
    if (needFit) {
      const ww = Math.max(wrap.clientWidth, 100);
      const wh = Math.max(wrap.clientHeight, 100);
      cam.zoom = 1;
      cam.x = (ww - cssW) / 2;
      cam.y = (wh - cssH) / 2;
      cam.caseId = currentId;
    }
    applyCamera();
  }

  function zoomAt(mx, my, factor) {
    const prev = cam.zoom;
    const next = Math.min(8, Math.max(0.15, prev * factor));
    if (next === prev) return;
    const sx = (mx - cam.x) / prev;
    const sy = (my - cam.y) / prev;
    cam.zoom = next;
    cam.x = mx - sx * next;
    cam.y = my - sy * next;
    applyCamera();
  }

  function resetCamera() {
    if (contentSize.w && contentSize.h) fitCamera(contentSize.w, contentSize.h, true);
    else applyCamera();
  }

  function worldToPx(x, y, raster) {
    const ox = raster.origin.x;
    const oy = raster.origin.y;
    const ps = raster.pixel_size || 0.35;
    const col = (x - ox) / ps;
    const row = (oy - y) / ps;
    return [col, row];
  }

  function drawState(state, opts = {}) {
    hitRegions = [];
    if (!state?.raster) return;
    const raster = state.raster;
    const w = raster.width || 800;
    const h = raster.height || 600;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const maxW = Math.min(1100, el("canvas-wrap").clientWidth - 8 || 900);
    const maxH = 640;
    const scale = Math.min(maxW / w, maxH / h, 2);
    const cssW = Math.round(w * scale);
    const cssH = Math.round(h * scale);
    canvas.width = Math.round(w * scale * dpr);
    canvas.height = Math.round(h * scale * dpr);
    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${cssH}px`;
    const imgEl = el("icm-img");
    imgEl.style.width = `${cssW}px`;
    imgEl.style.height = `${cssH}px`;
    imgEl.style.objectFit = "fill";
    fitCamera(cssW, cssH, false);

    ctx.setTransform(dpr * scale, 0, 0, dpr * scale, 0, 0);
    ctx.clearRect(0, 0, w, h);
    // Drawing is in raster-pixel space, then scaled to fit the viewport.
    const px = 1 / Math.max(scale, 0.05);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";

    const transparentBg = !!opts.transparentBg;
    if (!transparentBg) {
      ctx.fillStyle = "#0c1118";
      ctx.fillRect(0, 0, w, h);
    }

    const showObs = el("tog-obs").checked;
    const showLin = el("tog-linear").checked;
    const showDia = el("tog-dia").checked;
    const showRad = el("tog-rad").checked;
    const showLabels = el("tog-labels").checked;
    const showIds = el("tog-ids").checked;
    const movedIds = opts.movedIds instanceof Set ? opts.movedIds : new Set(opts.movedIds || []);
    const ghostState = opts.ghostState || null;
    const to = (p) => worldToPx(p.x, p.y, raster);

    if (showObs) {
      ctx.lineWidth = 1.1 * px;
      ctx.strokeStyle = transparentBg ? "rgba(220,230,240,0.35)" : "rgba(180,190,200,0.55)";
      for (const ob of state.obstacles || []) {
        const poly = ob.polyline || [];
        if (poly.length < 2) continue;
        ctx.beginPath();
        poly.forEach((pt, i) => {
          const [cx, cy] = to(pt);
          if (i === 0) ctx.moveTo(cx, cy);
          else ctx.lineTo(cx, cy);
        });
        ctx.stroke();
      }
    }

    function strokePath(pts, color, widthCss = 1.5, dashed = false) {
      if (!pts || pts.length < 2) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = widthCss * px;
      ctx.setLineDash(dashed ? [5 * px, 4 * px] : []);
      ctx.beginPath();
      pts.forEach((pt, i) => {
        const [cx, cy] = to(pt);
        if (i === 0) ctx.moveTo(cx, cy);
        else ctx.lineTo(cx, cy);
      });
      ctx.stroke();
      ctx.setLineDash([]);
    }

    /** CAD-style filled arrow: tip at `tip`, shaft toward tip along (dx,dy). */
    function drawArrowHead(tip, dx, dy, sizeWorld, color) {
      const len = Math.hypot(dx, dy) || 1;
      const ux = dx / len;
      const uy = dy / len;
      const [tx, ty] = to(tip);
      // Convert world arrow size → canvas px
      const [bx, by] = to({ x: tip.x - ux * sizeWorld, y: tip.y - uy * sizeWorld });
      const hx = tx - bx;
      const hy = ty - by;
      const px2 = -hy * 0.35;
      const py2 = hx * 0.35;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(tx, ty);
      ctx.lineTo(bx + px2, by + py2);
      ctx.lineTo(bx - px2, by - py2);
      ctx.closePath();
      ctx.fill();
    }

    function linearFeet(lin) {
      // Prefer CAD dimAngle projection (same as CadAlignedDimension)
      if (lin.dim_angle != null && Number.isFinite(Number(lin.dim_angle))) {
        const rad = (Number(lin.dim_angle) * Math.PI) / 180;
        const ax = Math.cos(rad);
        const ay = Math.sin(rad);
        const foot = (p) => {
          const dx = p.x - lin.point.x;
          const dy = p.y - lin.point.y;
          const t = dx * ax + dy * ay;
          return { x: lin.point.x + t * ax, y: lin.point.y + t * ay };
        };
        return { f1: foot(lin.p1), f2: foot(lin.p2) };
      }
      const horiz = Math.abs(lin.dir.y) >= Math.abs(lin.dir.x);
      if (horiz) {
        return {
          f1: { x: lin.p1.x, y: lin.point.y },
          f2: { x: lin.p2.x, y: lin.point.y },
        };
      }
      return {
        f1: { x: lin.point.x, y: lin.p1.y },
        f2: { x: lin.point.x, y: lin.p2.y },
      };
    }

    function linearGeom(lin) {
      const { f1, f2 } = linearFeet(lin);
      return [
        [lin.p1, f1],
        [lin.p2, f2],
        [f1, f2],
      ];
    }

    /** Extra dim-line segments: outer arrow stubs + outside-text leaders (CAD). */
    function linearExtraSegs(lin) {
      const { f1, f2 } = linearFeet(lin);
      const dx = f2.x - f1.x;
      const dy = f2.y - f1.y;
      const dist = Math.hypot(dx, dy);
      if (dist < 1e-9) return [];
      const ux = dx / dist;
      const uy = dy / dist;
      const aw = Number(lin.arrow_width) || 2.5;
      const segs = [];
      const arrowsOut = 2 * aw + 1 > dist;
      if (arrowsOut) {
        segs.push([
          { x: f1.x - ux * (aw + 3), y: f1.y - uy * (aw + 3) },
          { x: f1.x - ux * aw, y: f1.y - uy * aw },
        ]);
        segs.push([
          { x: f2.x + ux * aw, y: f2.y + uy * aw },
          { x: f2.x + ux * (aw + 3), y: f2.y + uy * (aw + 3) },
        ]);
      }
      const tp = lin.text_point;
      if (!tp || tp.x == null) return segs;
      const t = (tp.x - f1.x) * ux + (tp.y - f1.y) * uy;
      const outside = t < -0.5 || t > dist + 0.5;
      if (!outside) return segs;
      const gap = aw + 3;
      const comb =
        t < dist * 0.5
          ? { x: f1.x - ux * gap, y: f1.y - uy * gap }
          : { x: f2.x + ux * gap, y: f2.y + uy * gap };
      // extend main dim toward text side (short stubs already cover arrows)
      if (t < dist * 0.5) segs.push([comb, f1]);
      else segs.push([f2, comb]);
      const tw =
        (lin.text && Number(lin.text.width)) ||
        Math.max(6, String((lin.text && lin.text.content) || "").length * 3);
      const th = (lin.text && Number(lin.text.height)) || 3.5 * (lin.dim_scale || 1);
      // underline BELOW text (offset on dim normal), not through glyphs
      let nx = -uy;
      let ny = ux;
      const feat = {
        x: 0.5 * (lin.p1.x + lin.p2.x),
        y: 0.5 * (lin.p1.y + lin.p2.y),
      };
      if ((tp.x - feat.x) * nx + (tp.y - feat.y) * ny < 0) {
        nx = -nx;
        ny = -ny;
      }
      const off = th * 0.55;
      const half = tw / 2;
      const underA = { x: tp.x - ux * half + nx * off, y: tp.y - uy * half + ny * off };
      const underB = { x: tp.x + ux * half + nx * off, y: tp.y + uy * half + ny * off };
      segs.push([underA, underB]);
      const d0 = Math.hypot(underA.x - comb.x, underA.y - comb.y);
      const d1 = Math.hypot(underB.x - comb.x, underB.y - comb.y);
      const turn = d0 <= d1 ? underA : underB;
      if (Math.hypot(turn.x - comb.x, turn.y - comb.y) > 1e-3) segs.push([comb, turn]);
      return segs;
    }

    function drawLinearArrows(lin, color) {
      const { f1, f2 } = linearFeet(lin);
      const dx = f2.x - f1.x;
      const dy = f2.y - f1.y;
      const dist = Math.hypot(dx, dy);
      if (dist < 1e-9) return;
      const ux = dx / dist;
      const uy = dy / dist;
      const aw = Number(lin.arrow_width) || 2.5;
      if (2 * aw + 1 > dist) {
        // Outside: tip on feet, arrow body outside pointing inward
        drawArrowHead(f1, ux, uy, aw, color);
        drawArrowHead(f2, -ux, -uy, aw, color);
      } else {
        drawArrowHead(f1, -ux, -uy, aw, color);
        drawArrowHead(f2, ux, uy, aw, color);
      }
    }

    function linearMid(lin) {
      if (lin.text_point && lin.text_point.x != null && lin.text_point.y != null) {
        return lin.text_point;
      }
      const segs = linearGeom(lin);
      const dim = segs[2];
      return {
        x: (dim[0].x + dim[1].x) / 2,
        y: (dim[0].y + dim[1].y) / 2,
      };
    }

    function textAngleRad(lin) {
      // CAD: dimtih/dimtoh → force horizontal; else text parallel to dim line (readable)
      if (lin.dimtih || lin.dimtoh) return 0;
      if (lin.dim_angle == null) return 0;
      let deg = Number(lin.dim_angle);
      while (deg > 90) deg -= 180;
      while (deg <= -90) deg += 180;
      // Canvas y-down vs CAD y-up
      return (-deg * Math.PI) / 180;
    }

    function labelAt(p, text, color, aid, meta, opts = {}) {
      if (!showLabels && !showIds) return;
      const [cx, cy] = to(p);
      const label = showLabels ? text : "";
      const idPart = showIds ? (label ? ` [${aid}]` : aid) : "";
      const full = `${label}${idPart}`.trim();
      if (!full) return;

      // World text height → raster px (matches CAD dimtxt*dimScale)
      const worldH = Math.max(Number(opts.worldHeight) || 3.5, 1.5);
      const fontRaster = Math.max(worldH / (raster.pixel_size || 0.35), 8);
      const ang = opts.angleRad || 0;
      const active = selectedId === aid;
      const moved = movedIds.has(aid);

      ctx.save();
      ctx.translate(cx, cy);
      if (ang) ctx.rotate(ang);
      ctx.font = `600 ${fontRaster}px JetBrains Mono, ui-monospace, monospace`;
      const tw = ctx.measureText(full).width;
      const padX = fontRaster * 0.2;
      const padY = fontRaster * 0.15;
      const bw = tw + padX * 2;
      const bh = fontRaster + padY * 2;
      const x0 = -bw / 2;
      const y0 = -bh / 2;

      // CAD-like: light halo, not a heavy colored capsule
      ctx.fillStyle = active
        ? "rgba(61,155,253,0.88)"
        : moved
          ? "rgba(62,207,142,0.85)"
          : "rgba(8,12,18,0.72)";
      ctx.strokeStyle = moved ? "#3ecf8e" : color;
      ctx.lineWidth = (active || moved ? 1.6 : 1.0) * px;
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(x0, y0, bw, bh, Math.min(2 * px, fontRaster * 0.15));
      else ctx.rect(x0, y0, bw, bh);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = active || moved ? "#061018" : "#f4f7fb";
      ctx.textBaseline = "middle";
      ctx.textAlign = "left";
      // slight stroke for readability on busy drawings
      ctx.lineWidth = Math.max(fontRaster * 0.08, 0.8 * px);
      ctx.strokeStyle = "rgba(0,0,0,0.55)";
      ctx.strokeText(full, x0 + padX, 0);
      ctx.fillText(full, x0 + padX, 0);
      ctx.restore();

      const hitPad = Math.max(bw, bh) * 0.55;
      hitRegions.push({
        id: aid,
        x0: cx - hitPad,
        y0: cy - hitPad,
        x1: cx + hitPad,
        y1: cy + hitPad,
        meta,
        color,
      });
    }

    if (ghostState && showLin) {
      for (const lin of ghostState.linears || []) {
        if (movedIds.size && !movedIds.has(lin.id)) continue;
        for (const seg of linearGeom(lin)) strokePath(seg, "rgba(148,163,184,0.75)", 1.2, true);
      }
    }
    if (ghostState) {
      const glist = [
        ...(showDia ? ghostState.diameters || [] : []),
        ...(showRad ? ghostState.radials || [] : []),
      ];
      for (const d of glist) {
        if (movedIds.size && !movedIds.has(d.id)) continue;
        strokePath(d.path, "rgba(148,163,184,0.75)", 1.2, true);
      }
    }

    if (showLin) {
      for (const lin of state.linears || []) {
        const moved = movedIds.has(lin.id);
        const color = hex(lin.color);
        const strokeColor =
          viewMode === "diff" && movedIds.size && !moved ? "rgba(160,170,180,0.35)" : color;
        const segs = linearGeom(lin);
        const extras = linearExtraSegs(lin);
        const { f1, f2 } = linearFeet(lin);
        const dist = Math.hypot(f2.x - f1.x, f2.y - f1.y);
        const tp = lin.text_point;
        let textOutside = false;
        if (tp && dist > 1e-9) {
          const ux = (f2.x - f1.x) / dist;
          const uy = (f2.y - f1.y) / dist;
          const t = (tp.x - f1.x) * ux + (tp.y - f1.y) * uy;
          textOutside = t < -0.5 || t > dist + 0.5;
        }
        // extensions thinner
        strokePath(segs[0], strokeColor, selectedId === lin.id || moved ? 1.8 : 1.15, false);
        strokePath(segs[1], strokeColor, selectedId === lin.id || moved ? 1.8 : 1.15, false);
        // main dim line (skip if outside-text extras already redraw the span)
        if (!textOutside) {
          strokePath(segs[2], strokeColor, selectedId === lin.id || moved ? 2.4 : 1.7, false);
        }
        for (const seg of extras) {
          strokePath(seg, strokeColor, selectedId === lin.id || moved ? 2.4 : 1.7, false);
        }
        drawLinearArrows(lin, strokeColor);
        const content =
          (lin.text && lin.text.content) ||
          (lin.value != null ? Number(lin.value).toFixed(2) : lin.id);
        labelAt(linearMid(lin), content, color, lin.id, {
          kind: "linear",
          value: lin.value,
          content,
          color: lin.color,
        }, {
          angleRad: textAngleRad(lin),
          worldHeight: (lin.text && lin.text.height) || 3.5 * (lin.dim_scale || 1),
        });
      }
    }

    if (showDia) {
      for (const d of state.diameters || []) {
        const moved = movedIds.has(d.id);
        const color = hex(d.color);
        const strokeColor =
          viewMode === "diff" && movedIds.size && !moved ? "rgba(160,170,180,0.35)" : color;
        strokePath(d.path, strokeColor, moved || selectedId === d.id ? 2.4 : 1.7);
        const end = d.path?.[d.path.length - 1] || d.anchor;
        labelAt(end, (d.text && d.text.content) || "⌀", color, d.id, {
          kind: "diameter",
          content: d.text?.content,
          color: d.color,
        }, {
          worldHeight: (d.text && d.text.height) || 3.5 * (d.dim_scale || 1),
        });
      }
    }

    if (showRad) {
      for (const d of state.radials || []) {
        const moved = movedIds.has(d.id);
        const color = hex(d.color);
        const strokeColor =
          viewMode === "diff" && movedIds.size && !moved ? "rgba(160,170,180,0.35)" : color;
        strokePath(d.path, strokeColor, moved || selectedId === d.id ? 2.4 : 1.7);
        if (d.path && d.path.length >= 2) {
          const a0 = d.path[0];
          const a1 = d.path[1];
          const aw = 2.5 * (d.dim_scale || 1);
          drawArrowHead(a0, a0.x - a1.x, a0.y - a1.y, aw, strokeColor);
        }
        let labelPt = d.path?.[d.path.length - 1] || d.anchor;
        if (d.category === "chamfer" && d.path && d.path.length >= 3) {
          // path = [attach, dogleg, textEdge]; underline spans text away from dogleg
          const dog = d.path[d.path.length - 2];
          const edge = d.path[d.path.length - 1];
          const tw = (d.text && Number(d.text.width)) || 12;
          const dir = edge.x >= dog.x ? 1 : -1;
          strokePath(
            [
              { x: edge.x, y: edge.y },
              { x: edge.x + dir * tw, y: edge.y },
            ],
            strokeColor,
            moved || selectedId === d.id ? 2.0 : 1.5,
          );
          labelPt = { x: edge.x + (dir * tw) / 2, y: edge.y };
        }
        labelAt(labelPt, (d.text && d.text.content) || (d.category === "chamfer" ? "C" : "R"), color, d.id, {
          kind: d.category || "radial",
          content: d.text?.content,
          color: d.color,
        }, {
          worldHeight: (d.text && d.text.height) || 3.5 * (d.dim_scale || 1),
        });
      }
    }

    if (opts.changes?.length) {
      ctx.fillStyle = "#3d9bfd";
      ctx.strokeStyle = "#3d9bfd";
      for (const c of opts.changes) {
        const [x0, y0] = to(c.before);
        const [x1, y1] = to(c.after);
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();
        const ang = Math.atan2(y1 - y0, x1 - x0);
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x1 - 7 * Math.cos(ang - 0.4), y1 - 7 * Math.sin(ang - 0.4));
        ctx.lineTo(x1 - 7 * Math.cos(ang + 0.4), y1 - 7 * Math.sin(ang + 0.4));
        ctx.closePath();
        ctx.fill();
      }
    }

    canvas.classList.add("show");
  }

  function syncLayers() {
    const wrap = el("canvas-wrap");
    const img = el("icm-img");
    const empty = el("empty-view");
    const p = currentPayload;
    wrap.classList.toggle("mode-overlay", viewMode === "overlay");
    wrap.classList.toggle("mode-diff", viewMode === "diff");

    const showCanvas = ["anno", "overlay", "after", "diff"].includes(viewMode);
    const showImg = ["before", "overlay", "obstacle"].includes(viewMode);

    const changes = p?.state && p?.best_state ? computeDiff(p.state, p.best_state) : [];
    renderDiffPanel(changes, p?.run);

    if (!p) {
      canvas.classList.remove("show");
      img.classList.remove("show");
      empty.style.display = "block";
      return;
    }
    empty.style.display = "none";

    if (showImg) {
      let url = p.urls.instance;
      if (viewMode === "obstacle") url = p.urls.obstacle;
      img.onload = () => {
        img.classList.add("show");
        const maxW = Math.min(1100, wrap.clientWidth - 8 || 900);
        const maxH = 640;
        const nw = img.naturalWidth || 800;
        const nh = img.naturalHeight || 600;
        const scale = Math.min(maxW / nw, maxH / nh, 2);
        const cssW = Math.round(nw * scale);
        const cssH = Math.round(nh * scale);
        img.style.width = `${cssW}px`;
        img.style.height = `${cssH}px`;
        if (!showCanvas) fitCamera(cssW, cssH, false);
        if (viewMode === "overlay" && p.state) drawState(p.state, { transparentBg: true });
      };
      img.onerror = () => {
        img.classList.remove("show");
        empty.style.display = "block";
        empty.textContent = "图像加载失败";
      };
      img.src = url || "";
      if (!url) img.classList.remove("show");
    } else {
      img.classList.remove("show");
      img.style.width = "";
      img.style.height = "";
    }

    if (showCanvas) {
      if (viewMode === "diff") {
        if (!p.best_state) {
          canvas.classList.remove("show");
          empty.style.display = "block";
          empty.textContent = "尚无优化结果，请先运行 Agent";
          return;
        }
        drawState(p.best_state, {
          transparentBg: false,
          ghostState: p.state,
          movedIds: new Set(changes.map((c) => c.id)),
          changes,
        });
      } else if (viewMode === "after") {
        if (!p.best_state) {
          canvas.classList.remove("show");
          empty.style.display = "block";
          empty.textContent = "尚无优化后几何，请先运行 Agent";
          return;
        }
        drawState(p.best_state, { transparentBg: false });
      } else if (viewMode === "overlay") {
        drawState(p.state, { transparentBg: true });
      } else {
        drawState(p.state, { transparentBg: false });
      }
    } else {
      canvas.classList.remove("show");
    }
  }

  async function init() {
    try {
      const h = await fetchJson("/api/health");
      el("health").textContent =
        `model=${h.model} · llm=${h.use_llm ? "on" : "off"} · key=${h.llm_configured ? "yes" : "no"}`;
      const m = await fetchJson("/api/manifest");
      manifestMeta = m;
      cases = m.items.map((item) => {
        const id = item.state_path.split("/").pop().replace(".state.json", "");
        const viewType = item.view_type || (item.tags || []).find((t) => !["production", "prod"].includes(t) && !/^[LDR]\d+$/.test(t) && !/^TE[A-Z0-9]+$/i.test(t)) || "other";
        return {
          id,
          item,
          tags: item.tags || [],
          viewType: String(viewType),
          partNumber: item.part_number || m.part_number || "",
          solutionId: item.solution_id || m.solution_id || "",
          dimCount: item.dim_count ?? ((item.linear_count || 0) + (item.diameter_count || 0) + (item.radial_count || 0)),
        };
      });
      renderDatasetMeta();
      renderList();
      if (cases[0]) selectCase(cases[0].id);
    } catch (e) {
      el("health").textContent = "请先运行 python scripts/fetch_production_dataset.py";
      console.error(e);
    }
  }

  function renderDatasetMeta() {
    const nameEl = el("dataset-name");
    const linkEl = el("dataset-link");
    const countEl = el("view-count");
    if (!manifestMeta) return;
    const nSol = (manifestMeta.solution_ids && manifestMeta.solution_ids.length) || 0;
    const title =
      [
        nSol > 1
          ? `${nSol} 个方案`
          : manifestMeta.solution_id || manifestMeta.solution_name,
        (manifestMeta.part_numbers && manifestMeta.part_numbers.length > 1
          ? `${manifestMeta.part_numbers.length} 零件`
          : manifestMeta.part_number),
      ]
        .filter(Boolean)
        .join(" · ") ||
      manifestMeta.dataset_name ||
      "未命名方案";
    nameEl.textContent = title;
    if (countEl) countEl.textContent = `${cases.length} 视图`;
    if (manifestMeta.drawing_url) {
      linkEl.hidden = false;
      linkEl.href = manifestMeta.drawing_url;
      linkEl.textContent = "打开原图";
    } else {
      linkEl.hidden = true;
    }
  }

  async function reloadManifest(selectId) {
    const m = await fetchJson("/api/manifest");
    manifestMeta = m;
    cases = m.items.map((item) => {
      const id = item.state_path.split("/").pop().replace(".state.json", "");
      const viewType =
        item.view_type ||
        (item.tags || []).find(
          (t) =>
            !["production", "prod", "layout_after"].includes(t) &&
            !/^[LDR]\d+$/.test(t) &&
            !/^TE[A-Z0-9]+$/i.test(t) &&
            !t.includes("布局")
        ) ||
        "other";
      return {
        id,
        item,
        tags: item.tags || [],
        viewType: String(viewType),
        partNumber: item.part_number || m.part_number || "",
        solutionId: item.solution_id || m.solution_id || "",
        dimCount: item.dim_count || 0,
      };
    });
    renderDatasetMeta();
    renderList();
    const next =
      (selectId && cases.some((c) => c.id === selectId) && selectId) ||
      (currentId && cases.some((c) => c.id === currentId) && currentId) ||
      (cases[0] && cases[0].id) ||
      null;
    if (next) await selectCase(next);
    else {
      currentId = null;
      currentPayload = null;
      el("btn-run").disabled = true;
      const sub = el("run-cta-sub");
      if (sub) sub.textContent = "选择左侧视图后开始优化";
      const label = el("current-view-label");
      if (label) label.textContent = "—";
    }
  }

  async function deleteCase(caseId) {
    if (!confirm(`删除视图 ${caseId}？\n将移除快照文件并从数据树去掉。`)) return;
    const res = await fetch(`/api/cases/${encodeURIComponent(caseId)}`, { method: "DELETE" });
    if (!res.ok) {
      const t = await res.text();
      alert(`删除失败: ${t}`);
      return;
    }
    await reloadManifest(null);
  }

  async function deletePart(partNumber, solutionId) {
    if (!confirm(`删除零件「${partNumber}」下全部视图？`)) return;
    const q = new URLSearchParams({ part_number: partNumber });
    if (solutionId) q.set("solution_id", solutionId);
    const res = await fetch(`/api/parts?${q}`, { method: "DELETE" });
    if (!res.ok) {
      const t = await res.text();
      alert(`删除失败: ${t}`);
      return;
    }
    await reloadManifest(null);
  }

  const VIEW_TYPE_ORDER = ["front", "left", "right", "top", "bottom", "back", "iso", "section", "axonometric"];

  function renderList() {
    const box = el("case-list");
    box.innerHTML = "";
    const countEl = el("view-count");
    if (countEl) countEl.textContent = String(cases.length);

    // 方案 ID → 零件编号 → 视图
    const tree = {};
    cases.forEach((c) => {
      const sid = String(
        c.item.solution_id ||
          manifestMeta?.solution_id ||
          "unknown"
      );
      const part = String(
        c.item.part_number ||
          manifestMeta?.part_number ||
          "未指定零件"
      );
      ((tree[sid] ||= {})[part] ||= []).push(c);
    });

    Object.keys(tree)
      .sort()
      .forEach((sid, sIdx) => {
        const parts = tree[sid];
        const solDetails = document.createElement("details");
        solDetails.className = "view-group tree-solution";
        const solHasActive = Object.values(parts)
          .flat()
          .some((c) => c.id === currentId);
        solDetails.open = solHasActive || (!currentId && sIdx === 0);

        const nViews = Object.values(parts).reduce((n, arr) => n + arr.length, 0);
        const solSummary = document.createElement("summary");
        solSummary.innerHTML = `<span class="tree-label"><span class="tree-k">方案</span> ${sid}</span><span class="view-group-count">${nViews}</span>`;
        solDetails.appendChild(solSummary);

        const solBody = document.createElement("div");
        solBody.className = "view-group-body tree-nested";

        Object.keys(parts)
          .sort()
          .forEach((part, pIdx) => {
            const members = parts[part]
              .slice()
              .sort((a, b) => {
                const ia = VIEW_TYPE_ORDER.indexOf(a.viewType);
                const ib = VIEW_TYPE_ORDER.indexOf(b.viewType);
                if (ia !== ib) {
                  if (ia < 0 && ib < 0) return a.viewType.localeCompare(b.viewType);
                  if (ia < 0) return 1;
                  if (ib < 0) return -1;
                  return ia - ib;
                }
                return (b.dimCount || 0) - (a.dimCount || 0);
              });

            const partDetails = document.createElement("details");
            partDetails.className = "view-group tree-part";
            const partHasActive = members.some((c) => c.id === currentId);
            partDetails.open = partHasActive || (!currentId && sIdx === 0 && pIdx === 0);

            const partSummary = document.createElement("summary");
            partSummary.innerHTML = `<span class="tree-label"><span class="tree-k">零件</span> ${part}</span><span class="view-group-count">${members.length}</span>`;
            const partDel = document.createElement("button");
            partDel.type = "button";
            partDel.className = "btn-tree-del";
            partDel.title = "删除该零件全部视图";
            partDel.textContent = "删";
            partDel.onclick = (ev) => {
              ev.preventDefault();
              ev.stopPropagation();
              deletePart(part, sid);
            };
            partSummary.appendChild(partDel);
            partDetails.appendChild(partSummary);

            const partBody = document.createElement("div");
            partBody.className = "view-group-body";
            members.forEach((c) => {
              const row = document.createElement("div");
              row.className = "view-item-row";
              const btn = document.createElement("button");
              btn.type = "button";
              btn.className = "view-item" + (c.id === currentId ? " active" : "");
              const L = c.item.linear_count ?? "?";
              const D = c.item.diameter_count ?? "?";
              const R = c.item.radial_count ?? "?";
              const vt = c.viewType || "view";
              btn.innerHTML = `<div class="view-item-id"><span class="view-type-badge">${vt}</span> ${c.id}</div><div class="view-item-meta">L${L} · D${D} · R${R}</div>`;
              btn.onclick = () => selectCase(c.id);
              const del = document.createElement("button");
              del.type = "button";
              del.className = "btn-tree-del";
              del.title = "删除此视图";
              del.textContent = "删";
              del.onclick = (ev) => {
                ev.preventDefault();
                ev.stopPropagation();
                deleteCase(c.id);
              };
              row.appendChild(btn);
              row.appendChild(del);
              partBody.appendChild(row);
            });
            partDetails.appendChild(partBody);
            solBody.appendChild(partDetails);
          });

        solDetails.appendChild(solBody);
        box.appendChild(solDetails);
      });
  }

  async function selectCase(id, opts = {}) {
    const resetSteps = opts.resetSteps !== false;
    currentId = id;
    selectedId = null;
    cam.caseId = null; // force re-fit for new case
    renderList();
    el("btn-run").disabled = false;
    const sub = el("run-cta-sub");
    if (sub) sub.textContent = `当前：${id}`;
    const label = el("current-view-label");
    if (label) label.textContent = id;
    currentPayload = await fetchJson(`/api/cases/${encodeURIComponent(id)}`);
    if (resetSteps) {
      if (currentPayload.run) hydrateStepsFromRun(currentPayload.run);
      else clearStepCards();
    }
    renderAll();
  }

  function renderAll() {
    const p = currentPayload;
    if (!p) return;
    syncLayers();
    renderLegend(p.state.color_legend);
    renderAnnos(activeState() || p.state);
    if (viewMode === "diff" && p.run?.report_before && p.run?.report_after) {
      renderMetricsCompare(p.run.report_before, p.run.report_after);
    } else {
      const report =
        ((viewMode === "after" || viewMode === "diff") && p.run?.report_after) ||
        p.run?.report_before ||
        null;
      if (report) {
        renderMetrics(report);
      } else {
        el("metrics").innerHTML = `<div class="metric"><div class="k">status</div><div class="v">未跑</div></div>`;
        el("violations").innerHTML = "<li>点击「运行 Agent」生成评价</li>";
        el("violations-count").textContent = "";
      }
    }
    if (p.run) {
      el("actions").textContent = JSON.stringify(p.run.actions_applied, null, 2);
      if (!stepEvents.length) hydrateStepsFromRun(p.run);
    } else {
      el("actions").textContent = "—";
      if (!stepEvents.length) clearStepCards();
    }
    const src = p.state.source || {};
    if (src.drawing_url || src.view_type) {
      el("pick-info").textContent = `${src.view_type || ""} · ${src.view_uuid || currentId}`;
    }
  }

  function renderMetrics(report) {
    const m = report.metrics || {};
    const order = [
      "P",
      "V_hard",
      "S_total",
      "S_text_in_part",
      "S_text_in_cost",
      "S_leader_text",
      "S_layer",
      "S_path",
      "S_overcrowd",
      "S_balance",
      "IF_total",
      "Overcrowd",
      "PathLen",
    ];
    el("metrics").innerHTML = order
      .filter((k) => k in m)
      .map((k) => {
        const v = m[k];
        const cls =
          k === "P" || k === "V_hard" || k.startsWith("S_") || k.startsWith("IF")
            ? v > 0
              ? "bad"
              : "ok"
            : "";
        return `<div class="metric ${cls}"><div class="k">${k}</div><div class="v">${Number(v).toFixed(2)}</div></div>`;
      })
      .join("");
    const dens = report.side_density || {};
    el("metrics").innerHTML += Object.entries(dens)
      .map(([k, v]) => `<div class="metric"><div class="k">side ${k}</div><div class="v">${v}</div></div>`)
      .join("");

    const viol = report.hard_violations || [];
    el("violations-count").textContent = viol.length ? `(${viol.length})` : "(0)";
    el("violations").innerHTML = viol.length
      ? viol
          .map(
            (x) =>
              `<li><code>${x.code}</code> ${x.a_id} ↔ ${x.b_id} <span>${x.detail || ""}</span></li>`
          )
          .join("")
      : "<li>无硬冲突</li>";
  }

  function renderMetricsCompare(before, after) {
    const mb = before.metrics || {};
    const ma = after.metrics || {};
    const order = [
      "P",
      "V_hard",
      "S_total",
      "S_text_in_part",
      "S_text_in_cost",
      "S_leader_text",
      "S_layer",
      "S_path",
      "S_overcrowd",
      "S_balance",
    ];
    el("metrics").innerHTML = order
      .filter((k) => k in mb || k in ma)
      .map((k) => {
        const vb = Number(mb[k] ?? 0);
        const va = Number(ma[k] ?? 0);
        const d = va - vb;
        const improved = d < 0;
        const cls = Math.abs(d) < 1e-6 ? "" : improved ? "ok" : "bad";
        const arrow = Math.abs(d) < 1e-6 ? "→" : d < 0 ? "↓" : "↑";
        return `<div class="metric ${cls}"><div class="k">${k}</div><div class="v">${vb.toFixed(2)} → ${va.toFixed(2)} <span class="muted">${arrow}</span></div></div>`;
      })
      .join("");
    const viol = after.hard_violations || [];
    el("violations-count").textContent = viol.length ? `(${viol.length})` : "(0)";
    el("violations").innerHTML = viol.length
      ? viol
          .map(
            (x) =>
              `<li><code>${x.code}</code> ${x.a_id} ↔ ${x.b_id} <span>${x.detail || ""}</span></li>`
          )
          .join("")
      : "<li>无硬冲突</li>";
  }
  function selectAnno(id) {
    selectedId = id;
    el("pick-info").textContent = id ? `选中: ${id}` : "点击标注可高亮";
    syncLayers();
    renderAnnos(activeState() || currentPayload.state);
    renderLegend(currentPayload.state.color_legend);
  }

  function renderLegend(legend) {
    const entries = legend?.entries || {};
    el("legend").innerHTML =
      Object.entries(entries)
        .map(([color, e]) => {
          const h = hex(color);
          const active = selectedId === e.id ? "active" : "";
          return `<div class="legend-item ${active}" data-id="${e.id}">
          <div class="swatch" style="background:${h}"></div>
          <div><div>${e.id}</div><code>${color} · ${e.category}</code></div>
        </div>`;
        })
        .join("") || "<div class='hint'>无图例</div>";
    el("legend").querySelectorAll(".legend-item").forEach((node) => {
      node.onclick = () => selectAnno(node.dataset.id);
    });
  }

  function renderAnnos(state) {
    const n =
      (state.linears?.length || 0) +
      (state.diameters?.length || 0) +
      (state.radials?.length || 0);
    el("anno-count").textContent = `(${n})`;

    const blocks = [];
    const table = (title, rows, cols) => {
      if (!rows?.length) return "";
      const head = cols.map((c) => `<th>${c}</th>`).join("");
      const body = rows
        .map((r) => {
          const active = selectedId === r.id ? "active" : "";
          return `<tr class="${active}" data-id="${r.id}">${cols
            .map((c) => `<td>${r[c] ?? ""}</td>`)
            .join("")}</tr>`;
        })
        .join("");
      return `<div><strong>${title}</strong><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
    };
    blocks.push(
      table(
        "线性",
        (state.linears || []).map((x) => ({
          id: x.id,
          尺寸: x.text?.content || Number(x.value).toFixed(2),
          value: Number(x.value).toFixed(2),
          point: `(${x.point.x.toFixed(1)}, ${x.point.y.toFixed(1)})`,
        })),
        ["id", "尺寸", "value", "point"]
      )
    );
    blocks.push(
      table(
        "直径",
        (state.diameters || []).map((x) => ({
          id: x.id,
          标注: x.text?.content || "⌀",
          anchor: `(${x.anchor.x.toFixed(1)}, ${x.anchor.y.toFixed(1)})`,
        })),
        ["id", "标注", "anchor"]
      )
    );
    blocks.push(
      table(
        "半径",
        (state.radials || []).map((x) => ({
          id: x.id,
          标注: x.text?.content || "R",
          anchor: `(${x.anchor.x.toFixed(1)}, ${x.anchor.y.toFixed(1)})`,
        })),
        ["id", "标注", "anchor"]
      )
    );
    el("annos").innerHTML = blocks.join("") || "—";
    el("annos").querySelectorAll("tr[data-id]").forEach((tr) => {
      tr.onclick = () => selectAnno(tr.dataset.id);
    });
  }

  document.querySelectorAll("#view-tabs .tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#view-tabs .tab").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      viewMode = btn.dataset.view;
      renderAll();
    });
  });

  ["tog-obs", "tog-linear", "tog-dia", "tog-rad", "tog-labels", "tog-ids"].forEach((id) => {
    el(id).addEventListener("change", () => syncLayers());
  });

  (() => {
    const wrap = el("canvas-wrap");
    let dragging = false;
    let moved = false;
    let lastX = 0;
    let lastY = 0;
    let startX = 0;
    let startY = 0;
    let spaceDown = false;

    function hitAtClient(clientX, clientY) {
      if (!canvas.classList.contains("show")) return null;
      const state = activeState();
      if (!state?.raster || !hitRegions.length) return null;
      const rect = canvas.getBoundingClientRect();
      const raster = state.raster;
      const rx = (clientX - rect.left) / (rect.width / raster.width);
      const ry = (clientY - rect.top) / (rect.height / raster.height);
      return hitRegions.find((h) => rx >= h.x0 && rx <= h.x1 && ry >= h.y0 && ry <= h.y1) || null;
    }

    window.addEventListener("keydown", (ev) => {
      if (ev.code === "Space") spaceDown = true;
    });
    window.addEventListener("keyup", (ev) => {
      if (ev.code === "Space") spaceDown = false;
    });

    wrap.addEventListener(
      "wheel",
      (ev) => {
        ev.preventDefault();
        const rect = wrap.getBoundingClientRect();
        zoomAt(ev.clientX - rect.left, ev.clientY - rect.top, ev.deltaY < 0 ? 1.12 : 1 / 1.12);
      },
      { passive: false }
    );

    wrap.addEventListener("pointerdown", (ev) => {
      const middle = ev.button === 1;
      const left = ev.button === 0;
      if (!middle && !left) return;
      const onAnno = left && !spaceDown && hitAtClient(ev.clientX, ev.clientY);
      if (onAnno) return; // let canvas click select
      if (left && spaceDown) ev.preventDefault();
      dragging = true;
      moved = false;
      lastX = ev.clientX;
      lastY = ev.clientY;
      startX = ev.clientX;
      startY = ev.clientY;
      wrap.classList.add("is-panning");
      try {
        wrap.setPointerCapture(ev.pointerId);
      } catch (_) {}
    });

    wrap.addEventListener("pointermove", (ev) => {
      if (!dragging) return;
      const dx = ev.clientX - lastX;
      const dy = ev.clientY - lastY;
      lastX = ev.clientX;
      lastY = ev.clientY;
      if (Math.hypot(ev.clientX - startX, ev.clientY - startY) > 5) moved = true;
      cam.x += dx;
      cam.y += dy;
      applyCamera();
    });

    const endPan = (ev) => {
      if (!dragging) return;
      dragging = false;
      wrap.classList.remove("is-panning");
      cam.suppressClick = moved;
      try {
        wrap.releasePointerCapture(ev.pointerId);
      } catch (_) {}
    };
    wrap.addEventListener("pointerup", endPan);
    wrap.addEventListener("pointercancel", endPan);

    el("btn-zoom-in").onclick = () => {
      const rect = wrap.getBoundingClientRect();
      zoomAt(rect.width / 2, rect.height / 2, 1.2);
    };
    el("btn-zoom-out").onclick = () => {
      const rect = wrap.getBoundingClientRect();
      zoomAt(rect.width / 2, rect.height / 2, 1 / 1.2);
    };
    el("btn-zoom-reset").onclick = () => resetCamera();
  })();

  canvas.addEventListener("mousemove", (ev) => {
    const tip = el("tooltip");
    const rect = canvas.getBoundingClientRect();
    const state = activeState();
    if (!state?.raster) {
      tip.hidden = true;
      return;
    }
    const raster = state.raster;
    const cssScaleX = rect.width / raster.width;
    const cssScaleY = rect.height / raster.height;
    const rx = (ev.clientX - rect.left) / cssScaleX;
    const ry = (ev.clientY - rect.top) / cssScaleY;
    const hit = hitRegions.find((h) => rx >= h.x0 && rx <= h.x1 && ry >= h.y0 && ry <= h.y1);
    if (!hit) {
      tip.hidden = true;
      canvas.style.cursor = "crosshair";
      return;
    }
    canvas.style.cursor = "pointer";
    tip.hidden = false;
    const wrapRect = el("canvas-wrap").getBoundingClientRect();
    tip.style.left = `${ev.clientX - wrapRect.left + 12}px`;
    tip.style.top = `${ev.clientY - wrapRect.top + 12}px`;
    const m = hit.meta || {};
    tip.textContent = `${hit.id}\n${m.kind || ""} ${m.content || m.value || ""}\ncolor=${m.color || ""}`;
  });

  canvas.addEventListener("click", (ev) => {
    if (cam.suppressClick) {
      cam.suppressClick = false;
      return;
    }
    const state = activeState();
    if (!state?.raster) return;
    const rect = canvas.getBoundingClientRect();
    const raster = state.raster;
    const rx = (ev.clientX - rect.left) / (rect.width / raster.width);
    const ry = (ev.clientY - rect.top) / (rect.height / raster.height);
    const hit = hitRegions.find((h) => rx >= h.x0 && rx <= h.x1 && ry >= h.y0 && ry <= h.y1);
    if (hit) selectAnno(hit.id);
  });

  canvas.addEventListener("mouseleave", () => {
    el("tooltip").hidden = true;
  });

  window.addEventListener("resize", () => {
    if (currentPayload) syncLayers();
  });

  const importBtn = el("btn-import");
  const importFile = el("import-file");
  if (importFile) {
    importFile.addEventListener("change", () => {
      const nameEl = el("import-file-name");
      if (!nameEl) return;
      const f = importFile.files && importFile.files[0];
      nameEl.textContent = f ? f.name : "未选择文件";
      nameEl.title = f ? f.name : "";
    });
  }
  if (importBtn) {
    importBtn.onclick = async () => {
      const fileInput = el("import-file");
      const status = el("import-status");
      const ossKey = (el("import-oss-key") && el("import-oss-key").value.trim()) || "";
      const file = fileInput && fileInput.files && fileInput.files[0];
      if (!ossKey && !file) {
        alert("请填写 OSS 下载地址，或选择本地 output.json");
        return;
      }
      const fd = new FormData();
      if (file) fd.append("file", file);
      if (ossKey) fd.append("oss_key", ossKey);
      fd.append("part_number", (el("import-part") && el("import-part").value.trim()) || "");
      fd.append("part_suffix", (el("import-suffix") && el("import-suffix").value) || "·布局后");
      fd.append("phase", "layout_after");
      const runPara = el("import-run-parametric");
      fd.append("run_parametric", runPara && runPara.checked ? "1" : "0");
      importBtn.disabled = true;
      if (status) {
        status.hidden = false;
        status.textContent = runPara && runPara.checked
          ? "下载并跑投图中（可能较久）…"
          : "导入中…";
      }
      try {
        const res = await fetch("/api/import-oss", { method: "POST", body: fd });
        const body = await res.json().catch(() => ({}));
        if (!res.ok) {
          const detail = body && body.detail;
          let msg = "import-oss failed";
          if (typeof detail === "string") msg = detail;
          else if (detail && typeof detail === "object") {
            msg = detail.message || detail.stderr || detail.stdout || JSON.stringify(detail);
          } else if (body && body.message) {
            msg = body.message;
          }
          throw new Error(msg);
        }
        if (status) status.textContent = `已导入（数据树共 ${body.remaining} 视图）`;
        await reloadManifest(null);
      } catch (e) {
        if (status) status.textContent = `导入失败：${e.message || e}`;
        alert(`导入失败：${e.message || e}`);
      } finally {
        importBtn.disabled = false;
      }
    };
  }

  el("btn-run").onclick = async () => {
    if (!currentId) return;
    el("btn-run").disabled = true;
    const runLabel = el("run-cta-label");
    if (runLabel) runLabel.textContent = "运行中…";
    else el("btn-run").textContent = "运行中…";
    const box = el("run-progress");
    box.hidden = false;
    el("run-bar").style.width = "0%";
    el("run-pct").textContent = "0%";
    el("run-phase").textContent = "启动";
    el("run-msg").textContent = "连接 Agent 进度流…";
    clearStepCards("运行中…");
    el("steps-panel").scrollIntoView({ behavior: "smooth", block: "start" });

    try {
      await new Promise((resolve, reject) => {
        const es = new EventSource(
          `/api/cases/${encodeURIComponent(currentId)}/run/stream`
        );
        let finished = false;
        const stop = () => {
          if (!finished) {
            finished = true;
            es.close();
          }
        };
        es.addEventListener("progress", (ev) => {
          try {
            const p = JSON.parse(ev.data);
            const pct = Math.max(0, Math.min(100, p.percent || 0));
            el("run-bar").style.width = `${pct}%`;
            el("run-pct").textContent = `${pct}%`;
            el("run-phase").textContent = p.phase || "—";
            el("run-msg").textContent = p.message || "";
            appendStepCard(p);
          } catch (e) {
            appendStepCard({ phase: "error", message: String(e) });
          }
        });
        es.addEventListener("done", (ev) => {
          try {
            const r = JSON.parse(ev.data);
            el("run-bar").style.width = "100%";
            el("run-pct").textContent = "100%";
            el("run-phase").textContent = "done";
            el("run-msg").textContent = `完成 · improved=${r.improved} · steps=${r.steps} · ${r.reason}`;
            appendStepCard({
              phase: "done",
              message: `完成 · improved=${r.improved} · steps=${r.steps}`,
              result: r,
              percent: 100,
            });
            el("step-cards").querySelectorAll(".step-row.is-live").forEach((n) => n.classList.remove("is-live"));
            el("steps-subtitle").textContent = `${stepEvents.length} 步 · 点击展开详情`;
          } catch (e) {
            appendStepCard({ phase: "error", message: String(e) });
          }
          stop();
          resolve();
        });
        es.addEventListener("error", (ev) => {
          if (ev.data) {
            try {
              const p = JSON.parse(ev.data);
              appendStepCard({ phase: "error", message: p.message || "error" });
              el("run-msg").textContent = p.message || "运行失败";
              stop();
              reject(new Error(p.message || "run failed"));
              return;
            } catch (_) {}
          }
          if (es.readyState === EventSource.CLOSED) {
            stop();
            reject(new Error("进度流断开"));
          }
        });
      });
      await selectCase(currentId, { resetSteps: false });
      viewMode = "diff";
      document.querySelectorAll("#view-tabs .tab").forEach((b) => {
        b.classList.toggle("active", b.dataset.view === "diff");
      });
      renderAll();
    } catch (e) {
      alert("运行失败: " + e.message);
    } finally {
      el("btn-run").disabled = false;
      const runLabel = el("run-cta-label");
      if (runLabel) runLabel.textContent = "运行 Agent";
      else el("btn-run").textContent = "运行 Agent";
    }
  };

  document.querySelectorAll("#step-filter-tabs .tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#step-filter-tabs .tab").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      stepFilter = btn.dataset.filter;
      applyStepFilter();
    });
  });

  init();
})();
