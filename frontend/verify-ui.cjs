const { chromium } = require("playwright");
const { spawn } = require("child_process");
const fs = require("fs");
const http = require("http");
const path = require("path");

const projectRoot = path.resolve(__dirname, "..");
const frontendRoot = __dirname;
const portSeed = Number(process.env.VERIFY_PORT_SEED || (Date.now() % 500));
const backendPort = 8100 + portSeed;
const frontendPort = 5300 + portSeed;
const backendUrl = `http://127.0.0.1:${backendPort}`;
const frontendUrl = `http://127.0.0.1:${frontendPort}/`;
const outDir = path.join(frontendRoot, "verification-screenshots");

function waitFor(url, timeoutMs = 45000) {
  const started = Date.now();
  return new Promise((resolve, reject) => {
    const tick = () => {
      const req = http.get(url, { timeout: 2500 }, (res) => {
        res.resume();
        resolve(res.statusCode);
      });
      req.on("timeout", () => {
        req.destroy();
      });
      req.on("error", () => {
        if (Date.now() - started > timeoutMs) {
          reject(new Error(`Timed out waiting for ${url}`));
          return;
        }
        setTimeout(tick, 750);
      });
    };
    tick();
  });
}

function startProcess(command, args, cwd, env = {}) {
  const child = spawn(command, args, {
    cwd,
    env: { ...process.env, ...env },
    stdio: ["ignore", "pipe", "pipe"],
    windowsHide: true,
  });
  child.stdout.on("data", (data) => process.stdout.write(data));
  child.stderr.on("data", (data) => process.stderr.write(data));
  return child;
}

function stopProcess(child) {
  if (!child || child.killed) return;
  if (process.platform === "win32") {
    spawn("taskkill", ["/pid", String(child.pid), "/T", "/F"], { stdio: "ignore", windowsHide: true });
    return;
  }
  child.kill();
}

async function inspect(browser, viewport, label) {
  const page = await browser.newPage({ viewport });
  const consoleErrors = [];
  const failedRequests = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") consoleErrors.push(msg.text());
  });
  page.on("requestfailed", (req) => {
    const url = req.url();
    if (!url.includes("favicon")) {
      failedRequests.push({ url, failure: req.failure()?.errorText || "unknown" });
    }
  });

  await page.goto(frontendUrl, { waitUntil: "domcontentloaded", timeout: 30000 });
  await page.waitForSelector(".leaflet-map", { timeout: 15000 });
  await page.waitForFunction(
    () => document.querySelectorAll(".map-marker").length > 0 || document.querySelector(".error-state"),
    null,
    { timeout: 25000 }
  );
  await page.waitForTimeout(5500);

  const before = await page.evaluate(() => {
    const rect = (sel) => {
      const el = document.querySelector(sel);
      if (!el) return null;
      const r = el.getBoundingClientRect();
      return { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height) };
    };
    return {
      markers: document.querySelectorAll(".map-marker").length,
      tiles: document.querySelectorAll(".leaflet-tile").length,
      loadedTiles: document.querySelectorAll(".leaflet-tile-loaded").length,
      selectedMarkers: document.querySelectorAll(".map-marker.selected").length,
      detailTitle: document.querySelector(".detail-panel h3")?.textContent?.trim() || "",
      mapFrame: rect(".map-frame"),
      toolbar: rect(".map-toolbar"),
      intro: rect(".intro-row"),
      detail: rect(".detail-panel"),
      nav: rect(".service-header"),
      errorText: document.querySelector(".error-state")?.textContent?.trim() || "",
    };
  });

  const firstMarker = page.locator(".map-marker").first();
  if (await firstMarker.count()) {
    await firstMarker.click({ force: true, timeout: 10000 });
    await page.waitForTimeout(1200);
  }

  const selectedShot = path.join(outDir, `${label}-selected.png`);
  await page.screenshot({ path: selectedShot, fullPage: false });

  const afterSelect = await page.evaluate(() => ({
    detailTitle: document.querySelector(".detail-panel h3")?.textContent?.trim() || "",
    selectedMarkers: document.querySelectorAll(".map-marker.selected").length,
    mobilePanelOpen: document.querySelector(".detail-panel.mobile-open") != null,
  }));

  let nearbyRows = null;
  let nearbyError = null;
  if (label === "desktop-1440") {
    const btn = page.locator(".nearby-action button").first();
    if (await btn.count()) {
      await btn.click();
      await page.waitForTimeout(3500);
      nearbyRows = await page.locator(".nearby-row").count();
      nearbyError = await page.locator(".inline-error").first().textContent().catch(() => null);
      await page.screenshot({ path: path.join(outDir, `${label}-nearby.png`), fullPage: false });
    }
  }

  await page.close();
  return { label, viewport, before, afterSelect, nearbyRows, nearbyError, consoleErrors, failedRequests, screenshot: selectedShot };
}

(async () => {
  fs.mkdirSync(outDir, { recursive: true });
  const python = path.join(projectRoot, ".venv", "Scripts", "python.exe");
  const backend = startProcess(python, ["-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", String(backendPort)], projectRoot);
  const frontend = startProcess("cmd.exe", ["/c", "npm", "run", "dev", "--", "--host", "127.0.0.1", "--port", String(frontendPort), "--strictPort"], frontendRoot, {
    VITE_API_BASE: backendUrl,
  });

  try {
    await waitFor(`${backendUrl}/health`);
    await waitFor(frontendUrl);

    const browser = await chromium.launch({ channel: "chrome", headless: true });
    const results = [];
    results.push(await inspect(browser, { width: 1440, height: 900 }, "desktop-1440"));
    results.push(await inspect(browser, { width: 1024, height: 768 }, "tablet-1024"));
    results.push(await inspect(browser, { width: 390, height: 844 }, "mobile-390"));
    await browser.close();

    fs.writeFileSync(path.join(outDir, "playwright-report.json"), JSON.stringify(results, null, 2), "utf8");
    console.log(JSON.stringify(results, null, 2));
  } finally {
    stopProcess(frontend);
    stopProcess(backend);
  }
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
