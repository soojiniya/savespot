import "@fontsource-variable/inter";
import { smoothWheel } from "./smooth-wheel";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";

/*
 * SaveSpot scroll-driven landing.
 *
 * One pinned canvas is the "camera". Scroll progress (0 → 1 over the stage) is a pure
 * input: every frame the camera, the map layers and the overlay copy are computed from
 * it, so scrolling back up plays everything in reverse. Nothing here calls setState on
 * scroll; per-frame changes are written straight to the canvas and to a few DOM nodes
 * through refs.
 *
 * Data: regions/scores/hotspots come from the existing read-only API (passed in by Home).
 * The geometry file (/landing/korea-scroll.json) is a simplified visual layer derived from
 * open sigungu boundaries; it is not survey-grade and the page says so.
 */

export type LandingRegion = { name: string; rank: number; score: number };
export type LandingHotspot = { lat: number; lon: number; type: string };
type Geo = {
  koreaDots: [number, number][];
  busanDots: [number, number][];
  busanDistricts: { name: string; rings: [number, number][][] }[];
  nodes: Record<string, [number, number]>;
};
type View = "home" | "map" | "region" | "ai" | "guide";

const FOCUS = "부산광역시 중구";
const FEATURE_GROUPS = [
  ["01", "어린이 인구", "전년도 0–12세 어린이 인구"],
  ["02", "사고 건수 · 사고율", "어린이 사고 건수, 1만 명당 사고율, 최근 2년 평균"],
  ["03", "부상 · 사망", "전년도 부상자 수와 사망자 수"],
  ["04", "자동차 등록", "자동차 등록대수, 어린이 100명당 등록대수"],
  ["05", "최근 변화", "최근 1년 사고 건수 · 사고율 변화량"],
];

// ---------- small math helpers ----------
const clamp = (v: number, a = 0, b = 1) => Math.min(b, Math.max(a, v));
// how quickly the scene catches up with the scroll position (1/s); higher = snappier
const SCROLL_EASE = 7;

const smooth = (e0: number, e1: number, x: number) => {
  const t = clamp((x - e0) / (e1 - e0));
  return t * t * (3 - 2 * t);
};
// 0 before a, rises to 1 at b, holds to c, falls to 0 at d
const win = (p: number, a: number, b: number, c: number, d: number) =>
  Math.min(smooth(a, b, p), 1 - smooth(c, d, p));

// Camera keyframes: progress, look-at lon/lat, visible height in km, tilt°, yaw°, anchor x/y (0..1 of viewport)
type Key = [number, number, number, number, number, number, number, number];
const KEYS: Key[] = [
  [0.0, 127.8, 36.05, 600, 56, -8, 0.68, 0.5],
  [0.08, 127.8, 36.05, 600, 56, -8, 0.68, 0.5],
  [0.3, 127.85, 36.12, 540, 46, -6, 0.5, 0.5],
  [0.36, 128.2, 35.85, 380, 44, -8, 0.5, 0.52],
  [0.54, 129.035, 35.128, 34, 40, -14, 0.5, 0.54],
  [0.6, 129.035, 35.128, 30, 40, -14, 0.34, 0.54],
  [0.7, 129.06, 35.17, 50, 42, -14, 0.34, 0.54],
  [0.78, 127.9, 36.1, 620, 50, -8, 0.33, 0.5],
  [0.9, 127.9, 36.1, 620, 50, -8, 0.33, 0.5],
  [1.0, 127.9, 36.1, 700, 58, -8, 0.5, 0.42],
];
function cameraAt(p: number) {
  let i = 0;
  while (i < KEYS.length - 2 && p > KEYS[i + 1][0]) i++;
  const a = KEYS[i], b = KEYS[i + 1];
  const t = smooth(0, 1, (p - a[0]) / Math.max(1e-6, b[0] - a[0]));
  const span = Math.exp(Math.log(a[3]) + (Math.log(b[3]) - Math.log(a[3])) * t);
  // zooming in: travel to the destination early, then close in on it;
  // zooming out: pull back first, then drift to the new centre
  const w = b[3] < a[3] - 1 ? 1 - Math.pow(1 - t, 2.2) : b[3] > a[3] + 1 ? Math.pow(t, 1.8) : t;
  const lerp = (u: number, v: number, k: number) => u + (v - u) * k;
  return {
    lon: lerp(a[1], b[1], w), lat: lerp(a[2], b[2], w), span,
    tilt: lerp(a[4], b[4], t), yaw: lerp(a[5], b[5], t), ax: lerp(a[6], b[6], t), ay: lerp(a[7], b[7], t),
  };
}

function makeSprite(rgb: string, size = 64) {
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const g = c.getContext("2d")!;
  const gr = g.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  gr.addColorStop(0, `rgba(${rgb},1)`);
  gr.addColorStop(0.25, `rgba(${rgb},.45)`);
  gr.addColorStop(1, `rgba(${rgb},0)`);
  g.fillStyle = gr;
  g.fillRect(0, 0, size, size);
  return c;
}

const TEAL = "47,224,192";
const MINT = "150,255,226";
const CORAL = "255,122,92";

export function ScrollLanding({
  regions,
  hotspots,
  hotspotTotal,
  focusHotspotTotal,
  facilityTotal,
  ask,
  onNavigate,
  onOpenRegion,
}: {
  regions: LandingRegion[];
  hotspots: LandingHotspot[];
  hotspotTotal: number | null;
  focusHotspotTotal: number | null;
  facilityTotal: number | null;
  ask: ReactNode;
  onNavigate: (view: View) => void;
  onOpenRegion: (region: string) => void;
}) {
  const stageRef = useRef<HTMLElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const layerRefs = useRef<Record<string, HTMLElement | null>>({});
  const labelRefs = useRef<(HTMLElement | null)[]>([]);
  const focusLabelRef = useRef<HTMLDivElement>(null);
  const railRef = useRef<HTMLOListElement>(null);
  const meterRef = useRef<HTMLSpanElement>(null);
  const geoRef = useRef<Geo | null>(null);
  const dataRef = useRef({ regions, hotspots });
  dataRef.current = { regions, hotspots };
  const [reduced, setReduced] = useState(() => typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  const [finder, setFinder] = useState("");
  const [finderError, setFinderError] = useState("");

  const ranked = useMemo(() => [...regions].sort((a, b) => a.rank - b.rank), [regions]);
  const focus = ranked.find((r) => r.name === FOCUS) ?? null;
  const busan = useMemo(() => ranked.filter((r) => r.name.startsWith("부산광역시 ")), [ranked]);
  const total = ranked.length;
  const at = (rank: number) => ranked[rank - 1]?.score ?? null;
  const median = total ? ranked[Math.floor((total - 1) / 2)] : null;
  const fmt = (v: number | null | undefined, d = 1) => (typeof v === "number" ? v.toFixed(d) : "—");
  const num = (v: number | null | undefined) => (typeof v === "number" ? v.toLocaleString("ko-KR") : "—");

  // prefers-reduced-motion → static, stacked version (no pin, no scrubbing)
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const sync = () => setReduced(mq.matches);
    sync();
    mq.addEventListener("change", sync);
    return () => mq.removeEventListener("change", sync);
  }, []);

  // mouse-wheel notches glide instead of jumping (off for reduced motion)
  useEffect(() => (reduced ? undefined : smoothWheel()), [reduced]);

  // the pinned stage sits right under the sticky header, whatever height it wraps to
  useEffect(() => {
    const header = document.querySelector<HTMLElement>(".site-header");
    const root = stageRef.current?.closest<HTMLElement>(".sl-root");
    if (!header || !root) return;
    const sync = () => root.style.setProperty("--sl-top", `${Math.round(header.getBoundingClientRect().height)}px`);
    sync();
    const ro = new ResizeObserver(sync);
    ro.observe(header);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    let alive = true;
    fetch("/landing/korea-scroll.json")
      .then((r) => r.json())
      .then((g: Geo) => { if (alive) { geoRef.current = g; } })
      .catch(() => {});
    return () => { alive = false; };
  }, []);

  // ---------- the frame loop ----------
  useEffect(() => {
    const stage = stageRef.current, canvas = canvasRef.current;
    if (!stage || !canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    if (reduced) {
      // switching into the static layout: drop any scrubbed inline styles
      Object.values(layerRefs.current).forEach((el) => el?.removeAttribute("style"));
      railRef.current?.removeAttribute("style");
    }
    const small = window.matchMedia("(max-width: 760px)").matches || (navigator.hardwareConcurrency ?? 8) <= 4;
    const dprMax = small ? 1 : Math.min(2, window.devicePixelRatio || 1);
    let dpr = dprMax;
    const sprites = { teal: makeSprite(TEAL), coral: makeSprite(CORAL), mint: makeSprite(MINT) };
    let W = 0, H = 0, target = reduced ? 0.6 : 0, shown = target, raf = 0, running = false, visible = true;
    const t0 = performance.now();

    const resize = () => {
      const r = canvas.getBoundingClientRect();
      W = r.width; H = r.height;
      // keep the canvas under ~3.2M pixels so very large / high-DPI screens hold 60fps
      dpr = Math.max(1, Math.min(dprMax, Math.sqrt(3.2e6 / Math.max(1, W * H))));
      canvas.width = Math.max(1, Math.round(W * dpr));
      canvas.height = Math.max(1, Math.round(H * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      kick();
    };
    const readScroll = () => {
      if (reduced) return;
      const r = stage.getBoundingClientRect();
      const travel = r.height - canvas.getBoundingClientRect().height;
      target = clamp(-r.top / Math.max(1, travel));
      kick();
    };
    const kick = () => { if (!running && visible) { running = true; raf = requestAnimationFrame(frame); } };

    let last = 0, first = true;
    const frame = (now: number) => {
      running = false;
      // Inertial follow: the scene glides toward the scroll position instead of
      // jumping with each wheel notch. Time-based so it feels the same at 60/120Hz;
      // it always settles exactly on the scroll position (scrubbing stays reversible).
      const dt = last ? Math.min(0.05, (now - last) / 1000) : 1 / 60;
      last = now;
      if (first) { shown = target; first = false; }
      else shown += (target - shown) * (1 - Math.exp(-dt * SCROLL_EASE));
      if (Math.abs(target - shown) < 0.0002) shown = target;
      const t = reduced ? 0 : (now - t0) / 1000;
      draw(shown, t);
      if (!reduced) apply(shown);
      if (visible && !reduced) { running = true; raf = requestAnimationFrame(frame); }
    };

    // ---------- projection ----------
    let cam = cameraAt(0), S = 1, cT = 1, sT = 0, cY = 1, sY = 0, F = 1000, kx = 1;
    const setCam = (p: number) => {
      cam = cameraAt(p);
      S = H / cam.span; // px per km
      const tr = (cam.tilt * Math.PI) / 180, yr = (cam.yaw * Math.PI) / 180;
      cT = Math.cos(tr); sT = Math.sin(tr); cY = Math.cos(yr); sY = Math.sin(yr);
      F = 1.7 * Math.max(W, H);
      kx = 111.32 * Math.cos((cam.lat * Math.PI) / 180);
    };
    const out = { x: 0, y: 0, s: 0, ok: false };
    const proj = (lon: number, lat: number, h = 0) => {
      const X = (lon - cam.lon) * kx * S, Z = (cam.lat - lat) * 110.57 * S;
      const X2 = X * cY - Z * sY, Z2 = X * sY + Z * cY;
      const down = Z2 * cT - h * sT, depth = Z2 * sT + h * cT;
      const den = F - depth;
      out.ok = den > F * 0.2;
      const s = F / Math.max(den, F * 0.2);
      out.x = cam.ax * W + X2 * s; out.y = cam.ay * H + down * s; out.s = s;
      return out;
    };
    const onScreen = (m = 40) => out.ok && out.x > -m && out.x < W + m && out.y > -m * 3 && out.y < H + m;

    const draw = (p: number, t: number) => {
      ctx.clearRect(0, 0, W, H);
      const geo = geoRef.current;
      if (!geo || !W) return;
      setCam(p);
      const { regions: regs, hotspots: hs } = dataRef.current;
      const byName = new Map(regs.map((r) => [r.name, r]));
      // zoom-dependent layer weights
      const coarseGap = 5.5 * S; // px between nationwide dots
      const fineGap = 0.47 * S; // px between Busan dots
      const aCoarse = 1 - smooth(34, 150, coarseGap);
      const aFine = smooth(1.1, 4.5, fineGap);
      const aBorders = smooth(1.5, 5, fineGap) * (1 - smooth(0.74, 0.78, p) * 0.85);
      const intro = 1;
      const dim = 1 - 0.55 * smooth(0.82, 0.88, p);

      ctx.globalCompositeOperation = "lighter";
      // nationwide point cloud
      if (aCoarse > 0.01) {
        const sz = clamp(coarseGap * 0.34, 1.3, 3.6);
        const step = small ? 2 : 1;
        for (let i = 0; i < geo.koreaDots.length; i += step) {
          const d = geo.koreaDots[i];
          proj(d[0], d[1]);
          if (!onScreen(10)) continue;
          const tw = 0.62 + 0.38 * Math.sin(t * 1.1 + i * 0.37);
          const depthFade = clamp(0.45 + 0.55 * out.s);
          ctx.fillStyle = `rgba(${TEAL},${(0.24 + 0.5 * tw) * aCoarse * depthFade * intro * dim})`;
          ctx.fillRect(out.x - sz / 2, out.y - sz / 2, sz * out.s, sz * out.s * 0.8);
        }
      }
      // Busan detail: denser points as the camera arrives
      if (aFine > 0.01) {
        const sz = clamp(fineGap * 0.32, 1, 3.4);
        const step = small ? 2 : 1;
        for (let i = 0; i < geo.busanDots.length; i += step) {
          const d = geo.busanDots[i];
          proj(d[0], d[1]);
          if (!onScreen(10)) continue;
          const tw = 0.6 + 0.4 * Math.sin(t * 1.3 + i * 0.21);
          ctx.fillStyle = `rgba(${TEAL},${(0.14 + 0.4 * tw) * aFine * dim})`;
          ctx.fillRect(out.x - sz / 2, out.y - sz / 2, sz * out.s, sz * out.s * 0.8);
        }
      }
      ctx.globalCompositeOperation = "source-over";
      // district outlines (Busan), focus district highlighted
      if (aBorders > 0.01) {
        for (const dist of geo.busanDistricts) {
          const isFocus = dist.name === FOCUS;
          ctx.beginPath();
          for (const ring of dist.rings) {
            ring.forEach(([lon, lat], k) => { proj(lon, lat); k ? ctx.lineTo(out.x, out.y) : ctx.moveTo(out.x, out.y); });
            ctx.closePath();
          }
          if (isFocus) {
            ctx.fillStyle = `rgba(${MINT},${0.08 * aBorders})`;
            ctx.fill();
          }
          ctx.strokeStyle = isFocus ? `rgba(${MINT},${0.85 * aBorders})` : `rgba(${TEAL},${0.32 * aBorders})`;
          ctx.lineWidth = isFocus ? 1.6 : 0.8;
          ctx.stroke();
        }
      }
      ctx.globalCompositeOperation = "lighter";
      // TAAS hotspots inside Busan (real points from the API)
      const aHot = smooth(0.42, 0.5, p) * (1 - smooth(0.69, 0.73, p));
      if (aHot > 0.01) {
        hs.forEach((h, i) => {
          proj(h.lon, h.lat);
          if (!onScreen()) return;
          const pulse = (t * 0.6 + i * 0.137) % 1;
          const r = 10 + 10 * pulse;
          ctx.globalAlpha = aHot * (0.55 - 0.4 * pulse);
          ctx.drawImage(sprites.mint, out.x - r, out.y - r, r * 2, r * 2);
          ctx.globalAlpha = aHot;
          ctx.fillStyle = "rgba(235,255,250,.95)";
          ctx.fillRect(out.x - 1.5, out.y - 1.5, 3, 3);
        });
        ctx.globalAlpha = 1;
      }
      // region beams: one per sigungu, height from the AI relative score
      // the ten strongest signals are lit from the first frame; the rest rise in rank order
      const reveal = (rank: number) => (rank <= 10 ? 1 : smooth(0.1 + 0.16 * (rank / 229), 0.13 + 0.16 * (rank / 229), p));
      const national = 1 - smooth(0.4, 0.5, p) + smooth(0.72, 0.78, p);
      const emphTop = win(p, 0.74, 0.78, 0.9, 0.95); // TOP10 / TOP50 step
      const zoomBoost = clamp(Math.log(560 / cam.span) / Math.log(18), 0, 1);
      const order: { r: LandingRegion; x: number; y: number; s: number; tx: number; ty: number }[] = [];
      for (const r of regs) {
        const pos = geo.nodes[r.name];
        if (!pos) continue;
        const isBusan = r.name.startsWith("부산광역시 ");
        const vis = reveal(r.rank) * (isBusan ? 1 : clamp(national));
        if (vis < 0.01) continue;
        proj(pos[0], pos[1]);
        if (!onScreen(80)) continue;
        const bx = out.x, by = out.y, bs = out.s;
        const hBase = 6 + Math.pow(r.score / 100, 1.7) * 150;
        const h = hBase * (1 + zoomBoost * (isBusan ? 0.6 : 0)) * vis;
        proj(pos[0], pos[1], h);
        order.push({ r, x: bx, y: by, s: bs, tx: out.x, ty: out.y });
      }
      order.sort((a, b) => a.y - b.y);
      for (const o of order) {
        const { r } = o;
        const top10 = r.rank <= 10, top50 = r.rank <= 50, isFocus = r.name === FOCUS;
        let a = (0.35 + 0.6 * (r.score / 100)) * dim;
        if (emphTop > 0) a *= top50 ? 1 : 1 - 0.75 * emphTop;
        const hot = top10 || isFocus;
        const col = hot ? CORAL : top50 && emphTop > 0.2 ? MINT : TEAL;
        const w = (hot ? 3.6 : 1.4 + 1.6 * (r.score / 100)) * clamp(o.s, 0.6, 1.6);
        const gr = ctx.createLinearGradient(0, o.y, 0, o.ty);
        gr.addColorStop(0, `rgba(${col},${a})`);
        gr.addColorStop(1, `rgba(${col},${a * 0.08})`);
        ctx.fillStyle = gr;
        ctx.fillRect(o.x - w / 2, o.ty, w, o.y - o.ty);
        if (hot || isFocus) {
          const g = (isFocus ? 34 : 20) * clamp(o.s, 0.6, 1.5);
          ctx.globalAlpha = a * 0.9;
          ctx.drawImage(sprites.coral, o.x - g, o.y - g * 0.55, g * 2, g * 1.1);
          ctx.globalAlpha = 1;
        }
      }
      ctx.globalCompositeOperation = "source-over";

      // ----- DOM overlays that follow the map (positions only, no React) -----
      const focusPos = geo.nodes[FOCUS];
      const fl = focusLabelRef.current;
      if (fl && focusPos) {
        const f = byName.get(FOCUS);
        const hBase = f ? 6 + Math.pow(f.score / 100, 1.7) * 150 : 120;
        proj(focusPos[0], focusPos[1], hBase * (1 + zoomBoost * 0.6));
        const a = win(p, 0.47, 0.52, 0.62, 0.66);
        fl.style.opacity = String(a);
        fl.style.transform = `translate3d(${out.x.toFixed(1)}px, ${out.y.toFixed(1)}px, 0)`;
      }
      const la = win(p, 0.5, 0.56, 0.7, 0.74);
      busanLabels.forEach((name, i) => {
        const el = labelRefs.current[i];
        const pos = geo.nodes[name];
        if (!el || !pos) return;
        proj(pos[0], pos[1]);
        const a = name === FOCUS ? 0 : la * (onScreen(0) ? 1 : 0);
        el.style.opacity = String(a);
        el.style.transform = `translate3d(${out.x.toFixed(1)}px, ${out.y.toFixed(1)}px, 0)`;
      });
    };

    // ----- copy / panels: opacity + small travel from progress windows -----
    const apply = (p: number) => {
      const L = layerRefs.current;
      const set = (key: string, a: number, dy = 24) => {
        const el = L[key];
        if (!el) return;
        el.style.opacity = a.toFixed(3);
        el.style.transform = `translate3d(0, ${((1 - a) * dy).toFixed(1)}px, 0)`;
        el.style.visibility = a < 0.01 ? "hidden" : "visible";
      };
      const hero = 1 - smooth(0.04, 0.13, p);
      const heroEl = L.hero;
      if (heroEl) {
        heroEl.style.opacity = hero.toFixed(3);
        heroEl.style.transform = `translate3d(0, ${(-60 * (1 - hero)).toFixed(1)}px, 0) scale(${(1 + 0.04 * (1 - hero)).toFixed(4)})`;
        heroEl.style.filter = `blur(${((1 - hero) * 8).toFixed(1)}px)`;
        heroEl.style.visibility = hero < 0.01 ? "hidden" : "visible";
      }
      set("cue", 1 - smooth(0.01, 0.05, p), 0);
      set("s2", win(p, 0.13, 0.18, 0.31, 0.36));
      set("s3", win(p, 0.4, 0.46, 0.55, 0.58));
      const steps: [string, number][] = [
        ["a", win(p, 0.565, 0.59, 0.635, 0.655)],
        ["b", win(p, 0.655, 0.68, 0.725, 0.745)],
        ["c", win(p, 0.745, 0.775, 0.825, 0.845)],
        ["d", win(p, 0.845, 0.865, 0.9, 0.92)],
        ["e", win(p, 0.92, 0.94, 1.01, 1.02)],
      ];
      steps.forEach(([k, a]) => set(k, a, 32));
      const rail = win(p, 0.55, 0.58, 1.01, 1.02);
      if (railRef.current) {
        railRef.current.style.opacity = rail.toFixed(3);
        const active = steps.reduce((best, s, i) => (s[1] > steps[best][1] ? i : best), 0);
        Array.from(railRef.current.children).forEach((li, i) => li.classList.toggle("is-on", i === active && rail > 0.2));
      }
      if (meterRef.current) meterRef.current.style.transform = `scaleY(${p.toFixed(4)})`;
      const scene = p < 0.1 ? 0 : p < 0.38 ? 1 : p < 0.56 ? 2 : 3;
      L.chapters?.querySelectorAll("li").forEach((li, i) => li.classList.toggle("is-on", i === scene));
    };

    const busanLabels = dataRef.current.regions.filter((r) => r.name.startsWith("부산광역시 ")).map((r) => r.name);

    const ro = new ResizeObserver(resize);
    ro.observe(canvas);
    const io = new IntersectionObserver(([e]) => { visible = e.isIntersecting; if (visible) kick(); }, { rootMargin: "200px" });
    io.observe(stage);
    window.addEventListener("scroll", readScroll, { passive: true });
    window.addEventListener("resize", readScroll);
    resize();
    readScroll();
    // geometry arrives async; redraw once it does (the loop is idle in reduced mode)
    const wait = window.setInterval(() => { if (geoRef.current) { kick(); window.clearInterval(wait); } }, 120);
    return () => {
      cancelAnimationFrame(raf);
      window.clearInterval(wait);
      ro.disconnect();
      io.disconnect();
      window.removeEventListener("scroll", readScroll);
      window.removeEventListener("resize", readScroll);
    };
    // busan labels and regions are read through refs; the loop restarts if the set changes
  }, [reduced, busan.length]);

  const refFor = (key: string) => (el: HTMLElement | null) => { layerRefs.current[key] = el; };

  const submitFinder = () => {
    const text = finder.trim();
    if (!text) { setFinderError("지역 이름을 입력하세요."); return; }
    const names = ranked.map((r) => r.name);
    const match = names.find((n) => n === text) ?? names.find((n) => n.includes(text)) ?? names.find((n) => n.replace(/\s/g, "").includes(text.replace(/\s/g, "")));
    if (!match) { setFinderError("229개 시군구에서 찾지 못했습니다. 예: 동해시, 부산광역시 중구"); return; }
    onOpenRegion(match);
  };

  const top10Cut = at(10), top50Cut = at(50);
  const pos = (v: number | null) => `${clamp((v ?? 0) / 100) * 100}%`;

  return (
    <div className={`sl-root ${reduced ? "is-static" : ""}`}>
      <section ref={stageRef} className="sl-stage" aria-label="SaveSpot 소개: 전국에서 부산광역시 중구까지">
        <div className="sl-pin">
          <canvas ref={canvasRef} className="sl-canvas" aria-hidden="true" />
          <div className="sl-vignette" aria-hidden="true" />

          {/* chapter index + progress meter */}
          <ol ref={refFor("chapters") as (el: HTMLOListElement | null) => void} className="sl-chapters" aria-hidden="true">
            <li>01 SaveSpot</li><li>02 전국 229</li><li>03 부산광역시</li><li>04 분석</li>
          </ol>
          <div className="sl-meter" aria-hidden="true"><span ref={meterRef} /></div>

          {/* Scene 1 — brand */}
          <div ref={refFor("hero")} className="sl-hero">
            <p className="sl-kicker">AI REGIONAL RISK SIGNAL</p>
            <h1 className="sl-wordmark" aria-label="SaveSpot">
              <span className="sl-wm-save" aria-hidden="true">Save</span><span className="sl-wm-sp" aria-hidden="true">Sp</span>
              <span className="sl-wm-pin" aria-hidden="true">
                <svg viewBox="0 0 80 100">
                  <circle cx="40" cy="40" r="33" fill="#eafff9" />
                  <circle cx="40" cy="40" r="21" fill="#04090c" />
                  <circle cx="40" cy="40" r="13" fill="none" stroke="#2fe0c0" strokeWidth="3" />
                  <circle cx="40" cy="40" r="6" fill="#2fe0c0" />
                  <path d="M25 68 L40 97 L55 68 Z" fill="#eafff9" />
                </svg>
              </span>
              <span className="sl-wm-b" aria-hidden="true">t</span>
            </h1>
            <p className="sl-claim">위험을 예측하고,<br /><em>안전을 발견하다</em></p>
            <p className="sl-lede">
              AI가 전국 229개 시군구의 어린이 교통사고 데이터를 읽고, 다음 연도에 상대적으로 위험 신호가
              강한 지역을 찾아냅니다. 사고 발생확률이 아닌, 예방을 위한 참고 지표입니다.
            </p>
            <div className="sl-hero-ask">
              <span className="sl-step">AI에게 물어보기</span>
              {ask}
            </div>
            <div className="sl-hero-links">
              <button type="button" onClick={() => onNavigate("map")}>현황지도 열기 →</button>
              <button type="button" onClick={() => onNavigate("ai")}>AI 분석 방법 →</button>
              <span>답변은 공개 데이터로만 만들어요</span>
            </div>
          </div>
          <div ref={refFor("cue")} className="sl-cue" aria-hidden="true"><i /> 스크롤해서 지도로 들어가기</div>

          {/* Scene 2 — nationwide */}
          <div ref={refFor("s2")} className="sl-caption sl-caption-left">
            <span className="sl-step">02 — 전국</span>
            <h2>빛기둥 하나가<br />시군구 하나</h2>
            <p>높이는 229개 지역 안에서의 AI 상대 점수입니다. 상위 10곳은 산호색으로 먼저 떠오릅니다.</p>
            <dl className="sl-mini">
              <div><dt>AI 분석 지역</dt><dd>{num(total || null)}</dd></div>
              <div><dt>1위 점수</dt><dd>{fmt(at(1))}</dd></div>
              <div><dt>중앙값</dt><dd>{fmt(median?.score)}</dd></div>
            </dl>
          </div>

          {/* Scene 3 — entering Busan */}
          <div ref={refFor("s3")} className="sl-caption sl-caption-left">
            <span className="sl-step">03 — 지역 진입</span>
            <h2>부산광역시</h2>
            <p>
              16개 구·군 경계와 어린이 사고다발지역 선정지점 {num(focusHotspotTotal)}곳(TAAS)이 드러납니다.
              가장 밝은 기둥은 전국 1위 {FOCUS}입니다.
            </p>
            <p className="sl-fine">경계와 점 격자는 공개 시군구 경계를 단순화한 시각화입니다. 지점 위치는 TAAS 좌표를 그대로 사용했습니다.</p>
          </div>
          <div ref={focusLabelRef} className="sl-focus-label" aria-hidden="true">
            <div>
              <b>{FOCUS}</b>
              <span>AI 상대 점수 <strong>{fmt(focus?.score)}</strong></span>
              <small>{focus ? `${focus.rank}위 / ${total}` : "—"}</small>
            </div>
          </div>
          <div className="sl-labels" aria-hidden="true">
            {busan.map((r, i) => (
              <span key={r.name} ref={(el) => { labelRefs.current[i] = el; }}>
                {r.name.replace("부산광역시 ", "")} <b>{fmt(r.score)}</b>
              </span>
            ))}
          </div>

          {/* Scene 4 — analysis, one step at a time */}
          <ol ref={railRef} className="sl-rail" aria-hidden="true">
            <li>선택 지역</li><li>부산 16개 구·군</li><li>상위 10 · 50</li><li>AI가 보는 데이터</li><li>해석 방법</li>
          </ol>
          <div className="sl-panels">
            <article ref={refFor("a")} className="sl-panel">
              <span className="sl-step">04 — 선택 지역</span>
              <h2>{FOCUS}</h2>
              <div className="sl-score">
                <strong>{fmt(focus?.score)}</strong>
                <span>AI 상대 점수<br /><b>{focus ? `${focus.rank}위` : "—"}</b> / 전국 {total || "—"}개 시군구</span>
              </div>
              <p>다음 연도 어린이 1만 명당 사고율 상위 25% 여부를 학습한 모델의 신호를 229개 지역 안에서 0–100으로 비교한 상대 점수입니다. 발생확률이 아닙니다.</p>
            </article>
            <article ref={refFor("b")} className="sl-panel">
              <span className="sl-step">04 — 부산 16개 구·군</span>
              <h2>같은 도시, 다른 신호</h2>
              <ol className="sl-bars">
                {busan.map((r) => (
                  <li key={r.name} className={r.name === FOCUS ? "is-focus" : ""}>
                    <span>{r.name.replace("부산광역시 ", "")}</span>
                    <i><b style={{ width: `${Math.max(2, r.score)}%` }} /></i>
                    <em>{fmt(r.score)}</em>
                    <small>{r.rank}위</small>
                  </li>
                ))}
              </ol>
            </article>
            <article ref={refFor("c")} className="sl-panel">
              <span className="sl-step">04 — 상위 10 · 상위 50</span>
              <h2>전국에서 어디쯤인가</h2>
              <div className="sl-scale">
                <div className="sl-scale-track">
                  <span className="sl-cut median" style={{ left: pos(median?.score ?? null) }}><em>중앙값 {fmt(median?.score)}</em></span>
                  <span className="sl-cut t50" style={{ left: pos(top50Cut) }}><em>TOP 50 {fmt(top50Cut)}</em></span>
                  <span className="sl-cut t10" style={{ left: pos(top10Cut) }}><em>TOP 10 {fmt(top10Cut)}</em></span>
                  <span className="sl-dot" style={{ left: pos(focus?.score ?? null) }} />
                </div>
                <div className="sl-scale-axis"><span>0</span><span>100</span></div>
              </div>
              <p>지도에서 산호색은 상위 10곳, 민트색은 11~50위입니다. {focus ? `${FOCUS}은 ${focus.rank <= 10 ? "상위 10곳" : focus.rank <= 50 ? "상위 50곳" : "상위 50곳 밖"}에 있습니다.` : null}</p>
            </article>
            <article ref={refFor("d")} className="sl-panel">
              <span className="sl-step">04 — AI가 보는 데이터</span>
              <h2>10개 지표, 5가지 묶음</h2>
              <ul className="sl-features">
                {FEATURE_GROUPS.map(([n, label, desc]) => (
                  <li key={n}><span>{n}</span><b>{label}</b><small>{desc}</small></li>
                ))}
              </ul>
              <p className="sl-fine">Logistic Regression · Permutation importance로 변수 영향을 확인합니다. 안전시설 자료는 모델 입력이 아니며 49개 지역에서 따로 분석합니다.</p>
            </article>
            <article ref={refFor("e")} className="sl-panel">
              <span className="sl-step">04 — 해석 방법</span>
              <h2>점수는 출발점입니다</h2>
              <ul className="sl-rules">
                <li>AI 점수는 사고 발생확률이 아니라 229개 지역 안의 상대 비교입니다.</li>
                <li>사고다발지역 선정지점은 모든 사고 지점이 아니며, 미선정이 안전을 뜻하지 않습니다.</li>
                <li>시설 자료가 없는 지역을 시설 0개로 보지 않습니다.</li>
              </ul>
              <button type="button" className="sl-link" onClick={() => onNavigate("guide")}>데이터안내 전체 보기 →</button>
            </article>
          </div>
        </div>
      </section>

      {/* Scene 5 — call to action, ordinary page flow from here */}
      <section className="sl-cta" aria-labelledby="sl-cta-title">
        <div className="sl-cta-inner">
          <div className="sl-cta-copy">
            <span className="sl-step">05 — 직접 확인하기</span>
            <h2 id="sl-cta-title">우리 지역의<br />위험 신호 확인하기</h2>
            <form
              className="sl-finder"
              role="search"
              onSubmit={(e) => { e.preventDefault(); submitFinder(); }}
            >
              <label htmlFor="sl-finder-input" className="sr-only">시군구 이름</label>
              <input
                id="sl-finder-input"
                list="sl-finder-options"
                value={finder}
                placeholder="시군구 이름  예: 동해시"
                aria-invalid={finderError ? true : undefined}
                aria-describedby={finderError ? "sl-finder-error" : undefined}
                onChange={(e) => { setFinder(e.target.value); setFinderError(""); }}
              />
              <datalist id="sl-finder-options">
                {ranked.map((r) => <option key={r.name} value={r.name} />)}
              </datalist>
              <button type="submit" className="sl-btn sl-btn-primary">지역분석에서 보기 →</button>
            </form>
            {finderError ? <p id="sl-finder-error" className="sl-error" role="alert">{finderError}</p> : null}
            <div className="sl-cta-links">
              <button type="button" onClick={() => onNavigate("map")}>현황지도 열기 →</button>
              <button type="button" onClick={() => onNavigate("ai")}>AI 분석 보기 →</button>
            </div>
          </div>
        </div>
        <dl className="sl-figures">
          <div><dt>사고다발지역 선정지점</dt><dd>{num(hotspotTotal)}</dd><span>TAAS · 어린이보호구역 + 보행어린이</span></div>
          <div><dt>AI 분석 지역</dt><dd>{num(total || null)}</dd><span>시군구 단위 · 다음 연도 상대적 고위험</span></div>
          <div><dt>시설 상세분석 지역</dt><dd>{num(facilityTotal)}</dd><span>횡단보도·신호등·보호구역·CCTV 자료 보유</span></div>
        </dl>
        <footer className="sl-sources">
          <span>데이터</span>
          <b>TAAS 어린이 사고다발지역</b><b>어린이 교통사고 2020–2025</b><b>0–12세 어린이 인구</b><b>자동차 등록대수</b><b>교통안전시설 (49개 지역)</b>
          <button type="button" onClick={() => onNavigate("guide")}>데이터안내 →</button>
        </footer>
      </section>
    </div>
  );
}
