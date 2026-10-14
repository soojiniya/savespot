import { useEffect } from "react";
import { useMap } from "react-leaflet";
import L from "leaflet";
import { topoToGeoJSON } from "./topo";

/*
 * Base layer for the map page in the landing's language: Korea drawn as a teal
 * point grid (same simplified geometry the landing uses), sitting between the
 * dimmed tiles and the markers.
 *
 * Hover lift: the region under the cursor (a province, or a Busan district once
 * the Busan grid is showing) rises off the map on a spring — its dots lift, side
 * walls are stacked under its outline and its name appears above it. Purely
 * visual: the pane never takes pointer events, so markers and the map keep all
 * their existing clicks, and no data value is drawn or changed.
 */
type Pt = [number, number];
type Geo = {
  koreaDots: Pt[];
  busanDots: Pt[];
  busanDistricts: { name: string; rings: Pt[][] }[];
};
type Region = {
  name: string;
  polys: Pt[][][]; // polygon -> rings -> [lon, lat]
  bbox: [number, number, number, number];
  anchor: Pt; // label anchor [lon, lat]
};
type Spring = { x: number; v: number; target: number };

const LIFT_PX = 20;
const DOT_RGB = "47,224,192";
const LIT_RGB = "150,255,226";

function inRing(lon: number, lat: number, ring: Pt[]) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i];
    const [xj, yj] = ring[j];
    if (yi > lat !== yj > lat && lon < ((xj - xi) * (lat - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
function contains(r: Region, lon: number, lat: number) {
  const [x0, y0, x1, y1] = r.bbox;
  if (lon < x0 || lon > x1 || lat < y0 || lat > y1) return false;
  return r.polys.some((rings) => inRing(lon, lat, rings[0]) && !rings.slice(1).some((h) => inRing(lon, lat, h)));
}
function makeRegion(name: string, polys: Pt[][][]): Region {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  let biggest: Pt[] = polys[0][0];
  let biggestArea = -1;
  for (const rings of polys) {
    const outer = rings[0];
    let area = 0;
    for (let i = 0, j = outer.length - 1; i < outer.length; j = i++) {
      const [x, y] = outer[i];
      area += (outer[j][0] - x) * (outer[j][1] + y);
      x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
    }
    if (Math.abs(area) > biggestArea) { biggestArea = Math.abs(area); biggest = outer; }
  }
  // label anchor: the vertex mean of the largest ring, pulled inside if it falls out
  const mean: Pt = [
    biggest.reduce((s, p) => s + p[0], 0) / biggest.length,
    biggest.reduce((s, p) => s + p[1], 0) / biggest.length,
  ];
  const region: Region = { name, polys, bbox: [x0, y0, x1, y1], anchor: mean };
  if (!contains(region, mean[0], mean[1])) region.anchor = [(x0 + x1) / 2, (y0 + y1) / 2];
  return region;
}
const assign = (dots: Pt[], regions: Region[]) =>
  Int16Array.from(dots, ([lon, lat]) => regions.findIndex((r) => contains(r, lon, lat)));

export function DotLand() {
  const map = useMap();
  useEffect(() => {
    const pane = map.getPane("dots") ?? map.createPane("dots");
    pane.style.zIndex = "250";
    pane.style.pointerEvents = "none";
    const canvas = L.DomUtil.create("canvas", "dot-land", pane) as HTMLCanvasElement;
    const ctx = canvas.getContext("2d")!;
    const tagPane = map.getPane("dot-tags") ?? map.createPane("dot-tags");
    tagPane.style.zIndex = "640"; // above markers (600), below Leaflet tooltips (650)
    tagPane.style.pointerEvents = "none";
    const tag = L.DomUtil.create("div", "dot-land-tag", tagPane);
    tag.style.opacity = "0";
    tag.setAttribute("aria-hidden", "true");
    const reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;

    let geo: Geo | null = null;
    let provinces: Region[] = [];
    let districts: Region[] = [];
    let koreaOwner = new Int16Array(0); // dot -> province index
    let busanOwner = new Int16Array(0); // dot -> district index
    // projected container points, recomputed only when the view changes
    let koreaXY = new Float32Array(0);
    let busanXY = new Float32Array(0);
    let viewDirty = true;

    // one spring per region, keyed "p<i>" / "d<i>"
    const springs = new Map<string, Spring>();
    let hoverKey: string | null = null;
    let frame = 0;
    let last = 0;
    let disposed = false;

    const setHover = (key: string | null) => {
      if (key === hoverKey) return;
      if (hoverKey) springs.get(hoverKey)!.target = 0;
      hoverKey = key;
      if (key) {
        const s = springs.get(key) ?? { x: 0, v: 0, target: 0 };
        s.target = 1;
        springs.set(key, s);
      }
      if (reduceMotion) springs.forEach((s) => { s.x = s.target; s.v = 0; });
      schedule();
    };

    const project = (dots: Pt[]) => {
      const out = new Float32Array(dots.length * 2);
      dots.forEach(([lon, lat], i) => {
        const p = map.latLngToContainerPoint([lat, lon]);
        out[i * 2] = p.x;
        out[i * 2 + 1] = p.y;
      });
      return out;
    };

    const levels = () => {
      const z = map.getZoom();
      return {
        z,
        korea: Math.max(0, Math.min(1, (11 - z) / 3)),
        busan: Math.max(0, Math.min(1, (z - 9) / 2)),
      };
    };

    const tracePoly = (polys: Pt[][][], dy: number) => {
      ctx.beginPath();
      for (const rings of polys)
        for (const ring of rings)
          ring.forEach(([lon, lat], i) => {
            const p = map.latLngToContainerPoint([lat, lon]);
            if (i === 0) ctx.moveTo(p.x, p.y + dy);
            else ctx.lineTo(p.x, p.y + dy);
          });
    };

    const draw = (now: number) => {
      frame = 0;
      const dt = Math.min(0.05, last ? (now - last) / 1000 : 0.016);
      last = now;

      // advance springs (stiff, lightly under-damped: a small overshoot like a spring ease)
      let moving = false;
      springs.forEach((s, key) => {
        if (reduceMotion) return;
        const a = 210 * (s.target - s.x) - 19 * s.v;
        s.v += a * dt;
        s.x += s.v * dt;
        if (Math.abs(s.target - s.x) < 0.002 && Math.abs(s.v) < 0.01) {
          s.x = s.target;
          s.v = 0;
          if (s.x === 0 && key !== hoverKey) springs.delete(key);
        } else moving = true;
      });

      const size = map.getSize();
      if (viewDirty) {
        const dpr = Math.min(2, window.devicePixelRatio || 1);
        canvas.width = size.x * dpr;
        canvas.height = size.y * dpr;
        canvas.style.width = `${size.x}px`;
        canvas.style.height = `${size.y}px`;
        L.DomUtil.setPosition(canvas, map.containerPointToLayerPoint([0, 0]));
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        if (geo) {
          koreaXY = project(geo.koreaDots);
          busanXY = project(geo.busanDots);
        }
        viewDirty = false;
      }
      ctx.clearRect(0, 0, size.x, size.y);
      if (!geo) return;

      const { z, korea, busan } = levels();
      const lifts = (prefix: string, count: number) => {
        const h = new Float32Array(count);
        springs.forEach((s, key) => { if (key[0] === prefix) h[Number(key.slice(1))] = s.x; });
        return h;
      };
      const provLift = lifts("p", provinces.length);
      const distLift = lifts("d", districts.length);
      let peak = 0;
      springs.forEach((s) => { peak = Math.max(peak, Math.min(1, s.x)); });

      const sets: [Float32Array, Int16Array, Float32Array, number, number][] = [
        [koreaXY, koreaOwner, provLift, korea, 1.4 + Math.max(0, z - 7) * 0.5],
        [busanXY, busanOwner, distLift, busan, 1.2 + Math.max(0, z - 10) * 0.6],
      ];

      // 1) resting dots; the rest of the map dims a little while one region is up
      for (const [xy, owner, lift, alpha, r] of sets) {
        if (alpha <= 0) continue;
        ctx.fillStyle = `rgba(${DOT_RGB},${0.42 * alpha * (1 - 0.45 * peak)})`;
        for (let i = 0; i < owner.length; i++) {
          if (owner[i] >= 0 && lift[owner[i]] > 0.001) continue;
          const x = xy[i * 2], y = xy[i * 2 + 1];
          if (x < -4 || y < -4 || x > size.x + 4 || y > size.y + 4) continue;
          ctx.fillRect(x - r / 2, y - r / 2, r, r);
        }
      }

      // 2) lifted regions: footprint shadow, stacked walls, raised outline, raised dots, name
      const raised: [Region, number, Float32Array, Int16Array, number, number, number][] = [];
      provLift.forEach((h, i) => { if (h > 0.001 && korea > 0) raised.push([provinces[i], h, koreaXY, koreaOwner, i, korea, sets[0][4]]); });
      distLift.forEach((h, i) => { if (h > 0.001 && busan > 0) raised.push([districts[i], h, busanXY, busanOwner, i, busan, sets[1][4]]); });
      let labelAlpha = 0;
      let labelAt: { x: number; y: number; text: string } | null = null;
      for (const [region, h, xy, owner, idx, alpha, r] of raised) {
        const lift = -LIFT_PX * h;
        const k = Math.min(1, Math.max(0, h));

        tracePoly(region.polys, 2);
        ctx.fillStyle = `rgba(0,0,0,${0.45 * k * alpha})`;
        ctx.fill("evenodd");

        ctx.lineWidth = 1;
        ctx.lineJoin = "round";
        for (let step = 0; step > lift; step -= 1.5) {
          tracePoly(region.polys, step);
          ctx.strokeStyle = `rgba(${DOT_RGB},${0.07 * k * alpha})`;
          ctx.stroke();
        }

        tracePoly(region.polys, lift);
        ctx.fillStyle = `rgba(3,12,16,${0.78 * k * alpha})`;
        ctx.fill("evenodd");
        ctx.fillStyle = `rgba(${DOT_RGB},${0.07 * k * alpha})`;
        ctx.fill("evenodd");
        ctx.lineWidth = 1.2;
        ctx.strokeStyle = `rgba(${LIT_RGB},${0.8 * k * alpha})`;
        ctx.stroke();

        const rr = r * (1 + 0.35 * k);
        ctx.fillStyle = `rgba(${LIT_RGB},${(0.42 + 0.4 * k) * alpha})`;
        for (let i = 0; i < owner.length; i++) {
          if (owner[i] !== idx) continue;
          const x = xy[i * 2], y = xy[i * 2 + 1] + lift;
          if (x < -4 || y < -20 || x > size.x + 4 || y > size.y + 4) continue;
          ctx.fillRect(x - rr / 2, y - rr / 2, rr, rr);
        }

        if (k * alpha > labelAlpha) {
          labelAlpha = k * alpha;
          const at = map.latLngToContainerPoint([region.anchor[1], region.anchor[0]]);
          labelAt = { x: at.x, y: at.y + lift - 22 - 6 * (1 - k), text: region.name };
        }
      }

      // name tag lives in a pane above the markers so it is never covered
      if (labelAt && labelAlpha > 0.02) {
        if (tag.textContent !== labelAt.text) tag.textContent = labelAt.text;
        L.DomUtil.setPosition(tag, map.containerPointToLayerPoint([labelAt.x, labelAt.y]));
        tag.style.opacity = String(Math.min(1, labelAlpha));
      } else tag.style.opacity = "0";

      // markers standing in a raised region ride up with it
      liftMarkers(raised.map(([region, h, , , , alpha]) => [region, -LIFT_PX * h * Math.min(1, alpha * 2)]));

      if (moving) schedule();
      else last = 0;
    };
    // marker element -> region it stands in (cached per layer position and zoom)
    const markerRegion = new WeakMap<HTMLElement, { key: string; regions: Region[] }>();
    const shifted = new Set<HTMLElement>();
    const liftMarkers = (raised: [Region, number][]) => {
      const pane = map.getPane("markerPane");
      if (!pane) return;
      const next = new Set<HTMLElement>();
      if (raised.length) {
        const zoom = map.getZoom();
        for (const el of Array.from(pane.children) as HTMLElement[]) {
          const pos = (el as any)._leaflet_pos as L.Point | undefined;
          if (!pos) continue;
          const cacheKey = `${pos.x},${pos.y},${zoom}`;
          let hit = markerRegion.get(el);
          if (!hit || hit.key !== cacheKey) {
            const ll = map.layerPointToLatLng(pos);
            const regions = [...provinces, ...districts].filter((r) => contains(r, ll.lng, ll.lat));
            hit = { key: cacheKey, regions };
            markerRegion.set(el, hit);
          }
          const lifted = raised.find(([r]) => hit!.regions.includes(r));
          const inner = el.firstElementChild as HTMLElement | null;
          if (lifted && inner) {
            inner.style.translate = `0 ${lifted[1].toFixed(2)}px`;
            next.add(el);
          }
        }
      }
      shifted.forEach((el) => {
        if (!next.has(el)) (el.firstElementChild as HTMLElement | null)?.style.removeProperty("translate");
      });
      shifted.clear();
      next.forEach((el) => shifted.add(el));
    };

    const schedule = () => { if (!frame && !disposed) frame = requestAnimationFrame(draw); };
    const onView = () => { viewDirty = true; schedule(); };

    const onMove = (e: L.LeafletMouseEvent) => {
      if (!geo) return;
      const { lng, lat } = e.latlng;
      const { busan } = levels();
      // the raised region is drawn LIFT px higher; keep it while the cursor is on its raised shape
      if (hoverKey) {
        const s = springs.get(hoverKey);
        const region = (hoverKey[0] === "p" ? provinces : districts)[Number(hoverKey.slice(1))];
        if (s && region) {
          const base = map.containerPointToLatLng(e.containerPoint.add([0, LIFT_PX * Math.max(0, s.x)]));
          if (contains(region, base.lng, base.lat)) return;
        }
      }
      if (busan >= 0.5) {
        const d = districts.findIndex((r) => contains(r, lng, lat));
        if (d >= 0) return setHover(`d${d}`);
      }
      if (busan < 1) {
        const p = provinces.findIndex((r) => contains(r, lng, lat));
        return setHover(p >= 0 ? `p${p}` : null);
      }
      setHover(null);
    };
    const onOut = () => setHover(null);

    const base = import.meta.env.BASE_URL;
    Promise.all([
      fetch(`${base}landing/korea-scroll.json`).then((r) => r.json()),
      fetch(`${base}korea-states-topo.json`).then((r) => r.json()).then(topoToGeoJSON).catch(() => null),
    ])
      .then(([g, states]: [Geo, GeoJSON.FeatureCollection | null]) => {
        if (disposed) return;
        provinces = (states?.features ?? []).map((f) => {
          const geom = f.geometry as GeoJSON.Polygon | GeoJSON.MultiPolygon;
          const polys = (geom.type === "Polygon" ? [geom.coordinates] : geom.coordinates) as Pt[][][];
          return makeRegion(String(f.properties?.name ?? ""), polys);
        });
        districts = g.busanDistricts.map((d) => makeRegion(d.name.replace(/^부산광역시\s*/, "부산 "), d.rings.map((ring) => [ring])));
        koreaOwner = assign(g.koreaDots, provinces);
        busanOwner = assign(g.busanDots, districts);
        geo = g;
        onView();
      })
      .catch(() => {});

    map.on("move zoom resize viewreset zoomend moveend", onView);
    map.on("mousemove", onMove);
    const container = map.getContainer();
    container.addEventListener("mouseleave", onOut);
    map.on("zoomstart", onOut);
    schedule();
    return () => {
      disposed = true;
      map.off("move zoom resize viewreset zoomend moveend", onView);
      map.off("mousemove", onMove);
      container.removeEventListener("mouseleave", onOut);
      map.off("zoomstart", onOut);
      cancelAnimationFrame(frame);
      liftMarkers([]);
      canvas.remove();
      tag.remove();
    };
  }, [map]);
  return null;
}
