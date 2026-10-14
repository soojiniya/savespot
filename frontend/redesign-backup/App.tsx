import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  GeoJSON as LeafletGeoJSON,
  MapContainer,
  Marker,
  TileLayer,
  Tooltip,
  useMap,
  useMapEvents,
} from "react-leaflet";
import L from "leaflet";
import { BarChart } from "./components/charts/bar-chart";
import { Bar } from "./components/charts/bar";
import { RingChart } from "./components/charts/ring-chart";
import { Ring } from "./components/charts/ring";
import { RingCenter } from "./components/charts/ring-center";
import { LineChart } from "./components/charts/line-chart";
import { Line } from "./components/charts/line";
import { ScatterChart } from "./components/charts/scatter-chart";
import { Scatter } from "./components/charts/scatter";
import { Grid } from "./components/charts/grid";

const API_BASE = (
  import.meta.env.VITE_API_BASE ??
  import.meta.env.VITE_API_URL ??
  "http://127.0.0.1:8000"
).replace(/\/$/, "");
const AMBER = "#0d574c";
const CORAL = "#ed765d";
const MUTED_AMBER = "rgba(13,87,76,.42)";
const DEFAULT_CENTER: [number, number] = [36.35, 127.8];

type View = "map" | "region" | "ai" | "guide";
type SourceType = "schoolzone_hotspot" | "child_pedestrian_hotspot";
type FilterOption = { value: string; label: string };
type Hotspot = {
  hotspot_id: number;
  source_type: SourceType;
  location_name?: string;
  accident_year?: number | null;
  year_verification_status?: string;
  sido?: string;
  sigungu?: string;
  latitude: number;
  longitude: number;
  accident_count?: number;
  casualty_count?: number;
  death_count?: number;
  serious_injury_count?: number;
  minor_injury_count?: number;
  injury_report_count?: number;
};
type HotspotResponse = {
  items: Hotspot[];
  pagination: { limit: number; offset: number; count: number; total: number };
};
type NearbyItem = Hotspot & { distance_m?: number };
type NearbyResponse = { items: NearbyItem[] };
type Region = {
  sido_sgg_nm: string;
  SIDO_NM?: string;
  SIGUNGU_NM?: string;
  child_accident_count?: number;
  child_death_count?: number;
  child_injury_count?: number;
  child_population_0_12?: number;
  accident_per_10k_child?: number;
  crosswalk_count?: number;
  traffic_light_count?: number;
  schoolzone_count?: number;
  cctv_install_rate?: number;
  infrastructure_score?: number;
  speed_bump_count?: number | null;
};
type RegionResponse = {
  items: Region[];
  pagination: { total: number; count: number; limit?: number; offset?: number };
};
type AiRiskRegion = {
  risk_rank_2026: number;
  sido_sgg_nm: string;
  predicted_high_risk_probability_2026: number;
  prev_child_population?: number;
  prev_vehicle_count?: number;
  prev_accident_per_10k_child?: number;
  prev_accident_count?: number;
  prev_injury_count?: number;
  prev_death_count?: number;
  accident_count_change_1y?: number;
  accident_rate_change_1y?: number;
  accident_count_mean_2y?: number;
  [key: string]: string | number | undefined;
};
type AiRiskResponse = {
  items: AiRiskRegion[];
  pagination?: { total: number; count: number };
  model?: ModelInfo["model"];
};
type ImprovementCandidate = {
  improvement_priority_rank: number;
  sido_sgg_nm: string;
  predicted_high_risk_probability_2026: number;
  infrastructure_score: number;
  infrastructure_deficiency_score: number;
  improvement_priority_score: number;
  crosswalk_per_1000_child?: number;
  traffic_light_per_1000_child?: number;
  schoolzone_per_1000_child?: number;
  cctv_install_rate?: number;
};
type ImprovementResponse = {
  items: ImprovementCandidate[];
  pagination: { total: number; count: number };
};
type ModelInfo = {
  model: {
    model_available: boolean;
    model_name?: string;
    feature_cols: string[];
    target: string;
    prediction_unit?: string;
    positive_label?: string;
    not_supported?: string[];
  };
  metrics?: Array<Record<string, string | number>>;
};
type ImportanceItem = {
  feature: string;
  importance_mean: number;
  importance_std?: number;
};
type ImportanceResponse = {
  items: ImportanceItem[];
  explanation_method: string;
  shap_availbale?: boolean;
  shap_available?: boolean;
  model?: ModelInfo["model"];
};

const sourceMeta: Record<
  SourceType,
  { label: string; short: string; color: string }
> = {
  schoolzone_hotspot: {
    label: "어린이보호구역 사고다발지역",
    short: "보호구역",
    color: AMBER,
  },
  child_pedestrian_hotspot: {
    label: "보행어린이 사고다발지역",
    short: "보행어린이",
    color: CORAL,
  },
};

const featureLabels: Record<string, string> = {
  prev_child_population: "전년도 어린이 인구",
  prev_vehicle_count: "전년도 자동차 등록대수",
  prev_accident_count: "전년도 어린이 사고건수",
  prev_injury_count: "전년도 부상자수",
  prev_death_count: "전년도 사망자수",
  prev_accident_per_10k_child: "전년도 어린이 1만명당 사고율",
  accident_count_change_1y: "최근 1년 사고건수 변화",
  accident_rate_change_1y: "최근 1년 사고율 변화",
  accident_count_mean_2y: "최근 2년 평균 사고건수",
};

const nf = new Intl.NumberFormat("ko-KR");
function formatNumber(value?: number | null) {
  return typeof value === "number" ? nf.format(value) : "—";
}
function formatDecimal(value?: number | null, digits = 1) {
  return typeof value === "number"
    ? value.toLocaleString("ko-KR", { maximumFractionDigits: digits })
    : "—";
}
function scoreNumber(value?: number | null) {
  return typeof value === "number"
    ? Math.round((value <= 1 ? value * 100 : value) * 10) / 10
    : null;
}
function formatScore(value?: number | null) {
  const score = scoreNumber(value);
  return score == null ? "—" : `${score}점`;
}
function formatPercent(value?: number | null) {
  if (typeof value !== "number") return "—";
  return `${Math.round((value <= 1 ? value * 100 : value) * 10) / 10}%`;
}
function formatSigned(value?: number | null, unit = "") {
  if (typeof value !== "number") return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toLocaleString("ko-KR", { maximumFractionDigits: 1 })}${unit}`;
}
function formatYear(item: Hotspot) {
  return item.accident_year == null
    ? "연도 정보 확인 불가"
    : `${Math.round(item.accident_year)}년`;
}
function regionNameFromHotspot(item: Hotspot | null) {
  return item?.sido && item?.sigungu
    ? `${item.sido} ${item.sigungu}`.trim()
    : "";
}
function ratio(value: number | null | undefined, max: number) {
  if (typeof value !== "number" || !Number.isFinite(value) || max <= 0)
    return 0;
  return Math.max(3, Math.min(100, (value / max) * 100));
}
function average(values: Array<number | undefined>) {
  const nums = values.filter(
    (value): value is number =>
      typeof value === "number" && Number.isFinite(value),
  );
  return nums.length
    ? nums.reduce((sum, value) => sum + value, 0) / nums.length
    : null;
}
function percentile(value: number | undefined, values: number[]) {
  if (typeof value !== "number" || values.length === 0) return 0;
  const lower = values.filter((item) => item <= value).length;
  return Math.round((lower / values.length) * 100);
}
async function fetchJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { signal });
  if (!response.ok) throw new Error(`API ${response.status}`);
  return response.json() as Promise<T>;
}

function markerIcon(type: SourceType, selected: boolean) {
  return L.divIcon({
    className: `map-marker ${type === "schoolzone_hotspot" ? "filled" : "ring"} ${selected ? "selected" : ""}`,
    html: `<span style="--marker-color:${sourceMeta[type].color}"></span>`,
    iconSize: [22, 22],
    iconAnchor: [11, 11],
  });
}

function topoToGeoJSON(topology: any): GeoJSON.FeatureCollection {
  const [sx, sy] = topology.transform.scale;
  const [tx, ty] = topology.transform.translate;
  const arcs = topology.arcs.map((arc: number[][]) => {
    let x = 0;
    let y = 0;
    return arc.map(([dx, dy]) => {
      x += dx;
      y += dy;
      return [x * sx + tx, y * sy + ty];
    });
  });
  const getArc = (index: number) => {
    const arc = arcs[index < 0 ? ~index : index];
    return index < 0 ? [...arc].reverse() : arc;
  };
  const stitch = (indexes: number[]) =>
    indexes.flatMap((index, part) => {
      const arc = getArc(index);
      return part === 0 ? arc : arc.slice(1);
    });
  const object = Object.values(topology.objects)[0] as any;
  return {
    type: "FeatureCollection",
    features: object.geometries.map((geometry: any) => {
      const coordinates =
        geometry.type === "Polygon"
          ? geometry.arcs.map((ring: number[]) => stitch(ring))
          : geometry.arcs.map((polygon: number[][]) =>
              polygon.map((ring: number[]) => stitch(ring)),
            );
      return {
        type: "Feature",
        properties: geometry.properties,
        geometry: { type: geometry.type, coordinates },
      };
    }),
  };
}

function StateBoundaries() {
  const [data, setData] = useState<GeoJSON.FeatureCollection | null>(null);
  const [zoom, setZoom] = useState(7.5);
  const map = useMap();
  useEffect(() => {
    fetch("/korea-states-topo.json")
      .then((response) => response.json())
      .then((topology) => setData(topoToGeoJSON(topology)))
      .catch(() => setData(null));
  }, []);
  useMapEvents({ zoomend: () => setZoom(map.getZoom()) });
  if (!data) return null;
  return (
    <>
      <LeafletGeoJSON
        data={data}
        style={() => ({
          color: "rgba(255,255,255,.15)",
          weight: 1,
          fillColor: "#121214",
          fillOpacity: 0.9,
        })}
      />
      {zoom >= 8 &&
        data.features.map((feature, index) => {
          const geometry = feature.geometry as any;
          const points: number[][] = geometry.coordinates.flat(
            geometry.type === "Polygon" ? 1 : 2,
          );
          const lng =
            points.reduce((sum, point) => sum + point[0], 0) / points.length;
          const lat =
            points.reduce((sum, point) => sum + point[1], 0) / points.length;
          const name =
            feature.properties?.name_short ?? feature.properties?.name ?? "";
          return (
            <Marker
              key={`${feature.properties?.code ?? index}-label`}
              position={[lat, lng]}
              interactive={false}
              icon={L.divIcon({
                className: "state-label",
                html: `<span>${name}</span>`,
                iconSize: [64, 20],
                iconAnchor: [32, 10],
              })}
            />
          );
        })}
    </>
  );
}

function MapViewport({
  selected,
  onCenterChange,
}: {
  selected: Hotspot | null;
  onCenterChange: (center: [number, number]) => void;
}) {
  const map = useMap();
  const fitted = useRef(false);
  useEffect(() => {
    const frame = window.requestAnimationFrame(() => map.invalidateSize());
    return () => window.cancelAnimationFrame(frame);
  }, [map]);
  useEffect(() => {
    if (selected || fitted.current) return;
    map.fitBounds(
      L.latLngBounds([
        [33.05, 125.9],
        [38.6, 129.7],
      ]),
      {
        animate: false,
        paddingTopLeft: [40, 40],
        paddingBottomRight: [40, 40],
      },
    );
    fitted.current = true;
  }, [map, selected]);
  useEffect(() => {
    if (!selected) return;
    map.flyTo(
      [selected.latitude, selected.longitude],
      Math.max(map.getZoom(), 12),
      { duration: 0.55 },
    );
    const timer = window.setTimeout(
      () =>
        map.panBy([-Math.min(280, map.getSize().x * 0.2), 0], {
          duration: 0.35,
        }),
      600,
    );
    return () => window.clearTimeout(timer);
  }, [map, selected]);
  useMapEvents({
    moveend: () => {
      const point = map.getCenter();
      onCenterChange([point.lat, point.lng]);
    },
  });
  return null;
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
function BigMetric({
  label,
  value,
  tone = "default",
}: {
  label: string;
  value: string;
  tone?: "default" | "ai";
}) {
  return (
    <div className={`big-metric ${tone === "ai" ? "ai-tone" : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
function FilterSelect({
  label,
  value,
  options,
  disabled,
  displayValue,
  disabledReason,
  onChange,
}: {
  label: string;
  value: string;
  options: FilterOption[];
  disabled?: boolean;
  displayValue?: string;
  disabledReason?: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className={`filter-select ${disabled ? "is-disabled" : ""}`}>
      <span>{label}</span>
      <select
        value={value}
        disabled={disabled}
        title={disabled ? disabledReason : undefined}
        onChange={(event) => onChange(event.target.value)}
      >
        {displayValue ? <option value={value}>{displayValue}</option> : null}
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}

function FacilityMeter({
  label,
  value,
  max,
  format,
  nullable = false,
}: {
  label: string;
  value?: number | null;
  max: number;
  format?: "percent" | "score";
  nullable?: boolean;
}) {
  const unavailable = nullable && value == null;
  const display = unavailable
    ? "자료 없음"
    : format === "percent"
      ? formatPercent(value)
      : format === "score"
        ? formatScore(value)
        : formatNumber(value);
  return (
    <div className="facility-meter">
      <div>
        <span>{label}</span>
        <strong>{display}</strong>
      </div>
      <i>
        <b style={{ width: `${unavailable ? 0 : ratio(value, max)}%` }} />
      </i>
    </div>
  );
}

function BklitHorizontalBars({
  data,
  valueKey,
  labelKey,
  valueSuffix = "점",
  aspect = "1.8 / 1",
}: {
  data: Record<string, unknown>[];
  valueKey: string;
  labelKey: string;
  valueSuffix?: string;
  aspect?: string;
}) {
  return (
    <div className="bklit-bars">
      <BarChart
        data={data}
        xDataKey={labelKey}
        orientation="horizontal"
        aspectRatio={aspect}
        margin={{ top: 10, right: 20, bottom: 10, left: 8 }}
        barGap={0.45}
        animationDuration={650}
      >
        <Bar dataKey={valueKey} fill={AMBER} lineCap="round" />
      </BarChart>
      <div className="bar-label-overlay" aria-hidden="true">
        {data.map((item, index) => (
          <span key={`${String(item[labelKey])}-${index}`}>
            <b>{String(item[labelKey])}</b>
            <strong>
              {formatDecimal(item[valueKey] as number, 2)}
              {valueSuffix}
            </strong>
          </span>
        ))}
      </div>
    </div>
  );
}

function FacilityRing({
  facility,
  candidate,
}: {
  facility: Region | null;
  candidate: ImprovementCandidate | null;
}) {
  if (!facility) return null;
  const data = [
    {
      label: "인프라 공급수준",
      value: scoreNumber(facility.infrastructure_score),
      maxValue: 100,
      color: AMBER,
    },
    {
      label: "시설 부족도",
      value: scoreNumber(candidate?.infrastructure_deficiency_score),
      maxValue: 100,
      color: MUTED_AMBER,
    },
  ].filter(
    (
      item,
    ): item is {
      label: string;
      value: number;
      maxValue: number;
      color: string;
    } => typeof item.value === "number",
  );
  if (data.length === 0) return null;
  return (
    <div className="facility-ring-wrap">
      <RingChart
        data={data}
        size={188}
        strokeWidth={11}
        ringGap={9}
        baseInnerRadius={50}
        enterStaggerScale={0.75}
      >
        {data.map((_, index) => (
          <Ring key={index} index={index} showGlow={false} />
        ))}
        <RingCenter
          suffix="점"
          defaultLabel="분석용 지표"
          valueClassName="ring-center-value"
          labelClassName="ring-center-label"
        />
      </RingChart>
      <div className="ring-legend">
        {data.map((item) => (
          <span key={item.label}>
            <i style={{ background: item.color }} />
            {item.label}
            <strong>{formatDecimal(item.value)}점</strong>
          </span>
        ))}
      </div>
    </div>
  );
}

function ScatterRiskInfrastructure({
  items,
  large = false,
}: {
  items: ImprovementCandidate[];
  large?: boolean;
}) {
  const data = items
    .filter(
      (item) =>
        typeof item.infrastructure_score === "number" &&
        typeof item.predicted_high_risk_probability_2026 === "number",
    )
    .map((item) => ({
      region: item.sido_sgg_nm,
      infrastructure: scoreNumber(item.infrastructure_score) ?? 0,
      aiScore: scoreNumber(item.predicted_high_risk_probability_2026) ?? 0,
    }));
  if (data.length === 0) return null;
  return (
    <div className={`bklit-scatter ${large ? "large" : ""}`}>
      <ScatterChart
        data={data}
        xDataKey="infrastructure"
        aspectRatio={large ? "2.5 / 1" : "2 / 1"}
        margin={{ top: 24, right: 28, bottom: 28, left: 34 }}
        animationDuration={700}
      >
        <Grid strokeOpacity={0.16} vertical />
        <Scatter dataKey="aiScore" fill={AMBER} radius={large ? 6 : 5} />
      </ScatterChart>
      <div className="scatter-axis-note">
        <span>인프라 공급수준 →</span>
        <span>↑ AI 상대적 고위험 점수</span>
      </div>
    </div>
  );
}

function PageHeading({
  title,
  copy,
  warning,
}: {
  title: string;
  copy: string;
  warning?: string;
}) {
  return (
    <div className="analysis-heading product-heading">
      <div>
        <h1>{title}</h1>
        <p>{copy}</p>
      </div>
      {warning ? <strong>{warning}</strong> : null}
    </div>
  );
}

function App() {
  const [view, setView] = useState<View>("map");
  const [regionSeed, setRegionSeed] = useState("");
  const openRegion = useCallback((regionName: string) => {
    setRegionSeed(regionName);
    setView("region");
  }, []);
  const navItems: Array<{ id: View; label: string; number: string; icon: string }> = [
    { id: "map", label: "현황지도", number: "01", icon: "map" },
    { id: "region", label: "지역분석", number: "02", icon: "grid" },
    { id: "ai", label: "AI 분석", number: "03", icon: "spark" },
    { id: "guide", label: "데이터안내", number: "04", icon: "info" },
  ];
  const viewLabel = navItems.find((item) => item.id === view)?.label ?? "현황지도";
  return (
    <div className="app-shell">
      <aside className="service-sidebar">
        <button
          className="service-brand"
          onClick={() => setView("map")}
          aria-label="현황 지도로 이동"
        >
          <span className="brand-mark" aria-hidden="true">
            <i />
            <i />
            <i />
          </span>
          <span>
            <strong>어린이 교통안전</strong>
            <em>CHILD SAFETY LAB</em>
          </span>
        </button>
        <nav className="service-nav" aria-label="주요 메뉴">
          <span className="nav-kicker">ANALYSIS SPACE</span>
          {navItems.map((item) => (
            <button
              key={item.id}
              className={view === item.id ? "active" : ""}
              onClick={() => setView(item.id)}
            >
              <span className={`nav-icon ${item.icon}`} aria-hidden="true" />
              <span>{item.label}</span>
              <b>{item.number}</b>
            </button>
          ))}
        </nav>
        <div className="sidebar-coverage">
          <span>DATA COVERAGE</span>
          <strong>전국 229개 지역</strong>
          <p>사고다발지역 1,803개 · 시설 상세분석 가능 49개 지역</p>
          <i aria-hidden="true" />
        </div>
      </aside>
      <main className="service-main">
        <div className="content-topbar">
          <span>SAFE SPOT</span>
          <b>/</b>
          <strong>{viewLabel}</strong>
          <em>READ-ONLY DATA</em>
        </div>
        {view === "map" ? (
          <MapView onOpenRegion={openRegion} />
        ) : view === "region" ? (
          <RegionAnalysis initialRegion={regionSeed} />
        ) : view === "ai" ? (
          <AiAnalysis onOpenRegion={openRegion} />
        ) : (
          <Guide />
        )}
      </main>
    </div>
  );
}

function MapView({ onOpenRegion }: { onOpenRegion: (region: string) => void }) {
  const [items, setItems] = useState<Hotspot[]>([]);
  const [total, setTotal] = useState(0);
  const [typeTotals, setTypeTotals] = useState({
    schoolzone_hotspot: 646,
    child_pedestrian_hotspot: 1157,
  });
  const [selected, setSelected] = useState<Hotspot | null>(null);
  const [nearby, setNearby] = useState<NearbyItem[]>([]);
  const [filters, setFilters] = useState({
    source_type: "all" as SourceType | "all",
    accident_year: "all",
    sido: "all",
    sigungu: "all",
  });
  const [radius, setRadius] = useState(1000);
  const [mapCenter, setMapCenter] = useState<[number, number]>(DEFAULT_CENTER);
  const [loading, setLoading] = useState(true);
  const [nearbyLoading, setNearbyLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadHotspots = useCallback(
    async (nextFilters = filters, signal?: AbortSignal) => {
      setLoading(true);
      setError(null);
      try {
        const baseParams = new URLSearchParams();
        if (nextFilters.sido !== "all")
          baseParams.set("sido", nextFilters.sido);
        if (nextFilters.sigungu !== "all")
          baseParams.set("sigungu", nextFilters.sigungu);
        let data: HotspotResponse;
        if (nextFilters.source_type === "all") {
          const makeTypeParams = (
            sourceType: SourceType,
            includeYear: boolean,
          ) => {
            const params = new URLSearchParams(baseParams);
            params.set("source_type", sourceType);
            params.set(
              "limit",
              nextFilters.accident_year !== "all" ? "500" : "250",
            );
            if (includeYear && nextFilters.accident_year !== "all")
              params.set("accident_year", nextFilters.accident_year);
            return params;
          };
          if (nextFilters.accident_year !== "all") {
            const pedestrian = await fetchJson<HotspotResponse>(
              `/api/hotspots?${makeTypeParams("child_pedestrian_hotspot", true)}`,
              signal,
            );
            data = {
              items: pedestrian.items,
              pagination: { ...pedestrian.pagination, limit: 500 },
            };
            setTypeTotals((current) => ({
              ...current,
              child_pedestrian_hotspot: pedestrian.pagination.total,
            }));
          } else {
            const [schoolzone, pedestrian] = await Promise.all([
              fetchJson<HotspotResponse>(
                `/api/hotspots?${makeTypeParams("schoolzone_hotspot", false)}`,
                signal,
              ),
              fetchJson<HotspotResponse>(
                `/api/hotspots?${makeTypeParams("child_pedestrian_hotspot", false)}`,
                signal,
              ),
            ]);
            setTypeTotals({
              schoolzone_hotspot: schoolzone.pagination.total,
              child_pedestrian_hotspot: pedestrian.pagination.total,
            });
            data = {
              items: [...schoolzone.items, ...pedestrian.items],
              pagination: {
                limit: 500,
                offset: 0,
                count: schoolzone.items.length + pedestrian.items.length,
                total:
                  schoolzone.pagination.total + pedestrian.pagination.total,
              },
            };
          }
        } else {
          const params = new URLSearchParams(baseParams);
          params.set("source_type", nextFilters.source_type);
          params.set("limit", "500");
          if (
            nextFilters.source_type === "child_pedestrian_hotspot" &&
            nextFilters.accident_year !== "all"
          )
            params.set("accident_year", nextFilters.accident_year);
          data = await fetchJson<HotspotResponse>(
            `/api/hotspots?${params}`,
            signal,
          );
          setTypeTotals((current) => ({
            ...current,
            [nextFilters.source_type]: data.pagination.total,
          }));
        }
        setItems(data.items);
        setTotal(data.pagination.total);
        setSelected((current) =>
          current &&
          data.items.some((item) => item.hotspot_id === current.hotspot_id)
            ? current
            : null,
        );
      } catch (cause) {
        if ((cause as Error).name !== "AbortError")
          setError(
            "API 연결에 실패했습니다. FastAPI 서버가 실행 중인지 확인해 주세요.",
          );
      } finally {
        if (!signal?.aborted) setLoading(false);
      }
    },
    [filters],
  );

  useEffect(() => {
    const controller = new AbortController();
    loadHotspots(filters, controller.signal);
    return () => controller.abort();
  }, [filters, loadHotspots]);
  const options = useMemo(() => {
    const sidos = [
      ...new Set(items.map((item) => item.sido).filter(Boolean)),
    ].sort((a, b) => (a ?? "").localeCompare(b ?? "", "ko"));
    const sigungus =
      filters.sido === "all"
        ? []
        : [
            ...new Set(
              items
                .filter((item) => item.sido === filters.sido)
                .map((item) => item.sigungu)
                .filter(Boolean),
            ),
          ].sort((a, b) => (a ?? "").localeCompare(b ?? "", "ko"));
    return { sidos, sigungus };
  }, [filters.sido, items]);
  const loadNearby = useCallback(async () => {
    const target = selected
      ? [selected.latitude, selected.longitude]
      : mapCenter;
    setNearbyLoading(true);
    try {
      const params = new URLSearchParams({
        lat: String(target[0]),
        lng: String(target[1]),
        radius_m: String(radius),
        limit: "50",
      });
      if (filters.source_type !== "all")
        params.set("source_type", filters.source_type);
      const data = await fetchJson<NearbyResponse>(
        `/api/hotspots/nearby?${params}`,
      );
      setNearby(data.items);
    } finally {
      setNearbyLoading(false);
    }
  }, [filters.source_type, mapCenter, radius, selected]);
  const changeFilter = (key: keyof typeof filters, value: string) =>
    setFilters((current) => {
      const next = { ...current, [key]: value } as typeof filters;
      if (key === "source_type") next.accident_year = "all";
      if (key === "sido") next.sigungu = "all";
      return next;
    });
  const resetFilters = () => {
    setFilters({
      source_type: "all",
      accident_year: "all",
      sido: "all",
      sigungu: "all",
    });
    setSelected(null);
    setNearby([]);
  };
  const sourceOptions: FilterOption[] = [
    { value: "all", label: "전체 선정지점" },
    { value: "schoolzone_hotspot", label: "어린이보호구역" },
    { value: "child_pedestrian_hotspot", label: "보행어린이" },
  ];
  const yearDisabled = filters.source_type === "schoolzone_hotspot";
  const yearOptions: FilterOption[] = [
    { value: "all", label: "전체 연도" },
    ...Array.from({ length: 14 }, (_, index) => 2025 - index).map((year) => ({
      value: String(year),
      label: `${year}`,
    })),
  ];
  const sidoOptions: FilterOption[] = [
    { value: "all", label: "전국" },
    ...options.sidos.map((sido) => ({ value: sido ?? "", label: sido ?? "" })),
  ];
  const sigunguOptions: FilterOption[] = [
    {
      value: "all",
      label: filters.sido === "all" ? "시도 선택 후 표시" : "전체",
    },
    ...options.sigungus.map((sigungu) => ({
      value: sigungu ?? "",
      label: sigungu ?? "",
    })),
  ];

  return (
    <>
      <section className="intro-row">
        <div>
          <p>NATIONAL OVERVIEW</p>
          <h1>
            어린이의 길을,
            <br />
            <span>데이터로</span> 더 안전하게.
          </h1>
          <span>
            공식 사고다발지역 데이터를 지도 위에서 탐색하고, 지역별 사고
            패턴과 선정지점을 확인합니다.
          </span>
        </div>
        <div className="intro-status">
          <strong>{loading ? "—" : formatNumber(total)}</strong>
          <span>사고다발지역 선정지점</span>
          <p>
            어린이보호구역 {formatNumber(typeTotals.schoolzone_hotspot)} ·
            보행어린이 {formatNumber(typeTotals.child_pedestrian_hotspot)}
          </p>
        </div>
      </section>
      <section className="map-layout">
        <div className="map-stage">
          <div className="map-toolbar" aria-label="지도 필터">
            <FilterSelect
              label="유형"
              value={filters.source_type}
              options={sourceOptions}
              onChange={(value) => changeFilter("source_type", value)}
            />
            <FilterSelect
              label="연도"
              value={yearDisabled ? "disabled" : filters.accident_year}
              displayValue={yearDisabled ? "연도 미제공" : undefined}
              disabled={yearDisabled}
              disabledReason="이 데이터는 원본에 연도 필드가 제공되지 않습니다"
              options={
                yearDisabled
                  ? [{ value: "disabled", label: "연도 미제공" }]
                  : yearOptions
              }
              onChange={(value) => changeFilter("accident_year", value)}
            />
            <FilterSelect
              label="시도"
              value={filters.sido}
              options={sidoOptions}
              onChange={(value) => changeFilter("sido", value)}
            />
            <FilterSelect
              label="시군구"
              value={filters.sigungu}
              disabled={filters.sido === "all"}
              displayValue={
                filters.sido === "all" ? "시도 선택 후 이용" : undefined
              }
              disabledReason="시도를 먼저 선택하면 해당 시군구만 표시됩니다"
              options={sigunguOptions}
              onChange={(value) => changeFilter("sigungu", value)}
            />
            <button className="toolbar-reset" onClick={resetFilters}>
              초기화
            </button>
          </div>
          {filters.source_type === "all" && filters.accident_year !== "all" ? (
            <p className="year-filter-note">
              연도 선택 시 연도 정보가 있는 보행어린이 데이터에 적용됩니다.
            </p>
          ) : null}
          <span className="result-status">
            전체 {formatNumber(total)}개 중 지도에는 최대{" "}
            {formatNumber(items.length)}개 표시
          </span>
          <div className="map-frame">
            {loading ? (
              <div className="map-overlay">
                <span className="loader" />
                지점을 불러오는 중
              </div>
            ) : null}
            {error ? (
              <div className="map-overlay error-state">
                <strong>API 연결 실패</strong>
                <span>{error}</span>
              </div>
            ) : null}
            <MapContainer
              className="leaflet-map"
              center={DEFAULT_CENTER}
              zoom={7.5}
              zoomControl
              scrollWheelZoom
            >
              <TileLayer
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                subdomains="abc"
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>'
                maxZoom={19}
              />
              <StateBoundaries />
              <MapViewport selected={selected} onCenterChange={setMapCenter} />
              {items.map((item) => (
                <Marker
                  key={`${item.source_type}-${item.hotspot_id}`}
                  position={[item.latitude, item.longitude]}
                  icon={markerIcon(
                    item.source_type,
                    selected?.hotspot_id === item.hotspot_id,
                  )}
                  eventHandlers={{ click: () => setSelected(item) }}
                >
                  <Tooltip direction="top" offset={[0, -10]}>
                    {item.location_name ?? "사고다발지역 선정지점"}
                  </Tooltip>
                </Marker>
              ))}
            </MapContainer>
          </div>
          <div className="map-legend">
            <span>
              <i className="legend-marker square" /> 어린이보호구역
            </span>
            <span>
              <i className="legend-marker diamond" /> 보행어린이
            </span>
          </div>
        </div>
      </section>
      <HotspotList items={items} onSelect={setSelected} />
      {selected ? (
        <HotspotDetail
          selected={selected}
          nearby={nearby}
          radius={radius}
          nearbyLoading={nearbyLoading}
          onRadiusChange={setRadius}
          onNearby={loadNearby}
          onClose={() => setSelected(null)}
          onOpenRegion={onOpenRegion}
        />
      ) : null}
    </>
  );
}

function HotspotList({
  items,
  onSelect,
}: {
  items: Hotspot[];
  onSelect: (item: Hotspot) => void;
}) {
  const maxAccident = Math.max(
    1,
    ...items.map((item) => item.accident_count ?? 0),
  );
  return (
    <section className="hotspot-list-panel">
      <div className="section-rule-title">
        <span>RECENT HOTSPOTS</span>
        <h2>주요 선정지점</h2>
      </div>
      <div className="hotspot-list">
        {items.slice(0, 6).map((item, index) => (
          <button
            key={`${item.source_type}-${item.hotspot_id}`}
            onClick={() => onSelect(item)}
            className="hotspot-list-row"
          >
            <small>{String(index + 1).padStart(2, "0")}</small>
            <span>
              <strong>{item.location_name ?? "사고다발지역 선정지점"}</strong>
              <em>
                {sourceMeta[item.source_type].short} · {formatYear(item)}
              </em>
            </span>
            <b>{formatNumber(item.accident_count)}건</b>
            <i style={{ width: `${ratio(item.accident_count, maxAccident)}%` }} />
            <mark aria-hidden="true">→</mark>
          </button>
        ))}
      </div>
    </section>
  );
}

function HotspotDetail({
  selected,
  nearby,
  radius,
  nearbyLoading,
  onRadiusChange,
  onNearby,
  onClose,
  onOpenRegion,
}: {
  selected: Hotspot;
  nearby: NearbyItem[];
  radius: number;
  nearbyLoading: boolean;
  onRadiusChange: (radius: number) => void;
  onNearby: () => void;
  onClose: () => void;
  onOpenRegion: (region: string) => void;
}) {
  const regionName = regionNameFromHotspot(selected);
  return (
    <aside className="detail-panel mobile-open">
      <div className="detail-header">
        <h2>지점 상세</h2>
        <button
          className="close-detail"
          onClick={onClose}
          aria-label="상세 닫기"
        >
          ×
        </button>
      </div>
      <div className="detail-type">
        {sourceMeta[selected.source_type].label}
      </div>
      <h3>{selected.location_name ?? "사고다발지역 선정지점"}</h3>
      <p className="detail-location">
        {[selected.sido, selected.sigungu].filter(Boolean).join(" ")}
      </p>
      <div className="meaning-note">
        <strong>공식 기준에 따라 선정된 사고다발지역 지점입니다.</strong>
        <span>지역 전체 평가나 학교 단위 평가를 의미하지 않습니다.</span>
      </div>
      <div className="detail-metrics">
        <Metric label="사고연도" value={formatYear(selected)} />
        <Metric
          label="연도 확인 상태"
          value={selected.year_verification_status ?? "알 수 없음"}
        />
        <Metric
          label="사고건수"
          value={`${formatNumber(selected.accident_count)}건`}
        />
        <Metric
          label="사상자수"
          value={`${formatNumber(selected.casualty_count)}명`}
        />
      </div>
      <div className="injury-list">
        <span>사상자 상세</span>
        <Metric
          label="사망"
          value={`${formatNumber(selected.death_count)}명`}
        />
        <Metric
          label="중상"
          value={`${formatNumber(selected.serious_injury_count)}명`}
        />
        <Metric
          label="경상"
          value={`${formatNumber(selected.minor_injury_count)}명`}
        />
        <Metric
          label="부상신고"
          value={`${formatNumber(selected.injury_report_count)}명`}
        />
      </div>
      <div className="detail-actions">
        <button disabled={!regionName} onClick={() => onOpenRegion(regionName)}>
          이 지역 분석 보기
        </button>
      </div>
      <div className="nearby-action">
        <div>
          <strong>주변 사고다발지점</strong>
          <span>선택 지점 기준 거리순 조회</span>
        </div>
        <select
          value={radius}
          onChange={(event) => onRadiusChange(Number(event.target.value))}
        >
          <option value={500}>500m</option>
          <option value={1000}>1km</option>
          <option value={2000}>2km</option>
          <option value={5000}>5km</option>
        </select>
        <button onClick={onNearby}>
          {nearbyLoading ? "조회 중" : "주변 지점 보기"}
        </button>
      </div>
      {nearby.length > 0 ? (
        <div className="nearby-mini-list">
          {nearby.slice(0, 5).map((item) => (
            <div key={`${item.source_type}-${item.hotspot_id}`}>
              <strong>{item.location_name ?? "사고다발지역 선정지점"}</strong>
              <span>
                {sourceMeta[item.source_type].short} ·{" "}
                {formatNumber(item.distance_m)}m
              </span>
            </div>
          ))}
        </div>
      ) : null}
    </aside>
  );
}

function RegionAnalysis({ initialRegion }: { initialRegion: string }) {
  const [aiRegions, setAiRegions] = useState<AiRiskRegion[]>([]);
  const [facilities, setFacilities] = useState<Region[]>([]);
  const [candidates, setCandidates] = useState<ImprovementCandidate[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [compareIds, setCompareIds] = useState<string[]>([]);
  const [query, setQuery] = useState("");
  const [compareQuery, setCompareQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    Promise.all([
      fetchJson<AiRiskResponse>(
        "/api/ai/high-risk?limit=300",
        controller.signal,
      ),
      fetchJson<RegionResponse>("/api/facilities?limit=100", controller.signal),
      fetchJson<ImprovementResponse>(
        "/api/improvement-candidates?limit=100",
        controller.signal,
      ),
    ])
      .then(([aiData, facilityData, candidateData]) => {
        setAiRegions(aiData.items);
        setFacilities(facilityData.items);
        setCandidates(candidateData.items);
      })
      .catch((cause) => {
        if ((cause as Error).name !== "AbortError")
          setError("지역분석 API 연결에 실패했습니다.");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, []);
  useEffect(() => {
    if (!initialRegion || aiRegions.length === 0) return;
    const exact = aiRegions.find((item) => item.sido_sgg_nm === initialRegion);
    const loose = aiRegions.find((item) =>
      item.sido_sgg_nm.includes(initialRegion),
    );
    const next = exact ?? loose;
    if (next) setSelectedId(next.sido_sgg_nm);
  }, [aiRegions, initialRegion]);
  const facilityByRegion = useMemo(
    () => new Map(facilities.map((item) => [item.sido_sgg_nm, item])),
    [facilities],
  );
  const candidateByRegion = useMemo(
    () => new Map(candidates.map((item) => [item.sido_sgg_nm, item])),
    [candidates],
  );
  const selected =
    aiRegions.find((item) => item.sido_sgg_nm === selectedId) ?? null;
  const selectedFacility = selected
    ? (facilityByRegion.get(selected.sido_sgg_nm) ?? null)
    : null;
  const selectedCandidate = selected
    ? (candidateByRegion.get(selected.sido_sgg_nm) ?? null)
    : null;
  const regionOptions = query
    ? aiRegions.filter((item) => item.sido_sgg_nm.includes(query)).slice(0, 12)
    : [];
  const compareOptions = compareQuery
    ? aiRegions
        .filter(
          (item) =>
            item.sido_sgg_nm.includes(compareQuery) &&
            !compareIds.includes(item.sido_sgg_nm),
        )
        .slice(0, 8)
    : [];
  const comparison = compareIds
    .map((id) => aiRegions.find((item) => item.sido_sgg_nm === id))
    .filter(Boolean) as AiRiskRegion[];
  const national = useMemo(
    () => ({
      avgRate: average(
        aiRegions.map((item) => item.prev_accident_per_10k_child),
      ),
      avgAccident: average(aiRegions.map((item) => item.prev_accident_count)),
      medianScore:
        [...aiRegions]
          .map((item) => scoreNumber(item.predicted_high_risk_probability_2026))
          .filter((value): value is number => value != null)
          .sort((a, b) => a - b)[Math.floor(aiRegions.length / 2)] ?? null,
      rateValues: aiRegions
        .map((item) => item.prev_accident_per_10k_child)
        .filter((value): value is number => typeof value === "number"),
      scoreValues: aiRegions
        .map((item) => scoreNumber(item.predicted_high_risk_probability_2026))
        .filter((value): value is number => value != null),
    }),
    [aiRegions],
  );
  const facilityMax = {
    crosswalk: Math.max(
      1,
      ...facilities.map((item) => item.crosswalk_count ?? 0),
    ),
    traffic: Math.max(
      1,
      ...facilities.map((item) => item.traffic_light_count ?? 0),
    ),
    schoolzone: Math.max(
      1,
      ...facilities.map((item) => item.schoolzone_count ?? 0),
    ),
    cctv: Math.max(
      0.01,
      ...facilities.map((item) => item.cctv_install_rate ?? 0),
    ),
  };
  const toggleCompare = (id: string) =>
    setCompareIds((current) =>
      current.includes(id)
        ? current.filter((item) => item !== id)
        : current.length >= 4
          ? [...current.slice(1), id]
          : [...current, id],
    );
  const facilityCompareItems = comparison
    .map((item) => facilityByRegion.get(item.sido_sgg_nm))
    .filter(Boolean) as Region[];
  const canCompareFacilities =
    comparison.length > 0 && facilityCompareItems.length === comparison.length;
  return (
    <section className="analysis-view region-product-view">
      <PageHeading
        title="지역별로 보면, 다음이 보입니다."
        copy="사고 수준과 AI 분석, 안전시설 현황을 같은 기준으로 비교합니다. 시설 상세분석은 데이터가 확보된 49개 지역에만 표시됩니다."
      />
      {error ? <div className="inline-error">{error}</div> : null}
      {loading ? (
        <div className="analysis-loading">
          <span className="loader" />
          전국 229개 지역 데이터를 불러오는 중입니다
        </div>
      ) : (
        <div className="analysis-canvas">
          <section className="region-search-strip">
            <div>
              <strong>지역 선택</strong>
              <span>전국 229개 시군구 · 상세 시설분석은 49개 지역</span>
            </div>
            <label>
              <input
                value={query}
                placeholder="지역명 검색 예: 서울특별시 강남구"
                onChange={(event) => setQuery(event.target.value)}
              />
            </label>
            <select
              value={selectedId}
              onChange={(event) => setSelectedId(event.target.value)}
              aria-label="상세 지역 선택"
            >
              <option value="">지역을 선택하세요</option>
              {aiRegions.map((region) => (
                <option key={region.sido_sgg_nm} value={region.sido_sgg_nm}>
                  {region.sido_sgg_nm}
                </option>
              ))}
            </select>
            {query ? (
              <div className="region-option-popover">
                {regionOptions.map((region) => (
                  <button
                    key={region.sido_sgg_nm}
                    onClick={() => {
                      setSelectedId(region.sido_sgg_nm);
                      setQuery("");
                    }}
                  >
                    {region.sido_sgg_nm}
                  </button>
                ))}
              </div>
            ) : null}
          </section>
          <section className="region-summary-hero">
            <div className="region-story">
              <span>선택 지역 요약</span>
              <h2>{selected?.sido_sgg_nm ?? "지역 미선택"}</h2>
              <p>
                {selected
                  ? makeRegionDiagnosis(selected, aiRegions, national.avgRate)
                  : "지역을 선택하면 사고 현황과 AI 분석 결과를 확인할 수 있습니다."}
              </p>
            </div>
            <div className="summary-metrics">
              <div className="metric-group">
                <small>일반 현황</small>
                <BigMetric
                  label="어린이 사고건수"
                  value={`${formatNumber(selected?.prev_accident_count)}건`}
                />
                <BigMetric
                  label="어린이 1만명당 사고율"
                  value={formatDecimal(selected?.prev_accident_per_10k_child)}
                />
                <BigMetric
                  label="어린이 인구"
                  value={`${formatNumber(selected?.prev_child_population)}명`}
                />
              </div>
              <div className="metric-group ai">
                <small>AI 분석</small>
                <BigMetric
                  label="AI 상대적 고위험 점수"
                  value={formatScore(
                    selected?.predicted_high_risk_probability_2026,
                  )}
                  tone="ai"
                />
                <BigMetric
                  label="전국 AI 순위"
                  value={
                    selected
                      ? `${formatNumber(selected.risk_rank_2026)}위`
                      : "—"
                  }
                  tone="ai"
                />
              </div>
            </div>
          </section>
          {selected ? (
            <section className="region-insight-grid">
              <article className="analysis-panel wide">
                <div className="panel-title">
                  <strong>전국 대비 위치</strong>
                  <span>평균·분포 기준의 분석용 비교</span>
                </div>
                <PositionBars
                  selected={selected}
                  avgRate={national.avgRate}
                  avgAccident={national.avgAccident}
                  medianScore={national.medianScore}
                  ratePercentile={percentile(
                    selected.prev_accident_per_10k_child,
                    national.rateValues,
                  )}
                  scorePercentile={percentile(
                    scoreNumber(
                      selected.predicted_high_risk_probability_2026,
                    ) ?? undefined,
                    national.scoreValues,
                  )}
                />
              </article>
              <article className="analysis-panel">
                <div className="panel-title">
                  <strong>최근 변화 지표</strong>
                  <span>기존 변화량 필드 기반</span>
                </div>
                <RecentTrendLine selected={selected} />
                <div className="compact-metric-grid">
                  <Metric
                    label="최근 1년 사고건수 변화"
                    value={formatSigned(
                      selected.accident_count_change_1y,
                      "건",
                    )}
                  />
                  <Metric
                    label="최근 1년 사고율 변화"
                    value={formatSigned(selected.accident_rate_change_1y)}
                  />
                  <Metric
                    label="최근 2년 평균 사고건수"
                    value={`${formatDecimal(selected.accident_count_mean_2y)}건`}
                  />
                  <Metric
                    label="자동차 등록대수"
                    value={`${formatNumber(selected.prev_vehicle_count)}대`}
                  />
                </div>
              </article>
            </section>
          ) : null}
          <section className="region-insight-grid">
            <article className="analysis-panel compare-builder">
              <div className="panel-title">
                <strong>지역 비교</strong>
                <span>{compareIds.length}/4 선택</span>
              </div>
              <input
                value={compareQuery}
                placeholder="비교할 지역 검색"
                onChange={(event) => setCompareQuery(event.target.value)}
              />
              <div className="compare-chip-row">
                {compareIds.map((id) => (
                  <button key={id} onClick={() => toggleCompare(id)}>
                    {id}
                    <b>×</b>
                  </button>
                ))}
              </div>
              <div className="compare-option-grid">
                {compareOptions.map((region) => (
                  <button
                    key={region.sido_sgg_nm}
                    onClick={() => toggleCompare(region.sido_sgg_nm)}
                  >
                    {region.sido_sgg_nm}
                  </button>
                ))}
              </div>
            </article>
            <article className="analysis-panel wide">
              <div className="panel-title">
                <strong>229개 공통지표 비교</strong>
                <span>AI 점수는 실제 발생 가능성 수치가 아님</span>
              </div>
              {comparison.length === 0 ? (
                <p className="compact-empty">
                  비교할 지역을 최대 4개까지 선택하세요.
                </p>
              ) : (
                <RegionCompareChart items={comparison} />
              )}
            </article>
          </section>
          <section className="analysis-panel facility-section">
            <div className="panel-title">
              <strong>시설 상세분석</strong>
              <span>49개 시설 상세분석 가능 지역에만 표시</span>
            </div>
            {!selected ? (
              <p className="compact-empty">
                지역을 선택하면 시설 상세분석 가능 여부가 표시됩니다.
              </p>
            ) : selectedFacility ? (
              <div className="facility-layout">
                <FacilityRing
                  facility={selectedFacility}
                  candidate={selectedCandidate}
                />
                <div className="facility-side-metrics">
                  <Metric
                    label="시설 부족도"
                    value={formatScore(
                      selectedCandidate?.infrastructure_deficiency_score,
                    )}
                  />
                  <Metric
                    label="개선 검토 점수"
                    value={formatScore(
                      selectedCandidate?.improvement_priority_score,
                    )}
                  />
                  <Metric
                    label="개선 검토 후보 순위"
                    value={
                      selectedCandidate
                        ? `${formatNumber(selectedCandidate.improvement_priority_rank)}위`
                        : "자료 없음"
                    }
                  />
                </div>
                <div className="facility-meter-grid">
                  <FacilityMeter
                    label="횡단보도"
                    value={selectedFacility.crosswalk_count}
                    max={facilityMax.crosswalk}
                  />
                  <FacilityMeter
                    label="신호등"
                    value={selectedFacility.traffic_light_count}
                    max={facilityMax.traffic}
                  />
                  <FacilityMeter
                    label="어린이보호구역"
                    value={selectedFacility.schoolzone_count}
                    max={facilityMax.schoolzone}
                  />
                  <FacilityMeter
                    label="CCTV 설치율"
                    value={selectedFacility.cctv_install_rate}
                    max={facilityMax.cctv}
                    format="percent"
                  />
                </div>
              </div>
            ) : (
              <p className="facility-unavailable compact">
                현재 이 지역은 상세 안전시설 비교 데이터가 제공되지 않습니다.
                <br />
                시설 분석은 49개 지역을 대상으로 제공됩니다.
              </p>
            )}
            {comparison.length > 0 ? (
              <div className="facility-compare-note">
                {canCompareFacilities
                  ? `선택한 ${comparison.length}개 지역은 모두 시설 비교가 가능합니다.`
                  : "비교 지역 중 시설 상세분석 데이터가 없는 지역이 있어 시설값을 0으로 보정하지 않습니다."}
              </div>
            ) : null}
          </section>
        </div>
      )}
    </section>
  );
}

function makeRegionDiagnosis(
  selected: AiRiskRegion,
  all: AiRiskRegion[],
  avgRate: number | null,
) {
  const rate = selected.prev_accident_per_10k_child;
  const rateText =
    typeof rate === "number" && avgRate != null
      ? rate >= avgRate
        ? "전국 평균보다 높은 편"
        : "전국 평균보다 낮은 편"
      : "전국 평균 비교 자료가 제한적";
  const change =
    typeof selected.accident_count_change_1y === "number"
      ? selected.accident_count_change_1y > 0
        ? "최근 1년 사고건수는 증가했습니다"
        : selected.accident_count_change_1y < 0
          ? "최근 1년 사고건수는 감소했습니다"
          : "최근 1년 사고건수는 큰 변화가 없습니다"
      : "최근 변화 정보는 제한적입니다";
  return `${selected.sido_sgg_nm}은 전국 ${all.length || 229}개 지역 중 AI 상대적 고위험도 ${selected.risk_rank_2026}위입니다. 어린이 1만 명당 사고율은 ${rateText}이며, ${change}.`;
}
function PositionBars({
  selected,
  avgRate,
  avgAccident,
  medianScore,
  ratePercentile,
  scorePercentile,
}: {
  selected: AiRiskRegion;
  avgRate: number | null;
  avgAccident: number | null;
  medianScore: number | null;
  ratePercentile: number;
  scorePercentile: number;
}) {
  const rows = [
    {
      label: "사고율",
      base:
        avgRate == null ? "전국 평균 —" : `전국 평균 ${formatDecimal(avgRate)}`,
      selected: formatDecimal(selected.prev_accident_per_10k_child),
      pos: ratePercentile,
    },
    {
      label: "AI 점수",
      base:
        medianScore == null
          ? "전국 중앙값 —"
          : `전국 중앙값 ${formatDecimal(medianScore)}점`,
      selected: formatScore(selected.predicted_high_risk_probability_2026),
      pos: scorePercentile,
    },
    {
      label: "사고건수",
      base:
        avgAccident == null
          ? "전국 평균 —"
          : `전국 평균 ${formatDecimal(avgAccident)}건`,
      selected: `${formatNumber(selected.prev_accident_count)}건`,
      pos: 50,
    },
  ];
  return (
    <div className="position-bars">
      {rows.map((row) => (
        <div key={row.label} className="position-row">
          <span>{row.label}</span>
          <i>
            <b style={{ left: `${Math.max(4, Math.min(96, row.pos))}%` }} />
          </i>
          <strong>{row.selected}</strong>
          <em>{row.base}</em>
        </div>
      ))}
    </div>
  );
}
function RecentTrendLine({ selected }: { selected: AiRiskRegion }) {
  const rateNow =
    typeof selected.prev_accident_per_10k_child === "number"
      ? selected.prev_accident_per_10k_child
      : null;
  const countNow =
    typeof selected.prev_accident_count === "number"
      ? selected.prev_accident_count
      : null;
  const rateChange =
    typeof selected.accident_rate_change_1y === "number"
      ? selected.accident_rate_change_1y
      : null;
  const countChange =
    typeof selected.accident_count_change_1y === "number"
      ? selected.accident_count_change_1y
      : null;
  const data = [
    {
      date: new Date("2025-01-01"),
      rate:
        rateNow != null && rateChange != null
          ? rateNow - rateChange
          : (rateNow ?? 0),
      count:
        countNow != null && countChange != null
          ? countNow - countChange
          : (countNow ?? 0),
    },
    { date: new Date("2026-01-01"), rate: rateNow ?? 0, count: countNow ?? 0 },
  ];
  return (
    <div className="recent-trend-line">
      <LineChart
        data={data}
        xDataKey="date"
        aspectRatio="3.4 / 1"
        margin={{ top: 14, right: 18, bottom: 18, left: 20 }}
        animationDuration={620}
      >
        <Grid strokeOpacity={0.12} />
        <Line
          dataKey="rate"
          stroke={AMBER}
          strokeWidth={2.4}
          showMarkers
        markers={{ radius: 3.5, fill: AMBER }}
        />
        <Line
          dataKey="count"
          stroke="rgba(242,240,235,.54)"
          strokeWidth={1.8}
          showMarkers
        markers={{ radius: 3, fill: "rgba(242,240,235,.7)" }}
        />
      </LineChart>
      <div className="trend-line-legend">
        <span>사고율 변화</span>
        <span>사고건수 변화</span>
      </div>
    </div>
  );
}
function RegionCompareChart({ items }: { items: AiRiskRegion[] }) {
  const data = items.map((item) => ({
    region: item.sido_sgg_nm.replace("특별자치도 ", "").replace("광역시 ", ""),
    accident: item.prev_accident_count ?? 0,
    rate: item.prev_accident_per_10k_child ?? 0,
    score: scoreNumber(item.predicted_high_risk_probability_2026) ?? 0,
  }));
  return (
    <div className="compare-chart-product">
      <BarChart
        data={data}
        xDataKey="region"
        aspectRatio="3.4 / 1"
        margin={{ top: 18, right: 24, bottom: 24, left: 24 }}
        barGap={0.24}
        animationDuration={650}
      >
        <Grid strokeOpacity={0.14} />
        <Bar dataKey="accident" fill="rgba(232,163,61,.92)" lineCap="round" />
        <Bar dataKey="rate" fill="rgba(232,163,61,.48)" lineCap="round" />
        <Bar dataKey="score" fill="rgba(242,240,235,.5)" lineCap="round" />
      </BarChart>
      <div className="compare-chart-legend">
        <span>사고건수</span>
        <span>1만명당 사고율</span>
        <span>AI 점수</span>
      </div>
    </div>
  );
}

function AiAnalysis({
  onOpenRegion,
}: {
  onOpenRegion: (region: string) => void;
}) {
  const [top10, setTop10] = useState<AiRiskRegion[]>([]);
  const [highRisk, setHighRisk] = useState<AiRiskRegion[]>([]);
  const [candidates, setCandidates] = useState<ImprovementCandidate[]>([]);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [importance, setImportance] = useState<ImportanceItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    Promise.all([
      fetchJson<AiRiskResponse>("/api/ai/top10", controller.signal),
      fetchJson<AiRiskResponse>(
        "/api/ai/high-risk?limit=12",
        controller.signal,
      ),
      fetchJson<ImprovementResponse>(
        "/api/improvement-candidates?limit=49",
        controller.signal,
      ),
      fetchJson<ModelInfo>("/api/model/info", controller.signal),
      fetchJson<ImportanceResponse>("/api/model/importance", controller.signal),
    ])
      .then(([topData, riskData, candidateData, infoData, importanceData]) => {
        setTop10(topData.items);
        setHighRisk(riskData.items);
        setCandidates(candidateData.items);
        setModelInfo(infoData);
        setImportance(importanceData.items);
      })
      .catch((cause) => {
        if ((cause as Error).name !== "AbortError")
          setError("AI 분석 API 연결에 실패했습니다.");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, []);
  const top10Data = top10.map((item) => ({
    region: item.sido_sgg_nm.replace("특별자치도 ", "").replace("광역시 ", ""),
    score: scoreNumber(item.predicted_high_risk_probability_2026) ?? 0,
  }));
  const importanceData = importance.map((item) => ({
    feature: featureLabels[item.feature] ?? item.feature,
    value: Math.max(0, item.importance_mean),
  }));
  return (
    <section className="analysis-view ai-view">
      <PageHeading
        title="어디가 다음 위험이 될까요?"
        copy="Logistic Regression 모델이 전국 229개 지역의 다음 연도 상대적 고위험 여부를 분석합니다. 예측은 확률이 아니라 지역 간 비교를 위한 분석 점수입니다."
        warning="AI 점수는 실제 발생 가능성 수치가 아닌 상대적 고위험 분석 결과입니다."
      />
      {error ? <div className="inline-error">{error}</div> : null}
      {loading ? (
        <div className="analysis-loading">
          <span className="loader" />
          AI 분석 데이터를 불러오는 중입니다
        </div>
      ) : (
        <div className="ai-dashboard">
          <section className="model-strip">
            <span>
              최종 모델: {modelInfo?.model.model_name ?? "Logistic Regression"}
            </span>
            <span>분석 대상: 전국 229개 시군구</span>
            <span>예측 기준: 다음 연도 상대적 고위험군</span>
            <span>설명 방식: permutation importance</span>
          </section>
          <section className="ai-primary-grid">
            <article className="analysis-panel ai-top-panel">
              <div className="panel-title">
                <strong>AI 예측 기반 TOP 10</strong>
                <span>지역 단위 상대적 고위험 점수</span>
              </div>
              <BklitHorizontalBars
                data={top10Data}
                labelKey="region"
                valueKey="score"
                valueSuffix="점"
                aspect="1.6 / 1"
              />
            </article>
            <article className="analysis-panel highrisk-table">
              <div className="panel-title">
                <strong>AI 고위험 예상지역</strong>
                <span>행 선택 시 지역분석으로 이동</span>
              </div>
              <div className="compact-table">
                {highRisk.map((item) => (
                  <button
                    key={item.sido_sgg_nm}
                    onClick={() => onOpenRegion(item.sido_sgg_nm)}
                  >
                    <b>{item.risk_rank_2026}</b>
                    <span>{item.sido_sgg_nm}</span>
                    <strong>
                      {formatScore(item.predicted_high_risk_probability_2026)}
                    </strong>
                    <em>{formatSigned(item.accident_rate_change_1y)}</em>
                  </button>
                ))}
              </div>
            </article>
          </section>
          <section className="ai-secondary-grid">
            <article className="analysis-panel scatter-panel">
              <div className="panel-title">
                <strong>AI 위험 × 인프라 공급수준</strong>
                <span>49개 시설분석 지역 · 분석 보조선</span>
              </div>
              <ScatterRiskInfrastructure items={candidates} large />
              <p className="analysis-footnote">
                좌측 상단에 가까울수록 상대적 위험은 높지만 인프라 공급수준은
                낮은 후보로 읽을 수 있습니다. 이는 분석 보조 해석이며 공식 정책
                기준이 아닙니다.
              </p>
            </article>
            <article className="analysis-panel candidate-panel">
              <div className="panel-title">
                <strong>개선 검토 후보 TOP 5</strong>
                <span>분석용 휴리스틱</span>
              </div>
              <div className="candidate-list">
                {candidates.slice(0, 5).map((item) => (
                  <button
                    key={item.sido_sgg_nm}
                    onClick={() => onOpenRegion(item.sido_sgg_nm)}
                  >
                    <b>{item.improvement_priority_rank}</b>
                    <span>{item.sido_sgg_nm}</span>
                    <strong>
                      {formatScore(item.improvement_priority_score)}
                    </strong>
                    <em>
                      AI{" "}
                      {formatScore(item.predicted_high_risk_probability_2026)} ·
                      부족도 {formatScore(item.infrastructure_deficiency_score)}
                    </em>
                  </button>
                ))}
              </div>
              <p className="analysis-footnote">
                개선 검토 점수는 공식 정책 우선순위가 아닙니다.
              </p>
            </article>
          </section>
          <section className="analysis-panel importance-panel-product">
            <div className="panel-title">
              <strong>모델 설명 변수</strong>
              <span>Permutation importance · SHAP 미사용</span>
            </div>
            <BklitHorizontalBars
              data={importanceData}
              labelKey="feature"
              valueKey="value"
              valueSuffix=""
              aspect="2.8 / 1"
            />
            <p className="analysis-footnote">
              Permutation importance는 특정 변수를 섞었을 때 모델 성능이 얼마나
              감소하는지를 바탕으로 변수 중요도를 측정합니다.
            </p>
          </section>
        </div>
      )}
    </section>
  );
}

function Guide() {
  return (
    <section className="analysis-view guide-view product-guide">
      <PageHeading
        title="숫자보다 먼저, 맥락을 읽으세요."
        copy="세이프스팟의 지도와 AI 결과를 올바르게 해석하기 위한 기준입니다. 데이터 범위가 다른 지표를 하나의 의미로 합치지 않습니다."
      />
      <div className="guide-steps" aria-label="서비스 해석 흐름">
        <article>
          <i className="nav-icon map" aria-hidden="true" />
          <span>01</span>
          <strong>어디에서 나타나는가</strong>
          <p>공식 기준으로 선정된 사고다발지역을 지도에서 확인합니다.</p>
        </article>
        <article>
          <i className="nav-icon grid" aria-hidden="true" />
          <span>02</span>
          <strong>지역 수준은 어떤가</strong>
          <p>전국 229개 지역의 사고 수준과 AI 분석 점수를 비교합니다.</p>
        </article>
        <article>
          <i className="nav-icon spark" aria-hidden="true" />
          <span>03</span>
          <strong>무엇을 개선할까</strong>
          <p>시설 데이터가 있는 지역은 부족 신호와 개선 검토 후보를 함께 봅니다.</p>
        </article>
      </div>
      <div className="guide-product-grid">
        <article className="guide-card guide-wide">
          <span>A · MAP DATA</span>
          <h2>사고다발지역이란?</h2>
          <p>
            지도에 표시된 지점은 모든 사고가 발생한 장소가 아닙니다. 공식 선정
            기준에 따라 사고가 반복적으로 발생하거나 위험 신호가 확인된 지점을
            의미합니다.
          </p>
          <div className="guide-callout">
            <strong>1,803개 선정지점</strong>
            <em>선정지점 수는 전체 사고건수가 아닙니다.</em>
          </div>
        </article>
        <article className="guide-card">
          <span>B · YEAR POLICY</span>
          <h2>연도 정보는 정확하게</h2>
          <p>
            보행어린이 사고다발지역은 제공된 연도만 필터링합니다.
            어린이보호구역 사고다발지역은 원본에 공식 연도 필드가 없어 임의로
            연도를 붙이지 않습니다.
          </p>
          <div className="guide-compare-line">
            <b>확인된 연도만 표시</b>
            <i />
            <b>추정 연도 미생성</b>
          </div>
        </article>
        <article className="guide-card">
          <span>C · AI INTERPRETATION</span>
          <h2>AI 점수와 순위의 의미</h2>
          <p>
            AI 점수는 실제 발생 가능성 수치가 아니라 전국 229개 지역 안에서
            상대적으로 강하게 나타난 고위험 신호입니다. 전국 AI 순위도 공식
            평가나 확정 판단이 아닙니다.
          </p>
          <div className="guide-equation">
            <strong>분석점수</strong>
            <b>≠</b>
            <strong>확률값</strong>
          </div>
        </article>
        <article className="guide-card scope-numbers">
          <span>D · DATA COVERAGE</span>
          <h2>분석 범위</h2>
          <div>
            <strong>1,803</strong>
            <span>사고다발지역 선정지점</span>
          </div>
          <div>
            <strong>229</strong>
            <span>AI 분석 지역</span>
          </div>
          <div>
            <strong>49</strong>
            <span>시설 상세분석 가능 지역</span>
          </div>
          <p>229개 AI 분석과 49개 시설분석은 데이터 범위가 다릅니다.</p>
        </article>
        <article className="guide-card">
          <span>E · MODEL</span>
          <h2>모델 설명</h2>
          <dl className="guide-dl">
            <div>
              <dt>모델</dt>
              <dd>Logistic Regression</dd>
            </div>
            <div>
              <dt>예측 목표</dt>
              <dd>다음 연도 상대적 고위험 여부</dd>
            </div>
            <div>
              <dt>Target</dt>
              <dd>어린이 1만명당 사고율 연도별 Q75 이상</dd>
            </div>
            <div>
              <dt>설명 방식</dt>
              <dd>Permutation importance 기반 변수 영향 확인</dd>
            </div>
          </dl>
        </article>
        <article className="guide-card guide-warning">
          <span>F · LIMITATION</span>
          <h2>해석 시 주의사항</h2>
          <ul>
            <li>사고다발지역은 모든 사고 지점을 의미하지 않습니다.</li>
            <li>선정지점 미존재는 안전 지역을 의미하지 않습니다.</li>
            <li>시설 데이터가 없는 지역을 시설 0개로 해석하지 않습니다.</li>
            <li>개선 검토 점수는 공식 정책 우선순위가 아닙니다.</li>
          </ul>
        </article>
        <article className="guide-card guide-wide data-provenance">
          <span>DATA PROVENANCE</span>
          <h2>분석 결과는 의사결정의 시작점입니다.</h2>
          <p>
            시설 설치나 정책 결정을 위해서는 현장 확인과 추가 데이터 검토가 함께
            필요합니다. 세이프스팟은 데이터를 연결해 검토 대상을 좁히는 분석
            도구입니다.
          </p>
        </article>
      </div>
    </section>
  );
}
function DataSourceRow({
  title,
  scope,
  purpose,
  items,
}: {
  title: string;
  scope: string;
  purpose: string;
  items: string;
}) {
  return (
    <div>
      <strong>{title}</strong>
      <span>{scope}</span>
      <span>{purpose}</span>
      <em>{items}</em>
    </div>
  );
}

export default App;
