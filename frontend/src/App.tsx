import { useEffect, useMemo, useRef, useState } from 'react';
import L from 'leaflet';
import Plot from 'react-plotly.js';
import landData from './data/natural-earth-land.json';

type HeatPoint = { lat: number; lon: number; temperature: number };
type Heatmap = { points: HeatPoint[]; min_temperature: number; max_temperature: number; bounds: [[number, number], [number, number]] };
type Profile = { lat: number; lon: number; date: string; depths: number[]; temperatures: number[]; uncertainties: number[]; attention: Record<string, number[]> };
type OceanOverlay = L.Layer & { setHeatmap?: (value: Heatmap) => void };
const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const DEPTHS = [0, 25, 50, 75, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000];
const TODAY = new Date().toISOString().slice(0, 10);
const MODEL_BOUNDS = { south: -40, west: 20, north: 30, east: 130 };

function colorFor(value: number, min: number, max: number) { const ratio = Math.max(0, Math.min(1, (value - min) / Math.max(0.1, max - min))); const hue = 215 - ratio * 215; return `hsl(${hue}, 84%, ${47 + ratio * 5}%)`; }

type Ring = [number, number][];
type LandGeometry = { type: 'Polygon' | 'MultiPolygon'; coordinates: Ring[] | Ring[][] };
type LandFeature = { geometry: LandGeometry };
const landFeatures = (landData as unknown as { features: LandFeature[] }).features;

function validRing(ring: Ring) { if (ring.length < 4 || ring[0][0] !== ring[ring.length - 1][0] || ring[0][1] !== ring[ring.length - 1][1]) return false; return ring.every(([lon, lat], index) => Number.isFinite(lon) && Number.isFinite(lat) && (index === 0 || lon !== ring[index - 1][0] || lat !== ring[index - 1][1])); }
function ringsFor(feature: LandFeature) { return feature.geometry.type === 'Polygon' ? feature.geometry.coordinates as Ring[] : (feature.geometry.coordinates as Ring[][]).flat(); }
const validLandRings = landFeatures.flatMap(ringsFor).filter(validRing);
if (!validLandRings.length) throw new Error('Natural Earth land GeoJSON contains no valid closed rings');
const regionalLandRings = validLandRings.filter((ring) => ring.some(([lon, lat]) => lat >= MODEL_BOUNDS.south - 1 && lat <= MODEL_BOUNDS.north + 1 && lon >= MODEL_BOUNDS.west - 1 && lon <= MODEL_BOUNDS.east + 1));
function pointInRing(lat: number, lon: number, ring: Ring) { let inside = false; for (let index = 0, previous = ring.length - 1; index < ring.length; previous = index++) { const [lonA, latA] = ring[index]; const [lonB, latB] = ring[previous]; if ((latA > lat) !== (latB > lat) && lon < (lonB - lonA) * (lat - latA) / (latB - latA) + lonA) inside = !inside; } return inside; }
function isOceanPoint(lat: number, lon: number) { return lat >= MODEL_BOUNDS.south && lat <= MODEL_BOUNDS.north && lon >= MODEL_BOUNDS.west && lon <= MODEL_BOUNDS.east && !regionalLandRings.some((ring) => pointInRing(lat, lon, ring)); }

function hslToRgb(hue: number) { const saturation = 0.84; const lightness = 0.5; const chroma = (1 - Math.abs(2 * lightness - 1)) * saturation; const x = chroma * (1 - Math.abs((hue / 60) % 2 - 1)); const match = lightness - chroma / 2; const rgb = hue < 60 ? [chroma, x, 0] : hue < 120 ? [x, chroma, 0] : hue < 180 ? [0, chroma, x] : hue < 240 ? [0, x, chroma] : hue < 300 ? [x, 0, chroma] : [chroma, 0, x]; return rgb.map((channel) => Math.round((channel + match) * 255)); }


function createOceanOverlay(map: L.Map, onPointClick: (lat: number, lon: number) => void): OceanOverlay {
  const layer = new L.Layer() as OceanOverlay;
  let canvas: HTMLCanvasElement;
  let current: Heatmap | null = null;
  let drawFrame: number | null = null;
  const draw = () => {
    if (!canvas || !current) return;
    const context = canvas.getContext('2d'); if (!context) return;
    const size = map.getSize(); canvas.width = size.x; canvas.height = size.y; context.clearRect(0, 0, size.x, size.y);
    const lats = [...new Set(current.points.map((point) => point.lat))].sort((a, b) => a - b);
    const lons = [...new Set(current.points.map((point) => point.lon))].sort((a, b) => a - b);
    const grid = new Map(current.points.map((point) => [`${point.lat}|${point.lon}`, point.temperature]));
    const bounds = current.bounds; const latStep = (bounds[1][0] - bounds[0][0]) / (lats.length - 1); const lonStep = (bounds[1][1] - bounds[0][1]) / (lons.length - 1);
    const rasterWidth = 420; const rasterHeight = 360; const pixels = new ImageData(rasterWidth, rasterHeight); const min = current.min_temperature; const max = current.max_temperature;
    for (let y = 0; y < rasterHeight; y += 1) for (let x = 0; x < rasterWidth; x += 1) {
      const longitude = bounds[0][1] + (x / (rasterWidth - 1)) * (bounds[1][1] - bounds[0][1]); const latitude = bounds[1][0] - (y / (rasterHeight - 1)) * (bounds[1][0] - bounds[0][0]); const fx = (longitude - bounds[0][1]) / lonStep; const fy = (latitude - bounds[0][0]) / latStep; const ix = Math.floor(fx); const iy = Math.floor(fy); if (ix < 0 || iy < 0 || ix >= lons.length - 1 || iy >= lats.length - 1) continue;
      const tx = fx - ix; const ty = fy - iy; const value = (grid.get(`${lats[iy]}|${lons[ix]}`)! * (1 - tx) + grid.get(`${lats[iy]}|${lons[ix + 1]}`)! * tx) * (1 - ty) + (grid.get(`${lats[iy + 1]}|${lons[ix]}`)! * (1 - tx) + grid.get(`${lats[iy + 1]}|${lons[ix + 1]}`)! * tx) * ty; const rgb = hslToRgb(215 - Math.max(0, Math.min(1, (value - min) / Math.max(0.1, max - min))) * 215); const offset = (y * rasterWidth + x) * 4; pixels.data[offset] = rgb[0]; pixels.data[offset + 1] = rgb[1]; pixels.data[offset + 2] = rgb[2]; pixels.data[offset + 3] = 142;
    }
    const raster = document.createElement('canvas'); raster.width = rasterWidth; raster.height = rasterHeight; raster.getContext('2d')!.putImageData(pixels, 0, 0);
    const southwest = map.latLngToContainerPoint([bounds[0][0], bounds[0][1]]); const northeast = map.latLngToContainerPoint([bounds[1][0], bounds[1][1]]);
    context.save(); context.beginPath(); context.rect(southwest.x, northeast.y, northeast.x - southwest.x, southwest.y - northeast.y); regionalLandRings.forEach((ring) => { const first = map.latLngToContainerPoint([ring[0][1], ring[0][0]]); context.moveTo(first.x, first.y); ring.slice(1).forEach(([lon, lat]) => { const point = map.latLngToContainerPoint([lat, lon]); context.lineTo(point.x, point.y); }); context.closePath(); }); context.clip('evenodd'); context.imageSmoothingEnabled = true; context.globalAlpha = 0.72; context.drawImage(raster, southwest.x, northeast.y, northeast.x - southwest.x, southwest.y - northeast.y); context.restore();
  };
  const scheduleDraw = () => { if (drawFrame !== null) cancelAnimationFrame(drawFrame); drawFrame = requestAnimationFrame(() => { drawFrame = null; draw(); }); };
  layer.onAdd = () => { canvas = L.DomUtil.create('canvas', 'ocean-heatmap-canvas'); canvas.style.position = 'absolute'; canvas.style.left = '0'; canvas.style.top = '0'; canvas.style.zIndex = '400'; canvas.style.pointerEvents = 'auto'; canvas.style.opacity = '0.9'; canvas.width = map.getSize().x; canvas.height = map.getSize().y; map.getContainer().appendChild(canvas); map.on('move zoom resize', scheduleDraw); canvas.addEventListener('click', (event) => { const point = map.containerPointToLatLng([event.offsetX, event.offsetY]); if (isOceanPoint(point.lat, point.lng)) onPointClick(point.lat, point.lng); }); scheduleDraw(); return layer; };
  layer.onRemove = () => { map.off('move zoom resize', scheduleDraw); if (drawFrame !== null) cancelAnimationFrame(drawFrame); canvas?.remove(); return layer; };
  layer.setHeatmap = (value) => { current = value; scheduleDraw(); };
  return layer;
}

export default function App() {
  const mapRef = useRef<HTMLDivElement>(null); const leafletRef = useRef<L.Map | null>(null); const layerRef = useRef<OceanOverlay | null>(null);
  const [depthIndex, setDepthIndex] = useState(0); const [selectedDate, setSelectedDate] = useState(TODAY); const [heatmap, setHeatmap] = useState<Heatmap | null>(null); const [profile, setProfile] = useState<Profile | null>(null); const [loading, setLoading] = useState(true); const [chatOpen, setChatOpen] = useState(false); const [question, setQuestion] = useState(''); const [messages, setMessages] = useState<string[]>(['Ask about temperature, depth, or the warmest region.']);
  const depth = DEPTHS[depthIndex];
  useEffect(() => { if (!mapRef.current || leafletRef.current) return; const map = L.map(mapRef.current, { zoomControl: false, minZoom: 2, maxBounds: [[-42, 18], [32, 132]] }); L.control.zoom({ position: 'bottomright' }).addTo(map); L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { attribution: '&copy; OpenStreetMap contributors', opacity: 0.72 }).addTo(map); map.fitBounds([[MODEL_BOUNDS.south, MODEL_BOUNDS.west], [MODEL_BOUNDS.north, MODEL_BOUNDS.east]], { padding: [12, 12] }); const overlay = createOceanOverlay(map, loadProfile); map.addLayer(overlay); layerRef.current = overlay; leafletRef.current = map; return () => { map.remove(); leafletRef.current = null; }; }, []);
  useEffect(() => { let active = true; setLoading(true); fetch(`${API}/heatmap?date=${selectedDate}&depth=${depth}`).then((response) => response.json()).then((data) => { if (active) setHeatmap(data); }).finally(() => active && setLoading(false)); return () => { active = false; }; }, [selectedDate, depth]);
  useEffect(() => { if (heatmap) layerRef.current?.setHeatmap?.(heatmap); }, [heatmap]);
  async function loadProfile(lat: number, lon: number) { const response = await fetch(`${API}/predict`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ lat, lon, date: selectedDate }) }); setProfile(await response.json()); }
  function download() { if (!profile) return; window.open(`${API}/export?lat=${profile.lat}&lon=${profile.lon}&date=${profile.date}`, '_blank'); }
  async function askAgent(event: React.FormEvent) { event.preventDefault(); if (!question.trim()) return; const asked = question; setQuestion(''); setMessages((items) => [...items, `You: ${asked}`]); const response = await fetch(`${API}/agent`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message: asked }) }); const data = await response.json(); setMessages((items) => [...items, data.answer || 'Assistant unavailable.']); }
  const chart = useMemo(() => profile && ({ x: profile.temperatures, y: profile.depths, lower: profile.temperatures.map((v, i) => v - profile.uncertainties[i]), upper: profile.temperatures.map((v, i) => v + profile.uncertainties[i]) }), [profile]);
    // Plotly's legacy React typings reject string axis titles although the runtime accepts them.
    // @ts-expect-error Plotly layout accepts the compact axis title form.
    return <main className="app-shell"><div ref={mapRef} className="map" /><header className="topbar"><div className="brand"><span className="brand-mark">∿</span><span>Ocean<span>Embed</span></span></div><div className="region-pill"><i /> INDIAN OCEAN <small>MODEL REGION</small></div><div className="status">● LIVE SYNTHETIC FEED</div></header><aside className="toolbar"><div className="tool-title">PREDICTION LAYERS</div><label className="control-label">Observation date <strong>{selectedDate}</strong></label><input aria-label="Observation date" type="date" value={selectedDate} onChange={(event) => setSelectedDate(event.target.value)} /><label className="control-label depth-label">Depth <strong>{depth} m</strong></label><input aria-label="Depth" type="range" min="0" max="14" value={depthIndex} onChange={(event) => setDepthIndex(Number(event.target.value))} /><div className="depth-ticks"><span>0 m</span><span>1,000 m</span></div><div className="gradient-legend"><div className="gradient-bar" /><div><span>{heatmap?.max_temperature.toFixed(1) || '—'} °C</span><span>{heatmap?.min_temperature.toFixed(1) || '—'} °C</span></div></div><div className="legend-caption">PREDICTED TEMPERATURE</div><div className="divider" /><div className="model-note"><span className="pulse" /><div><b>OceanEmbed v0.1</b><small>Multi-modal ocean encoder<br />15 depth levels · uncertainty-aware</small></div></div><button className="agent-toggle" onClick={() => setChatOpen(!chatOpen)}>✦ {chatOpen ? 'Close assistant' : 'Ask OceanEmbed'}</button></aside>{loading && <div className="loading">UPDATING FIELD <span>•••</span></div>}{profile && <section className="profile-panel"><button className="close" aria-label="Close profile" onClick={() => setProfile(null)}>×</button><div className="eyebrow">POINT PROFILE</div><h1>{profile.lat.toFixed(2)}°N <span>/</span> {profile.lon.toFixed(2)}°E</h1><p className="muted">{profile.date} · Indian Ocean</p>{chart && <Plot data={[{ x: chart.upper, y: chart.y, type: 'scatter', mode: 'lines', line: { color: 'transparent' }, hoverinfo: 'skip' }, { x: chart.lower, y: chart.y, type: 'scatter', mode: 'lines', fill: 'tonextx', fillcolor: 'rgba(69, 201, 216, .18)', line: { color: 'transparent' }, hoverinfo: 'skip' }, { x: chart.x, y: chart.y, type: 'scatter', mode: 'lines+markers', line: { color: '#f5bd63', width: 3 }, marker: { color: '#f5bd63', size: 5 }, name: 'Predicted' }]} layout={{ autosize: true, height: 300, margin: { l: 48, r: 15, t: 10, b: 42 }, paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: { color: '#9fb1bc', family: 'DM Mono' }, xaxis: { title: '°C', gridcolor: '#29404b' }, yaxis: { title: 'Depth (m)', autorange: 'reversed', gridcolor: '#29404b' }, showlegend: false }} config={{ displayModeBar: false, responsive: true }} useResizeHandler style={{ width: '100%' }} />}{profile && <div className="profile-footer"><span>95% CONFIDENCE BAND</span><button onClick={download}>↓ DOWNLOAD CSV</button></div>}</section>}{chatOpen && <section className="chat-panel"><div className="chat-head"><b>OceanEmbed assistant</b><button onClick={() => setChatOpen(false)}>×</button></div><div className="chat-messages">{messages.map((message, index) => <p key={index} className={message.startsWith('You:') ? 'user-message' : ''}>{message}</p>)}</div><form onSubmit={askAgent}><input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask about the ocean…" /><button aria-label="Send">↑</button></form></section>}<footer className="coordinates">{profile ? `POINT ${profile.lat.toFixed(3)}N ${profile.lon.toFixed(3)}E` : 'CLICK THE FIELD TO INSPECT A PROFILE'} <span>•</span> {depth} M</footer></main>;
}
