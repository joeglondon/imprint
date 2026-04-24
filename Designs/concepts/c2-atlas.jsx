// c2-atlas.jsx — Concept 2: Atlas
// Regions as territory. 2D contour map; hand-drawn-feeling paper cartography.
// Left: query + legend. Center: the atlas. Bottom: current route.

const C2Atlas = () => {
  // contour blobs — each region is a stack of concentric "elevation" rings
  const regions = [
    { id:'r1', cx: 240, cy: 200, rx: 130, ry: 95, tone: 'amber',   name: 'Onboarding research', chunks: 148 },
    { id:'r2', cx: 540, cy: 220, rx: 170, ry: 120, tone: 'teal',    name: 'API architecture',   chunks: 312 },
    { id:'r3', cx: 300, cy: 460, rx: 110, ry: 90,  tone: 'magenta', name: 'Legal & privacy',    chunks: 88 },
    { id:'r4', cx: 720, cy: 440, rx: 150, ry: 100, tone: 'green',   name: 'Engineering journal', chunks: 210 },
    { id:'r5', cx: 470, cy: 380, rx: 60,  ry: 50,  tone: 'amber',   name: 'Reference papers',   chunks: 64 },
  ];
  const toneColor = { amber: AMBER, teal: TEAL, magenta: MAGENTA, green: GREEN };

  const focus = regions[1]; // API architecture is focused
  const routePts = [
    { x: 240, y: 200 }, { x: 470, y: 380 }, { x: 540, y: 220 }, { x: 720, y: 440 },
  ];

  return (
    <Mac title="imprint" subtitle="— atlas · 02"
      chromeRight={<>
        <button style={btn2}>Survey</button>
        <button style={btn2Primary}>Route</button>
      </>}>
      {/* LEFT */}
      <aside style={{ width: 240, background: PAPER2, borderRight: `0.5px solid ${RULE}`, padding: '14px 12px', display:'flex', flexDirection:'column', gap: 14 }}>
        <div>
          <SectionRow label="Ask" />
          <div style={{ background: PAPER, border: `0.5px solid ${RULE}`, borderRadius: 6, padding: 10 }}>
            <div className="serif" style={{ fontSize: 13.5, lineHeight: 1.45, color: INK }}>
              Where does the retry logic live, and what does legal say about it?
            </div>
            <div style={{ display:'flex', gap:6, marginTop: 8 }}>
              <Tag tone="teal">api</Tag><Tag tone="magenta">legal</Tag>
            </div>
          </div>
        </div>

        <div>
          <SectionRow label="Regions" right={<span className="mono" style={{ fontSize: 9, color: INK4 }}>5 of 34</span>} />
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {regions.map(r => (
              <div key={r.id} style={{
                display:'flex', alignItems:'center', gap:8, padding:'6px 8px', borderRadius: 5,
                background: r.id === focus.id ? 'rgba(40,30,20,0.06)' : 'transparent',
              }}>
                <span style={{ width:10, height:10, borderRadius:2, background: toneColor[r.tone] }} />
                <span style={{ flex:1, fontSize: 12, color: r.id===focus.id ? INK : INK2, fontWeight: r.id===focus.id ? 500 : 400 }}>{r.name}</span>
                <span className="mono" style={{ fontSize: 10, color: INK4 }}>{r.chunks}</span>
              </div>
            ))}
          </div>
        </div>

        <div>
          <SectionRow label="Cartography" />
          <div style={{ display:'flex', flexDirection:'column', gap: 5, fontSize: 11.5, color: INK2 }}>
            <LegendRow swatch={<Ring />} label="contour = chunk density" />
            <LegendRow swatch={<Star />} label="peak = region centroid" />
            <LegendRow swatch={<Route />} label="route = semantic path" />
            <LegendRow swatch={<Pin color={AMBER} />} label="you" />
            <LegendRow swatch={<Pin color={TEAL} />} label="claude" />
          </div>
        </div>

        <div style={{ marginTop:'auto' }}>
          <SectionRow label="Today" />
          <div style={{ fontSize: 11.5, color: INK3, lineHeight: 1.5 }}>
            <Actor who="ai" /> mapped 23 new chunks into <b style={{ color: INK }}>API architecture</b> at 14:02.<br />
            <Actor who="human" /> imported 3 PDFs into <b style={{ color: INK }}>Reference papers</b>.
          </div>
        </div>
      </aside>

      {/* CENTER — the atlas */}
      <main style={{ flex: 1, position:'relative', background: PAPER, overflow: 'hidden' }}>
        {/* paper grid */}
        <svg viewBox="0 0 900 620" preserveAspectRatio="xMidYMid slice"
          style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}>
          <defs>
            <pattern id="c2-grid" width="40" height="40" patternUnits="userSpaceOnUse">
              <path d="M 40 0 L 0 0 0 40" fill="none" stroke="oklch(80% 0.010 85)" strokeWidth="0.4" />
            </pattern>
            <pattern id="c2-grid-minor" width="8" height="8" patternUnits="userSpaceOnUse">
              <path d="M 8 0 L 0 0 0 8" fill="none" stroke="oklch(86% 0.010 85)" strokeWidth="0.3" />
            </pattern>
            <radialGradient id="c2-focus" cx="50%" cy="50%">
              <stop offset="0%" stopColor={TEAL} stopOpacity="0.18" />
              <stop offset="100%" stopColor={TEAL} stopOpacity="0" />
            </radialGradient>
          </defs>
          <rect width="900" height="620" fill={PAPER} />
          <rect width="900" height="620" fill="url(#c2-grid-minor)" />
          <rect width="900" height="620" fill="url(#c2-grid)" />

          {/* contour rings per region (elevation) */}
          {regions.map(r => (
            <g key={r.id}>
              {[1.25, 1, 0.75, 0.5, 0.28].map((s, i) => (
                <ellipse key={i} cx={r.cx} cy={r.cy} rx={r.rx * s} ry={r.ry * s}
                  fill={i === 4 ? toneColor[r.tone] : 'none'}
                  stroke={toneColor[r.tone]}
                  strokeWidth={i === 0 ? 1.2 : 0.7}
                  opacity={i === 4 ? 0.55 : (0.18 + i * 0.08)} />
              ))}
              {/* centroid marker */}
              <g transform={`translate(${r.cx},${r.cy})`}>
                <circle r="4" fill={PAPER} stroke={toneColor[r.tone]} strokeWidth="1.5" />
                <circle r="1.6" fill={toneColor[r.tone]} />
              </g>
              <text x={r.cx} y={r.cy - r.ry - 6}
                fontFamily='"IBM Plex Mono"' fontSize="10.5" fontWeight="500"
                textAnchor="middle" fill={INK2} letterSpacing="0.4">
                {r.name.toUpperCase()}
              </text>
              <text x={r.cx} y={r.cy - r.ry + 6}
                fontFamily='"IBM Plex Mono"' fontSize="9"
                textAnchor="middle" fill={INK4}>
                {r.chunks} chunks · el. {Math.round(r.chunks/3)}
              </text>
            </g>
          ))}

          {/* focus glow */}
          <circle cx={focus.cx} cy={focus.cy} r="140" fill="url(#c2-focus)" />

          {/* the route — dotted line through waypoints */}
          <path d={routePts.map((p, i) => `${i===0?'M':'L'}${p.x},${p.y}`).join(' ')}
            fill="none" stroke={INK2} strokeWidth="1.4" strokeDasharray="4 4" opacity="0.8" />
          {routePts.map((p, i) => (
            <g key={i} transform={`translate(${p.x},${p.y})`}>
              <circle r="6" fill={PAPER} stroke={INK2} strokeWidth="1.2" />
              <text y="3" textAnchor="middle" fontSize="9" fontFamily='"IBM Plex Mono"' fontWeight="600" fill={INK}>{i+1}</text>
            </g>
          ))}

          {/* scattered chunk dots (texture) */}
          {Array.from({ length: 80 }).map((_, i) => {
            const r = regions[i % regions.length];
            const a = (i * 137.5) * Math.PI / 180;
            const dist = 10 + (i % 11) * 8;
            return <circle key={i} cx={r.cx + Math.cos(a)*dist} cy={r.cy + Math.sin(a)*dist*0.75}
              r="1.2" fill={toneColor[r.tone]} opacity="0.6" />;
          })}

          {/* cursors */}
          <Cursor x={470} y={380} color={AMBER} label="you" />
          <Cursor x={720} y={440} color={TEAL}  label="claude" offsetX={-70} />

          {/* compass rose — bottom right */}
          <g transform="translate(830, 560)">
            <circle r="22" fill={PAPER} stroke="oklch(75% 0.010 85)" strokeWidth="0.6" />
            <path d="M0,-16 L3,0 L0,16 L-3,0 Z" fill={INK2} opacity="0.75" />
            <path d="M-16,0 L0,3 L16,0 L0,-3 Z" fill={INK4} opacity="0.55" />
            <text y="-28" textAnchor="middle" fontSize="8" fontFamily='"IBM Plex Mono"' fill={INK3}>semantic N</text>
          </g>

          {/* scale bar */}
          <g transform="translate(32, 580)">
            <line x1="0" y1="0" x2="120" y2="0" stroke={INK2} strokeWidth="1" />
            <line x1="0" y1="-3" x2="0" y2="3" stroke={INK2} strokeWidth="1" />
            <line x1="60" y1="-2" x2="60" y2="2" stroke={INK2} strokeWidth="1" />
            <line x1="120" y1="-3" x2="120" y2="3" stroke={INK2} strokeWidth="1" />
            <text y="14" fontSize="8.5" fontFamily='"IBM Plex Mono"' fill={INK3}>0</text>
            <text x="120" y="14" textAnchor="end" fontSize="8.5" fontFamily='"IBM Plex Mono"' fill={INK3}>0.4 cos-sim</text>
          </g>
        </svg>

        {/* Floating card — focused region */}
        <div style={{
          position: 'absolute', top: 16, right: 16, width: 260,
          background: PAPER, border: `0.5px solid ${RULE}`, borderRadius: 10,
          boxShadow: '0 8px 30px rgba(40,30,20,0.08)',
          padding: '12px 14px',
        }}>
          <div style={{ display:'flex', alignItems:'center', gap:8 }}>
            <span style={{ width:10, height:10, borderRadius:3, background: TEAL }} />
            <span className="mono" style={{ fontSize: 10, color: INK4 }}>REGION · r-api</span>
          </div>
          <div className="serif" style={{ fontSize: 17, fontWeight: 500, marginTop: 4, color: INK }}>
            API architecture
          </div>
          <div style={{ fontSize: 12, color: INK2, lineHeight: 1.5, marginTop: 6 }}>
            Gateway, retries, rate limiting. 312 chunks across 12 docs. Centroid drifted <span className="mono">+0.04</span> since last rebuild.
          </div>
          <div style={{ display:'flex', gap:6, marginTop: 8, flexWrap: 'wrap' }}>
            <Tag tone="teal">retry-budget</Tag><Tag tone="teal">sliding-window</Tag><Tag>gateway</Tag>
          </div>
        </div>
      </main>
    </Mac>
  );
};

function Cursor({ x, y, color, label, offsetX = 12 }) {
  const tint = color === AMBER ? 'oklch(92% 0.05 75)' : 'oklch(92% 0.04 200)';
  return (
    <g>
      <circle cx={x} cy={y} r="10" fill="none" stroke={color} strokeWidth="1.5" opacity="0.9" />
      <circle cx={x} cy={y} r="3.5" fill={color} />
      <g transform={`translate(${x+offsetX},${y-10})`}>
        <rect x="0" y="0" width="60" height="18" rx="9" fill={PAPER} stroke={color} strokeOpacity="0.7" strokeWidth="0.6" />
        <circle cx="9" cy="9" r="3.2" fill={color} />
        <text x="17" y="12" fontFamily='"IBM Plex Mono"' fontSize="10" fill="oklch(30% 0.008 80)">{label}</text>
      </g>
    </g>
  );
}

function LegendRow({ swatch, label }) {
  return (
    <div style={{ display:'flex', alignItems:'center', gap:8 }}>
      <div style={{ width: 22, display:'flex', justifyContent:'center' }}>{swatch}</div>
      <span>{label}</span>
    </div>
  );
}
const Ring = () => <svg width="22" height="14"><ellipse cx="11" cy="7" rx="9" ry="5" fill="none" stroke={INK3} strokeWidth="0.8" /><ellipse cx="11" cy="7" rx="5" ry="3" fill="none" stroke={INK3} strokeWidth="0.8" /></svg>;
const Star = () => <svg width="14" height="14"><circle cx="7" cy="7" r="5" fill="none" stroke={INK3} strokeWidth="0.8" /><circle cx="7" cy="7" r="1.6" fill={INK3} /></svg>;
const Route = () => <svg width="22" height="8"><line x1="0" y1="4" x2="22" y2="4" stroke={INK3} strokeDasharray="3 3" /></svg>;
const Pin = ({ color }) => <svg width="12" height="12"><circle cx="6" cy="6" r="5" fill="none" stroke={color} strokeWidth="1.4" /><circle cx="6" cy="6" r="2" fill={color} /></svg>;

const btn2 = {
  height: 24, padding:'0 10px', borderRadius:5,
  background: PAPER, color: INK2, border: `0.5px solid ${RULE}`,
  fontSize: 11.5, cursor:'pointer', fontFamily:'inherit',
};
const btn2Primary = { ...btn2, background: INK, color: PAPER, border: 'none' };

window.C2Atlas = C2Atlas;
