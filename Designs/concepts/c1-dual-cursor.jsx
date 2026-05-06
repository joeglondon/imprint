// c1-dual-cursor.jsx — Concept 1, now theme-driven.
// A theme object is passed in; all colors read from it. 5 variants live in
// c1-themes.jsx so they can be scrolled side-by-side in the canvas.

const C1DualCursor = ({ theme: T = C1_THEMES.graphite, label = 'graphite' }) => {
  const regions = [
    { id: 'r-onb',  x: 300, y: 190, label: 'Onboarding research',  tone: 'a', size: 28 },
    { id: 'r-api',  x: 520, y: 340, label: 'API architecture',     tone: 'b', size: 34 },
    { id: 'r-leg',  x: 180, y: 410, label: 'Legal & privacy',      tone: 'c', size: 24 },
    { id: 'r-eng',  x: 720, y: 220, label: 'Engineering journal',  tone: 'd', size: 26 },
    { id: 'r-mkt',  x: 660, y: 500, label: 'Marketing notes',      tone: 'a', size: 20 },
    { id: 'r-ref',  x: 380, y: 540, label: 'Reference papers',     tone: 'b', size: 22 },
  ];
  const humanAt = regions[1];
  const aiAt    = regions[3];
  const edges = [
    ['r-onb','r-api'],['r-api','r-eng'],['r-api','r-leg'],['r-eng','r-mkt'],
    ['r-ref','r-api'],['r-ref','r-leg'],['r-onb','r-ref'],['r-mkt','r-api'],
  ];
  const chunks = [];
  regions.forEach((r, i) => {
    const count = 5 + (i % 3);
    for (let k = 0; k < count; k++) {
      const a = (k / count) * Math.PI * 2 + i;
      const rad = r.size + 14 + (k % 2) * 8;
      chunks.push({ id: `${r.id}-c${k}`, x: r.x + Math.cos(a) * rad, y: r.y + Math.sin(a) * rad, tone: r.tone });
    }
  });
  const toneColor = { a: T.accentA, b: T.accentB, c: T.accentC, d: T.accentD };

  // button styles, derived from theme
  const btnPrimary = { height: 28, padding: '0 12px', borderRadius: 6, background: T.btnBg, color: T.btnFg, border: 'none', fontSize: 12, fontWeight: 500, cursor: 'pointer', fontFamily: 'inherit' };
  const btnGhost = { height: 28, padding: '0 10px', borderRadius: 6, background: 'transparent', color: T.ink2, border: `0.5px solid ${T.rule}`, fontSize: 12, cursor: 'pointer', fontFamily: 'inherit', display: 'inline-flex', alignItems: 'center', gap: 5 };

  // Overlay a locally-scoped mac frame so we don't recolor the global Mac component.
  return (
    <div style={{
      position: 'relative', width: '100%', height: '100%',
      borderRadius: 14, overflow: 'hidden',
      background: T.surface1,
      boxShadow: T.windowShadow,
      display: 'flex', flexDirection: 'column',
      fontFamily: '"IBM Plex Sans", -apple-system, system-ui, sans-serif',
      color: T.ink,
    }}>
      {/* titlebar */}
      <div style={{
        height: 38, flexShrink: 0, display: 'flex', alignItems: 'center', gap: 12,
        padding: '0 14px', borderBottom: `0.5px solid ${T.rule}`, background: T.chromeBg,
      }}>
        <div style={{ display: 'flex', gap: 8 }}>
          <Dot c="#ED6A5E" /><Dot c="#F5BF4F" /><Dot c="#62C554" />
        </div>
        <div style={{ flex: 1, textAlign: 'center', fontSize: 12, color: T.ink2 }}>
          <span style={{ fontWeight: 500 }}>imprint</span>
          <span style={{ color: T.ink4, marginLeft: 10 }}>— {label}</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span className="mono" style={{ fontSize: 10.5, color: T.ink3 }}>418 docs · 6,204 chunks</span>
          <div style={{ width: 1, height: 14, background: T.rule, margin: '0 4px' }} />
          <ActorT who="human" T={T} />
          <ActorT who="ai" T={T} label="Agent" />
        </div>
      </div>

      <div style={{ flex: 1, minHeight: 0, display: 'flex' }}>
        {/* LEFT SIDEBAR */}
        <aside style={{
          width: 208, background: T.surface2, borderRight: `0.5px solid ${T.rule}`,
          padding: '14px 12px', display: 'flex', flexDirection: 'column', gap: 14,
        }}>
          <div>
            <Row label="Store" T={T} right={<span className="mono" style={{ fontSize: 9, color: T.ink4 }}>~/Library</span>} />
            <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              {[['Library', 'tray', true, '418'],['Sources', 'doc', false, '34'],['Model', 'cpu', false, null],['Map', 'globe', false, '34']].map(([n, ic, on, r]) => (
                <div key={n} style={{
                  display: 'flex', alignItems: 'center', gap: 8,
                  padding: '5px 8px', borderRadius: 5,
                  background: on ? T.selBg : 'transparent',
                  color: on ? T.ink : T.ink2, fontSize: 12.5, fontWeight: on ? 500 : 400,
                }}>
                  <Icon name={ic} size={13} color={on ? T.ink : T.ink3} />
                  <span style={{ flex: 1 }}>{n}</span>
                  {r && <span className="mono" style={{ fontSize: 10, color: T.ink4 }}>{r}</span>}
                </div>
              ))}
            </div>
          </div>

          <div>
            <Row label="Regions" T={T} right={<Icon name="plus" size={12} color={T.ink4} />} />
            <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              {regions.map(r => (
                <div key={r.id} style={{ display: 'flex', alignItems: 'center', gap: 7, fontSize: 12, padding: '3px 4px', color: T.ink2 }}>
                  <span style={{ width: 8, height: 8, borderRadius: '50%', background: toneColor[r.tone] }} />
                  <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.label}</span>
                  <span className="mono" style={{ fontSize: 10, color: T.ink4 }}>{r.size}</span>
                </div>
              ))}
            </div>
          </div>

          <div style={{ marginTop: 'auto' }}>
            <Row label="Presence" T={T} />
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6, fontSize: 11.5, color: T.ink2 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}><ActorT who="human" T={T} /><span style={{ color: T.ink3 }}>on API</span></div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}><ActorT who="ai" T={T} /><span style={{ color: T.ink3 }}>on Eng. journal</span></div>
            </div>
          </div>
        </aside>

        {/* CENTER */}
        <main style={{ flex: 1, background: T.surface1, position: 'relative', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
          <div style={{ height: 48, borderBottom: `0.5px solid ${T.rule}`, padding: '0 16px', display: 'flex', alignItems: 'center', gap: 10, background: T.surface1 }}>
            <div style={{ flex: 1, height: 28, borderRadius: 6, background: T.surface2, border: `0.5px solid ${T.rule}`, display: 'flex', alignItems: 'center', gap: 8, padding: '0 10px' }}>
              <Icon name="search" size={13} color={T.ink3} />
              <span style={{ fontSize: 12.5, color: T.ink2 }}>How is rate limiting implemented across the gateway?</span>
              <span style={{ flex: 1 }} />
              <span className="mono" style={{ fontSize: 10, color: T.ink4 }}>⏎ to route</span>
            </div>
            <button style={btnPrimary}>Route</button>
            <button style={btnGhost}><Icon name="back" size={12} color={T.ink2} /> Backtrack</button>
          </div>

          <div style={{ flex: 1, position: 'relative', background: T.cloudBg }}>
            <svg viewBox="0 0 900 680" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}>
              <defs>
                <radialGradient id={`c1-halo-h-${label}`} cx="50%" cy="50%">
                  <stop offset="0%" stopColor={T.human} stopOpacity="0.45" />
                  <stop offset="60%" stopColor={T.human} stopOpacity="0.08" />
                  <stop offset="100%" stopColor={T.human} stopOpacity="0" />
                </radialGradient>
                <radialGradient id={`c1-halo-a-${label}`} cx="50%" cy="50%">
                  <stop offset="0%" stopColor={T.ai} stopOpacity="0.45" />
                  <stop offset="60%" stopColor={T.ai} stopOpacity="0.08" />
                  <stop offset="100%" stopColor={T.ai} stopOpacity="0" />
                </radialGradient>
              </defs>
              {edges.map(([s, t], i) => {
                const src = regions.find(r => r.id === s), tgt = regions.find(r => r.id === t);
                return <line key={i} x1={src.x} y1={src.y} x2={tgt.x} y2={tgt.y} stroke={T.edge} strokeWidth="1" />;
              })}
              {chunks.map(c => <circle key={c.id} cx={c.x} cy={c.y} r="2.5" fill={toneColor[c.tone]} opacity="0.55" />)}
              {regions.map(r => (
                <g key={r.id}>
                  <circle cx={r.x} cy={r.y} r={r.size} fill={toneColor[r.tone]} opacity="0.18" />
                  <circle cx={r.x} cy={r.y} r={r.size * 0.55} fill={toneColor[r.tone]} opacity="0.55" />
                  <circle cx={r.x} cy={r.y} r={r.size * 0.25} fill={toneColor[r.tone]} />
                  <text x={r.x} y={r.y + r.size + 14} fontFamily='"IBM Plex Mono"' fontSize="10" fill={T.ink2} textAnchor="middle" letterSpacing="0.3">{r.label}</text>
                </g>
              ))}
              <circle cx={humanAt.x} cy={humanAt.y} r="70" fill={`url(#c1-halo-h-${label})`} />
              <circle cx={aiAt.x}    cy={aiAt.y}    r="70" fill={`url(#c1-halo-a-${label})`} />
              <g>
                <circle cx={humanAt.x} cy={humanAt.y} r="9" fill="none" stroke={T.human} strokeWidth="2" />
                <circle cx={humanAt.x} cy={humanAt.y} r="3" fill={T.human} />
              </g>
              <g>
                <circle cx={aiAt.x} cy={aiAt.y} r="9" fill="none" stroke={T.ai} strokeWidth="2" strokeDasharray="2 2" />
                <circle cx={aiAt.x} cy={aiAt.y} r="3" fill={T.ai} />
              </g>
              <line x1={humanAt.x} y1={humanAt.y} x2={aiAt.x} y2={aiAt.y} stroke={T.ink3} strokeOpacity="0.35" strokeWidth="1" strokeDasharray="3 4" />
              <g transform={`translate(${humanAt.x + 14}, ${humanAt.y - 14})`}>
                <rect x="0" y="0" width="70" height="18" rx="9" fill={T.surface1} stroke={T.rule} />
                <circle cx="10" cy="9" r="3" fill={T.human} />
                <text x="18" y="12" fontSize="10" fontFamily='"IBM Plex Mono"' fill={T.ink2}>you · here</text>
              </g>
              <g transform={`translate(${aiAt.x + 14}, ${aiAt.y - 14})`}>
                <rect x="0" y="0" width="88" height="18" rx="9" fill={T.surface1} stroke={T.rule} />
                <circle cx="10" cy="9" r="3" fill={T.ai} />
                <text x="18" y="12" fontSize="10" fontFamily='"IBM Plex Mono"' fill={T.ink2}>Agent · citing</text>
              </g>
            </svg>
          </div>

          <div style={{ height: 30, borderTop: `0.5px solid ${T.rule}`, padding: '0 14px', display: 'flex', alignItems: 'center', gap: 14, background: T.surface2, fontSize: 11, color: T.ink3 }}>
            <span className="mono">↔ drag to pan · ⌘+scroll to zoom</span>
            <span style={{ flex: 1 }} />
            <span className="mono">embeddinggemma:300m</span>
            <span style={{ width: 1, height: 12, background: T.rule }} />
            <span className="mono" style={{ color: T.ok }}>● local</span>
          </div>
        </main>

        {/* RIGHT INSPECTOR */}
        <aside style={{ width: 320, background: T.surface1, borderLeft: `0.5px solid ${T.rule}`, display: 'flex', flexDirection: 'column' }}>
          <div style={{ padding: '14px 16px 10px', borderBottom: `0.5px solid ${T.rule}` }}>
            <Row label="Selection" T={T} right={<TagT T={T} tone="ai">region</TagT>} />
            <div className="serif" style={{ fontSize: 19, fontWeight: 500, lineHeight: 1.25, color: T.ink }}>API architecture</div>
            <div className="mono" style={{ fontSize: 10.5, color: T.ink4, marginTop: 3 }}>r-api · 68 chunks · 12 docs</div>
            <div style={{ fontSize: 12.5, color: T.ink2, marginTop: 8, lineHeight: 1.5 }}>
              Gateway design, service boundaries, rate limiting, retries. Neighboring: Engineering journal, Reference papers, Legal & privacy.
            </div>
          </div>
          <div style={{ padding: '12px 16px', borderBottom: `0.5px solid ${T.rule}` }}>
            <Row label="Excerpt" T={T} right={<span className="mono" style={{ fontSize: 9, color: T.ink4 }}>gateway-spec.md · §3.2</span>} />
            <div className="serif" style={{ fontSize: 13, lineHeight: 1.55, color: T.ink }}>
              &ldquo;Rate limits are applied per-tenant using a sliding window counter in Redis. The gateway rejects requests above <mark style={{ background: T.highlight, color: T.ink, padding: '0 2px', borderRadius: 2 }}>1,200 rpm</mark> with HTTP 429 and a <mark style={{ background: T.highlight, color: T.ink, padding: '0 2px', borderRadius: 2 }}>Retry-After</mark> header…&rdquo;
            </div>
          </div>
          <div style={{ flex: 1, padding: '12px 16px', overflow: 'hidden' }}>
            <Row label="Trails" T={T} right={<span className="mono" style={{ fontSize: 10, color: T.ink4 }}>last 10 min</span>} />
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, fontSize: 11.5 }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}><ActorT who="human" T={T} /></div>
                {['Onboarding', 'Reference papers', 'API arch.', '§3.2 gateway'].map((s, i) => (
                  <TrailStepT key={i} label={s} color={T.human} last={i===3} T={T} />
                ))}
              </div>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}><ActorT who="ai" T={T} /></div>
                {['Eng. journal', 'API arch.', 'gateway-spec', 'rate-limit.rs'].map((s, i) => (
                  <TrailStepT key={i} label={s} color={T.ai} last={i===3} T={T} />
                ))}
              </div>
            </div>
            <div style={{ marginTop: 14 }}>
              <Row label="Links from here" T={T} />
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                {[['semantic', 'Retry budgets (eng. journal)', 0.82, 'ai'],
                  ['same-doc', 'Gateway: auth section', 0.77, 'ink'],
                  ['citation', 'Lamport, 1978', 0.64, 'c'],
                  ['entity', 'Redis · sliding-window', 0.58, 'd']].map(([kind, label, score, tone], i) => (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '6px 8px', borderRadius: 5, background: T.surface2, border: `0.5px solid ${T.rule}` }}>
                    <TagT T={T} tone={tone} style={{ minWidth: 54, justifyContent: 'center' }}>{kind}</TagT>
                    <span style={{ flex: 1, color: T.ink, fontSize: 12 }}>{label}</span>
                    <span className="mono" style={{ fontSize: 10.5, color: T.ink3 }}>{score.toFixed(2)}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
};

function Row({ label, right, T }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
      <span className="mono" style={{ fontSize: 10, color: T.ink3, letterSpacing: 0.8, textTransform: 'uppercase' }}>{label}</span>
      {right}
    </div>
  );
}
function ActorT({ who, T, label }) {
  const color = who === 'human' ? T.human : T.ai;
  return (
    <span className="mono" style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 10.5, color: T.ink2, letterSpacing: 0.3 }}>
      <span style={{ width: 8, height: 8, borderRadius: '50%', background: color, boxShadow: `0 0 0 2px ${T.surface1}, 0 0 0 3px ${color}33` }} />
      {label || (who === 'human' ? 'you' : 'Agent')}
    </span>
  );
}
function TagT({ children, tone = 'ink', T, style = {} }) {
  const map = {
    ink:     { bg: T.tagInk,     fg: T.ink2 },
    human:   { bg: T.tagHuman,   fg: T.humanInk },
    ai:      { bg: T.tagAi,      fg: T.aiInk },
    a:       { bg: T.tagHuman,   fg: T.humanInk },
    b:       { bg: T.tagAi,      fg: T.aiInk },
    c:       { bg: T.tagC,       fg: T.cInk },
    d:       { bg: T.tagD,       fg: T.dInk },
  };
  const t = map[tone] || map.ink;
  return (
    <span className="mono" style={{
      display: 'inline-flex', alignItems: 'center', gap: 4, padding: '2px 6px', borderRadius: 4,
      background: t.bg, color: t.fg, fontSize: 10, fontWeight: 500, letterSpacing: 0.4, textTransform: 'uppercase', ...style,
    }}>{children}</span>
  );
}
function TrailStepT({ label, color, last, T }) {
  return (
    <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start', paddingBottom: last ? 0 : 4 }}>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', paddingTop: 4 }}>
        <div style={{ width: 7, height: 7, borderRadius: '50%', background: last ? color : 'transparent', border: `1.5px solid ${color}` }} />
        {!last && <div style={{ width: 1, flex: 1, minHeight: 14, background: color, opacity: 0.4, marginTop: 2 }} />}
      </div>
      <div style={{ paddingTop: 1, color: last ? T.ink : T.ink2, fontWeight: last ? 500 : 400, lineHeight: 1.35 }}>{label}</div>
    </div>
  );
}

window.C1DualCursor = C1DualCursor;
