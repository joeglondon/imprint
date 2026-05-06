// c4-stratigraphy.jsx — Concept 4: Memory as geological strata.
// Time runs left→right. Regions stack vertically. Each chunk is an "outcrop"
// at a point in time. Hits glow. You can scrub a vertical time cursor.

const C4Stratigraphy = () => {
  const strata = [
    { id:'r-api', label:'API architecture',     tone: TEAL,    h: 78, weight: 0.9 },
    { id:'r-eng', label:'Engineering journal',  tone: GREEN,   h: 64, weight: 0.7 },
    { id:'r-onb', label:'Onboarding research',  tone: AMBER,   h: 70, weight: 0.8 },
    { id:'r-leg', label:'Legal & privacy',      tone: MAGENTA, h: 54, weight: 0.5 },
    { id:'r-ref', label:'Reference papers',     tone: AMBER,   h: 58, weight: 0.6 },
  ];
  const totalH = strata.reduce((a, s) => a + s.h, 0);

  // chunks deterministically scattered along time x-axis [0..900]
  const mkChunks = (stratumIdx, weight) => {
    const n = Math.floor(weight * 60 + 10);
    return Array.from({ length: n }).map((_, i) => {
      const seed = (stratumIdx+1) * 131 + i * 37;
      const x = ((seed * 0.618) % 1) * 880 + 10;
      const y = ((seed * 0.382) % 1);
      return { x, y, size: 1.6 + ((seed % 7) * 0.2) };
    });
  };

  // hit markers — query results in the current scrub
  const hits = [
    { stratum: 0, x: 640, label: 'MAX_RPM=1200' },
    { stratum: 0, x: 720, label: 'gateway-spec' },
    { stratum: 3, x: 310, label: 'Q1 approval' },
  ];

  const cursorX = 700; // time cursor

  return (
    <Mac title="imprint" subtitle="— stratigraphy · 04"
      chromeRight={<>
        <span className="mono" style={{ fontSize: 10.5, color: INK3 }}>jun 04 → apr 23 · 318 days</span>
      </>}>
      {/* LEFT — query + stats */}
      <aside style={{ width: 240, background: PAPER2, borderRight:`0.5px solid ${RULE}`, padding:'14px 12px', display:'flex', flexDirection:'column', gap: 14 }}>
        <div>
          <SectionRow label="Ask at this moment" />
          <div style={{ background: PAPER, border:`0.5px solid ${RULE}`, borderRadius: 6, padding:10 }}>
            <div className="serif" style={{ fontSize: 13, color: INK, lineHeight: 1.45 }}>
              What did I know about rate limits on <b>Apr 18</b>?
            </div>
            <div style={{ display:'flex', gap:6, marginTop: 8 }}>
              <Tag>time-travel</Tag><Tag tone="teal">api</Tag>
            </div>
          </div>
        </div>

        <div>
          <SectionRow label="Scrub" right={<span className="mono" style={{ fontSize: 10, color: INK4 }}>apr 18, 14:02</span>} />
          <div style={{ display:'flex', flexDirection:'column', gap: 6 }}>
            {[['Today', false],['This week', false],['Q2', true],['All time', false]].map(([l, on],i)=>(
              <button key={i} style={{
                height: 26, borderRadius: 5, padding:'0 10px',
                background: on ? INK : PAPER, color: on ? PAPER : INK2,
                border: on ? 'none' : `0.5px solid ${RULE}`,
                textAlign:'left', fontSize:12, fontFamily:'inherit', cursor:'pointer',
              }}>{l}</button>
            ))}
          </div>
        </div>

        <div>
          <SectionRow label="Hits in window" right={<span className="mono" style={{ fontSize: 10, color: INK4 }}>3</span>} />
          <div style={{ display:'flex', flexDirection:'column', gap: 6 }}>
            {hits.map((h,i)=>(
              <div key={i} style={{ display:'flex', alignItems:'center', gap:6, padding:'4px 6px', background: PAPER, borderRadius: 5, border:`0.5px solid ${RULE}` }}>
                <span style={{ width:6, height:6, borderRadius:'50%', background: strata[h.stratum].tone }} />
                <span style={{ fontSize: 12, color: INK, flex:1 }}>{h.label}</span>
                <span className="mono" style={{ fontSize: 9, color: INK4 }}>{strata[h.stratum].label.split(' ')[0].toLowerCase()}</span>
              </div>
            ))}
          </div>
        </div>

        <div style={{ marginTop:'auto' }}>
          <SectionRow label="Authors in layer" />
          <div style={{ display:'flex', flexDirection:'column', gap: 5, fontSize: 11.5, color: INK2 }}>
            <div><Actor who="human" /><span style={{ marginLeft:8, color: INK3 }}>wrote 4 chunks</span></div>
            <div><Actor who="ai" /><span style={{ marginLeft:8, color: INK3 }}>added 19 annotations</span></div>
          </div>
        </div>
      </aside>

      {/* CENTER */}
      <main style={{ flex:1, background: PAPER, display:'flex', flexDirection:'column', minWidth: 0 }}>
        {/* time axis */}
        <div style={{ height: 34, borderBottom:`0.5px solid ${RULE}`, display:'flex', alignItems:'center', padding:'0 16px', background: PAPER2 }}>
          <span className="mono" style={{ fontSize: 10.5, color: INK3, letterSpacing: 0.5, flex: 1 }}>TIME · months →</span>
          <span className="mono" style={{ fontSize: 10.5, color: INK4 }}>zoom: months</span>
        </div>

        <div style={{ flex: 1, position: 'relative', minHeight: 0 }}>
          <svg viewBox={`0 0 900 ${totalH * 1.05 + 40}`} preserveAspectRatio="none"
            style={{ position:'absolute', inset:0, width:'100%', height:'100%' }}>
            {/* month gridlines */}
            {['Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr'].map((m, i) => {
              const x = (i+1) * (900/9);
              return (
                <g key={m}>
                  <line x1={x} y1="0" x2={x} y2={totalH + 30} stroke="oklch(86% 0.010 85)" strokeWidth="0.4" />
                  <text x={x+4} y="12" fontFamily='"IBM Plex Mono"' fontSize="9" fill={INK4}>{m}</text>
                </g>
              );
            })}

            {/* strata */}
            {(() => {
              let y = 20;
              return strata.map((s, idx) => {
                const top = y;
                const chunks = mkChunks(idx, s.weight);
                const band = (
                  <g key={s.id}>
                    {/* layer bg */}
                    <rect x="0" y={top} width="900" height={s.h}
                      fill={s.tone} opacity="0.06" />
                    {/* upper edge — wavy stratum line */}
                    <path d={wavyPath(top + 2, 900, 3, idx)} fill="none" stroke={s.tone} strokeOpacity="0.6" strokeWidth="1" />
                    <path d={wavyPath(top + s.h - 2, 900, 4, idx + 10)} fill="none" stroke={s.tone} strokeOpacity="0.3" strokeWidth="0.8" />
                    {/* chunk dots (outcrops) */}
                    {chunks.map((c, i) => (
                      <circle key={i} cx={c.x} cy={top + 8 + c.y * (s.h - 16)}
                        r={c.size} fill={s.tone} opacity="0.7" />
                    ))}
                    {/* hits */}
                    {hits.filter(h => h.stratum === idx).map((h, i) => (
                      <g key={i}>
                        <circle cx={h.x} cy={top + s.h/2} r="8" fill={PAPER} stroke={s.tone} strokeWidth="1.5" />
                        <circle cx={h.x} cy={top + s.h/2} r="3" fill={s.tone} />
                      </g>
                    ))}
                    {/* label */}
                    <text x="12" y={top + 16}
                      fontFamily='"IBM Plex Mono"' fontSize="10.5" fontWeight="500"
                      fill={INK2} letterSpacing="0.4">
                      {s.label.toUpperCase()}
                    </text>
                    <text x="12" y={top + 30}
                      fontFamily='"IBM Plex Mono"' fontSize="9" fill={INK4}>
                      {chunks.length} outcrops · density {Math.round(s.weight*100)}%
                    </text>
                  </g>
                );
                y += s.h + 4;
                return band;
              });
            })()}

            {/* time cursor */}
            <g>
              <line x1={cursorX} y1="0" x2={cursorX} y2={totalH + 30}
                stroke={INK} strokeWidth="1" strokeDasharray="0" />
              <rect x={cursorX - 44} y="0" width="88" height="20" fill={INK} rx="4" />
              <text x={cursorX} y="13" textAnchor="middle"
                fontFamily='"IBM Plex Mono"' fontSize="10" fontWeight="500" fill={PAPER}>apr 18 · 14:02</text>
              {/* drag handle at bottom */}
              <circle cx={cursorX} cy={totalH + 28} r="5" fill={INK} />
            </g>
          </svg>
        </div>

        {/* bottom — timeline mini */}
        <div style={{ height: 60, borderTop:`0.5px solid ${RULE}`, padding: '10px 16px', background: PAPER2, display:'flex', alignItems:'center', gap: 12 }}>
          <span className="mono" style={{ fontSize: 10, color: INK3, letterSpacing: 0.6 }}>ACTIVITY</span>
          <div style={{ flex: 1, position:'relative', height: 28, background: PAPER, borderRadius: 4, border:`0.5px solid ${RULE}` }}>
            <svg viewBox="0 0 900 28" preserveAspectRatio="none" style={{ width:'100%', height:'100%' }}>
              {/* activity heatmap rects */}
              {Array.from({length:90}).map((_,i) => {
                const v = (Math.sin(i*0.5)*0.5 + 0.5) * 0.6 + ((i*17)%5)*0.08;
                return <rect key={i} x={i*10} y={28 - v*28} width="9" height={v*28} fill={INK2} opacity={0.2 + v*0.5} />;
              })}
              <line x1={cursorX} y1="0" x2={cursorX} y2="28" stroke={INK} strokeWidth="1" />
            </svg>
          </div>
          <span className="mono" style={{ fontSize: 10.5, color: INK3 }}>◀ scrub ▶</span>
        </div>
      </main>

      {/* RIGHT — excerpt as-of timestamp */}
      <aside style={{ width: 320, borderLeft:`0.5px solid ${RULE}`, background: PAPER, padding:'14px 16px', display:'flex', flexDirection:'column', gap: 14 }}>
        <div>
          <SectionRow label="As of apr 18, 14:02" right={<Tag tone="teal">api</Tag>} />
          <div className="serif" style={{ fontSize: 16.5, fontWeight: 500, color: INK, lineHeight: 1.3 }}>
            gateway-spec.md · §3.2
          </div>
          <div className="mono" style={{ fontSize: 10, color: INK4, marginTop: 2 }}>written by you · 2 edits · last touched by Agent</div>
        </div>

        <div style={{
          background: PAPER2, border:`0.5px solid ${RULE}`, borderRadius: 8, padding: 12,
        }}>
          <div className="serif" style={{ fontSize: 13, color: INK, lineHeight: 1.55 }}>
            &ldquo;Gateway rejects above <mark style={{ background: 'oklch(92% 0.09 75)', padding:'0 2px', borderRadius: 2, color: INK }}>1,200 rpm</mark>. Redis sliding window, 60s. Returns <code className="mono">Retry-After</code>.&rdquo;
          </div>
          <div className="mono" style={{ fontSize: 10, color: INK4, marginTop: 8, letterSpacing: 0.4 }}>
            HASH 8f3a…04 · EMBED v2
          </div>
        </div>

        <div>
          <SectionRow label="Diff from today" />
          <div style={{ background: PAPER2, border:`0.5px solid ${RULE}`, borderRadius: 8, padding:'10px 12px', fontSize: 12, color: INK2, lineHeight: 1.5, fontFamily: '"IBM Plex Mono", monospace' }}>
            <div style={{ color: 'oklch(52% 0.13 25)', background:'oklch(94% 0.04 25)', padding:'1px 4px', borderRadius:2, marginBottom:2 }}>− 1,200 rpm (this snapshot)</div>
            <div style={{ color: 'oklch(40% 0.10 150)', background:'oklch(94% 0.04 150)', padding:'1px 4px', borderRadius:2 }}>+ 1,000 rpm (legal-approved)</div>
          </div>
        </div>

        <div style={{ marginTop:'auto' }}>
          <SectionRow label="Neighboring layers" />
          <div style={{ display:'flex', flexDirection:'column', gap:5 }}>
            {[
              ['Legal · Q1 approval', MAGENTA, '−28 days'],
              ['Eng. journal · retry', GREEN, '−6 days'],
              ['Reference · Lamport',  AMBER, '−104 days'],
            ].map(([label, tone, delta], i) => (
              <div key={i} style={{ display:'flex', alignItems:'center', gap:8, padding:'6px 8px', background: PAPER2, border:`0.5px solid ${RULE}`, borderRadius: 5 }}>
                <span style={{ width: 8, height:8, borderRadius:'50%', background: tone }} />
                <span style={{ flex:1, fontSize: 12, color: INK }}>{label}</span>
                <span className="mono" style={{ fontSize: 10, color: INK4 }}>{delta}</span>
              </div>
            ))}
          </div>
        </div>
      </aside>
    </Mac>
  );
};

function wavyPath(y, w, amp, seed) {
  const steps = 12;
  let d = `M 0 ${y}`;
  for (let i = 1; i <= steps; i++) {
    const x = (i / steps) * w;
    const wave = Math.sin((i + seed) * 1.3) * amp;
    d += ` L ${x.toFixed(1)} ${(y + wave).toFixed(1)}`;
  }
  return d;
}

window.C4Stratigraphy = C4Stratigraphy;
