// app.jsx — Design canvas: Concept 1 across 5 color palettes.

function App() {
  const themes = [
    ['graphite', C1_THEMES.graphite],
    ['midnight', C1_THEMES.midnight],
    ['slate',    C1_THEMES.slate],
    ['plum',     C1_THEMES.plum],
    ['terminal', C1_THEMES.terminal],
  ];

  return (
    <DesignCanvas>
      <DCSection
        id="intro"
        title="Concept 1 · Dual cursor — palette studies"
        subtitle="Same layout, five color worlds. No paper/cream. Each palette picks a surface tone, an ink scale, and four shared-chroma accents that double as actor (you / claude) and tag colors."
      >
        <DCArtboard id="legend" label="Palettes overview" width={620} height={560}>
          <Overview themes={themes} />
        </DCArtboard>
      </DCSection>

      {themes.map(([key, T], i) => (
        <DCSection
          key={key}
          id={`c1-${key}`}
          title={`0${i+1} · ${T.name}`}
          subtitle={T.desc}
        >
          <DCArtboard id={`c1-${key}-a`} label={`${T.name}`} width={1280} height={800}>
            <C1DualCursor theme={T} label={T.name.toLowerCase()} />
          </DCArtboard>
        </DCSection>
      ))}
    </DesignCanvas>
  );
}

function Overview({ themes }) {
  return (
    <div style={{ height: '100%', padding: 28, background: '#f4f2ed', display: 'flex', flexDirection: 'column', gap: 18, fontFamily: '"IBM Plex Sans", system-ui' }}>
      <div>
        <div className="mono" style={{ fontSize: 10, letterSpacing: 1.2, color: '#777', textTransform: 'uppercase' }}>imprint / concept 01 / palette studies</div>
        <div className="serif" style={{ fontSize: 28, fontWeight: 500, marginTop: 6, color: '#1a1a1a', lineHeight: 1.15 }}>Five color worlds, same shape.</div>
        <div style={{ fontSize: 13, color: '#555', marginTop: 6, lineHeight: 1.5, maxWidth: 520 }}>
          Warm paper is gone. Each palette picks a surface tone (cool gray, night, slate, plum, phosphor) and locks the four region accents to shared chroma so nothing fights.
        </div>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
        {themes.map(([key, T], i) => (
          <div key={key} style={{
            background: T.surface1,
            borderRadius: 10, padding: 14, color: T.ink,
            boxShadow: T.windowShadow,
          }}>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 6 }}>
              <div className="mono" style={{ fontSize: 10, color: T.ink4, letterSpacing: 0.6 }}>0{i+1}</div>
              <div className="serif" style={{ fontSize: 18, fontWeight: 500 }}>{T.name}</div>
            </div>
            <div style={{ fontSize: 11.5, color: T.ink3, lineHeight: 1.45, marginBottom: 10 }}>{T.desc}</div>
            <div style={{ display: 'flex', gap: 4 }}>
              {[T.surface1, T.surface2, T.surface3, T.ink, T.human, T.ai, T.accentC, T.accentD].map((c, j) => (
                <div key={j} style={{ flex: 1, height: 26, borderRadius: 4, background: c, border: `0.5px solid ${T.rule}` }} />
              ))}
            </div>
            <div style={{ display: 'flex', gap: 10, marginTop: 10, fontSize: 10.5 }}>
              <span className="mono" style={{ display: 'inline-flex', alignItems: 'center', gap: 5, color: T.ink2 }}>
                <span style={{ width: 8, height: 8, borderRadius: '50%', background: T.human }} /> you
              </span>
              <span className="mono" style={{ display: 'inline-flex', alignItems: 'center', gap: 5, color: T.ink2 }}>
                <span style={{ width: 8, height: 8, borderRadius: '50%', background: T.ai }} /> claude
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
