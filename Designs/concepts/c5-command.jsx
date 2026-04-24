// c5-command.jsx — Concept 5: Command deck.
// Dense, inspector-first. Four panes always visible. Cloud is a small context
// map in the upper-right. Designed for power users who live in this app.

const C5Command = () => {
  return (
    <Mac title="imprint" subtitle="— command · 05"
      chromeRight={<>
        <span className="mono" style={{ fontSize: 10, color: INK4 }}>⌘1-4 · focus pane</span>
        <span style={{ width: 1, height: 12, background: RULE }} />
        <Tag tone="green">● 11ms</Tag>
      </>}>
      <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '260px 1fr 1fr', gridTemplateRows: '44px 1fr 1fr', minWidth: 0, minHeight: 0 }}>
        {/* Top strip — full width query bar */}
        <div style={{ gridColumn: '1 / -1', borderBottom: `0.5px solid ${RULE}`, background: PAPER2, display: 'flex', alignItems: 'center', gap: 10, padding: '0 14px' }}>
          <span className="mono" style={{ fontSize: 10, color: INK3, letterSpacing: 0.6 }}>QUERY</span>
          <div style={{ flex: 1, height: 26, borderRadius: 5, border: `0.5px solid ${RULE}`, background: PAPER, display: 'flex', alignItems: 'center', padding: '0 10px', gap: 8 }}>
            <Icon name="search" size={12} color={INK3} />
            <span className="mono" style={{ fontSize: 12, color: INK }}>region:api AND (retry OR rate-limit) since:&ldquo;apr 1&rdquo;</span>
            <span style={{ flex: 1 }} />
            <Tag tone="teal">api</Tag><Tag tone="magenta">legal</Tag>
          </div>
          <button style={cBtn}>route</button>
          <button style={cBtn}>grep</button>
          <button style={cBtnPrimary}>semantic</button>
        </div>

        {/* LEFT rail — sources + regions, spans both rows */}
        <aside style={{ gridRow: '2 / 4', borderRight: `0.5px solid ${RULE}`, background: PAPER2, padding: '12px 10px', display: 'flex', flexDirection: 'column', gap: 12, minHeight: 0 }}>
          <div>
            <SectionRow label="Store" right={<Icon name="plus" size={11} color={INK4} />} />
            <div style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              {[['Library', '418'], ['Inbox', '12'], ['Tombstones', '3']].map(([n, v]) => (
                <div key={n} style={navRow(n === 'Library')}>
                  <span style={{ flex: 1 }}>{n}</span>
                  <span className="mono" style={{ fontSize: 10, color: INK4 }}>{v}</span>
                </div>
              ))}
            </div>
          </div>
          <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
            <SectionRow label="Regions" right={<span className="mono" style={{ fontSize: 10, color: INK4 }}>34</span>} />
            <div style={{ overflow: 'hidden', display: 'flex', flexDirection: 'column', gap: 2 }}>
              {[
                ['API architecture', TEAL, 312, true],
                ['Engineering journal', GREEN, 210, false],
                ['Onboarding research', AMBER, 148, false],
                ['Reference papers', AMBER, 64, false],
                ['Legal & privacy', MAGENTA, 88, false],
                ['Marketing notes', AMBER, 42, false],
                ['Incident reviews', MAGENTA, 29, false],
                ['Weekly syncs', GREEN, 51, false],
              ].map(([n, c, k, active]) => (
                <div key={n} style={navRow(active)}>
                  <span style={{ width: 8, height: 8, borderRadius: 2, background: c, flexShrink: 0 }} />
                  <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{n}</span>
                  <span className="mono" style={{ fontSize: 10, color: INK4 }}>{k}</span>
                </div>
              ))}
            </div>
          </div>
          <div>
            <SectionRow label="Presence" />
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 11, color: INK3 }}>
              <div><Actor who="human" /> <span style={{ color: INK4 }}>· routing</span></div>
              <div><Actor who="ai" /> <span style={{ color: INK4 }}>· reading r-api</span></div>
            </div>
          </div>
        </aside>

        {/* TOP LEFT pane — Routed regions + map */}
        <section style={paneStyle}>
          <div style={paneHeader}>
            <span className="mono" style={paneLabel}>⌘1 · ROUTED</span>
            <span style={{ flex: 1 }} />
            <span className="mono" style={{ fontSize: 10, color: INK4 }}>3 regions · rationale ↓</span>
          </div>
          <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '1fr 180px', gap: 12, padding: 12, minHeight: 0 }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6, overflow: 'hidden' }}>
              {[
                ['r-api', 'API architecture', 0.91, TEAL],
                ['r-leg', 'Legal & privacy', 0.78, MAGENTA],
                ['r-eng', 'Engineering journal', 0.64, GREEN],
              ].map(([id, name, score, c]) => (
                <div key={id} style={{
                  display: 'flex', alignItems: 'center', gap: 8, padding: '7px 10px',
                  background: id === 'r-api' ? 'oklch(94% 0.04 200)' : PAPER2,
                  border: `0.5px solid ${id === 'r-api' ? TEAL : RULE}`,
                  borderRadius: 5,
                }}>
                  <span style={{ width: 10, height: 10, borderRadius: 2, background: c }} />
                  <span className="mono" style={{ fontSize: 10, color: INK4, width: 38 }}>{id}</span>
                  <span style={{ flex: 1, fontSize: 12.5, color: INK }}>{name}</span>
                  <div style={{ width: 60, height: 4, background: PAPER3, borderRadius: 2, overflow: 'hidden' }}>
                    <div style={{ width: `${score*100}%`, height: '100%', background: c }} />
                  </div>
                  <span className="mono" style={{ fontSize: 10.5, color: INK2, width: 32, textAlign: 'right' }}>{score.toFixed(2)}</span>
                </div>
              ))}
              <div style={{
                marginTop: 4, padding: '8px 10px', background: PAPER2,
                border: `0.5px solid ${RULE}`, borderRadius: 5,
                fontSize: 11.5, color: INK2, lineHeight: 1.5,
              }}>
                <span className="mono" style={{ fontSize: 10, color: INK4, letterSpacing: 0.5 }}>RATIONALE </span>
                Matched <i>rate-limit</i> tokens and <i>retry</i> entities in <b>r-api</b> (strong); cited as neighbor-of in <b>r-leg</b> via <i>approval</i> link.
              </div>
            </div>
            {/* Mini cloud */}
            <div style={{ position: 'relative', background: INK, borderRadius: 6, overflow: 'hidden' }}>
              <svg viewBox="0 0 180 170" style={{ width: '100%', height: '100%' }}>
                <line x1="60" y1="80" x2="110" y2="60" stroke="#fff" strokeOpacity="0.3" />
                <line x1="110" y1="60" x2="140" y2="110" stroke="#fff" strokeOpacity="0.3" />
                <line x1="60" y1="80" x2="140" y2="110" stroke="#fff" strokeOpacity="0.15" />
                <circle cx="110" cy="60" r="16" fill={TEAL} opacity="0.3" />
                <circle cx="110" cy="60" r="8" fill={TEAL} />
                <circle cx="60" cy="80" r="6" fill={MAGENTA} />
                <circle cx="140" cy="110" r="6" fill={GREEN} />
                <circle cx="40" cy="130" r="4" fill={AMBER} opacity="0.7" />
                <circle cx="150" cy="40" r="3" fill="#fff" opacity="0.4" />
                {Array.from({length: 22}).map((_,i) => (
                  <circle key={i} cx={20 + (i*13)%160} cy={20 + ((i*29)%130)} r="1" fill="#fff" opacity="0.35" />
                ))}
              </svg>
              <div style={{ position: 'absolute', bottom: 6, left: 8, fontFamily: '"IBM Plex Mono", monospace', fontSize: 9, color: '#fff', opacity: 0.6 }}>context map</div>
            </div>
          </div>
        </section>

        {/* TOP RIGHT pane — Passage hits */}
        <section style={{ ...paneStyle, borderLeft: `0.5px solid ${RULE}` }}>
          <div style={paneHeader}>
            <span className="mono" style={paneLabel}>⌘2 · PASSAGES</span>
            <span style={{ flex: 1 }} />
            <span className="mono" style={{ fontSize: 10, color: INK4 }}>8 hits · sorted by score</span>
          </div>
          <div style={{ flex: 1, overflow: 'hidden', padding: 12, display: 'flex', flexDirection: 'column', gap: 6 }}>
            {[
              ['0.94', 'rate-limit.rs:47',     TEAL,    'pub const MAX_RPM: u32 = 1_200;'],
              ['0.89', 'gateway-spec.md §3.2', TEAL,    'Rate limits per-tenant via sliding window in Redis…'],
              ['0.81', 'approvals-q1.pdf p.8', MAGENTA, 'Tenant ceiling shall not exceed 1,000 rpm.'],
              ['0.72', 'retry-notes.md',       GREEN,   'Budgeted retries decay exponentially over 30s.'],
              ['0.65', 'journal-2024-03-04',   GREEN,   'Chose sliding-window because Lamport\'s token-bucket…'],
            ].map(([s, src, c, txt], i) => (
              <div key={i} style={{
                display: 'grid', gridTemplateColumns: '36px 140px 1fr', gap: 10,
                padding: '7px 10px', borderRadius: 5,
                background: i === 0 ? 'oklch(95% 0.04 75)' : PAPER2,
                border: `0.5px solid ${i === 0 ? 'oklch(82% 0.08 75)' : RULE}`,
              }}>
                <span className="mono" style={{ fontSize: 11, color: INK, fontWeight: 600 }}>{s}</span>
                <span className="mono" style={{ fontSize: 10.5, color: INK3, display: 'flex', alignItems: 'center', gap: 5 }}>
                  <span style={{ width: 6, height: 6, borderRadius: '50%', background: c }} />
                  {src}
                </span>
                <span className="serif" style={{ fontSize: 12.5, color: INK, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{txt}</span>
              </div>
            ))}
          </div>
        </section>

        {/* BOTTOM LEFT pane — Navigation trail */}
        <section style={{ ...paneStyle, borderTop: `0.5px solid ${RULE}` }}>
          <div style={paneHeader}>
            <span className="mono" style={paneLabel}>⌘3 · TRAIL</span>
            <span style={{ flex: 1 }} />
            <button style={cBtnSmall}><Icon name="back" size={10} color={INK2} /> backtrack</button>
          </div>
          <div style={{ flex: 1, overflow: 'hidden', padding: '12px 14px', display: 'flex', flexDirection: 'column', gap: 0 }}>
            {[
              ['14:00', 'human', 'Ask: rate limits across gateway', null],
              ['14:00', 'system', 'Routed → r-api (0.91), r-leg (0.78)', null],
              ['14:01', 'ai', 'Read gateway-spec.md §3.2', 'semantic'],
              ['14:01', 'ai', 'Jumped to rate-limit.rs:47', 'same-doc'],
              ['14:02', 'ai', 'Followed citation → approvals-q1.pdf', 'citation'],
              ['14:02', 'ai', 'Flagged mismatch · MAX_RPM vs ceiling', 'entity'],
              ['14:03', 'human', 'Pinned mismatch as memory note', null],
            ].map(([time, who, text, kind], i, arr) => (
              <div key={i} style={{ display: 'flex', gap: 10, paddingBottom: i === arr.length - 1 ? 0 : 6 }}>
                <span className="mono" style={{ fontSize: 10, color: INK4, width: 40, paddingTop: 2 }}>{time}</span>
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', paddingTop: 3 }}>
                  <span style={{ width: 7, height: 7, borderRadius: '50%',
                    background: who === 'human' ? AMBER : who === 'ai' ? TEAL : INK4,
                  }} />
                  {i !== arr.length - 1 && <div style={{ width: 1, flex: 1, minHeight: 10, background: RULE, marginTop: 1 }} />}
                </div>
                <div style={{ flex: 1, fontSize: 12, color: INK, display: 'flex', gap: 8, alignItems: 'center' }}>
                  <span>{text}</span>
                  {kind && <Tag tone={kind === 'citation' ? 'magenta' : kind === 'entity' ? 'green' : kind === 'same-doc' ? 'ink' : 'teal'}>{kind}</Tag>}
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* BOTTOM RIGHT pane — Excerpt + links */}
        <section style={{ ...paneStyle, borderTop: `0.5px solid ${RULE}`, borderLeft: `0.5px solid ${RULE}` }}>
          <div style={paneHeader}>
            <span className="mono" style={paneLabel}>⌘4 · INSPECTOR</span>
            <span style={{ flex: 1 }} />
            <span className="mono" style={{ fontSize: 10, color: INK4 }}>rate-limit.rs · ch-042</span>
          </div>
          <div style={{ flex: 1, overflow: 'hidden', padding: '12px 14px', display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div style={{
              background: INK, color: PAPER, borderRadius: 6, padding: '10px 12px',
              fontFamily: '"IBM Plex Mono", monospace', fontSize: 11.5, lineHeight: 1.7,
            }}>
              <div style={{ color: 'oklch(70% 0.008 85)' }}>// src/gateway/rate-limit.rs · line 45</div>
              <div><span style={{ color: MAGENTA }}>/// </span><span style={{ color: 'oklch(75% 0.008 85)' }}>Per-tenant ceiling; must stay ≤ approvals-q1.pdf §legal.</span></div>
              <div>pub const <span style={{ color: TEAL }}>MAX_RPM</span>: u32 = <span style={{ color: AMBER, background: 'oklch(35% 0.12 60)', padding: '0 4px', borderRadius: 2 }}>1_200</span>;</div>
            </div>
            <div>
              <SectionRow label="Links out" />
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                {[
                  ['semantic', 'retry-budget.md', 0.82, TEAL],
                  ['same-doc', 'rate-limit.rs · tests', 0.77, INK2],
                  ['citation', 'Lamport 1978', 0.64, MAGENTA],
                  ['entity',   'sliding-window (redis)', 0.58, GREEN],
                ].map(([k, n, s, c], i) => (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '5px 8px', background: PAPER2, border: `0.5px solid ${RULE}`, borderRadius: 4 }}>
                    <Tag tone={k === 'citation' ? 'magenta' : k === 'entity' ? 'green' : k === 'same-doc' ? 'ink' : 'teal'} style={{ minWidth: 54, justifyContent: 'center' }}>{k}</Tag>
                    <span style={{ flex: 1, fontSize: 12, color: INK }}>{n}</span>
                    <span className="mono" style={{ fontSize: 10.5, color: INK3 }}>{s.toFixed(2)}</span>
                    <Icon name="chev" size={11} color={INK4} />
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>
      </div>
    </Mac>
  );
};

const paneStyle = { display: 'flex', flexDirection: 'column', minWidth: 0, minHeight: 0, background: PAPER };
const paneHeader = { height: 28, flexShrink: 0, display: 'flex', alignItems: 'center', gap: 8, padding: '0 12px', borderBottom: `0.5px solid ${RULE}`, background: PAPER2 };
const paneLabel = { fontSize: 10, color: INK3, letterSpacing: 0.8, fontWeight: 500 };
const cBtn = { height: 22, padding: '0 8px', borderRadius: 4, background: PAPER, color: INK2, border: `0.5px solid ${RULE}`, fontSize: 11, cursor: 'pointer', fontFamily: 'inherit' };
const cBtnPrimary = { ...cBtn, background: INK, color: PAPER, border: 'none' };
const cBtnSmall = { ...cBtn, height: 18, fontSize: 10, display: 'inline-flex', alignItems: 'center', gap: 3 };
const navRow = (active) => ({
  display: 'flex', alignItems: 'center', gap: 7,
  padding: '5px 7px', borderRadius: 4,
  background: active ? 'rgba(40,30,20,0.07)' : 'transparent',
  color: active ? INK : INK2,
  fontSize: 12,
  fontWeight: active ? 500 : 400,
});

window.C5Command = C5Command;
