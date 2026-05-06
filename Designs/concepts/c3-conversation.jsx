// c3-conversation.jsx — Concept 3: Conversation-first.
// Chat on the left is primary. Right side is a reactive mini-graph that
// animates to whatever the AI just cited, plus an inline citations panel.

const C3Conversation = () => {
  return (
    <Mac title="imprint" subtitle="— conversation · 03"
      chromeRight={<>
        <Tag tone="green">local</Tag>
        <span className="mono" style={{ fontSize: 10.5, color: INK3 }}>session · tuesday</span>
      </>}>
      {/* LEFT rail (compact) */}
      <aside style={{ width: 56, background: PAPER2, borderRight: `0.5px solid ${RULE}`, padding: '10px 0', display:'flex', flexDirection:'column', alignItems:'center', gap: 10 }}>
        {[['chat',true],['search',false],['layers',false],['cpu',false]].map(([n,on],i)=>(
          <div key={i} style={{ width: 36, height: 36, borderRadius: 8, background: on? 'rgba(40,30,20,0.08)':'transparent', display:'grid', placeItems:'center' }}>
            <Icon name={n==='chat'?'sparkle':n} size={16} color={on?INK:INK3} />
          </div>
        ))}
        <div style={{ flex:1 }} />
        <div style={{ width: 28, height: 28, borderRadius: '50%', background: AMBER, color: PAPER, fontWeight:600, fontSize:11, display:'grid', placeItems:'center' }}>MM</div>
      </aside>

      {/* CENTER — chat */}
      <main style={{ flex: 1, display:'flex', flexDirection:'column', background: PAPER, minWidth: 0 }}>
        {/* Session header */}
        <div style={{ padding:'14px 22px', borderBottom:`0.5px solid ${RULE}`, display:'flex', alignItems:'center', gap:12 }}>
          <div style={{ flex: 1 }}>
            <div className="serif" style={{ fontSize: 17, fontWeight: 500, color: INK }}>
              Rate limiting across the gateway
            </div>
            <div className="mono" style={{ fontSize: 10.5, color: INK4, marginTop:2 }}>
              session · 14 turns · 23 citations · 4 regions touched
            </div>
          </div>
          <button style={btn3}>Export thread</button>
          <button style={btn3Primary}>+ New thread</button>
        </div>

        {/* Chat transcript */}
        <div style={{ flex: 1, overflow:'hidden', padding: '18px 22px', display:'flex', flexDirection:'column', gap: 18 }}>
          {/* user turn */}
          <ChatTurn who="human">
            <p style={{ margin:0 }}>How is rate limiting implemented across the gateway, and does that match what legal approved last quarter?</p>
          </ChatTurn>

          {/* ai turn — with inline citations */}
          <ChatTurn who="ai">
            <p style={{ margin: '0 0 8px' }}>
              Rate limits use a per-tenant sliding window in Redis, rejecting above 1,200 rpm with HTTP 429 <Cite n="1" />. The Retry-After header is set from the remaining-window size <Cite n="2" />.
            </p>
            <p style={{ margin:'0 0 10px' }}>
              Legal's Q1 approval specifies 1,000 rpm as the maximum exposed to external tenants <Cite n="3" />&nbsp;— the current gateway value is 1,200.{' '}
              <b>There's a 200-rpm gap.</b>
            </p>
            <div style={{
              display:'inline-flex', alignItems:'center', gap:8,
              padding:'6px 10px', background:'oklch(92% 0.05 75)',
              borderRadius: 6, fontSize: 11.5, color: AMBER_INK,
            }}>
              <Icon name="sparkle" size={12} color={AMBER_INK} />
              <span className="mono" style={{ letterSpacing:0.4 }}>MISMATCH · opened a review note in Legal region</span>
            </div>
          </ChatTurn>

          {/* user follow-up */}
          <ChatTurn who="human">
            <p style={{ margin:0 }}>Show me where the 1,200 is set in code.</p>
          </ChatTurn>

          {/* ai turn — routed */}
          <ChatTurn who="ai">
            <div className="mono" style={{ fontSize: 10, color: INK4, marginBottom: 6, letterSpacing: 0.5 }}>ROUTED · API architecture → gateway-spec.md, rate-limit.rs</div>
            <div style={{
              background: PAPER2, border:`0.5px solid ${RULE}`, borderRadius: 8,
              padding: 12, fontFamily: '"IBM Plex Mono", monospace', fontSize: 12, color: INK2, lineHeight: 1.55,
            }}>
              <div style={{ color: INK4 }}>// src/gateway/rate-limit.rs · line 47</div>
              <div>pub const <span style={{ color: MAGENTA }}>MAX_RPM</span>: u32 = <mark style={{ background: 'oklch(92% 0.09 75)', color: INK, padding:'0 3px', borderRadius: 2 }}>1_200</mark>;</div>
            </div>
          </ChatTurn>
        </div>

        {/* Composer */}
        <div style={{ padding: '12px 22px 18px', borderTop: `0.5px solid ${RULE}`, background: PAPER }}>
          <div style={{
            background: PAPER2, border: `0.5px solid ${RULE}`, borderRadius: 10,
            padding: '10px 12px',
          }}>
            <div style={{ display:'flex', gap:6, marginBottom: 8, flexWrap:'wrap' }}>
              <Tag tone="teal">API</Tag><Tag tone="magenta">Legal</Tag>
              <span className="mono" style={{ fontSize: 10, color: INK4, alignSelf:'center' }}>scope · 2 regions</span>
              <span style={{ flex:1 }} />
              <span className="mono" style={{ fontSize: 10, color: INK4, alignSelf:'center' }}>⌘K · change scope</span>
            </div>
            <div style={{ fontSize: 13.5, color: INK, lineHeight: 1.5 }}>
              Draft a mitigation plan to close the 200-rpm gap — suggest a two-week rollout…
            </div>
            <div style={{ display:'flex', alignItems:'center', gap: 10, marginTop: 10 }}>
              <button style={chipBtn}><Icon name="plus" size={12} color={INK3} /> Attach doc</button>
              <button style={chipBtn}><Icon name="mic" size={12} color={INK3} /> Voice</button>
              <span style={{ flex:1 }} />
              <span className="mono" style={{ fontSize: 10, color: INK4 }}>Agent · local</span>
              <button style={btn3Primary}>Send ↵</button>
            </div>
          </div>
        </div>
      </main>

      {/* RIGHT — reactive graph + citations */}
      <aside style={{ width: 380, borderLeft:`0.5px solid ${RULE}`, background: PAPER, display:'flex', flexDirection:'column' }}>
        <div style={{ padding:'12px 16px', borderBottom:`0.5px solid ${RULE}` }}>
          <SectionRow label="What Agent is reading" right={<span className="mono" style={{ fontSize:10, color: INK4 }}>live</span>} />
        </div>
        {/* mini graph */}
        <div style={{ height: 240, position:'relative', background: `radial-gradient(400px 200px at 50% 40%, oklch(97% 0.008 85), ${PAPER})`, borderBottom:`0.5px solid ${RULE}` }}>
          <svg viewBox="0 0 380 240" style={{ position:'absolute', inset:0, width:'100%', height:'100%' }}>
            <defs>
              <radialGradient id="c3-focus" cx="50%" cy="50%"><stop offset="0%" stopColor={TEAL} stopOpacity="0.35" /><stop offset="100%" stopColor={TEAL} stopOpacity="0" /></radialGradient>
            </defs>
            {/* edges */}
            <line x1="110" y1="160" x2="200" y2="110" stroke={INK3} strokeOpacity="0.3" />
            <line x1="200" y1="110" x2="290" y2="80" stroke={INK3} strokeOpacity="0.3" />
            <line x1="200" y1="110" x2="270" y2="180" stroke={INK3} strokeOpacity="0.3" />
            <line x1="110" y1="160" x2="270" y2="180" stroke={INK3} strokeOpacity="0.15" strokeDasharray="3 3" />
            {/* regions */}
            <circle cx="200" cy="110" r="60" fill="url(#c3-focus)" />
            {[
              { x: 110, y: 160, t: MAGENTA, s: 10, l: 'Legal' },
              { x: 200, y: 110, t: TEAL,    s: 14, l: 'API' },
              { x: 290, y: 80,  t: GREEN,   s: 9,  l: 'Eng. journal' },
              { x: 270, y: 180, t: AMBER,   s: 8,  l: 'Reference' },
            ].map((n, i) => (
              <g key={i}>
                <circle cx={n.x} cy={n.y} r={n.s+6} fill={n.t} opacity="0.2" />
                <circle cx={n.x} cy={n.y} r={n.s} fill={n.t} />
                <text x={n.x} y={n.y + n.s + 12} textAnchor="middle" fontFamily='"IBM Plex Mono"' fontSize="9.5" fill={INK2}>{n.l}</text>
              </g>
            ))}
            {/* AI cursor pulse on API */}
            <circle cx="200" cy="110" r="22" fill="none" stroke={TEAL} strokeWidth="1.2" opacity="0.8" />
            <circle cx="200" cy="110" r="30" fill="none" stroke={TEAL} strokeWidth="0.8" opacity="0.4" />
          </svg>
        </div>

        {/* Citations panel */}
        <div style={{ flex:1, overflow: 'hidden', padding: '12px 16px' }}>
          <SectionRow label="Citations in this turn" right={<span className="mono" style={{ fontSize: 10, color: INK4 }}>3</span>} />
          <div style={{ display:'flex', flexDirection:'column', gap: 8 }}>
            <Citation n="1" region="API" tone="teal"    title="gateway-spec.md · §3.2" snippet="Rate limits are applied per-tenant using a sliding window counter in Redis…" />
            <Citation n="2" region="API" tone="teal"    title="rate-limit.rs · line 47" snippet="MAX_RPM: u32 = 1_200; // returned via Retry-After on 429" />
            <Citation n="3" region="Legal" tone="magenta" title="approvals-q1.pdf · p.8" snippet="External tenant ceiling shall not exceed 1,000 requests per minute." />
          </div>

          <div style={{ marginTop: 14, padding: 10, borderRadius: 8, background: 'oklch(94% 0.03 75)', border:`0.5px solid oklch(82% 0.05 75)` }}>
            <div className="mono" style={{ fontSize: 10, color: AMBER_INK, letterSpacing:0.6 }}>FLAG · MEMORY WRITE</div>
            <div style={{ fontSize: 12.5, color: INK, marginTop: 4, lineHeight: 1.45 }}>
              Save this mismatch as a note linked to both regions?
            </div>
            <div style={{ display:'flex', gap: 6, marginTop: 8 }}>
              <button style={btn3}>Dismiss</button>
              <button style={{ ...btn3Primary, flex: 1 }}>Save to memory</button>
            </div>
          </div>
        </div>
      </aside>
    </Mac>
  );
};

function ChatTurn({ who, children }) {
  const isAi = who === 'ai';
  return (
    <div style={{ display:'flex', gap: 12 }}>
      <div style={{ width: 28, flexShrink: 0 }}>
        {isAi
          ? <div style={{ width:26, height:26, borderRadius:'50%', background: 'oklch(92% 0.04 200)', border:`1px solid ${TEAL}`, display:'grid', placeItems:'center' }}><Icon name="sparkle" size={14} color={TEAL} /></div>
          : <div style={{ width:26, height:26, borderRadius:'50%', background: AMBER, color: PAPER, fontWeight:600, fontSize:11, display:'grid', placeItems:'center' }}>MM</div>}
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div className="mono" style={{ fontSize: 10, color: INK4, letterSpacing: 0.5, marginBottom: 4 }}>
          {isAi ? 'CLAUDE · 14:02' : 'YOU · 14:02'}
        </div>
        <div className="serif" style={{ fontSize: 14, lineHeight: 1.6, color: INK }}>{children}</div>
      </div>
    </div>
  );
}

function Cite({ n }) {
  return (
    <sup className="mono" style={{
      display:'inline-block', fontSize: 9.5, fontWeight: 600, lineHeight: 1,
      padding:'2px 5px', margin:'0 2px', borderRadius: 3,
      background: 'oklch(92% 0.04 200)', color: 'oklch(38% 0.08 210)',
      verticalAlign: 'super', cursor:'pointer',
    }}>[{n}]</sup>
  );
}

function Citation({ n, title, snippet, region, tone }) {
  return (
    <div style={{
      border: `0.5px solid ${RULE}`, borderRadius: 7, padding: '8px 10px',
      background: PAPER2,
    }}>
      <div style={{ display:'flex', alignItems:'center', gap:6, marginBottom: 4 }}>
        <Cite n={n} />
        <Tag tone={tone}>{region}</Tag>
        <span style={{ flex:1 }} />
        <span className="mono" style={{ fontSize: 10, color: INK4 }}>{title}</span>
      </div>
      <div className="serif" style={{ fontSize: 12, color: INK2, lineHeight: 1.45 }}>&ldquo;{snippet}&rdquo;</div>
    </div>
  );
}

const btn3 = { height: 26, padding:'0 10px', borderRadius:5, background: PAPER, color: INK2, border:`0.5px solid ${RULE}`, fontSize: 11.5, cursor:'pointer', fontFamily:'inherit' };
const btn3Primary = { ...btn3, background: INK, color: PAPER, border:'none' };
const chipBtn = { height: 24, padding:'0 8px', borderRadius:5, background: PAPER, color: INK2, border:`0.5px solid ${RULE}`, fontSize: 11, cursor:'pointer', fontFamily:'inherit', display:'inline-flex', alignItems:'center', gap:4 };

window.C3Conversation = C3Conversation;
