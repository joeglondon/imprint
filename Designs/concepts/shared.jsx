// shared.jsx — shared primitives across concepts

const INK = 'oklch(18% 0.008 80)';
const INK2 = 'oklch(30% 0.008 80)';
const INK3 = 'oklch(46% 0.008 80)';
const INK4 = 'oklch(62% 0.008 80)';
const PAPER = 'oklch(96% 0.006 85)';
const PAPER2 = 'oklch(92% 0.008 85)';
const PAPER3 = 'oklch(88% 0.010 85)';
const RULE = 'oklch(82% 0.010 85)';
const AMBER = 'oklch(72% 0.13 70)';
const AMBER_INK = 'oklch(42% 0.12 60)';
const TEAL = 'oklch(68% 0.10 200)';
const MAGENTA = 'oklch(62% 0.14 340)';
const GREEN = 'oklch(66% 0.12 150)';

// Tiny mac-window scaffold local to this project so all 5 concepts share the
// same chrome feel. Uses a warm paper titlebar to match the overall design
// system, not the glassy Tahoe default.
function Mac({ title, subtitle, children, style = {}, chromeRight = null }) {
  return (
    <div style={{
      position: 'relative', width: '100%', height: '100%',
      borderRadius: 14, overflow: 'hidden',
      background: PAPER,
      boxShadow: '0 20px 60px rgba(40,30,20,0.18), 0 2px 8px rgba(40,30,20,0.08), 0 0 0 0.5px rgba(40,30,20,0.12)',
      display: 'flex', flexDirection: 'column',
      fontFamily: '"IBM Plex Sans", -apple-system, system-ui, sans-serif',
      color: INK,
      ...style,
    }}>
      <div style={{
        height: 38, flexShrink: 0,
        display: 'flex', alignItems: 'center', gap: 12,
        padding: '0 14px',
        borderBottom: `0.5px solid ${RULE}`,
        background: PAPER2,
      }}>
        <div style={{ display: 'flex', gap: 8 }}>
          <Dot c="#ED6A5E" /><Dot c="#F5BF4F" /><Dot c="#62C554" />
        </div>
        <div style={{ flex: 1, textAlign: 'center', fontSize: 12, color: INK2, letterSpacing: 0.2 }}>
          <span style={{ fontWeight: 500 }}>{title}</span>
          {subtitle && <span style={{ color: INK4, marginLeft: 10 }}>{subtitle}</span>}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>{chromeRight}</div>
      </div>
      <div style={{ flex: 1, minHeight: 0, display: 'flex' }}>{children}</div>
    </div>
  );
}

function Dot({ c }) {
  return <div style={{ width: 12, height: 12, borderRadius: '50%', background: c, boxShadow: 'inset 0 0 0 0.5px rgba(0,0,0,0.15)' }} />;
}

// Mono label stuck into a tinted pill — the workhorse tag style across all concepts.
function Tag({ children, tone = 'ink', style = {} }) {
  const tones = {
    ink:     { bg: 'rgba(40,30,20,0.06)', fg: INK2 },
    amber:   { bg: 'oklch(92% 0.05 75)',  fg: AMBER_INK },
    teal:    { bg: 'oklch(92% 0.04 200)', fg: 'oklch(40% 0.08 210)' },
    magenta: { bg: 'oklch(92% 0.04 340)', fg: 'oklch(40% 0.10 340)' },
    green:   { bg: 'oklch(92% 0.04 150)', fg: 'oklch(38% 0.08 150)' },
    human:   { bg: 'oklch(92% 0.05 75)',  fg: AMBER_INK },
    ai:      { bg: 'oklch(92% 0.04 200)', fg: 'oklch(40% 0.08 210)' },
  };
  const t = tones[tone] || tones.ink;
  return (
    <span className="mono" style={{
      display: 'inline-flex', alignItems: 'center', gap: 4,
      padding: '2px 6px', borderRadius: 4,
      background: t.bg, color: t.fg,
      fontSize: 10, fontWeight: 500, letterSpacing: 0.4, textTransform: 'uppercase',
      ...style,
    }}>
      {children}
    </span>
  );
}

// A pair of 7px dots that marks actor (human=amber, ai=teal).
function Actor({ who = 'human', label, size = 8 }) {
  const color = who === 'human' ? AMBER : TEAL;
  return (
    <span className="mono" style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 10.5, color: INK2, letterSpacing: 0.3 }}>
      <span style={{
        width: size, height: size, borderRadius: '50%',
        background: color,
        boxShadow: `0 0 0 2px oklch(96% 0.006 85), 0 0 0 3px ${color === AMBER ? 'oklch(92% 0.05 75)' : 'oklch(92% 0.04 200)'}`,
      }} />
      {label || (who === 'human' ? 'you' : 'claude')}
    </span>
  );
}

// Placeholder block for imagery
function Placeholder({ label, h = 120, style = {} }) {
  return (
    <div className="stripe-bg" style={{
      height: h, borderRadius: 6,
      background: PAPER2,
      border: `0.5px dashed ${RULE}`,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      ...style,
    }}>
      <span className="mono" style={{ fontSize: 10, color: INK4, letterSpacing: 0.5 }}>{label}</span>
    </div>
  );
}

// Reusable titlebar row with a section label and a small action area
function SectionRow({ label, right = null, style = {} }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8, ...style }}>
      <span className="mono" style={{ fontSize: 10, color: INK3, letterSpacing: 0.8, textTransform: 'uppercase' }}>{label}</span>
      {right}
    </div>
  );
}

// SVG icon set — minimal, stroke-based, no emoji
function Icon({ name, size = 14, color = 'currentColor', style = {} }) {
  const s = { width: size, height: size, display: 'inline-block', verticalAlign: 'middle', flexShrink: 0, ...style };
  const stroke = { stroke: color, strokeWidth: 1.5, fill: 'none', strokeLinecap: 'round', strokeLinejoin: 'round' };
  switch (name) {
    case 'search': return <svg viewBox="0 0 16 16" style={s}><circle cx="7" cy="7" r="4.5" {...stroke} /><path d="M10.5 10.5L14 14" {...stroke} /></svg>;
    case 'sparkle': return <svg viewBox="0 0 16 16" style={s}><path d="M8 2v4M8 10v4M2 8h4M10 8h4" {...stroke} /><path d="M4 4l1.5 1.5M10.5 10.5L12 12M12 4l-1.5 1.5M5.5 10.5L4 12" {...stroke} /></svg>;
    case 'layers': return <svg viewBox="0 0 16 16" style={s}><path d="M2 5.5l6-3 6 3-6 3-6-3z" {...stroke} /><path d="M2 8.5l6 3 6-3M2 11.5l6 3 6-3" {...stroke} /></svg>;
    case 'doc': return <svg viewBox="0 0 16 16" style={s}><path d="M3.5 2h6l3 3v9h-9V2z" {...stroke} /><path d="M9.5 2v3h3" {...stroke} /></svg>;
    case 'link': return <svg viewBox="0 0 16 16" style={s}><path d="M7 9a3 3 0 0 0 4 0l2-2a3 3 0 0 0-4-4L8 4" {...stroke} /><path d="M9 7a3 3 0 0 0-4 0L3 9a3 3 0 0 0 4 4l1-1" {...stroke} /></svg>;
    case 'dot': return <svg viewBox="0 0 16 16" style={s}><circle cx="8" cy="8" r="2" fill={color} /></svg>;
    case 'chev': return <svg viewBox="0 0 16 16" style={s}><path d="M6 4l4 4-4 4" {...stroke} /></svg>;
    case 'plus': return <svg viewBox="0 0 16 16" style={s}><path d="M8 3v10M3 8h10" {...stroke} /></svg>;
    case 'back': return <svg viewBox="0 0 16 16" style={s}><path d="M10 4L6 8l4 4" {...stroke} /></svg>;
    case 'grid': return <svg viewBox="0 0 16 16" style={s}><rect x="2.5" y="2.5" width="4" height="4" {...stroke} /><rect x="9.5" y="2.5" width="4" height="4" {...stroke} /><rect x="2.5" y="9.5" width="4" height="4" {...stroke} /><rect x="9.5" y="9.5" width="4" height="4" {...stroke} /></svg>;
    case 'globe': return <svg viewBox="0 0 16 16" style={s}><circle cx="8" cy="8" r="5.5" {...stroke} /><ellipse cx="8" cy="8" rx="2.5" ry="5.5" {...stroke} /><path d="M2.5 8h11" {...stroke} /></svg>;
    case 'cpu': return <svg viewBox="0 0 16 16" style={s}><rect x="4" y="4" width="8" height="8" rx="1" {...stroke} /><rect x="6" y="6" width="4" height="4" {...stroke} /><path d="M6 2v2M10 2v2M6 12v2M10 12v2M2 6h2M2 10h2M12 6h2M12 10h2" {...stroke} /></svg>;
    case 'tray': return <svg viewBox="0 0 16 16" style={s}><path d="M2.5 9.5V12a1.5 1.5 0 0 0 1.5 1.5h8A1.5 1.5 0 0 0 13.5 12V9.5" {...stroke} /><path d="M5 9.5l1 1h4l1-1M5 9.5V3a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v6.5" {...stroke} /></svg>;
    case 'mic': return <svg viewBox="0 0 16 16" style={s}><rect x="6" y="2" width="4" height="8" rx="2" {...stroke} /><path d="M4 8a4 4 0 0 0 8 0M8 12v2" {...stroke} /></svg>;
    default: return null;
  }
}

// Legend card shown as concept 00
function Legend() {
  return (
    <div style={{ height: '100%', background: PAPER, padding: 32, display: 'flex', flexDirection: 'column', gap: 20, color: INK }}>
      <div>
        <div className="mono" style={{ fontSize: 10, color: INK3, letterSpacing: 1.2, textTransform: 'uppercase' }}>imprint / design system</div>
        <div className="serif" style={{ fontSize: 32, fontWeight: 500, lineHeight: 1.15, marginTop: 6 }}>One memory, two cursors.</div>
        <div style={{ fontSize: 13.5, color: INK2, lineHeight: 1.55, marginTop: 8, maxWidth: 440 }}>
          A local macOS database shared by a human and an AI. Both read, write, cite, and navigate the same semantic graph. Each concept tries a different answer to: <i>what's the shape of a mind two things share?</i>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <div>
          <SectionRow label="Type" />
          <div className="serif" style={{ fontSize: 22, color: INK }}>IBM Plex Serif</div>
          <div style={{ fontSize: 15, color: INK2 }}>IBM Plex Sans</div>
          <div className="mono" style={{ fontSize: 12, color: INK3, marginTop: 2 }}>IBM Plex Mono · tags, ids</div>
        </div>
        <div>
          <SectionRow label="Palette" />
          <div style={{ display: 'flex', gap: 6 }}>
            {[
              ['paper', PAPER],['paper-2', PAPER2],['ink', INK],['amber', AMBER],['teal', TEAL],['magenta', MAGENTA],
            ].map(([n,c]) => (
              <div key={n} style={{ flex: 1 }}>
                <div style={{ height: 36, borderRadius: 4, background: c, border: `0.5px solid ${RULE}` }} />
                <div className="mono" style={{ fontSize: 9, color: INK3, marginTop: 4, letterSpacing: 0.4 }}>{n}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div>
        <SectionRow label="Actors" />
        <div style={{ display: 'flex', gap: 14, alignItems: 'center' }}>
          <Actor who="human" /><Actor who="ai" />
          <span style={{ color: INK4, fontSize: 12 }}>— same graph, different cursors</span>
        </div>
      </div>

      <div>
        <SectionRow label="Concepts ahead" />
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: 12, color: INK2 }}>
          <div><b style={{ color: INK }}>01 Dual cursor</b> — presence</div>
          <div><b style={{ color: INK }}>02 Atlas</b> — territory</div>
          <div><b style={{ color: INK }}>03 Conversation</b> — chat-first</div>
          <div><b style={{ color: INK }}>04 Stratigraphy</b> — time as depth</div>
          <div><b style={{ color: INK }}>05 Command deck</b> — dense pro</div>
        </div>
      </div>

      <div style={{ marginTop: 'auto', fontSize: 11, color: INK4, borderTop: `0.5px solid ${RULE}`, paddingTop: 10 }}>
        All concepts reuse the same data model from your Rust backend: documents → chunks → regions → links. The difference is the <i>verb</i>: navigate, survey, converse, replay, command.
      </div>
    </div>
  );
}

Object.assign(window, {
  INK, INK2, INK3, INK4, PAPER, PAPER2, PAPER3, RULE, AMBER, AMBER_INK, TEAL, MAGENTA, GREEN,
  Mac, Dot, Tag, Actor, Placeholder, SectionRow, Icon, Legend,
});
