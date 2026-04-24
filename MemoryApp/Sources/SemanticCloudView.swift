import AppKit
import SwiftUI

struct SemanticCloudView: View {
    @Environment(\.memoryTheme) private var theme
    @State private var hoveredNode: GraphNode?
    let snapshot: VisualizationSnapshot?
    let selectedNode: GraphNode?
    var onSelect: (GraphNode) -> Void

    var body: some View {
        ZStack {
            if let snapshot, !snapshot.nodes.isEmpty {
                let focusNode = hoveredNode ?? selectedNode
                SemanticMap2DView(
                    snapshot: snapshot,
                    selectedNode: selectedNode,
                    hoveredNode: hoveredNode,
                    onSelect: onSelect,
                    onHover: { node in
                        if hoveredNode?.id != node?.id {
                            hoveredNode = node
                        }
                    }
                )
                .overlay(alignment: .topLeading) {
                    GraphMapHeader(snapshot: snapshot)
                        .padding(16)
                }
                .overlay(alignment: .topTrailing) {
                    GraphLegend()
                        .padding(16)
                }
                .overlay(alignment: .bottomLeading) {
                    GraphReadout(snapshot: snapshot, selectedNode: focusNode)
                        .padding(16)
                }
            } else {
                emptyState(snapshot == nil ? "Import files to generate your 2D memory map." : "No graph nodes are available for this memory store.")
            }
        }
    }

    private func emptyState(_ message: String) -> some View {
        Rectangle()
            .fill(theme.surfaces.surface1)
            .overlay(GraphPaperGrid().opacity(0.72))
            .overlay {
                VStack(spacing: 12) {
                    Image(systemName: "point.3.connected.trianglepath.dotted")
                        .font(.system(size: 42))
                        .foregroundStyle(theme.accents.ai)
                    Text(message)
                        .font(.system(size: 13))
                        .foregroundStyle(theme.ink.secondary)
                }
            }
    }
}

private struct SemanticMap2DView: View {
    @Environment(\.memoryTheme) private var theme
    @State private var viewport = GraphViewportTransform()
    @StateObject private var layoutCache = GraphLayoutCache()
    let snapshot: VisualizationSnapshot
    let selectedNode: GraphNode?
    let hoveredNode: GraphNode?
    var onSelect: (GraphNode) -> Void
    var onHover: (GraphNode?) -> Void

    var body: some View {
        GeometryReader { proxy in
            let size = proxy.size
            let layout = layoutCache.layout(snapshot: snapshot, viewportSize: size, theme: theme)
            let focusID = hoveredNode?.id ?? selectedNode?.id

            ZStack(alignment: .topLeading) {
                GraphBackground()

                ZStack {
                    Canvas { context, _ in
                        drawAmbientEdges(layout, in: &context)
                        drawBaseNodes(layout, in: &context)
                    }

                    Canvas { context, _ in
                        drawFocusOverlay(layout, focusID: focusID, hoverID: hoveredNode?.id, in: &context)
                    }

                    ForEach(layout.labelledNodes(focusID: focusID, scale: viewport.scale)) { display in
                        NodeLabel(display: display, selected: display.node.id == focusID, scale: viewport.scale)
                            .position(x: display.point.x, y: display.point.y + display.radius + (display.node.kind == .region ? 17 : 14))
                            .allowsHitTesting(false)
                    }
                }
                .frame(width: layout.worldSize.width, height: layout.worldSize.height, alignment: .topLeading)
                .scaleEffect(viewport.scale, anchor: .topLeading)
                .offset(viewport.offset)

                MapInteractionLayer(
                    onHover: { location in
                        onHover(layout.nearestNode(to: viewport.mapPoint(from: location))?.node)
                    },
                    onExit: {
                        onHover(nil)
                    },
                    onTap: { location in
                        if let node = layout.nearestNode(to: viewport.mapPoint(from: location))?.node {
                            onSelect(node)
                        }
                    },
                    onPan: { delta in
                        viewport.pan(by: delta)
                    },
                    onZoom: { factor, location in
                        viewport.zoom(by: factor, around: location)
                    }
                )
                .frame(width: size.width, height: size.height)
            }
            .clipped()
            .onAppear {
                viewport.centerIfNeeded(viewSize: size, worldSize: layout.worldSize)
            }
            .onChange(of: layout.signature) { _, _ in
                viewport.recenter(viewSize: size, worldSize: layout.worldSize)
            }
            .onChange(of: size) { _, newSize in
                viewport.centerIfNeeded(viewSize: newSize, worldSize: layout.worldSize)
            }
        }
    }

    private func drawAmbientEdges(_ layout: Graph2DLayout, in context: inout GraphicsContext) {
        for edge in layout.ambientEdges {
            var path = Path()
            path.move(to: edge.source.point)
            path.addLine(to: edge.target.point)
            let width = max(0.45, CGFloat(edge.edge.weight) * 0.9)
            context.stroke(path, with: .color(theme.surfaces.edge.opacity(0.24)), style: StrokeStyle(lineWidth: width, lineCap: .round, dash: [3, 5]))
        }
    }

    private func drawBaseNodes(_ layout: Graph2DLayout, in context: inout GraphicsContext) {
        for display in layout.nodes {
            drawNode(display, color: layout.color(for: display), opacity: 1.0, radiusScale: 1.0, isFocused: false, in: &context)
        }
    }

    private func drawFocusOverlay(_ layout: Graph2DLayout, focusID: String?, hoverID: String?, in context: inout GraphicsContext) {
        guard let focusID, let connectedIDs = layout.connectedNodeIDs(focusID: focusID) else { return }

        context.fill(
            Path(CGRect(origin: .zero, size: layout.worldSize)),
            with: .color(theme.surfaces.surface1.opacity(0.48))
        )

        for edge in layout.focusEdges(focusID: focusID) {
            var path = Path()
            path.move(to: edge.source.point)
            path.addLine(to: edge.target.point)
            let color = layout.color(for: edge.source).opacity(0.78)
            context.stroke(path, with: .color(color), style: StrokeStyle(lineWidth: 1.45, lineCap: .round))
        }

        for display in layout.nodes where connectedIDs.contains(display.node.id) && display.node.id != focusID {
            drawNode(display, color: layout.color(for: display), opacity: 1.0, radiusScale: 1.0, isFocused: false, in: &context)
        }

        if let focus = layout.nodeByID[focusID] {
            drawNode(
                focus,
                color: layout.color(for: focus),
                opacity: 1.0,
                radiusScale: focus.node.id == hoverID ? 1.18 : 1.0,
                isFocused: true,
                in: &context
            )
        }
    }

    private func drawNode(_ display: GraphDisplayNode, color base: Color, opacity: Double, radiusScale: CGFloat, isFocused: Bool, in context: inout GraphicsContext) {
            let center = display.point
            let radius = display.radius * radiusScale

            if isFocused {
                let haloRadius = radius * 2.8
                context.fill(Path(ellipseIn: CGRect(x: center.x - haloRadius, y: center.y - haloRadius, width: haloRadius * 2, height: haloRadius * 2)), with: .color(base.opacity(0.16)))
            }

            if display.node.kind == .document {
                let ring = Path(ellipseIn: CGRect(x: center.x - radius * 1.55, y: center.y - radius * 1.55, width: radius * 3.1, height: radius * 3.1))
                context.stroke(ring, with: .color(base.opacity(0.50 * opacity)), style: StrokeStyle(lineWidth: isFocused ? 1.4 : 0.85))
            }

            let outer = Path(ellipseIn: CGRect(x: center.x - radius, y: center.y - radius, width: radius * 2, height: radius * 2))
            let innerRadius = radius * (display.node.kind == .chunk ? 0.62 : 0.50)
            let inner = Path(ellipseIn: CGRect(x: center.x - innerRadius, y: center.y - innerRadius, width: innerRadius * 2, height: innerRadius * 2))
            context.fill(outer, with: .color(base.opacity((display.node.kind == .chunk ? 0.42 : 0.22) * opacity)))
            context.fill(inner, with: .color(base.opacity((display.node.kind == .chunk ? 0.70 : 0.62) * opacity)))

            if isFocused {
                context.stroke(Path(ellipseIn: CGRect(x: center.x - radius * 1.85, y: center.y - radius * 1.85, width: radius * 3.7, height: radius * 3.7)), with: .color(theme.accents.highlight.opacity(0.95)), style: StrokeStyle(lineWidth: 2.0))
                context.stroke(Path(ellipseIn: CGRect(x: center.x - radius * 2.35, y: center.y - radius * 2.35, width: radius * 4.7, height: radius * 4.7)), with: .color(base.opacity(0.42)), style: StrokeStyle(lineWidth: 0.9, dash: [3, 4]))
            }
    }
}

private struct GraphViewportTransform {
    var scale: CGFloat = 1
    var offset: CGSize = .zero
    private var hasCentered = false
    private let minScale: CGFloat = 0.58
    private let maxScale: CGFloat = 3.0

    func mapPoint(from viewPoint: CGPoint) -> CGPoint {
        CGPoint(
            x: (viewPoint.x - offset.width) / scale,
            y: (viewPoint.y - offset.height) / scale
        )
    }

    mutating func pan(by delta: CGSize) {
        offset.width += delta.width
        offset.height += delta.height
    }

    mutating func zoom(by factor: CGFloat, around anchor: CGPoint) {
        let nextScale = min(max(scale * factor, minScale), maxScale)
        guard nextScale != scale else { return }
        let ratio = nextScale / scale
        offset.width = anchor.x - (anchor.x - offset.width) * ratio
        offset.height = anchor.y - (anchor.y - offset.height) * ratio
        scale = nextScale
    }

    mutating func centerIfNeeded(viewSize: CGSize, worldSize: CGSize) {
        guard !hasCentered else { return }
        recenter(viewSize: viewSize, worldSize: worldSize)
    }

    mutating func recenter(viewSize: CGSize, worldSize: CGSize) {
        scale = 1
        offset = CGSize(
            width: max(0, (viewSize.width - worldSize.width) / 2),
            height: max(0, (viewSize.height - worldSize.height) / 2)
        )
        hasCentered = true
    }
}

private struct MapInteractionLayer: NSViewRepresentable {
    var onHover: (CGPoint) -> Void
    var onExit: () -> Void
    var onTap: (CGPoint) -> Void
    var onPan: (CGSize) -> Void
    var onZoom: (CGFloat, CGPoint) -> Void

    func makeNSView(context: Context) -> MapInteractionNSView {
        let view = MapInteractionNSView()
        view.onHover = onHover
        view.onExit = onExit
        view.onTap = onTap
        view.onPan = onPan
        view.onZoom = onZoom
        return view
    }

    func updateNSView(_ view: MapInteractionNSView, context: Context) {
        view.onHover = onHover
        view.onExit = onExit
        view.onTap = onTap
        view.onPan = onPan
        view.onZoom = onZoom
    }
}

private final class MapInteractionNSView: NSView {
    var onHover: ((CGPoint) -> Void)?
    var onExit: (() -> Void)?
    var onTap: ((CGPoint) -> Void)?
    var onPan: ((CGSize) -> Void)?
    var onZoom: ((CGFloat, CGPoint) -> Void)?
    private var trackingAreaRef: NSTrackingArea?

    override var acceptsFirstResponder: Bool { true }
    override var isFlipped: Bool { true }

    override func updateTrackingAreas() {
        if let trackingAreaRef {
            removeTrackingArea(trackingAreaRef)
        }
        let area = NSTrackingArea(
            rect: bounds,
            options: [.activeInKeyWindow, .mouseMoved, .mouseEnteredAndExited, .inVisibleRect],
            owner: self
        )
        addTrackingArea(area)
        trackingAreaRef = area
        super.updateTrackingAreas()
    }

    override func acceptsFirstMouse(for event: NSEvent?) -> Bool {
        true
    }

    override func mouseMoved(with event: NSEvent) {
        onHover?(convert(event.locationInWindow, from: nil))
    }

    override func mouseDragged(with event: NSEvent) {
        onPan?(CGSize(width: event.deltaX, height: event.deltaY))
    }

    override func mouseExited(with event: NSEvent) {
        onExit?()
    }

    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
    }

    override func mouseUp(with event: NSEvent) {
        onTap?(convert(event.locationInWindow, from: nil))
    }

    override func scrollWheel(with event: NSEvent) {
        let location = convert(event.locationInWindow, from: nil)
        if event.modifierFlags.contains(.command) || event.modifierFlags.contains(.control) {
            let factor = exp(-event.scrollingDeltaY * 0.006)
            onZoom?(factor, location)
        } else {
            onPan?(CGSize(width: event.scrollingDeltaX, height: event.scrollingDeltaY))
        }
    }

    override func magnify(with event: NSEvent) {
        let location = convert(event.locationInWindow, from: nil)
        onZoom?(1 + event.magnification, location)
    }
}

private struct GraphBackground: View {
    @Environment(\.memoryTheme) private var theme

    var body: some View {
        ZStack {
            RadialGradient(
                colors: [theme.surfaces.surface1, theme.surfaces.surface2.opacity(0.82)],
                center: UnitPoint(x: 0.42, y: 0.36),
                startRadius: 40,
                endRadius: 920
            )
            GraphPaperGrid()
        }
    }
}

private struct GraphPaperGrid: View {
    @Environment(\.memoryTheme) private var theme

    var body: some View {
        Canvas { context, size in
            let minor: CGFloat = 8
            let major: CGFloat = 40
            var minorPath = Path()
            var x: CGFloat = 0
            while x <= size.width {
                minorPath.move(to: CGPoint(x: x, y: 0))
                minorPath.addLine(to: CGPoint(x: x, y: size.height))
                x += minor
            }
            var y: CGFloat = 0
            while y <= size.height {
                minorPath.move(to: CGPoint(x: 0, y: y))
                minorPath.addLine(to: CGPoint(x: size.width, y: y))
                y += minor
            }
            context.stroke(minorPath, with: .color(theme.surfaces.rule.opacity(0.20)), style: StrokeStyle(lineWidth: 0.35))

            var majorPath = Path()
            x = 0
            while x <= size.width {
                majorPath.move(to: CGPoint(x: x, y: 0))
                majorPath.addLine(to: CGPoint(x: x, y: size.height))
                x += major
            }
            y = 0
            while y <= size.height {
                majorPath.move(to: CGPoint(x: 0, y: y))
                majorPath.addLine(to: CGPoint(x: size.width, y: y))
                y += major
            }
            context.stroke(majorPath, with: .color(theme.surfaces.rule.opacity(0.36)), style: StrokeStyle(lineWidth: 0.55))
        }
    }
}

private struct GraphMapHeader: View {
    @Environment(\.memoryTheme) private var theme
    let snapshot: VisualizationSnapshot

    var body: some View {
        HStack(spacing: 8) {
            Text("Semantic Map")
                .font(.system(size: 10, weight: .medium, design: .monospaced))
                .foregroundStyle(theme.ink.tertiary)
                .textCase(.uppercase)
            Text("2D")
                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                .foregroundStyle(theme.ink.primary)
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(theme.surfaces.selection, in: RoundedRectangle(cornerRadius: 4, style: .continuous))
            Text("\(snapshot.nodes.count) nodes")
                .font(.system(size: 10, weight: .medium, design: .monospaced))
                .foregroundStyle(theme.ink.secondary)
        }
        .padding(.horizontal, 10)
        .frame(height: 28)
        .background(theme.surfaces.surface1.opacity(0.82), in: RoundedRectangle(cornerRadius: 7, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 7, style: .continuous).stroke(theme.surfaces.rule, lineWidth: 0.5))
        .shadow(color: .black.opacity(0.06), radius: 14, y: 8)
    }
}

private struct GraphLegend: View {
    @Environment(\.memoryTheme) private var theme

    var body: some View {
        HStack(spacing: 10) {
            legendDot("cluster hue", theme.accents.ai)
            legendDot("center = bright", theme.accents.highlight)
            Text("blend = between")
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(theme.ink.tertiary)
        }
        .padding(.horizontal, 10)
        .frame(height: 28)
        .background(theme.surfaces.surface1.opacity(0.78), in: RoundedRectangle(cornerRadius: 7, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 7, style: .continuous).stroke(theme.surfaces.rule, lineWidth: 0.5))
        .shadow(color: .black.opacity(0.05), radius: 12, y: 7)
    }

    private func legendDot(_ label: String, _ color: Color) -> some View {
        HStack(spacing: 5) {
            Circle().fill(color).frame(width: 6, height: 6)
            Text(label)
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(theme.ink.tertiary)
        }
    }
}

private struct NodeLabel: View {
    @Environment(\.memoryTheme) private var theme
    let display: GraphDisplayNode
    let selected: Bool
    let scale: CGFloat

    var body: some View {
        Text(label)
            .font(.system(size: selected ? 10.5 : fontSize, weight: selected || display.node.kind == .region ? .medium : .regular, design: .monospaced))
            .foregroundStyle(selected ? theme.ink.primary : theme.ink.secondary)
            .lineLimit(1)
            .padding(.horizontal, selected ? 7 : 0)
            .padding(.vertical, selected ? 3 : 0)
            .background {
                if selected {
                    RoundedRectangle(cornerRadius: 4, style: .continuous)
                        .fill(theme.surfaces.surface1.opacity(0.9))
                        .overlay(RoundedRectangle(cornerRadius: 4, style: .continuous).stroke(theme.surfaces.rule, lineWidth: 0.5))
                }
            }
            .frame(maxWidth: selected ? 220 : 150)
    }

    private var fontSize: CGFloat {
        switch display.node.kind {
        case .region:
            return 9.8
        case .document:
            return scale >= 1.45 ? 9.4 : 9.0
        case .chunk:
            return 8.6
        }
    }

    private var label: String {
        let limit: Int
        if selected {
            limit = scale >= 1.6 ? 64 : 42
        } else {
            switch display.node.kind {
            case .region:
                limit = scale >= 1.3 ? 42 : 30
            case .document:
                limit = scale >= 1.7 ? 48 : 30
            case .chunk:
                limit = scale >= 2.05 ? 40 : 28
            }
        }
        if display.node.label.count <= limit { return display.node.label }
        let end = display.node.label.index(display.node.label.startIndex, offsetBy: max(1, limit - 1))
        return "\(display.node.label[..<end])..."
    }
}

private struct GraphReadout: View {
    @Environment(\.memoryTheme) private var theme
    let snapshot: VisualizationSnapshot
    let selectedNode: GraphNode?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 12) {
                metric("regions", snapshot.summary.regions)
                metric("docs", snapshot.summary.documents)
                metric("links", snapshot.summary.links)
            }
            if let selectedNode {
                HStack(spacing: 8) {
                    Text(selectedNode.kind.rawValue.uppercased())
                        .font(.system(size: 9, weight: .medium, design: .monospaced))
                        .foregroundStyle(tagForeground(for: selectedNode))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(tagBackground(for: selectedNode), in: RoundedRectangle(cornerRadius: 4, style: .continuous))
                    Text(selectedNode.label)
                        .font(.system(size: 11.5, weight: .medium))
                        .foregroundStyle(theme.ink.primary)
                        .lineLimit(1)
                }
            } else {
                Text("click any node to inspect links, excerpt, and trail")
                    .font(.system(size: 11.5))
                    .foregroundStyle(theme.ink.tertiary)
            }
        }
        .padding(12)
        .frame(width: 360, alignment: .leading)
        .background(theme.surfaces.surface1.opacity(0.86), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous).stroke(theme.surfaces.rule, lineWidth: 0.5))
        .shadow(color: .black.opacity(0.07), radius: 18, y: 10)
    }

    private func metric(_ label: String, _ value: Int) -> some View {
        HStack(spacing: 5) {
            Text("\(value)")
                .font(.system(size: 11, weight: .semibold, design: .monospaced))
                .foregroundStyle(theme.ink.secondary)
            Text(label)
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(theme.ink.quaternary)
        }
    }

    private func tagBackground(for node: GraphNode) -> Color {
        switch node.kind {
        case .region: return theme.tags.ai
        case .document: return theme.tags.human
        case .chunk: return theme.tags.d
        }
    }

    private func tagForeground(for node: GraphNode) -> Color {
        switch node.kind {
        case .region: return theme.accents.aiInk
        case .document: return theme.accents.humanInk
        case .chunk: return theme.accents.dInk
        }
    }
}

private struct GraphDisplayNode: Identifiable {
    let id: String
    let node: GraphNode
    let point: CGPoint
    let radius: CGFloat
    let rank: Int
}

private struct GraphDisplayEdge: Identifiable {
    let id: String
    let edge: GraphEdge
    let source: GraphDisplayNode
    let target: GraphDisplayNode
}

private struct GraphLayoutSignature: Equatable {
    let themeID: String
    let viewportWidth: Int
    let viewportHeight: Int
    let nodeCount: Int
    let edgeCount: Int
    let firstNodeID: String?
    let lastNodeID: String?
}

@MainActor
private final class GraphLayoutCache: ObservableObject {
    private var cachedSignature: GraphLayoutSignature?
    private var cachedLayout: Graph2DLayout?

    func layout(snapshot: VisualizationSnapshot, viewportSize: CGSize, theme: MemoryTheme) -> Graph2DLayout {
        let signature = GraphLayoutSignature(
            themeID: theme.id,
            viewportWidth: Int(viewportSize.width.rounded()),
            viewportHeight: Int(viewportSize.height.rounded()),
            nodeCount: snapshot.nodes.count,
            edgeCount: snapshot.edges.count,
            firstNodeID: snapshot.nodes.first?.id,
            lastNodeID: snapshot.nodes.last?.id
        )
        if cachedSignature == signature, let cachedLayout {
            return cachedLayout
        }
        let layout = Graph2DLayout.build(snapshot: snapshot, viewportSize: viewportSize, signature: signature, theme: theme)
        cachedSignature = signature
        cachedLayout = layout
        return layout
    }
}

private struct GraphTerritory: Identifiable {
    let id: String
    let center: CGPoint
    let radius: CGFloat
    let color: SemanticClusterColor
}

private struct SemanticClusterColor {
    let hue: Double
    let saturation: Double
    let brightness: Double

    var color: Color {
        Color(hue: hue, saturation: saturation, brightness: brightness)
    }

    func withBrightness(_ value: Double) -> SemanticClusterColor {
        SemanticClusterColor(hue: hue, saturation: saturation, brightness: min(max(value, 0), 1))
    }

    func blended(with other: SemanticClusterColor, amount: Double) -> SemanticClusterColor {
        let t = min(max(amount, 0), 1)
        var delta = other.hue - hue
        if delta > 0.5 { delta -= 1 }
        if delta < -0.5 { delta += 1 }
        let blendedHue = (hue + delta * t).truncatingRemainder(dividingBy: 1)
        return SemanticClusterColor(
            hue: blendedHue < 0 ? blendedHue + 1 : blendedHue,
            saturation: saturation + (other.saturation - saturation) * t,
            brightness: brightness + (other.brightness - brightness) * t
        )
    }
}

private struct Graph2DLayout {
    let signature: GraphLayoutSignature
    let worldSize: CGSize
    let nodes: [GraphDisplayNode]
    let territories: [GraphTerritory]
    let territoryByNodeID: [String: String]
    let nodeByID: [String: GraphDisplayNode]
    let territoryByID: [String: GraphTerritory]
    let colorByNodeID: [String: Color]
    let ambientEdges: [GraphDisplayEdge]
    let focusEdgesByNodeID: [String: [GraphDisplayEdge]]
    let connectedNodeIDsByFocusID: [String: Set<String>]
    let baseLabelledNodes: [GraphDisplayNode]

    static func build(snapshot: VisualizationSnapshot, viewportSize: CGSize, signature: GraphLayoutSignature, theme: MemoryTheme) -> Graph2DLayout {
        let canvas = CGSize(
            width: max(viewportSize.width * 1.34, viewportSize.width + 240, 900),
            height: max(viewportSize.height * 1.28, viewportSize.height + 180, 680)
        )
        let nodes = snapshot.nodes
        let regions = nodes.filter { $0.kind == .region }.sorted { $0.id < $1.id }
        let documents = nodes.filter { $0.kind == .document }.sorted { $0.id < $1.id }
        let chunks = nodes.filter { $0.kind == .chunk }.sorted { $0.id < $1.id }
        let regionIDs = regionIDs(from: nodes)
        let centers = separatedRegionCenters(regionIDs: regionIDs, size: canvas)
        let chunkCounts = Dictionary(grouping: chunks, by: { $0.regionId ?? "unassigned" }).mapValues(\.count)
        let documentCounts = Dictionary(grouping: documents, by: { $0.regionId ?? "unassigned" }).mapValues(\.count)

        var territories = regionIDs.map { id in
            let mass = chunkCounts[id, default: 0] + documentCounts[id, default: 0] * 2
            let radius = CGFloat(78 + min(170, sqrt(Double(max(mass, 1))) * 13.5))
            return GraphTerritory(id: id, center: centers[id] ?? CGPoint(x: canvas.width / 2, y: canvas.height / 2), radius: radius, color: stableClusterColor(for: id))
        }
        if territories.isEmpty {
            territories = [GraphTerritory(id: "unassigned", center: CGPoint(x: canvas.width / 2, y: canvas.height / 2), radius: 110, color: stableClusterColor(for: "unassigned"))]
        }

        var display = [GraphDisplayNode]()
        var territoryByNodeID = [String: String]()
        for (rank, region) in regions.enumerated() {
            let id = region.regionId ?? region.id.replacingOccurrences(of: "region:", with: "")
            let center = centers[id] ?? projected(region.position, size: canvas)
            display.append(GraphDisplayNode(id: region.id, node: region, point: center, radius: 20 + CGFloat(min(18, chunkCounts[id, default: 0])) / 3, rank: rank))
            territoryByNodeID[region.id] = id
        }

        let docsByRegion = Dictionary(grouping: documents, by: { $0.regionId ?? nearestRegionID(for: $0, centers: centers) })
        for (regionID, docs) in docsByRegion {
            let center = centers[regionID] ?? CGPoint(x: canvas.width / 2, y: canvas.height / 2)
            let territoryRadius = territories.first(where: { $0.id == regionID })?.radius ?? 105
            for (index, doc) in docs.enumerated() {
                let angle = goldenAngle(index)
                let ring = (0.22 + 0.52 * sqrt(Double(index + 1) / Double(max(docs.count, 1)))) * territoryRadius
                let jitter = deterministicJitter(doc.id, amount: 11)
                let x = center.x + CGFloat(Darwin.cos(Double(angle))) * ring + jitter.x
                let y = center.y + CGFloat(Darwin.sin(Double(angle))) * ring * 0.72 + jitter.y
                let point = CGPoint(x: x, y: y)
                display.append(GraphDisplayNode(id: doc.id, node: doc, point: point, radius: 12.5, rank: index))
                territoryByNodeID[doc.id] = regionID
            }
        }

        let chunksByRegion = Dictionary(grouping: chunks, by: { $0.regionId ?? nearestRegionID(for: $0, centers: centers) })
        for (regionID, regionChunks) in chunksByRegion {
            let center = centers[regionID] ?? CGPoint(x: canvas.width / 2, y: canvas.height / 2)
            let territoryRadius = territories.first(where: { $0.id == regionID })?.radius ?? 105
            for (index, chunk) in regionChunks.enumerated() {
                let angle = goldenAngle(index)
                let normalized = sqrt(Double(index + 1) / Double(max(regionChunks.count, 1)))
                let ring = normalized * territoryRadius * 0.78
                let jitter = deterministicJitter(chunk.id, amount: 14)
                let x = center.x + CGFloat(Darwin.cos(Double(angle))) * ring + jitter.x
                let y = center.y + CGFloat(Darwin.sin(Double(angle))) * ring * 0.72 + jitter.y
                let point = CGPoint(x: x, y: y)
                display.append(GraphDisplayNode(id: chunk.id, node: chunk, point: point, radius: 4.8, rank: index))
                territoryByNodeID[chunk.id] = regionID
            }
        }

        let nodeByID = Dictionary(uniqueKeysWithValues: display.map { ($0.node.id, $0) })
        let territoryByID = Dictionary(uniqueKeysWithValues: territories.map { ($0.id, $0) })
        let colorByNodeID = Dictionary(uniqueKeysWithValues: display.map { display in
            (display.node.id, Self.color(for: display, territoryByNodeID: territoryByNodeID, territoryByID: territoryByID, territories: territories))
        })
        let displayEdges = snapshot.edges.compactMap { edge -> GraphDisplayEdge? in
            guard let source = nodeByID[edge.source], let target = nodeByID[edge.target] else { return nil }
            return GraphDisplayEdge(id: edge.id, edge: edge, source: source, target: target)
        }
        let ambientEdges = displayEdges
            .filter { edge in
                !Self.isSourceEdge(edge.edge) && edge.edge.weight >= 0.62
            }
            .sorted { left, right in
                left.edge.weight == right.edge.weight ? left.id < right.id : left.edge.weight > right.edge.weight
            }
            .prefix(180)
        var focusEdgesByNodeID = [String: [GraphDisplayEdge]]()
        var connectedNodeIDsByFocusID = [String: Set<String>]()
        for edge in displayEdges {
            focusEdgesByNodeID[edge.edge.source, default: []].append(edge)
            focusEdgesByNodeID[edge.edge.target, default: []].append(edge)
            connectedNodeIDsByFocusID[edge.edge.source, default: [edge.edge.source]].insert(edge.edge.target)
            connectedNodeIDsByFocusID[edge.edge.target, default: [edge.edge.target]].insert(edge.edge.source)
        }
        for key in focusEdgesByNodeID.keys {
            focusEdgesByNodeID[key]?.sort {
                $0.edge.weight == $1.edge.weight ? $0.id < $1.id : $0.edge.weight > $1.edge.weight
            }
        }
        let baseLabelledNodes = display.filter { display in
            display.node.kind == .region || (display.node.kind == .document && display.rank < 6)
        }

        return Graph2DLayout(
            signature: signature,
            worldSize: canvas,
            nodes: display,
            territories: territories,
            territoryByNodeID: territoryByNodeID,
            nodeByID: nodeByID,
            territoryByID: territoryByID,
            colorByNodeID: colorByNodeID,
            ambientEdges: Array(ambientEdges),
            focusEdgesByNodeID: focusEdgesByNodeID,
            connectedNodeIDsByFocusID: connectedNodeIDsByFocusID,
            baseLabelledNodes: baseLabelledNodes
        )
    }

    func labelledNodes(focusID: String?, scale: CGFloat) -> [GraphDisplayNode] {
        var labelled = baseLabelledNodes
        if scale >= 1.25 {
            labelled.append(contentsOf: zoomLabels(for: .document, limit: scale >= 1.7 ? 48 : 18))
        }
        if scale >= 2.0 {
            labelled.append(contentsOf: zoomLabels(for: .chunk, limit: scale >= 2.55 ? 90 : 36))
        }

        guard let focusID, let focusNode = nodeByID[focusID] else {
            return uniqued(labelled)
        }
        labelled.append(focusNode)
        return uniqued(labelled)
    }

    private func zoomLabels(for kind: GraphNodeKind, limit: Int) -> [GraphDisplayNode] {
        nodes
            .filter { $0.node.kind == kind }
            .sorted { left, right in
                if left.rank != right.rank { return left.rank < right.rank }
                return left.id < right.id
            }
            .prefix(limit)
            .map { $0 }
    }

    private func uniqued(_ displays: [GraphDisplayNode]) -> [GraphDisplayNode] {
        var seen = Set<String>()
        return displays.filter { display in
            seen.insert(display.node.id).inserted
        }
    }

    func nearestNode(to point: CGPoint) -> GraphDisplayNode? {
        var best: GraphDisplayNode?
        var bestScore = CGFloat.greatestFiniteMagnitude
        for display in nodes {
            let dx = display.point.x - point.x
            let dy = display.point.y - point.y
            let threshold = hitRadius(for: display)
            let score = sqrt(dx * dx + dy * dy) / threshold
            guard score <= 1 else { continue }
            if score < bestScore || (score == bestScore && display.id < (best?.id ?? "")) {
                best = display
                bestScore = score
            }
        }
        return best
    }

    func color(for display: GraphDisplayNode) -> Color {
        colorByNodeID[display.node.id] ?? Self.stableClusterColor(for: territoryKey(for: display.node)).color
    }

    func focusEdges(focusID: String?) -> [GraphDisplayEdge] {
        guard let focusID else { return [] }
        return focusEdgesByNodeID[focusID] ?? []
    }

    func connectedNodeIDs(focusID: String) -> Set<String>? {
        connectedNodeIDsByFocusID[focusID] ?? (nodeByID[focusID] == nil ? nil : [focusID])
    }

    private func territoryKey(for node: GraphNode) -> String {
        territoryByNodeID[node.id] ?? node.regionId ?? node.id.replacingOccurrences(of: "region:", with: "")
    }

    private static func color(for display: GraphDisplayNode, territoryByNodeID: [String: String], territoryByID: [String: GraphTerritory], territories: [GraphTerritory]) -> Color {
        let key = territoryByNodeID[display.node.id] ?? display.node.regionId ?? display.node.id.replacingOccurrences(of: "region:", with: "")
        guard let territory = territoryByID[key] else {
            return Self.stableClusterColor(for: key).color
        }

        let assignedDistance = distance(display.point, territory.center)
        let assignedNorm = min(1.4, assignedDistance / max(territory.radius, 1))
        let centerBrightness = 0.96 - min(0.40, Double(assignedNorm) * 0.34)
        var clusterColor = territory.color.withBrightness(centerBrightness)

        if let alternate = nearestAlternateTerritory(to: display.point, excluding: key, territories: territories) {
            let alternateNorm = min(1.6, distance(display.point, alternate.center) / max(alternate.radius, 1))
            let assignedInfluence = max(0.10, 1.18 - assignedNorm)
            let alternateInfluence = max(0, 1.08 - alternateNorm)
            if alternateInfluence > 0, alternateInfluence > assignedInfluence * 0.28 {
                let blend = min(0.48, Double(alternateInfluence / (assignedInfluence + alternateInfluence)) * 0.72)
                let alternateBrightness = 0.96 - min(0.40, Double(alternateNorm) * 0.34)
                clusterColor = clusterColor.blended(with: alternate.color.withBrightness(alternateBrightness), amount: blend)
            }
        }

        return clusterColor.color
    }

    private static func nearestAlternateTerritory(to point: CGPoint, excluding key: String, territories: [GraphTerritory]) -> GraphTerritory? {
        territories
            .filter { $0.id != key }
            .min { left, right in
                distance(point, left.center) / max(left.radius, 1) < distance(point, right.center) / max(right.radius, 1)
            }
    }

    private static func distance(_ a: CGPoint, _ b: CGPoint) -> CGFloat {
        let dx = a.x - b.x
        let dy = a.y - b.y
        return sqrt(dx * dx + dy * dy)
    }

    private static func regionIDs(from nodes: [GraphNode]) -> [String] {
        let explicit = nodes
            .filter { $0.kind == .region }
            .map { $0.regionId ?? $0.id.replacingOccurrences(of: "region:", with: "") }
        let referenced = nodes.compactMap(\.regionId)
        let ids = Array(Set(explicit + referenced)).sorted()
        return ids.isEmpty ? ["unassigned"] : ids
    }

    private static func separatedRegionCenters(regionIDs: [String], size: CGSize) -> [String: CGPoint] {
        let count = max(regionIDs.count, 1)
        let shortest = min(size.width, size.height)
        let baseRadius = min(size.width, size.height) * min(0.34, 0.19 + CGFloat(count) * 0.019)
        let center = CGPoint(x: size.width * 0.50, y: size.height * 0.48)
        if count == 1, let id = regionIDs.first {
            return [id: center]
        }
        return Dictionary(uniqueKeysWithValues: regionIDs.enumerated().map { index, id in
            let angle = goldenAngle(index) - .pi / 8
            let ring = baseRadius * CGFloat(0.38 + 0.66 * sqrt(Double(index + 1) / Double(count)))
            let x = center.x + CGFloat(Darwin.cos(Double(angle))) * ring * 1.16
            let y = center.y + CGFloat(Darwin.sin(Double(angle))) * ring * 0.76
            let point = CGPoint(x: x, y: y)
            let margin = max(90, shortest * 0.12)
            return (id, CGPoint(x: min(max(point.x, margin), size.width - margin), y: min(max(point.y, margin), size.height - margin)))
        })
    }

    private static func projected(_ position: GraphPosition, size: CGSize) -> CGPoint {
        CGPoint(x: size.width * (0.5 + CGFloat(position.x) * 0.018), y: size.height * (0.5 + CGFloat(position.z) * 0.018))
    }

    private static func nearestRegionID(for node: GraphNode, centers: [String: CGPoint]) -> String {
        centers.keys.sorted().first ?? node.regionId ?? "unassigned"
    }

    private static func goldenAngle(_ index: Int) -> CGFloat {
        CGFloat(index) * 2.39996323
    }

    private static func deterministicJitter(_ id: String, amount: CGFloat) -> CGPoint {
        var hash: UInt64 = 14695981039346656037
        for byte in id.utf8 {
            hash ^= UInt64(byte)
            hash &*= 1099511628211
        }
        func unit(_ shift: UInt64) -> CGFloat {
            let value = (hash >> shift) & 0xffff
            return CGFloat(value) / CGFloat(UInt16.max) * 2 - 1
        }
        return CGPoint(x: unit(0) * amount, y: unit(16) * amount)
    }

    private func hitRadius(for display: GraphDisplayNode) -> CGFloat {
        switch display.node.kind {
        case .region:
            return max(30, display.radius * 1.65)
        case .document:
            return max(22, display.radius * 1.75)
        case .chunk:
            return max(14, display.radius * 2.1)
        }
    }

    private static func stableClusterColor(for key: String) -> SemanticClusterColor {
        let hash = stableHash(key)
        let goldenRatioConjugate = 0.618033988749895
        let seed = Double(hash % 10_000) / 10_000
        let hue = (seed + goldenRatioConjugate).truncatingRemainder(dividingBy: 1)
        let saturation = 0.58 + Double((hash >> 12) % 160) / 1000
        return SemanticClusterColor(hue: hue, saturation: min(saturation, 0.78), brightness: 0.92)
    }

    private static func stableHash(_ key: String) -> UInt64 {
        var hash: UInt64 = 14695981039346656037
        for byte in key.utf8 {
            hash ^= UInt64(byte)
            hash &*= 1099511628211
        }
        return hash
    }

    private static func isSourceEdge(_ edge: GraphEdge) -> Bool {
        edge.id.hasPrefix("chunk-doc:") || edge.label == "source text"
    }
}
