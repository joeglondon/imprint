import AppKit
import SwiftUI

struct MemoryTheme {
    let id: String
    let name: String
    let description: String
    let surfaces: SurfaceTokens
    let ink: InkTokens
    let accents: AccentTokens
    let tags: TagTokens
    let buttons: ButtonTokens

    struct SurfaceTokens {
        let surface1: Color
        let surface2: Color
        let surface3: Color
        let chrome: Color
        let rule: Color
        let edge: Color
        let selection: Color
    }

    struct InkTokens {
        let primary: Color
        let secondary: Color
        let tertiary: Color
        let quaternary: Color
    }

    struct AccentTokens {
        let human: Color
        let humanInk: Color
        let ai: Color
        let aiInk: Color
        let c: Color
        let cInk: Color
        let d: Color
        let dInk: Color
        let highlight: Color
        let success: Color

        var graphPalette: [Color] { [human, ai, c, d] }
    }

    struct TagTokens {
        let ink: Color
        let human: Color
        let ai: Color
        let c: Color
        let d: Color
    }

    struct ButtonTokens {
        let background: Color
        let foreground: Color
    }
}

enum ThemeCatalog {
    static let graphite = MemoryTheme(
        id: "graphite",
        name: "Graphite",
        description: "Cool neutral gray. Ink black. Electric amber + cobalt accents.",
        surfaces: .init(
            surface1: Color(hex: 0xF0F2F4),
            surface2: Color(hex: 0xE6E8EA),
            surface3: Color(hex: 0xD5D8DB),
            chrome: Color(hex: 0xDFE1E4),
            rule: Color(hex: 0xC1C4C8),
            edge: Color(hex: 0x7D8185),
            selection: Color(hex: 0xCCD2D7)
        ),
        ink: .init(
            primary: Color(hex: 0x101214),
            secondary: Color(hex: 0x36383B),
            tertiary: Color(hex: 0x616467),
            quaternary: Color(hex: 0x8C8F93)
        ),
        accents: .init(
            human: Color(hex: 0xF77F00),
            humanInk: Color(hex: 0x762300),
            ai: Color(hex: 0x0077EC),
            aiInk: Color(hex: 0x003094),
            c: Color(hex: 0xC841A5),
            cInk: Color(hex: 0x74065C),
            d: Color(hex: 0x00A159),
            dInk: Color(hex: 0x005423),
            highlight: Color(hex: 0xFFE36E),
            success: Color(hex: 0x54A859)
        ),
        tags: .init(
            ink: Color(hex: 0xCED1D4),
            human: Color(hex: 0xFFDFC0),
            ai: Color(hex: 0xD2EAFF),
            c: Color(hex: 0xFBDEF0),
            d: Color(hex: 0xD4F0DC)
        ),
        buttons: .init(
            background: Color(hex: 0x101214),
            foreground: Color(hex: 0xF0F2F4)
        )
    )
}

private struct MemoryThemeKey: EnvironmentKey {
    static let defaultValue = ThemeCatalog.graphite
}

extension EnvironmentValues {
    var memoryTheme: MemoryTheme {
        get { self[MemoryThemeKey.self] }
        set { self[MemoryThemeKey.self] = newValue }
    }
}

extension Color {
    init(hex: UInt, alpha: Double = 1) {
        self.init(
            .sRGB,
            red: Double((hex >> 16) & 0xFF) / 255,
            green: Double((hex >> 8) & 0xFF) / 255,
            blue: Double(hex & 0xFF) / 255,
            opacity: alpha
        )
    }

    var nsColor: NSColor {
        NSColor(self)
    }
}
