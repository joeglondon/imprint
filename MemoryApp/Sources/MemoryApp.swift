import SwiftUI

@main
struct MemoryAppMain: App {
    @StateObject private var appState = AppState()

    var body: some Scene {
        WindowGroup("imprint") {
            ContentView()
                .environmentObject(appState)
                .frame(minWidth: 1280, minHeight: 820)
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 1440, height: 900)
    }
}
