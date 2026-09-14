import SwiftUI

@main
struct MbenchdApp: App {
    @Environment(\.scenePhase) private var scenePhase
    @State private var state = AppState.shared

    init() {
        AppState.shared.startServices()
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(state)
                .task { state.activate() }
        }
        .onChange(of: scenePhase) { _, phase in
            AppLogger.info(.lifecycle, "scene phase \(phase)")
        }
    }
}
