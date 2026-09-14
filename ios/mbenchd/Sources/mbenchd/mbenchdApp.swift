import SwiftUI

@main
struct MbenchdApp: App {
    @State private var state = AppState.shared

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(state)
                .task { state.bootstrap() }
        }
    }
}
