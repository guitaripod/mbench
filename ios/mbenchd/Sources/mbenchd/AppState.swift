import Foundation
import Observation
import UIKit

@Observable
final class AppState: @unchecked Sendable {
    static let shared = AppState()

    @MainActor var telemetry: TelemetrySnapshot?
    @MainActor var server: ServerStatus = ServerStatus(state: "idle", args: [])
    @MainActor var models: [StoredModel] = []
    @MainActor var logLines: [String] = []
    @MainActor var controlError: String?

    private let lock = NSLock()
    private var control: ControlServer?
    private var ticker: Timer?
    private var started = false

    private init() {}

    /// Everything that does not need a window, started from the process entry point so a phone that launched the
    /// app with the screen off still logs why nothing answered.
    func startServices() {
        lock.lock()
        let first = !started
        started = true
        lock.unlock()
        guard first else { return }
        AppLogger.info(.lifecycle, "mbenchd \(Router.version) starting on \(DeviceTelemetry.hardware()) "
                       + "\(UIDevice.current.systemName) \(UIDevice.current.systemVersion)")
        DeviceTelemetry.shared.start()
        let server = ControlServer(port: 8081)
        do {
            try server.start()
            lock.lock()
            control = server
            lock.unlock()
        } catch {
            AppLogger.error(.control, "control server failed to start: \(error)")
            Task { @MainActor in self.controlError = String(describing: error) }
        }
    }

    @MainActor
    func activate() {
        UIApplication.shared.isIdleTimerDisabled = true
        AppLogger.info(.lifecycle, "scene active; screen kept awake")
        refresh()
        guard ticker == nil else { return }
        ticker = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { _ in
            Task { @MainActor in self.refresh() }
        }
    }

    @MainActor
    func refresh() {
        telemetry = DeviceTelemetry.shared.snapshot()
        server = LlamaServerRunner.shared.snapshot()
        models = ModelStore.all()
        logLines = LogFileWriter.shared.tail(lines: 14)
    }

    @MainActor
    func unload() {
        Task { await LlamaServerRunner.shared.unload() }
    }
}
