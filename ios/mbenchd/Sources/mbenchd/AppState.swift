import Foundation
import Observation
import UIKit

@Observable
final class AppState: @unchecked Sendable {
    static let shared = AppState()
    static let historyLength = 60

    @MainActor var telemetry: TelemetrySnapshot?
    @MainActor var server: ServerStatus = ServerStatus(state: "idle", args: [])
    @MainActor var metrics: ServerMetrics?
    @MainActor var history: [Double] = []
    @MainActor var lastDecode: Double?
    @MainActor var models: [StoredModel] = []
    @MainActor var logLines: [String] = []
    @MainActor var controlError: String?
    @MainActor var progress: RunProgress = RunProgress()

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
        server = Router.active()
        progress = ProgressStore.shared.snapshot()
        models = ModelStore.all()
        logLines = LogFileWriter.shared.tail(lines: 40)
        guard server.state == "running", let port = server.port else {
            metrics = nil
            return
        }
        Task { [port] in
            let reading = await MetricsReader.shared.read(port: port)
            await MainActor.run { self.record(reading) }
        }
    }

    @MainActor
    func load(_ model: StoredModel) {
        Task {
            do {
                let request = LoadRequest(model: model.file, nCtx: 8192, parallel: 4, flashAttn: "on",
                                          cacheTypeK: "q8_0", cacheTypeV: "q8_0", extraArgs: ["-kvu"],
                                          engine: ModelStore.isModelDirectory(ModelStore.directory
                                              .appendingPathComponent(model.file)) ? "mlx" : "llama.cpp")
                _ = try await Router.loadEngine(request)
            } catch {
                AppLogger.error(.server, "load from the screen failed: \(error)")
            }
        }
    }

    @MainActor
    func unload() {
        Task { await Router.unloadAll() }
    }

    /// Keeps the last minute of decode speed so the screen can show the shape of the throttle, not just a number.
    @MainActor
    private func record(_ reading: ServerMetrics?) {
        metrics = reading
        guard let value = reading?.decodeTokensPerSecond, value > 0 else { return }
        lastDecode = value
        guard reading?.isBusy == true else { return }
        history.append(value)
        if history.count > AppState.historyLength { history.removeFirst(history.count - AppState.historyLength) }
    }
}
