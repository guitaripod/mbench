import Foundation
import Observation
import UIKit

@MainActor
@Observable
final class AppState {
    static let shared = AppState()

    var telemetry: TelemetrySnapshot?
    var server: ServerStatus = ServerStatus(state: "idle", args: [])
    var models: [StoredModel] = []
    var controlError: String?
    var logLines: [String] = []

    private var control: ControlServer?
    private var ticker: Timer?

    private init() {}

    func bootstrap() {
        UIApplication.shared.isIdleTimerDisabled = true
        DeviceTelemetry.shared.start()
        AppLogger.info(.lifecycle, "mbenchd \(Router.version) on \(DeviceTelemetry.hardware()) "
                       + "\(UIDevice.current.systemName) \(UIDevice.current.systemVersion)")
        startControlServer()
        refresh()
        ticker = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { _ in
            Task { @MainActor in self.refresh() }
        }
    }

    func refresh() {
        telemetry = DeviceTelemetry.shared.snapshot()
        server = LlamaServerRunner.shared.snapshot()
        models = ModelStore.all()
        logLines = LogFileWriter.shared.tail(lines: 14)
    }

    func unload() {
        Task { await LlamaServerRunner.shared.unload() }
    }

    private func startControlServer() {
        let server = ControlServer(port: 8081)
        do {
            try server.start()
            control = server
        } catch {
            controlError = String(describing: error)
            AppLogger.error(.control, "control server failed to start: \(error)")
        }
    }
}
