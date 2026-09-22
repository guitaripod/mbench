import Foundation
import LlamaServerBridge
import UIKit

struct AppInfo: Codable, Sendable {
    let version: String
    let llamaCommit: String
    let llamaBuild: Int32
    let logPath: String
    let serverLogPath: String
    let modelsPath: String
    let engines: [String]
}

struct DeviceInfo: Codable, Sendable {
    let hardware: String
    let system: String
    let processors: Int
}

struct ModelList: Codable, Sendable {
    let object: String
    let data: [StoredModel]
}

struct HealthPayload: Codable, Sendable {
    let app: AppInfo
    let device: DeviceInfo
    let telemetry: TelemetrySnapshot
    let server: ServerStatus
    let progress: RunProgress
}

enum Router {
    static let version = "0.1.0"

    static func handle(_ request: HTTPRequest) async -> HTTPResponse {
        AppLogger.debug(.control, "\(request.method) \(request.path)")
        switch (request.method, path(of: request)) {
        case ("GET", "/mb/health"), ("GET", "/health"):
            return HTTPResponse.json(health())
        case ("GET", "/mb/models"), ("GET", "/v1/models"):
            return HTTPResponse.json(ModelList(object: "list", data: ModelStore.all()))
        case ("POST", "/mb/load"):
            return await load(request)
        case ("POST", "/mb/progress"):
            return progress(request)
        case ("POST", "/mb/unload"), ("GET", "/unload"):
            await unloadAll()
            return HTTPResponse.json(["ok": true])
        case ("GET", "/running"):
            return HTTPResponse.json(["running": running()])
        default:
            return HTTPResponse.error("no route for \(request.method) \(request.path)", status: 404)
        }
    }

    static func health() -> HealthPayload {
        HealthPayload(
            app: AppInfo(
                version: version,
                llamaCommit: String(cString: mb_llama_commit()),
                llamaBuild: mb_llama_build_number(),
                logPath: LogFileWriter.shared.path,
                serverLogPath: ModelStore.logDirectory.appendingPathComponent("llama-server.log").path,
                modelsPath: ModelStore.directory.path,
                engines: engines
            ),
            device: DeviceInfo(
                hardware: DeviceTelemetry.hardware(),
                system: "\(UIDevice.current.systemName) \(UIDevice.current.systemVersion)",
                processors: ProcessInfo.processInfo.processorCount
            ),
            telemetry: DeviceTelemetry.shared.snapshot(),
            server: active(),
            progress: ProgressStore.shared.snapshot()
        )
    }

    /// What the host says its run is doing. Nothing here changes what the phone measures; it is what the screen
    /// shows while a run is under way.
    private static func progress(_ request: HTTPRequest) -> HTTPResponse {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        guard let reported = try? decoder.decode(RunProgress.self, from: request.body) else {
            return HTTPResponse.error("bad progress report", status: 400)
        }
        ProgressStore.shared.record(reported)
        return HTTPResponse.json(["ok": true])
    }

    /// The engine that is loaded, or llama.cpp when nothing is: a run reads one server status, whichever engine
    /// answered it.
    static func active() -> ServerStatus {
        #if MBENCHD_MLX
        if MLXRunner.shared.isRunning { return MLXRunner.shared.snapshot() }
        #endif
        return LlamaServerRunner.shared.snapshot()
    }

    static var engines: [String] {
        #if MBENCHD_MLX
        return ["llama.cpp", "mlx"]
        #else
        return ["llama.cpp"]
        #endif
    }

    /// Loads a model into the engine the request names, leaving nothing of the other engine behind: two engines
    /// cannot hold the port, and a phone has room for one model at a time.
    static func loadEngine(_ request: LoadRequest) async throws -> ServerStatus {
        #if MBENCHD_MLX
        if request.engine == "mlx" {
            await LlamaServerRunner.shared.unload()
            return try await MLXRunner.shared.load(request)
        }
        await MLXRunner.shared.unload()
        #endif
        return try await LlamaServerRunner.shared.load(request)
    }

    static func unloadAll() async {
        ProgressStore.shared.clear()
        await LlamaServerRunner.shared.unload()
        #if MBENCHD_MLX
        await MLXRunner.shared.unload()
        #endif
    }

    private static func load(_ request: HTTPRequest) async -> HTTPResponse {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        guard let payload = try? decoder.decode(LoadRequest.self, from: request.body) else {
            return HTTPResponse.error("bad load request", status: 400)
        }
        do {
            let status = try await loadEngine(payload)
            return HTTPResponse.json(status)
        } catch let error as LoadError {
            AppLogger.error(.server, "load failed: \(error)")
            return HTTPResponse.error(String(describing: error), status: 400)
        } catch {
            AppLogger.error(.server, "load failed: \(error)")
            return HTTPResponse.error(error.localizedDescription, status: 500)
        }
    }

    private static func running() -> [[String: String]] {
        let status = active()
        guard status.state == "running", let model = status.model else { return [] }
        return [["model": model, "state": "ready"]]
    }

    private static func path(of request: HTTPRequest) -> String {
        String(request.path.split(separator: "?").first ?? "")
    }
}
