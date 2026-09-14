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
        case ("POST", "/mb/unload"), ("GET", "/unload"):
            await LlamaServerRunner.shared.unload()
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
                modelsPath: ModelStore.directory.path
            ),
            device: DeviceInfo(
                hardware: DeviceTelemetry.hardware(),
                system: "\(UIDevice.current.systemName) \(UIDevice.current.systemVersion)",
                processors: ProcessInfo.processInfo.processorCount
            ),
            telemetry: DeviceTelemetry.shared.snapshot(),
            server: LlamaServerRunner.shared.snapshot()
        )
    }

    private static func load(_ request: HTTPRequest) async -> HTTPResponse {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        guard let payload = try? decoder.decode(LoadRequest.self, from: request.body) else {
            return HTTPResponse.error("bad load request", status: 400)
        }
        do {
            let status = try await LlamaServerRunner.shared.load(payload)
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
        let status = LlamaServerRunner.shared.snapshot()
        guard status.state == "running", let model = status.model else { return [] }
        return [["model": model, "state": "ready"]]
    }

    private static func path(of request: HTTPRequest) -> String {
        String(request.path.split(separator: "?").first ?? "")
    }
}
