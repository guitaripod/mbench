import Foundation
import LlamaServerBridge

struct LoadRequest: Codable, Sendable {
    var model: String
    var nCtx: Int?
    var parallel: Int?
    var nGpuLayers: Int?
    var flashAttn: String?
    var cacheTypeK: String?
    var cacheTypeV: String?
    var jinja: Bool?
    var reasoningFormat: String?
    var chatTemplate: String?
    var port: Int?
    var extraArgs: [String]?
    var engine: String?
    var reasoningOpen: Bool?
}

struct ServerStatus: Codable, Sendable {
    var state: String
    var model: String?
    var modelPath: String?
    var port: Int?
    var args: [String]
    var loadSeconds: Double?
    var startedAt: Double?
    var modelSha256: String?
    var exitCode: Int32?
    var error: String?
    var engine: String?
    var slots: Int?
}

enum LoadError: Error {
    case unknownModel(String)
    case timedOut(Double)
    case exited(Int32)
}

actor LoadGate {
    /// Serialises load and unload. Two control requests arriving together — a run finishing as the next one starts —
    /// used to race inside the runner and drop the connection, which the host could only read as the phone dying.
    func run<T>(_ body: () async throws -> T) async rethrows -> T {
        try await body()
    }
}

final class LlamaServerRunner: @unchecked Sendable {
    static let shared = LlamaServerRunner()

    private let gate = LoadGate()
    private let lock = NSLock()
    private var status = ServerStatus(state: "idle", args: [], engine: "llama.cpp")
    private var thread: Thread?
    private var finished = DispatchSemaphore(value: 0)
    private var argumentStorage: [UnsafeMutablePointer<CChar>?] = []

    private init() {}

    func snapshot() -> ServerStatus {
        lock.lock()
        defer { lock.unlock() }
        return status
    }

    var isRunning: Bool {
        lock.lock()
        defer { lock.unlock() }
        return status.state == "running" || status.state == "loading"
    }

    func load(_ request: LoadRequest) async throws -> ServerStatus {
        try await gate.run { try await self.start(request) }
    }

    func unload() async {
        await gate.run { await self.stop() }
    }

    private func start(_ request: LoadRequest) async throws -> ServerStatus {
        guard let model = ModelStore.resolve(request.model) else { throw LoadError.unknownModel(request.model) }
        await stop()

        let port = request.port ?? 8080
        let arguments = LlamaServerRunner.arguments(for: request, model: model, port: port)
        let started = Date()

        let digest = ModelDigest.of(model)
        lock.lock()
        status = ServerStatus(state: "loading", model: request.model, modelPath: model.path, port: port,
                              args: arguments, startedAt: started.timeIntervalSince1970, modelSha256: digest,
                              engine: "llama.cpp", slots: request.parallel ?? 1)
        finished = DispatchSemaphore(value: 0)
        lock.unlock()

        AppLogger.info(.server, "starting llama-server: \(arguments.joined(separator: " "))")
        spawn(arguments: arguments)

        try await waitUntilReady(port: port, deadline: started.addingTimeInterval(3600))

        lock.lock()
        status.state = "running"
        status.loadSeconds = round(Date().timeIntervalSince(started) * 100) / 100
        let current = status
        lock.unlock()
        AppLogger.info(.server, "llama-server ready in \(current.loadSeconds ?? 0)s")
        return current
    }

    private func stop() async {
        lock.lock()
        let running = thread != nil
        let semaphore = finished
        lock.unlock()
        guard running else { return }

        AppLogger.info(.server, "stopping llama-server")
        lock.lock()
        status.state = "stopping"
        lock.unlock()

        for _ in 0..<60 {
            mb_server_stop()
            if semaphore.wait(timeout: .now() + 1) == .success {
                break
            }
        }
        lock.lock()
        thread = nil
        argumentStorage.forEach { free($0) }
        argumentStorage = []
        status = ServerStatus(state: "idle", args: [], exitCode: status.exitCode, engine: "llama.cpp")
        lock.unlock()
        AppLogger.info(.server, "llama-server stopped")
    }

    /// Runs llama-server's own entry point on a thread of its own, keeping the C strings it parsed alive for
    /// as long as it runs.
    private func spawn(arguments: [String]) {
        let pointers: [UnsafeMutablePointer<CChar>?] = arguments.map { strdup($0) }
        lock.lock()
        argumentStorage = pointers
        lock.unlock()

        let worker = Thread { [weak self] in
            let code = pointers.withUnsafeBufferPointer { buffer in
                mb_server_run(Int32(buffer.count), buffer.baseAddress)
            }
            AppLogger.info(.server, "llama-server exited with \(code)")
            guard let self else { return }
            self.lock.lock()
            self.status.exitCode = code
            if self.status.state != "stopping" {
                self.status.state = code == 0 ? "idle" : "failed"
                self.status.error = code == 0 ? nil : "llama-server exited with \(code)"
            }
            let semaphore = self.finished
            self.lock.unlock()
            semaphore.signal()
        }
        worker.name = "llama-server"
        worker.stackSize = 8 * 1024 * 1024
        worker.qualityOfService = QualityOfService.userInitiated
        lock.lock()
        thread = worker
        lock.unlock()
        worker.start()
    }

    private func waitUntilReady(port: Int, deadline: Date) async throws {
        while Date() < deadline {
            lock.lock()
            let state = status.state
            let code = status.exitCode
            lock.unlock()
            if state == "failed" || (state != "loading" && state != "running") {
                throw LoadError.exited(code ?? -1)
            }
            if await LlamaServerRunner.probe(port: port) { return }
            try? await Task.sleep(nanoseconds: 500_000_000)
        }
        throw LoadError.timedOut(deadline.timeIntervalSinceNow)
    }

    private static func probe(port: Int) async -> Bool {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/health")!)
        request.timeoutInterval = 5
        guard let (_, response) = try? await URLSession.shared.data(for: request) else { return false }
        return (response as? HTTPURLResponse)?.statusCode == 200
    }

    private static func arguments(for request: LoadRequest, model: URL, port: Int) -> [String] {
        var arguments = [
            "llama-server",
            "-m", model.path,
            "--host", "127.0.0.1",
            "--port", String(port),
            "-ngl", String(request.nGpuLayers ?? 999),
            "-c", String(request.nCtx ?? 8192),
            "-np", String(request.parallel ?? 1),
            "-fa", request.flashAttn ?? "on",
            "--alias", request.model,
            "--metrics",
            "--slots",
            "--no-ui",
            "--log-file", ModelStore.logDirectory.appendingPathComponent("llama-server.log").path,
        ]
        if request.jinja ?? true { arguments.append("--jinja") }
        if let format = request.reasoningFormat { arguments += ["--reasoning-format", format] }
        if let template = request.chatTemplate { arguments += ["--chat-template", template] }
        if let key = request.cacheTypeK { arguments += ["-ctk", key] }
        if let value = request.cacheTypeV { arguments += ["-ctv", value] }
        arguments += request.extraArgs ?? []
        return arguments
    }
}
