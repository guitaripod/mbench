#if MBENCHD_MLX
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM
import Tokenizers

/// The sampling a model asks for in its own generation_config.json, used when a request names none of its own.
struct SamplingDefaults: Sendable {
    var temperature: Float = 0.7
    var topP: Float = 1.0
    var topK: Int = 0
    var repetitionPenalty: Float?

    static func read(from directory: URL) -> SamplingDefaults {
        var found = SamplingDefaults()
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("generation_config.json")),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return found }
        if let temperature = json["temperature"] as? Double { found.temperature = Float(temperature) }
        if let topP = json["top_p"] as? Double { found.topP = Float(topP) }
        if let topK = json["top_k"] as? Int { found.topK = topK }
        if let penalty = json["repetition_penalty"] as? Double, penalty != 1 { found.repetitionPenalty = Float(penalty) }
        return found
    }
}

struct Sampling: Sendable {
    var temperature: Float?
    var topP: Float?
    var topK: Int?
    var seed: UInt64?
    var maxTokens: Int?
}

enum ServeError: Error {
    case notLoaded
    case contextExceeded(prompt: Int, limit: Int)
}

enum CompletionEvent: Sendable {
    case reasoning(String)
    case content(String)
    case toolCall(ToolCall)
    case finished(promptTokens: Int, completionTokens: Int, reason: String)
}

/// Splits a model's thinking from its answer the way a server with a reasoning parser does: everything up to
/// `</think>` is reasoning and everything after it is the answer. A model that is already inside a `<think>` block
/// when generation starts — LFM2.5 opens one in its chat template — is told so at load, because the opening tag
/// itself never reaches the stream.
struct ReasoningSplitter {
    private static let open = "<think>"
    private static let close = "</think>"

    private var thinking: Bool
    private var started = false
    private var buffer = ""

    init(open: Bool) {
        thinking = open
    }

    mutating func feed(_ text: String) -> (reasoning: String, content: String) {
        buffer += text
        if !started {
            started = true
            if buffer.hasPrefix(Self.open) {
                buffer.removeFirst(Self.open.count)
                thinking = true
            }
        }
        guard thinking else {
            let content = buffer
            buffer = ""
            return ("", content)
        }
        if let found = buffer.range(of: Self.close) {
            let reasoning = String(buffer[buffer.startIndex ..< found.lowerBound])
            let content = String(buffer[found.upperBound...])
            buffer = ""
            thinking = false
            return (reasoning, content)
        }
        return (released(), "")
    }

    /// Thinking is reported as it is written, holding back only enough characters that a closing tag split across
    /// two chunks is still recognised. Holding all of it back instead would land the whole answer in one delta, and
    /// a client measuring time to first token would measure the whole generation.
    private mutating func released() -> String {
        let keep = Self.close.count - 1
        guard buffer.count > keep else { return "" }
        let cut = buffer.index(buffer.endIndex, offsetBy: -keep)
        let ready = String(buffer[buffer.startIndex ..< cut])
        buffer = String(buffer[cut...])
        return ready
    }

    /// What is still held back, released once the model has stopped: a tail that never closed its thinking is
    /// reasoning, and anything after a closing tag is answer.
    mutating func flush() -> (reasoning: String, content: String) {
        let rest = buffer
        buffer = ""
        return thinking ? (rest, "") : ("", rest)
    }
}

final class MLXRunner: @unchecked Sendable {
    static let shared = MLXRunner()

    static let mlxSwiftLM = "3.31.4"

    private let gate = LoadGate()
    private let lock = NSLock()
    private var container: ModelContainer?
    private var server: InferenceServer?
    private var status = ServerStatus(state: "idle", args: [], engine: "mlx")
    private var defaults = SamplingDefaults()
    private var contextLimit: Int?
    private var reasoningOpen = false

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

    var slots: Int {
        lock.lock()
        defer { lock.unlock() }
        return status.slots ?? 1
    }

    var context: Int? {
        lock.lock()
        defer { lock.unlock() }
        return contextLimit
    }

    var modelName: String {
        lock.lock()
        defer { lock.unlock() }
        return status.model ?? "mlx"
    }

    func load(_ request: LoadRequest) async throws -> ServerStatus {
        try await gate.run { try await self.start(request) }
    }

    func unload() async {
        await gate.run { await self.stop() }
    }

    private func start(_ request: LoadRequest) async throws -> ServerStatus {
        guard let directory = ModelStore.resolveDirectory(request.model) else {
            throw LoadError.unknownModel(request.model)
        }
        await stop()

        let port = request.port ?? 8080
        let started = Date()
        let arguments = ["mlx", "-m", directory.path, "--port", String(port),
                         "-c", String(request.nCtx ?? 0), "-np", String(request.parallel ?? 1)]
        lock.lock()
        status = ServerStatus(state: "loading", model: request.model, modelPath: directory.path, port: port,
                              args: arguments, startedAt: started.timeIntervalSince1970,
                              engine: "mlx", slots: request.parallel ?? 1)
        contextLimit = (request.nCtx ?? 0) > 0 ? request.nCtx : nil
        reasoningOpen = request.reasoningOpen ?? false
        defaults = SamplingDefaults.read(from: directory)
        lock.unlock()

        AppLogger.info(.server, "loading MLX model: \(arguments.joined(separator: " "))")
        MLXRunner.holdMemoryBelowTheLimit()
        do {
            let loaded = try await MLXRunner.container(for: directory)
            let listening = InferenceServer(port: UInt16(port))
            try listening.start()
            lock.lock()
            container = loaded
            server = listening
            status.state = "running"
            status.loadSeconds = round(Date().timeIntervalSince(started) * 100) / 100
            let current = status
            lock.unlock()
            AppLogger.info(.server, "MLX model ready in \(current.loadSeconds ?? 0)s")
            return current
        } catch {
            lock.lock()
            status = ServerStatus(state: "failed", args: arguments, error: "\(error)", engine: "mlx")
            lock.unlock()
            AppLogger.error(.server, "MLX load failed: \(error)")
            throw error
        }
    }

    /// iOS kills an app that asks for more memory than it is given, and MLX will happily ask: it keeps a buffer
    /// cache and allocates whatever a prompt needs. A ceiling well under what the app is allowed turns that into a
    /// failed request — which a run records and reports — instead of a dead app and a lost run.
    private static func holdMemoryBelowTheLimit() {
        MLX.GPU.set(cacheLimit: 32 * 1024 * 1024)
        MLX.GPU.set(memoryLimit: memoryCeiling, relaxed: false)
        MLX.GPU.clearCache()
    }

    private static var memoryCeiling: Int {
        let available = Int(DeviceTelemetry.shared.snapshot().availableMib * 1024 * 1024)
        return max(768 * 1024 * 1024, Int(Double(available) * 0.7))
    }

    /// Text models load through the LLM factory and multimodal ones through the VLM factory. A folder says which
    /// it is by whether a processor sits beside the weights, so neither factory's failure is ever hidden behind the
    /// other's.
    private static func container(for directory: URL) async throws -> ModelContainer {
        let loader = TransformersTokenizerLoader()
        if multimodal(directory) {
            return try await VLMModelFactory.shared.loadContainer(from: directory, using: loader)
        }
        return try await LLMModelFactory.shared.loadContainer(from: directory, using: loader)
    }

    private static func multimodal(_ directory: URL) -> Bool {
        ["processor_config.json", "preprocessor_config.json"].contains {
            FileManager.default.fileExists(atPath: directory.appendingPathComponent($0).path)
        }
    }

    private func stop() async {
        lock.lock()
        let listening = server
        let loaded = container
        server = nil
        container = nil
        status = ServerStatus(state: "idle", args: [], engine: "mlx")
        lock.unlock()
        guard listening != nil || loaded != nil else { return }
        listening?.stop()
        MLX.GPU.clearCache()
        AppLogger.info(.server, "MLX model unloaded")
    }

    /// Runs one completion, reporting the answer as it arrives. Thinking and answer are reported apart, so a client
    /// reads this server the way it reads llama-server with a reasoning parser.
    func complete(messages: [Chat.Message], tools: [ToolSpec]?, sampling: Sampling,
                  emit: (CompletionEvent) async -> Void) async throws {
        lock.lock()
        let loaded = container
        let limit = contextLimit
        let settings = defaults
        let open = reasoningOpen
        lock.unlock()
        guard let loaded else { throw ServeError.notLoaded }

        let input = try await loaded.prepare(input: UserInput(chat: messages, tools: tools))
        let promptTokens = input.text.tokens.size
        if let limit, promptTokens > limit {
            throw ServeError.contextExceeded(prompt: promptTokens, limit: limit)
        }

        let parameters = GenerateParameters(
            maxTokens: sampling.maxTokens,
            maxKVSize: limit,
            temperature: sampling.temperature ?? settings.temperature,
            topP: sampling.topP ?? settings.topP,
            topK: sampling.topK ?? settings.topK,
            repetitionPenalty: settings.repetitionPenalty,
            prefillStepSize: 256,
            seed: sampling.seed)

        var splitter = ReasoningSplitter(open: open)
        var completionTokens = 0
        var reason = "stop"
        let stream = try await loaded.generate(input: input, parameters: parameters)
        for await generation in stream {
            switch generation {
            case .chunk(let text):
                let split = splitter.feed(text)
                if !split.reasoning.isEmpty { await emit(.reasoning(split.reasoning)) }
                if !split.content.isEmpty { await emit(.content(split.content)) }
            case .toolCall(let call):
                await emit(.toolCall(call))
                reason = "tool_calls"
            case .info(let info):
                completionTokens = info.generationTokenCount
                if info.stopReason == .length { reason = "length" }
            }
        }
        MLX.GPU.clearCache()
        let tail = splitter.flush()
        if !tail.reasoning.isEmpty { await emit(.reasoning(tail.reasoning)) }
        if !tail.content.isEmpty { await emit(.content(tail.content)) }
        await emit(.finished(promptTokens: promptTokens, completionTokens: completionTokens, reason: reason))
    }
}
#endif
