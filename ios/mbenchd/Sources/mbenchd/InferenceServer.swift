#if MBENCHD_MLX
import Foundation
import MLXLMCommon
import Network

/// The OpenAI-shaped server in front of the MLX engine. llama.cpp brings its own; MLX is a library, so this is the
/// part of a run that goes through our code — it streams each answer as it is generated and reports the token counts
/// the engine measured, never its own.
final class InferenceServer: @unchecked Sendable {
    private let port: NWEndpoint.Port
    private let queue = DispatchQueue(label: "mbenchd.inference", attributes: .concurrent)
    private var listener: NWListener?

    init(port: UInt16) {
        self.port = NWEndpoint.Port(rawValue: port)!
    }

    func start() throws {
        let parameters = NWParameters.tcp
        parameters.allowLocalEndpointReuse = true
        let listener = try NWListener(using: parameters, on: port)
        listener.newConnectionHandler = { [weak self] connection in
            self?.accept(connection)
        }
        listener.stateUpdateHandler = { state in
            AppLogger.info(.server, "inference listener \(String(describing: state))")
        }
        listener.start(queue: queue)
        self.listener = listener
        AppLogger.info(.server, "inference server listening on \(port.rawValue)")
    }

    func stop() {
        listener?.cancel()
        listener = nil
        AppLogger.info(.server, "inference server stopped")
    }

    private func accept(_ connection: NWConnection) {
        connection.start(queue: queue)
        receive(connection, buffer: Data())
    }

    private func receive(_ connection: NWConnection, buffer: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 1 << 16) { [weak self] data, _, complete, error in
            guard let self else { return }
            var current = buffer
            if let data { current.append(data) }
            if error != nil || (complete && current.isEmpty) {
                connection.cancel()
                return
            }
            guard let request = ControlServer.parse(current) else {
                if complete { connection.cancel() } else { self.receive(connection, buffer: current) }
                return
            }
            Task { await self.handle(request, on: connection) }
        }
    }

    private func handle(_ request: HTTPRequest, on connection: NWConnection) async {
        let path = String(request.path.split(separator: "?").first ?? "")
        AppLogger.debug(.server, "\(request.method) \(path)")
        switch (request.method, path) {
        case ("GET", "/health"):
            await send(HTTPResponse.json(["status": "ok"]), on: connection)
        case ("GET", "/props"):
            await send(HTTPResponse.json(InferenceServer.props()), on: connection)
        case ("GET", "/v1/models"), ("GET", "/models"):
            await send(HTTPResponse.json(InferenceServer.models()), on: connection)
        case ("POST", "/v1/chat/completions"), ("POST", "/chat/completions"):
            await complete(request, on: connection)
        default:
            await send(HTTPResponse.error("no route for \(request.method) \(path)", status: 404), on: connection)
        }
    }

    private static func props() -> [String: JSONAny] {
        let runner = MLXRunner.shared
        return [
            "version": .string(MLXRunner.mlxSwiftLM),
            "build_info": .string("mlx-swift-lm \(MLXRunner.mlxSwiftLM)"),
            "model_path": .string(runner.snapshot().modelPath ?? ""),
            "total_slots": .int(runner.slots),
            "default_generation_settings": .object(["n_ctx": .int(runner.context ?? 0)]),
        ]
    }

    private static func models() -> [String: JSONAny] {
        let runner = MLXRunner.shared
        return [
            "object": .string("list"),
            "data": .array([.object([
                "id": .string(runner.modelName),
                "object": .string("model"),
                "max_model_len": .int(runner.context ?? 0),
            ])]),
        ]
    }

    private func complete(_ request: HTTPRequest, on connection: NWConnection) async {
        guard let body = try? JSONSerialization.jsonObject(with: request.body) as? [String: Any] else {
            await send(HTTPResponse.error("the request body is not JSON", status: 400), on: connection)
            return
        }
        let messages = InferenceServer.messages(from: body["messages"])
        guard !messages.isEmpty else {
            await send(HTTPResponse.error("no messages", status: 400), on: connection)
            return
        }
        let tools = (body["tools"] as? [Any])?.compactMap { InferenceServer.sendable($0) as? ToolSpec }
        let sampling = InferenceServer.sampling(from: body)
        let identifier = "chatcmpl-" + UUID().uuidString.prefix(20)
        let model = body["model"] as? String ?? MLXRunner.shared.modelName
        let created = Int(Date().timeIntervalSince1970)
        let streaming = body["stream"] as? Bool ?? false
        let wantsUsage = ((body["stream_options"] as? [String: Any])?["include_usage"] as? Bool) ?? false

        if streaming {
            await stream(messages: messages, tools: tools, sampling: sampling, identifier: identifier,
                         model: model, created: created, wantsUsage: wantsUsage, on: connection)
        } else {
            await whole(messages: messages, tools: tools, sampling: sampling, identifier: identifier,
                        model: model, created: created, on: connection)
        }
    }

    private func whole(messages: [Chat.Message], tools: [ToolSpec]?, sampling: Sampling, identifier: String,
                       model: String, created: Int, on connection: NWConnection) async {
        var reasoning = ""
        var content = ""
        var calls: [ToolCall] = []
        var usage = (prompt: 0, completion: 0)
        var reason = "stop"
        do {
            try await MLXRunner.shared.complete(messages: messages, tools: tools, sampling: sampling) { event in
                switch event {
                case .reasoning(let text): reasoning += text
                case .content(let text): content += text
                case .toolCall(let call): calls.append(call)
                case .finished(let prompt, let completion, let stop):
                    usage = (prompt, completion)
                    reason = stop
                }
            }
        } catch {
            await send(InferenceServer.failure(error), on: connection)
            return
        }
        var message: [String: JSONAny] = ["role": .string("assistant"), "content": .string(content)]
        if !reasoning.isEmpty { message["reasoning_content"] = .string(reasoning) }
        if !calls.isEmpty { message["tool_calls"] = .array(calls.enumerated().map(InferenceServer.toolCallPayload)) }
        let payload: [String: JSONAny] = [
            "id": .string(identifier),
            "object": .string("chat.completion"),
            "created": .int(created),
            "model": .string(model),
            "choices": .array([.object([
                "index": .int(0),
                "message": .object(message),
                "finish_reason": .string(reason),
            ])]),
            "usage": .object(InferenceServer.usagePayload(prompt: usage.prompt, completion: usage.completion)),
        ]
        await send(HTTPResponse.json(payload), on: connection)
    }

    private func stream(messages: [Chat.Message], tools: [ToolSpec]?, sampling: Sampling, identifier: String,
                        model: String, created: Int, wantsUsage: Bool, on connection: NWConnection) async {
        var headed = false
        var calls: [ToolCall] = []
        var usage = (prompt: 0, completion: 0)
        var reason = "stop"

        func head() async {
            guard !headed else { return }
            headed = true
            var lines = "HTTP/1.1 200 OK\r\n"
            lines += "Content-Type: text/event-stream\r\n"
            lines += "Cache-Control: no-cache\r\n"
            lines += "Connection: close\r\n\r\n"
            _ = await write(connection, Data(lines.utf8))
        }

        func event(_ payload: [String: JSONAny]) async {
            await head()
            guard let data = try? JSONEncoder().encode(payload) else { return }
            var line = Data("data: ".utf8)
            line.append(data)
            line.append(Data("\n\n".utf8))
            _ = await write(connection, line)
        }

        func delta(_ fields: [String: JSONAny], finish: String?) -> [String: JSONAny] {
            [
                "id": .string(identifier),
                "object": .string("chat.completion.chunk"),
                "created": .int(created),
                "model": .string(model),
                "choices": .array([.object([
                    "index": .int(0),
                    "delta": .object(fields),
                    "finish_reason": finish.map { JSONAny.string($0) } ?? .null,
                ])]),
            ]
        }

        do {
            try await MLXRunner.shared.complete(messages: messages, tools: tools, sampling: sampling) { generated in
                switch generated {
                case .reasoning(let text):
                    await event(delta(["reasoning_content": .string(text)], finish: nil))
                case .content(let text):
                    await event(delta(["content": .string(text)], finish: nil))
                case .toolCall(let call):
                    calls.append(call)
                    await event(delta(["tool_calls": .array([InferenceServer.toolCallPayload(
                        index: calls.count - 1, call: call)])], finish: nil))
                case .finished(let prompt, let completion, let stop):
                    usage = (prompt, completion)
                    reason = stop
                }
            }
        } catch {
            if headed {
                await event(delta([:], finish: "error"))
            } else {
                await send(InferenceServer.failure(error), on: connection)
                return
            }
        }
        await event(delta([:], finish: reason))
        if wantsUsage {
            await event([
                "id": .string(identifier),
                "object": .string("chat.completion.chunk"),
                "created": .int(created),
                "model": .string(model),
                "choices": .array([]),
                "usage": .object(InferenceServer.usagePayload(prompt: usage.prompt, completion: usage.completion)),
            ])
        }
        await head()
        _ = await write(connection, Data("data: [DONE]\n\n".utf8))
        connection.cancel()
    }

    private static func usagePayload(prompt: Int, completion: Int) -> [String: JSONAny] {
        ["prompt_tokens": .int(prompt), "completion_tokens": .int(completion),
         "total_tokens": .int(prompt + completion)]
    }

    private static func toolCallPayload(index: Int, call: ToolCall) -> JSONAny {
        let arguments = (try? JSONEncoder().encode(call.function.arguments)).map { String(decoding: $0, as: UTF8.self) }
        return .object([
            "index": .int(index),
            "id": .string(call.id ?? "call_\(index)"),
            "type": .string("function"),
            "function": .object([
                "name": .string(call.function.name),
                "arguments": .string(arguments ?? "{}"),
            ]),
        ])
    }

    private static func failure(_ error: Error) -> HTTPResponse {
        if case ServeError.contextExceeded(let prompt, let limit) = error {
            return HTTPResponse.error("Context size has been exceeded: \(prompt) tokens for a \(limit)-token context",
                                      status: 400)
        }
        return HTTPResponse.error("\(error)", status: 500)
    }

    private static func sampling(from body: [String: Any]) -> Sampling {
        var sampling = Sampling()
        sampling.maxTokens = body["max_tokens"] as? Int ?? body["max_completion_tokens"] as? Int
        if let temperature = body["temperature"] as? Double { sampling.temperature = Float(temperature) }
        if let topP = body["top_p"] as? Double { sampling.topP = Float(topP) }
        if let topK = body["top_k"] as? Int { sampling.topK = topK }
        if let seed = body["seed"] as? Int, seed >= 0 { sampling.seed = UInt64(seed) }
        return sampling
    }

    private static func messages(from value: Any?) -> [Chat.Message] {
        guard let entries = value as? [[String: Any]] else { return [] }
        return entries.map { entry in
            let content = text(of: entry["content"])
            switch entry["role"] as? String {
            case "system":
                return .system(content)
            case "assistant":
                let calls = (entry["tool_calls"] as? [[String: Any]])?.compactMap(toolCall)
                return .assistant(content, toolCalls: calls?.isEmpty == false ? calls : nil)
            case "tool":
                return .tool(content, id: entry["tool_call_id"] as? String)
            default:
                return .user(content)
            }
        }
    }

    /// A message's content is a string in every request mbench sends, and a list of parts in the wider OpenAI
    /// shape; both are read, so a client that sends parts is answered rather than refused.
    private static func text(of value: Any?) -> String {
        if let string = value as? String { return string }
        guard let parts = value as? [[String: Any]] else { return "" }
        return parts.compactMap { $0["text"] as? String }.joined()
    }

    private static func toolCall(_ entry: [String: Any]) -> ToolCall? {
        guard let function = entry["function"] as? [String: Any], let name = function["name"] as? String else {
            return nil
        }
        var arguments: [String: any Sendable] = [:]
        if let raw = function["arguments"] as? String, let data = raw.data(using: .utf8),
           let parsed = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            arguments = parsed.mapValues { sendable($0) }
        } else if let parsed = function["arguments"] as? [String: Any] {
            arguments = parsed.mapValues { sendable($0) }
        }
        return ToolCall(function: .init(name: name, arguments: arguments), id: entry["id"] as? String)
    }

    /// JSONSerialization hands back `Any`; a chat template is given `Sendable` values, so the tree is retyped once
    /// on the way in rather than force-cast at every use.
    private static func sendable(_ value: Any) -> any Sendable {
        switch value {
        case let dictionary as [String: Any]:
            return dictionary.mapValues { sendable($0) }
        case let array as [Any]:
            return array.map { sendable($0) }
        case let string as String:
            return string
        case let number as NSNumber:
            if CFGetTypeID(number) == CFBooleanGetTypeID() { return number.boolValue }
            return CFNumberIsFloatType(number) ? number.doubleValue : number.intValue
        default:
            return String(describing: value)
        }
    }

    private func send(_ response: HTTPResponse, on connection: NWConnection) async {
        var head = "HTTP/1.1 \(response.status) \(response.status == 200 ? "OK" : "Error")\r\n"
        head += "Content-Type: \(response.contentType)\r\n"
        head += "Content-Length: \(response.body.count)\r\n"
        head += "Connection: close\r\n\r\n"
        var payload = Data(head.utf8)
        payload.append(response.body)
        _ = await write(connection, payload)
        connection.cancel()
    }

    private func write(_ connection: NWConnection, _ data: Data) async -> Bool {
        await withCheckedContinuation { continuation in
            connection.send(content: data, completion: .contentProcessed { error in
                continuation.resume(returning: error == nil)
            })
        }
    }
}

/// A JSON value built by hand, so a response can carry the fields OpenAI clients expect without a struct for
/// every shape.
enum JSONAny: Encodable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([JSONAny])
    case object([String: JSONAny])

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null: try container.encodeNil()
        case .bool(let value): try container.encode(value)
        case .int(let value): try container.encode(value)
        case .double(let value): try container.encode(value)
        case .string(let value): try container.encode(value)
        case .array(let value): try container.encode(value)
        case .object(let value): try container.encode(value)
        }
    }
}
#endif
