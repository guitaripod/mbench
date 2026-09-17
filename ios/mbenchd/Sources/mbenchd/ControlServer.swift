import Foundation
import Network
import UIKit

struct HTTPRequest {
    let method: String
    let path: String
    let body: Data
}

struct HTTPResponse {
    var status: Int = 200
    var body: Data = Data()
    var contentType = "application/json"

    static func json(_ value: some Encodable, status: Int = 200) -> HTTPResponse {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let body = (try? encoder.encode(value)) ?? Data("{}".utf8)
        return HTTPResponse(status: status, body: body)
    }

    static func error(_ message: String, status: Int) -> HTTPResponse {
        HTTPResponse.json(["error": ["message": message]], status: status)
    }
}

final class ControlServer: @unchecked Sendable {
    private let port: NWEndpoint.Port
    private let queue = DispatchQueue(label: "mbenchd.control")
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
            AppLogger.info(.control, "control listener \(String(describing: state))")
        }
        listener.start(queue: queue)
        self.listener = listener
        AppLogger.info(.control, "control server listening on \(port.rawValue)")
    }

    private func accept(_ connection: NWConnection) {
        connection.start(queue: queue)
        receive(connection, buffer: Data())
    }

    /// Reads until the headers are complete and the declared body has arrived, so a POST that spans several
    /// TCP reads is still handled as one request.
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
            Task {
                let response = await Router.handle(request)
                self.send(response, on: connection)
            }
        }
    }

    private func send(_ response: HTTPResponse, on connection: NWConnection) {
        var head = "HTTP/1.1 \(response.status) \(response.status == 200 ? "OK" : "Error")\r\n"
        head += "Content-Type: \(response.contentType)\r\n"
        head += "Content-Length: \(response.body.count)\r\n"
        head += "Connection: close\r\n\r\n"
        var payload = Data(head.utf8)
        payload.append(response.body)
        connection.send(content: payload, completion: .contentProcessed { _ in connection.cancel() })
    }

    static func parse(_ data: Data) -> HTTPRequest? {
        guard let separator = data.range(of: Data("\r\n\r\n".utf8)) else { return nil }
        let header = String(decoding: data[..<separator.lowerBound], as: UTF8.self)
        let lines = header.split(separator: "\r\n", omittingEmptySubsequences: false)
        guard let request = lines.first?.split(separator: " "), request.count >= 2 else { return nil }
        let length = lines.compactMap { line -> Int? in
            let parts = line.split(separator: ":", maxSplits: 1)
            guard parts.count == 2, parts[0].lowercased() == "content-length" else { return nil }
            return Int(parts[1].trimmingCharacters(in: .whitespaces))
        }.first ?? 0
        let body = data[separator.upperBound...]
        guard body.count >= length else { return nil }
        return HTTPRequest(method: String(request[0]), path: String(request[1]), body: Data(body.prefix(length)))
    }
}
