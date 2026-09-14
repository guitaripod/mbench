import Foundation

struct ServerMetrics: Sendable {
    var decodeTokensPerSecond: Double?
    var prefillTokensPerSecond: Double?
    var requestsProcessing: Double?
    var requestsDeferred: Double?
    var kvCacheRatio: Double?
    var tokensPredicted: Double?

    var isBusy: Bool { (requestsProcessing ?? 0) > 0 }
}

actor MetricsReader {
    static let shared = MetricsReader()

    private var session: URLSession = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 3
        return URLSession(configuration: configuration)
    }()

    func read(port: Int) async -> ServerMetrics? {
        guard let url = URL(string: "http://127.0.0.1:\(port)/metrics"),
              let (data, response) = try? await session.data(from: url),
              (response as? HTTPURLResponse)?.statusCode == 200,
              let text = String(data: data, encoding: .utf8) else { return nil }
        return MetricsReader.parse(text)
    }

    /// llama-server publishes Prometheus text: one `name value` line per sample, comments first.
    static func parse(_ text: String) -> ServerMetrics {
        var values: [String: Double] = [:]
        for line in text.split(separator: "\n") where !line.hasPrefix("#") {
            let parts = line.split(separator: " ")
            guard parts.count >= 2, let value = Double(parts[parts.count - 1]) else { continue }
            values[String(parts[0]).replacingOccurrences(of: "llamacpp:", with: "")] = value
        }
        return ServerMetrics(
            decodeTokensPerSecond: values["predicted_tokens_seconds"],
            prefillTokensPerSecond: values["prompt_tokens_seconds"],
            requestsProcessing: values["requests_processing"],
            requestsDeferred: values["requests_deferred"],
            kvCacheRatio: values["kv_cache_usage_ratio"],
            tokensPredicted: values["tokens_predicted_total"]
        )
    }
}
