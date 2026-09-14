import Foundation

final class LogFileWriter: @unchecked Sendable {
    static let shared = LogFileWriter(name: "mbenchd")

    private let queue = DispatchQueue(label: "mbenchd.log", qos: .utility)
    private let current: URL
    private let previous: URL
    private let limit = 4 * 1024 * 1024
    private let formatter: DateFormatter

    init(name: String) {
        let logs = LogFileWriter.directory()
        current = logs.appendingPathComponent("\(name).log")
        previous = logs.appendingPathComponent("\(name).previous.log")
        formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
        formatter.timeZone = TimeZone(identifier: "UTC")
        try? FileManager.default.createDirectory(at: logs, withIntermediateDirectories: true)
    }

    var path: String { current.path }

    func write(_ level: String, _ category: String, _ message: String) {
        let stamp = formatter.string(from: Date())
        queue.async { [self] in
            append("\(stamp) [\(level)] [\(category)] \(message)\n")
        }
    }

    func tail(lines: Int) -> [String] {
        queue.sync {
            guard let text = try? String(contentsOf: current, encoding: .utf8) else { return [] }
            return Array(text.split(separator: "\n").suffix(lines).map(String.init))
        }
    }

    /// Appends to the current file, rotating it aside once it passes the size limit so a long run keeps
    /// the newest lines without growing without bound.
    private func append(_ line: String) {
        guard let data = line.data(using: .utf8) else { return }
        let manager = FileManager.default
        if let size = try? manager.attributesOfItem(atPath: current.path)[.size] as? Int, size > limit {
            try? manager.removeItem(at: previous)
            try? manager.moveItem(at: current, to: previous)
        }
        guard let handle = try? FileHandle(forWritingTo: current) else {
            try? data.write(to: current)
            return
        }
        defer { try? handle.close() }
        try? handle.seekToEnd()
        try? handle.write(contentsOf: data)
    }

    private static func directory() -> URL {
        let library = FileManager.default.urls(for: .libraryDirectory, in: .userDomainMask)[0]
        return library.appendingPathComponent("Logs")
    }
}
