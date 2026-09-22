import Foundation

/// Where a benchmark has got to, as the host running it reports. The phone measures the model but never drives the
/// run, so without this the screen can only say that something is loaded.
struct RunProgress: Codable, Sendable {
    var run: String?
    var model: String?
    var suite: String?
    var phase: String?
    var done: Int?
    var total: Int?
    var note: String?
    var startedAt: Double?
    var updatedAt: Double?

    var share: Double? {
        guard let done, let total, total > 0 else { return nil }
        return min(1, Double(done) / Double(total))
    }

    /// A report stops counting as news once nothing has arrived for a while: a run that was cancelled on the host
    /// leaves its last phase behind, and a stale phase on screen reads as a run still going.
    var fresh: Bool {
        guard let updatedAt else { return false }
        return Date().timeIntervalSince1970 - updatedAt < 300
    }
}

final class ProgressStore: @unchecked Sendable {
    static let shared = ProgressStore()

    private let lock = NSLock()
    private var current = RunProgress()

    private init() {}

    func snapshot() -> RunProgress {
        lock.lock()
        defer { lock.unlock() }
        return current
    }

    func record(_ reported: RunProgress) {
        lock.lock()
        var merged = reported
        if reported.run == current.run {
            merged.startedAt = current.startedAt ?? reported.startedAt
            merged.model = reported.model ?? current.model
            merged.suite = reported.suite ?? current.suite
        }
        merged.startedAt = merged.startedAt ?? Date().timeIntervalSince1970
        merged.updatedAt = Date().timeIntervalSince1970
        current = merged
        lock.unlock()
    }

    func clear() {
        lock.lock()
        current = RunProgress()
        lock.unlock()
    }
}
