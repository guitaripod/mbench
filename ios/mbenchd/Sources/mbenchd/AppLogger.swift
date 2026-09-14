import Foundation
import OSLog

enum LogCategory: String {
    case lifecycle
    case server
    case control
    case telemetry
    case models
}

enum AppLogger {
    private static let subsystem = "com.midgar.mbenchd"

    static func debug(_ category: LogCategory, _ message: String) {
        emit("DEBUG", category, message) { $0.debug("\($1, privacy: .public)") }
    }

    static func info(_ category: LogCategory, _ message: String) {
        emit("INFO", category, message) { $0.info("\($1, privacy: .public)") }
    }

    static func error(_ category: LogCategory, _ message: String) {
        emit("ERROR", category, message) { $0.error("\($1, privacy: .public)") }
    }

    /// Fans one call out to OSLog and to the file the host pulls over USB, since a phone under test has no
    /// console attached.
    private static func emit(_ level: String, _ category: LogCategory, _ message: String,
                             _ osLog: (Logger, String) -> Void) {
        osLog(Logger(subsystem: subsystem, category: category.rawValue), message)
        LogFileWriter.shared.write(level, category.rawValue, message)
    }
}
