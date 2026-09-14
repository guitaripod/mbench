import Foundation

struct StoredModel: Codable, Sendable, Identifiable {
    let id: String
    let file: String
    let bytes: Int64
}

enum ModelStore {
    static var documents: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    static var directory: URL {
        let url = documents.appendingPathComponent("models")
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    static var logDirectory: URL {
        let url = documents.appendingPathComponent("logs")
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    static func all() -> [StoredModel] {
        let manager = FileManager.default
        let entries = (try? manager.contentsOfDirectory(atPath: directory.path)) ?? []
        return entries.filter { $0.hasSuffix(".gguf") }.sorted().map { name in
            let path = directory.appendingPathComponent(name)
            let attributes = try? manager.attributesOfItem(atPath: path.path)
            let bytes = (attributes?[.size] as? NSNumber)?.int64Value ?? 0
            return StoredModel(id: String(name.dropLast(5)), file: name, bytes: bytes)
        }
    }

    /// Resolves what a request asked for: a model id, a file name, or an absolute path already on the device.
    static func resolve(_ wanted: String) -> URL? {
        if wanted.hasPrefix("/"), FileManager.default.fileExists(atPath: wanted) {
            return URL(fileURLWithPath: wanted)
        }
        let candidates = [wanted, wanted + ".gguf"]
        for candidate in candidates {
            let url = directory.appendingPathComponent(candidate)
            if FileManager.default.fileExists(atPath: url.path) { return url }
        }
        return nil
    }
}
