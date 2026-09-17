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
        let files = entries.filter { $0.hasSuffix(".gguf") }.sorted().map { name -> StoredModel in
            let path = directory.appendingPathComponent(name)
            let attributes = try? manager.attributesOfItem(atPath: path.path)
            let bytes = (attributes?[.size] as? NSNumber)?.int64Value ?? 0
            return StoredModel(id: String(name.dropLast(5)), file: name, bytes: bytes)
        }
        let folders = entries.filter { isModelDirectory(directory.appendingPathComponent($0)) }.sorted()
            .map { name in StoredModel(id: name, file: name, bytes: weight(of: directory.appendingPathComponent(name))) }
        return files + folders
    }

    /// An MLX model is a folder of weights and its tokenizer rather than one file, and config.json is what every
    /// such folder has and nothing else in here does.
    static func isModelDirectory(_ url: URL) -> Bool {
        var directory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &directory), directory.boolValue else {
            return false
        }
        return FileManager.default.fileExists(atPath: url.appendingPathComponent("config.json").path)
    }

    static func weight(of url: URL) -> Int64 {
        let manager = FileManager.default
        let entries = (try? manager.contentsOfDirectory(atPath: url.path)) ?? []
        return entries.reduce(Int64(0)) { total, name in
            let attributes = try? manager.attributesOfItem(atPath: url.appendingPathComponent(name).path)
            return total + ((attributes?[.size] as? NSNumber)?.int64Value ?? 0)
        }
    }

    /// Resolves what an MLX request asked for: a folder name under models, or an absolute path already on the device.
    static func resolveDirectory(_ wanted: String) -> URL? {
        let candidate = wanted.hasPrefix("/") ? URL(fileURLWithPath: wanted) : directory.appendingPathComponent(wanted)
        return isModelDirectory(candidate) ? candidate : nil
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
