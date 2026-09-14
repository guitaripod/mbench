import CryptoKit
import Foundation

enum ModelDigest {
    private static let store = UserDefaults.standard
    private static let chunk = 4 * 1024 * 1024

    /// The sha256 of the weights the phone actually loaded, so a desktop run can only lend its quality score to a
    /// phone run that served the identical file. Cached against size and modification date, because hashing a
    /// multi-gigabyte file on a phone is not free.
    static func of(_ url: URL) -> String? {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
              let size = (attributes[.size] as? NSNumber)?.int64Value else { return nil }
        let modified = (attributes[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
        let key = "digest:\(url.lastPathComponent):\(size):\(modified)"
        if let cached = store.string(forKey: key) { return cached }
        guard let digest = compute(url) else { return nil }
        store.set(digest, forKey: key)
        return digest
    }

    private static func compute(_ url: URL) -> String? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        var hasher = SHA256()
        let started = Date()
        while let data = try? handle.read(upToCount: chunk), !data.isEmpty {
            hasher.update(data: data)
        }
        let digest = hasher.finalize().map { String(format: "%02x", $0) }.joined()
        AppLogger.info(.models, "hashed \(url.lastPathComponent) in \(Int(Date().timeIntervalSince(started)))s: \(digest.prefix(12))")
        return digest
    }
}
