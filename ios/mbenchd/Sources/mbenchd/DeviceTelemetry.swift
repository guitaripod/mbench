import Foundation
import UIKit
import os

struct ThermalTransition: Codable, Sendable {
    let state: String
    let at: Double
}

struct TelemetrySnapshot: Codable, Sendable {
    let thermalState: String
    let thermalSince: Double
    let thermalTransitions: [ThermalTransition]
    let footprintMib: Double
    let peakFootprintMib: Double
    let availableMib: Double
    let physicalMib: Double
    let batteryLevel: Double
    let batteryState: String
    let lowPowerMode: Bool
    let uptime: Double
}

final class DeviceTelemetry: @unchecked Sendable {
    static let shared = DeviceTelemetry()

    private let lock = NSLock()
    private let started = Date()
    private var transitions: [ThermalTransition] = []
    private var stateSince = Date()
    private var peakFootprint: Double = 0

    private init() {}

    func start() {
        UIDevice.current.isBatteryMonitoringEnabled = true
        record(state: DeviceTelemetry.name(ProcessInfo.processInfo.thermalState))
        NotificationCenter.default.addObserver(
            forName: ProcessInfo.thermalStateDidChangeNotification, object: nil, queue: nil
        ) { [weak self] _ in
            let state = DeviceTelemetry.name(ProcessInfo.processInfo.thermalState)
            AppLogger.info(.telemetry, "thermal state \(state)")
            self?.record(state: state)
        }
    }

    func snapshot() -> TelemetrySnapshot {
        let footprint = DeviceTelemetry.footprintMib()
        lock.lock()
        peakFootprint = max(peakFootprint, footprint)
        let peak = peakFootprint
        let since = Date().timeIntervalSince(stateSince)
        let history = transitions
        lock.unlock()
        return TelemetrySnapshot(
            thermalState: DeviceTelemetry.name(ProcessInfo.processInfo.thermalState),
            thermalSince: round(since * 100) / 100,
            thermalTransitions: history,
            footprintMib: footprint,
            peakFootprintMib: peak,
            availableMib: Double(os_proc_available_memory()) / 1048576,
            physicalMib: Double(ProcessInfo.processInfo.physicalMemory) / 1048576,
            batteryLevel: Double(UIDevice.current.batteryLevel),
            batteryState: DeviceTelemetry.batteryName(UIDevice.current.batteryState),
            lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled,
            uptime: round(Date().timeIntervalSince(started) * 100) / 100
        )
    }

    /// The phone's model identifier ("iPhone18,5"), which is the only reliable way to tell one A19 variant
    /// from another; the marketing name never reaches the process.
    static func hardware() -> String {
        var info = utsname()
        uname(&info)
        return withUnsafePointer(to: &info.machine) { pointer in
            pointer.withMemoryRebound(to: CChar.self, capacity: Int(_SYS_NAMELEN)) { String(cString: $0) }
        }
    }

    static func footprintMib() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        return Double(info.phys_footprint) / 1048576
    }

    private func record(state: String) {
        lock.lock()
        stateSince = Date()
        transitions.append(ThermalTransition(state: state, at: round(Date().timeIntervalSince(started) * 100) / 100))
        if transitions.count > 500 { transitions.removeFirst(transitions.count - 500) }
        lock.unlock()
    }

    private static func name(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    private static func batteryName(_ state: UIDevice.BatteryState) -> String {
        switch state {
        case .charging: return "charging"
        case .full: return "full"
        case .unplugged: return "unplugged"
        case .unknown: return "unknown"
        @unknown default: return "unknown"
        }
    }
}
