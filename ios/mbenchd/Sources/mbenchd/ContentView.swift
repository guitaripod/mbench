import SwiftUI

struct ContentView: View {
    @Environment(AppState.self) private var state

    var body: some View {
        NavigationStack {
            List {
                Section("server") { serverRows }
                Section("device") { deviceRows }
                Section("models") { modelRows }
                Section("log") { logRows }
            }
            .navigationTitle("mbenchd")
            .toolbar {
                Button("Unload") { state.unload() }
                    .disabled(state.server.state == "idle")
            }
        }
    }

    @ViewBuilder private var serverRows: some View {
        row("state", state.server.state)
        if let model = state.server.model { row("model", model) }
        if let port = state.server.port { row("port", String(port)) }
        if let load = state.server.loadSeconds { row("loaded in", "\(load) s") }
        if let error = state.server.error { row("error", error) }
        if let error = state.controlError { row("control", error) }
        row("control port", "8081")
    }

    @ViewBuilder private var deviceRows: some View {
        if let telemetry = state.telemetry {
            row("thermal", "\(telemetry.thermalState) · \(Int(telemetry.thermalSince)) s")
            row("footprint", "\(Int(telemetry.footprintMib)) MiB (peak \(Int(telemetry.peakFootprintMib)))")
            row("available", "\(Int(telemetry.availableMib)) MiB")
            row("battery", "\(Int(telemetry.batteryLevel * 100))% \(telemetry.batteryState)")
        }
        row("hardware", DeviceTelemetry.hardware())
    }

    @ViewBuilder private var modelRows: some View {
        if state.models.isEmpty {
            Text("push .gguf files into Documents/models").foregroundStyle(.secondary)
        } else {
            ForEach(state.models) { model in
                row(model.id, String(format: "%.2f GB", Double(model.bytes) / 1_073_741_824))
            }
        }
    }

    @ViewBuilder private var logRows: some View {
        ForEach(Array(state.logLines.enumerated()), id: \.offset) { _, line in
            Text(line).font(.system(size: 10, design: .monospaced)).lineLimit(2)
        }
    }

    private func row(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label).foregroundStyle(.secondary)
            Spacer()
            Text(value).font(.system(.body, design: .monospaced)).multilineTextAlignment(.trailing)
        }
    }
}
