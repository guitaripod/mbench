import SwiftUI

struct ContentView: View {
    @Environment(AppState.self) private var state
    @State private var showLog = false

    var body: some View {
        ZStack {
            Backdrop(tint: accent)
            ScrollView {
                VStack(spacing: 14) {
                    header
                    StatusCard(server: state.server, accent: accent)
                    if state.progress.fresh {
                        ProgressCard(progress: state.progress, accent: accent)
                    }
                    if state.server.state == "running" {
                        LiveCard(metrics: state.metrics, history: state.history, last: state.lastDecode, accent: accent)
                    }
                    tiles
                    models
                    logSection
                }
                .padding(16)
                .padding(.bottom, 40)
            }
        }
        .preferredColorScheme(.dark)
    }

    private var accent: Color {
        Theme.thermal(state.telemetry?.thermalState ?? "nominal")
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 3) {
                Text("mbench").font(.system(size: 26, weight: .heavy, design: .rounded))
                Text(subtitle).font(.system(size: 11, design: .monospaced)).foregroundStyle(.white.opacity(0.45))
            }
            Spacer()
            Pulse(active: state.server.state == "running", color: accent)
        }
        .padding(.bottom, 2)
    }

    private var subtitle: String {
        let device = DeviceTelemetry.hardware()
        let port = state.controlError == nil ? "control :8081" : "control down"
        return "\(device) · \(UIDevice.current.systemVersion) · \(port)"
    }

    private var tiles: some View {
        HStack(spacing: 10) {
            ThermalTile(state: state.telemetry?.thermalState ?? "–",
                        seconds: state.telemetry?.thermalSince ?? 0)
            MemoryTile(held: state.telemetry?.footprintMib ?? 0,
                       available: state.telemetry?.availableMib ?? 0,
                       peak: state.telemetry?.peakFootprintMib ?? 0)
            BatteryTile(level: state.telemetry?.batteryLevel ?? 0,
                        charging: (state.telemetry?.batteryState ?? "") != "unplugged")
        }
    }

    private var models: some View {
        Card {
            HStack {
                Label("models", systemImage: "shippingbox.fill").font(Theme.label)
                Spacer()
                Text("\(state.models.count)").font(Theme.label).foregroundStyle(.white.opacity(0.4))
            }
            if state.models.isEmpty {
                Text("push a .gguf into Documents/models")
                    .font(.system(size: 12)).foregroundStyle(.white.opacity(0.4)).padding(.top, 4)
            }
            ForEach(state.models) { model in
                ModelRow(model: model, loaded: state.server.model == model.file || state.server.model == model.id,
                         busy: state.server.state == "loading", accent: accent,
                         load: { state.load(model) }, unload: { state.unload() })
            }
        }
    }

    private var logSection: some View {
        Card {
            Button { withAnimation(.snappy) { showLog.toggle() } } label: {
                HStack {
                    Label("log", systemImage: "text.alignleft").font(Theme.label)
                    Spacer()
                    Image(systemName: showLog ? "chevron.up" : "chevron.down")
                        .font(.system(size: 11, weight: .bold)).foregroundStyle(.white.opacity(0.4))
                }
            }
            .buttonStyle(.plain)
            if showLog {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(Array(state.logLines.suffix(24).enumerated()), id: \.offset) { _, line in
                        Text(line).font(.system(size: 9, design: .monospaced))
                            .foregroundStyle(.white.opacity(0.55)).lineLimit(2)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.top, 6)
            }
        }
    }
}

enum Theme {
    static let label = Font.system(size: 11, weight: .semibold, design: .rounded)
    static let value = Font.system(size: 15, weight: .semibold, design: .rounded)

    static func thermal(_ state: String) -> Color {
        switch state {
        case "fair": return Color(red: 0.98, green: 0.78, blue: 0.25)
        case "serious": return Color(red: 1.0, green: 0.55, blue: 0.2)
        case "critical": return Color(red: 1.0, green: 0.32, blue: 0.36)
        default: return Color(red: 0.35, green: 0.9, blue: 0.65)
        }
    }

    static let states = ["nominal", "fair", "serious", "critical"]
}

struct Backdrop: View {
    let tint: Color

    var body: some View {
        ZStack {
            Color(red: 0.04, green: 0.045, blue: 0.06)
            RadialGradient(colors: [tint.opacity(0.22), .clear], center: .top, startRadius: 10, endRadius: 420)
        }
        .ignoresSafeArea()
        .animation(.easeInOut(duration: 0.8), value: tint)
    }
}

struct Card<Content: View>: View {
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 8) { content }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(14)
            .background(RoundedRectangle(cornerRadius: 18, style: .continuous).fill(.white.opacity(0.055)))
            .overlay(RoundedRectangle(cornerRadius: 18, style: .continuous).stroke(.white.opacity(0.07)))
    }
}

struct Pulse: View {
    let active: Bool
    let color: Color
    @State private var wide = false

    var body: some View {
        ZStack {
            Circle().fill(color.opacity(0.25)).frame(width: wide ? 26 : 14, height: wide ? 26 : 14)
                .opacity(wide ? 0 : 1)
            Circle().fill(active ? color : .white.opacity(0.25)).frame(width: 9, height: 9)
        }
        .frame(width: 28, height: 28)
        .onAppear {
            guard active else { return }
            withAnimation(.easeOut(duration: 1.4).repeatForever(autoreverses: false)) { wide = true }
        }
    }
}

struct StatusCard: View {
    let server: ServerStatus
    let accent: Color

    var body: some View {
        Card {
            Text(server.state.uppercased())
                .font(.system(size: 34, weight: .black, design: .rounded))
                .foregroundStyle(server.state == "running" ? accent : .white.opacity(0.85))
                .contentTransition(.numericText())
            Text(server.model ?? "no model loaded")
                .font(.system(size: 12, design: .monospaced))
                .foregroundStyle(.white.opacity(0.55))
                .lineLimit(2)
            if let error = server.error {
                Text(error).font(.system(size: 11)).foregroundStyle(Theme.thermal("critical"))
            }
            HStack(spacing: 18) {
                if let seconds = server.loadSeconds { Stat(name: "loaded in", value: "\(Int(seconds))s") }
                if let port = server.port { Stat(name: "port", value: String(port)) }
                if let context = contextArgument { Stat(name: "context", value: context) }
                if let slots = argument("-np") { Stat(name: "slots", value: slots) }
            }
            .padding(.top, 2)
        }
    }

    private var contextArgument: String? {
        guard let raw = argument("-c"), let value = Int(raw) else { return nil }
        return value >= 1024 ? "\(value / 1024)k" : String(value)
    }

    private func argument(_ flag: String) -> String? {
        guard let index = server.args.firstIndex(of: flag), index + 1 < server.args.count else { return nil }
        return server.args[index + 1]
    }
}

struct Stat: View {
    let name: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(name).font(.system(size: 9, weight: .medium)).foregroundStyle(.white.opacity(0.35))
            Text(value).font(Theme.value)
        }
    }
}

/// What the run on the host is doing right now: which task, how far through it, and how long it has been going.
struct ProgressCard: View {
    let progress: RunProgress
    let accent: Color

    var body: some View {
        Card {
            VStack(alignment: .leading, spacing: 9) {
                HStack(alignment: .firstTextBaseline) {
                    Text(progress.phase ?? "starting")
                        .font(.system(size: 20, weight: .bold, design: .rounded))
                        .foregroundStyle(accent)
                    Spacer()
                    if let done = progress.done, let total = progress.total, total > 0 {
                        Text("\(done)/\(total)")
                            .font(.system(size: 13, weight: .semibold, design: .monospaced))
                            .foregroundStyle(.white.opacity(0.75))
                            .contentTransition(.numericText())
                    }
                }
                if let share = progress.share {
                    GeometryReader { frame in
                        ZStack(alignment: .leading) {
                            Capsule().fill(.white.opacity(0.09))
                            Capsule().fill(accent.opacity(0.85))
                                .frame(width: max(3, frame.size.width * share))
                        }
                    }
                    .frame(height: 6)
                    .animation(.easeOut(duration: 0.4), value: share)
                }
                HStack(spacing: 10) {
                    if let model = progress.model {
                        Text(model).font(Theme.label).foregroundStyle(.white.opacity(0.5))
                            .lineLimit(1).truncationMode(.middle)
                    }
                    Spacer()
                    if let suite = progress.suite {
                        Text(suite).font(Theme.label).foregroundStyle(.white.opacity(0.35))
                    }
                    Text(elapsed).font(Theme.label).foregroundStyle(.white.opacity(0.5))
                        .monospacedDigit()
                }
                if let note = progress.note, !note.isEmpty {
                    Text(note).font(.system(size: 11)).foregroundStyle(.white.opacity(0.45))
                        .lineLimit(2)
                }
            }
        }
    }

    private var elapsed: String {
        guard let started = progress.startedAt else { return "–" }
        let seconds = Int(max(0, Date().timeIntervalSince1970 - started))
        let hours = seconds / 3600, minutes = (seconds % 3600) / 60
        return hours > 0 ? "\(hours)h\(String(format: "%02d", minutes))m"
                         : "\(minutes)m\(String(format: "%02d", seconds % 60))s"
    }
}

struct LiveCard: View {
    let metrics: ServerMetrics?
    let history: [Double]
    let last: Double?
    let accent: Color

    var body: some View {
        Card {
            HStack(alignment: .lastTextBaseline, spacing: 6) {
                if let value = metrics?.decodeTokensPerSecond, value > 0 {
                    Text(String(format: "%.1f", value))
                        .font(.system(size: 42, weight: .black, design: .rounded))
                        .foregroundStyle(accent).contentTransition(.numericText())
                } else if let last {
                    Text(String(format: "%.1f", last))
                        .font(.system(size: 42, weight: .black, design: .rounded))
                        .foregroundStyle(.white.opacity(0.35))
                } else {
                    Text(busy ? "measuring" : "idle")
                        .font(.system(size: 26, weight: .bold, design: .rounded))
                        .foregroundStyle(.white.opacity(0.3))
                }
                Text(busy || last == nil ? "tok/s" : "tok/s last answer")
                    .font(.system(size: 13, weight: .semibold)).foregroundStyle(.white.opacity(0.4))
                Spacer()
                if busy {
                    Label("\(Int(metrics?.requestsProcessing ?? 0))", systemImage: "bolt.fill")
                        .font(Theme.label).foregroundStyle(accent)
                }
            }
            Sparkline(values: history, color: accent,
                      placeholder: busy ? "the first answer sets the speed" : "waiting for a request")
                .frame(height: 46)
            HStack(spacing: 18) {
                Stat(name: "prefill", value: rate(metrics?.prefillTokensPerSecond))
                Stat(name: "kv cache", value: percent(metrics?.kvCacheRatio))
                Stat(name: "tokens", value: whole(metrics?.tokensPredicted))
            }
        }
    }

    private var busy: Bool {
        metrics?.isBusy ?? false
    }

    private func rate(_ value: Double?) -> String {
        guard let value, value > 0 else { return "–" }
        return "\(Int(value)) tok/s"
    }

    private func percent(_ value: Double?) -> String {
        guard let value else { return "–" }
        return "\(Int(value * 100))%"
    }

    private func whole(_ value: Double?) -> String {
        guard let value else { return "–" }
        return value >= 1000 ? String(format: "%.1fk", value / 1000) : String(Int(value))
    }
}

struct Sparkline: View {
    let values: [Double]
    let color: Color
    var placeholder = "waiting for a request"

    var body: some View {
        GeometryReader { geometry in
            let points = values.suffix(AppState.historyLength)
            let top = max(points.max() ?? 1, 1)
            let bottom = min(points.min() ?? 0, top)
            let span = max(top - bottom, top * 0.15)
            let step = points.count > 1 ? geometry.size.width / CGFloat(points.count - 1) : 0
            let coordinates = points.enumerated().map { index, value in
                CGPoint(x: CGFloat(index) * step,
                        y: geometry.size.height * (1 - CGFloat((value - bottom) / span)) * 0.92 + 2)
            }
            ZStack {
                if coordinates.count > 1 {
                    Path { path in
                        path.addLines(coordinates)
                    }
                    .stroke(color, style: StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round))
                    Path { path in
                        path.addLines(coordinates)
                        path.addLine(to: CGPoint(x: coordinates.last!.x, y: geometry.size.height))
                        path.addLine(to: CGPoint(x: coordinates.first!.x, y: geometry.size.height))
                        path.closeSubpath()
                    }
                    .fill(LinearGradient(colors: [color.opacity(0.28), .clear], startPoint: .top, endPoint: .bottom))
                } else {
                    Text(placeholder)
                        .font(.system(size: 10)).foregroundStyle(.white.opacity(0.3))
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
        }
    }
}

struct ThermalTile: View {
    let state: String
    let seconds: Double

    var body: some View {
        Tile(name: "thermal", value: state, detail: seconds >= 1 ? "\(Int(seconds))s" : nil,
             tint: Theme.thermal(state)) {
            HStack(spacing: 3) {
                ForEach(Theme.states, id: \.self) { name in
                    Capsule()
                        .fill(reached(name) ? Theme.thermal(state) : Color.white.opacity(0.12))
                        .frame(height: 4)
                }
            }
        }
    }

    private func reached(_ name: String) -> Bool {
        guard let current = Theme.states.firstIndex(of: state), let step = Theme.states.firstIndex(of: name) else {
            return false
        }
        return step <= current
    }
}

struct MemoryTile: View {
    let held: Double
    let available: Double
    let peak: Double

    var body: some View {
        Tile(name: "memory", value: MemoryTile.size(held),
             detail: "\(MemoryTile.size(available)) free", tint: .white.opacity(0.8)) {
            Meter(share: held / max(held + available, 1), tint: Theme.thermal(pressure))
        }
    }

    /// Megabytes until it is worth a gigabyte, so an idle app does not read as 0.0 GB.
    static func size(_ mib: Double) -> String {
        mib >= 1024 ? String(format: "%.1f GB", mib / 1024) : String(format: "%.0f MB", mib)
    }

    private var pressure: String {
        let share = available / max(held + available, 1)
        return share < 0.15 ? "critical" : share < 0.35 ? "serious" : "nominal"
    }
}

struct BatteryTile: View {
    let level: Double
    let charging: Bool

    var body: some View {
        Tile(name: "battery", value: "\(Int(max(level, 0) * 100))%",
             detail: charging ? "charging" : "on battery", tint: .white.opacity(0.8)) {
            Meter(share: max(level, 0), tint: charging ? Theme.thermal("nominal") : .white.opacity(0.5))
        }
    }
}

struct Tile<Footer: View>: View {
    let name: String
    let value: String
    var detail: String?
    let tint: Color
    @ViewBuilder var footer: Footer

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(name).font(.system(size: 9, weight: .semibold)).foregroundStyle(.white.opacity(0.35))
            Text(value).font(.system(size: 16, weight: .bold, design: .rounded)).foregroundStyle(tint)
                .lineLimit(1).minimumScaleFactor(0.7)
            footer
            if let detail {
                Text(detail).font(.system(size: 9, design: .monospaced)).foregroundStyle(.white.opacity(0.35))
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 16, style: .continuous).fill(.white.opacity(0.055)))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(.white.opacity(0.07)))
    }
}

struct Meter: View {
    let share: Double
    let tint: Color

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                Capsule().fill(.white.opacity(0.12))
                Capsule().fill(tint).frame(width: geometry.size.width * min(max(share, 0), 1))
            }
        }
        .frame(height: 4)
    }
}

struct ModelRow: View {
    let model: StoredModel
    let loaded: Bool
    let busy: Bool
    let accent: Color
    let load: () -> Void
    let unload: () -> Void

    var body: some View {
        HStack(spacing: 10) {
            VStack(alignment: .leading, spacing: 2) {
                Text(model.id).font(.system(size: 12, weight: .medium, design: .monospaced))
                    .lineLimit(1).truncationMode(.middle)
                Text(String(format: "%.2f GB", Double(model.bytes) / 1_073_741_824))
                    .font(.system(size: 10)).foregroundStyle(.white.opacity(0.4))
            }
            Spacer()
            Button(action: loaded ? unload : load) {
                Text(loaded ? "unload" : "load")
                    .font(.system(size: 11, weight: .bold, design: .rounded))
                    .padding(.horizontal, 12).padding(.vertical, 7)
                    .background(Capsule().fill(loaded ? Color.white.opacity(0.12) : accent.opacity(0.9)))
                    .foregroundStyle(loaded ? .white : .black)
            }
            .buttonStyle(.plain)
            .disabled(busy)
            .opacity(busy ? 0.4 : 1)
        }
        .padding(.vertical, 6)
    }
}
