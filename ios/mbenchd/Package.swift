// swift-tools-version: 6.0

import Foundation
import PackageDescription

let vendorLib = Context.packageDirectory + "/Vendor/lib"
let archives = ((try? FileManager.default.contentsOfDirectory(atPath: vendorLib)) ?? [])
    .filter { $0.hasSuffix(".a") }
    .sorted()
    .map { vendorLib + "/" + $0 }

let package = Package(
    name: "mbenchd",
    platforms: [.iOS(.v18)],
    products: [
        .library(name: "mbenchd", targets: ["mbenchd"]),
    ],
    targets: [
        .target(
            name: "LlamaServerBridge",
            cxxSettings: [.unsafeFlags(["-std=c++17"])]
        ),
        .target(
            name: "mbenchd",
            dependencies: ["LlamaServerBridge"],
            swiftSettings: [.swiftLanguageMode(.v5)],
            linkerSettings: [
                .unsafeFlags(archives.isEmpty ? [] : ["-Xlinker", "-all_load"] + archives),
                .linkedFramework("Metal"),
                .linkedFramework("MetalKit"),
                .linkedFramework("Accelerate"),
                .linkedLibrary("c++"),
            ]
        ),
    ]
)
