// swift-tools-version: 6.1

import Foundation
import PackageDescription

let mlx = Context.environment["MBENCHD_MLX"] == "1"

let vendorLib = Context.packageDirectory + "/Vendor/lib"

/// The archives to link, named by whoever built them: macOS evaluates a manifest inside a sandbox that never
/// answers a directory read, so the build script lists Vendor/lib and passes the names in. The fallback is what a
/// Linux cross-build leaves behind.
let named = Context.environment["MBENCHD_ARCHIVES"]?
    .split(separator: ",")
    .map(String.init)
    .filter { $0.hasSuffix(".a") } ?? []
let archives = (named.isEmpty ? [
    "libcpp-httplib.a", "libggml-base.a", "libggml-cpu.a", "libggml-metal.a",
    "libggml.a", "libllama-common.a", "libllama.a",
] : named).sorted().map { vendorLib + "/" + $0 }

let package = Package(
    name: "mbenchd",
    platforms: [.iOS(.v18)],
    products: [
        .library(name: "mbenchd", targets: ["mbenchd"]),
    ],
    dependencies: mlx ? [
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMinor(from: "3.31.4")),
        .package(url: "https://github.com/huggingface/swift-transformers", .upToNextMinor(from: "1.3.4")),
    ] : [],
    targets: [
        .target(
            name: "LlamaServerBridge",
            cxxSettings: [.unsafeFlags(["-std=c++17"])]
        ),
        .target(
            name: "mbenchd",
            dependencies: ["LlamaServerBridge"] + (mlx ? [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ] : []),
            swiftSettings: [.swiftLanguageMode(.v5)] + (mlx ? [.define("MBENCHD_MLX")] : []),
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
