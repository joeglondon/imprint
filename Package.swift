// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MemoryApp",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "MemoryApp", targets: ["MemoryApp"])
    ],
    targets: [
        .systemLibrary(
            name: "AIMemoryFFI",
            path: ".ffi"
        ),
        .executableTarget(
            name: "MemoryApp",
            dependencies: ["AIMemoryFFI"],
            path: "MemoryApp/Sources",
            linkerSettings: [
                .unsafeFlags(["-L", ".ffi/lib"]),
                .linkedLibrary("ai_memory")
            ]
        )
    ]
)
