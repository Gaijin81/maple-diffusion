// swift-tools-version:5.7
import PackageDescription

let package = Package(
    name: "MapleDiffusion",
    platforms: [
        .macOS(.v12)
    ],
    products: [
        .library(
            name: "MapleDiffusion",
            targets: ["MapleDiffusion"]),
        .executable(
            name: "MapleDiffusionTests",
            targets: ["MapleDiffusionTests"]),
        .executable(
            name: "MapleDiffusionApp",
            targets: ["MapleDiffusionApp"])
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-docc-plugin", from: "1.0.0"),
    ],
    targets: [
        .target(
            name: "MapleDiffusion",
            dependencies: [],
            path: "maple-diffusion",
            exclude: [
                "maple_diffusion.entitlements",
                "Preview Content/Preview Assets.xcassets",
                "Assets.xcassets",
                "maple_diffusionApp.swift"
            ],
            resources: [
                .copy("bins")
            ]
        ),
        .executableTarget(
            name: "MapleDiffusionApp",
            dependencies: ["MapleDiffusion"],
            path: "maple-diffusion",
            sources: ["maple_diffusionApp.swift"],
            resources: [
                .copy("Preview Content"),
                .copy("Assets.xcassets")
            ]
        ),
        .executableTarget(
            name: "MapleDiffusionTests",
            dependencies: ["MapleDiffusion"],
            path: "Tests/MapleDiffusionTests"
        )
    ]
)
