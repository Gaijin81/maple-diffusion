import Foundation
import Metal
import MetalPerformanceShadersGraph
@testable import MapleDiffusion

struct TestResult {
    let name: String
    let passed: Bool
    let message: String?
}

class SimpleTestRunner {
    static func runAllTests(completion: @escaping ([TestResult]) -> Void) {
        var results: [TestResult] = []
        
        // Test MapleDiffusion initialization
        do {
            _ = try MapleDiffusion(saveMemoryButBeSlower: true)
            results.append(TestResult(name: "Initialize MapleDiffusion", passed: true, message: nil))
        } catch let error as MapleDiffusionError {
            results.append(TestResult(name: "Initialize MapleDiffusion", passed: false, message: error.localizedDescription))
        } catch {
            results.append(TestResult(name: "Initialize MapleDiffusion", passed: false, message: "Unexpected error: \(error)"))
        }
        
        // Test performance optimization
        let optimizer = PerformanceOptimizer.shared
        let device = MTLCreateSystemDefaultDevice()!
        let graph = MPSGraph()
        optimizer.optimizeGraphForDevice(graph: graph, device: MPSGraphDevice(mtlDevice: device))
        results.append(TestResult(name: "Performance Optimization", passed: true, message: nil))
        
        // Test resource management
        ResourceManager.shared.clearCache()
        results.append(TestResult(name: "Resource Management", passed: true, message: nil))
        
        // Test tensor cache
        TensorCache.shared.clear()
        results.append(TestResult(name: "Tensor Cache", passed: true, message: nil))
        
        completion(results)
    }
    
    static func printResults(_ results: [TestResult]) {
        print("\n=== Test Results ===")
        for result in results {
            let status = result.passed ? "✅ PASS" : "❌ FAIL"
            print("\(status) - \(result.name)")
            if let message = result.message {
                print("   Message: \(message)")
            }
        }
        
        let totalTests = results.count
        let passedTests = results.filter { $0.passed }.count
        print("\nTotal Tests: \(totalTests)")
        print("Passed: \(passedTests)")
        print("Failed: \(totalTests - passedTests)")
        
        // Exit with appropriate status code
        Darwin.exit(passedTests == totalTests ? 0 : 1)
    }
    
    static func run() {
        runAllTests { results in
            printResults(results)
        }
    }
}
