import XCTest
@testable import MapleDiffusion

final class MapleDiffusionPerformanceTests: XCTestCase {
    var diffusion: MapleDiffusion!
    
    override func setUpWithError() throws {
        try super.setUpWithError()
        diffusion = try MapleDiffusion(saveMemoryButBeSlower: false)
    }
    
    override func tearDownWithError() throws {
        diffusion = nil
        try super.tearDownWithError()
    }
    
    // MARK: - Batch Processing Tests
    func testBatchProcessingPerformance() throws {
        let prompts = [
            "a beautiful mountain landscape",
            "a serene ocean view",
            "a bustling cityscape",
            "a peaceful forest scene"
        ]
        
        measure {
            let expectation = XCTestExpectation(description: "Batch processing completed")
            expectation.expectedFulfillmentCount = prompts.count
            
            for prompt in prompts {
                diffusion.generate(prompt: prompt, negativePrompt: "", seed: 42, steps: 20, guidanceScale: 7.5) { image, progress, stage in
                    if progress == 1.0 {
                        expectation.fulfill()
                    }
                }
            }
            
            wait(for: [expectation], timeout: 120.0)
        }
    }
    
    // MARK: - Memory Pressure Tests
    func testMemoryPressureHandling() throws {
        let iterations = 5
        var memoryUsage: [Int] = []
        
        let expectation = XCTestExpectation(description: "Memory pressure test completed")
        expectation.expectedFulfillmentCount = iterations
        
        for i in 0..<iterations {
            diffusion.generate(prompt: "test \(i)", negativePrompt: "", seed: 42, steps: 15, guidanceScale: 7.5) { image, progress, stage in
                if progress == 1.0 {
                    memoryUsage.append(self.reportMemoryUsage())
                    expectation.fulfill()
                }
            }
        }
        
        wait(for: [expectation], timeout: 180.0)
        
        // Vérifier que la mémoire ne croît pas de manière excessive
        for i in 1..<memoryUsage.count {
            let memoryDelta = abs(memoryUsage[i] - memoryUsage[i-1])
            XCTAssertLessThan(memoryDelta, 100 * 1024 * 1024, "Memory growth exceeds 100MB between iterations")
        }
    }
    
    // MARK: - Concurrent Processing Tests
    func testConcurrentProcessing() throws {
        let concurrentTasks = 3
        let expectation = XCTestExpectation(description: "Concurrent processing completed")
        expectation.expectedFulfillmentCount = concurrentTasks
        
        measure {
            for i in 0..<concurrentTasks {
                DispatchQueue.global().async {
                    self.diffusion.generate(prompt: "concurrent test \(i)", negativePrompt: "", seed: 42, steps: 15, guidanceScale: 7.5) { image, progress, stage in
                        if progress == 1.0 {
                            expectation.fulfill()
                        }
                    }
                }
            }
            
            wait(for: [expectation], timeout: 180.0)
        }
    }
    
    // MARK: - Resource Usage Tests
    func testResourceUsageOverTime() throws {
        let duration: TimeInterval = 60 // 1 minute test
        let interval: TimeInterval = 5 // Check every 5 seconds
        var measurements: [(Date, Int)] = []
        
        let expectation = XCTestExpectation(description: "Resource usage monitoring completed")
        
        let startTime = Date()
        let timer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { timer in
            let currentMemory = self.reportMemoryUsage()
            measurements.append((Date(), currentMemory))
            
            if Date().timeIntervalSince(startTime) >= duration {
                timer.invalidate()
                expectation.fulfill()
            }
        }
        
        // Generate images during monitoring
        diffusion.generate(prompt: "resource test", negativePrompt: "", seed: 42, steps: 30, guidanceScale: 7.5) { _, _, _ in }
        
        wait(for: [expectation], timeout: duration + 10)
        timer.invalidate()
        
        // Analyser les mesures
        var maxMemorySpike: Int = 0
        for i in 1..<measurements.count {
            let memoryDelta = abs(measurements[i].1 - measurements[i-1].1)
            maxMemorySpike = max(maxMemorySpike, memoryDelta)
        }
        
        XCTAssertLessThan(maxMemorySpike, 500 * 1024 * 1024, "Memory spike exceeds 500MB")
    }
    
    // MARK: - Cache Performance Tests
    func testTensorCachePerformance() {
        measure {
            let device = MTLCreateSystemDefaultDevice()!
            let graph = MPSGraph()
            
            for i in 0..<100 {
                let tensor = graph.placeholder(shape: [1, 64, 64, 3], dataType: .float16, name: "test_\(i)")
                let data = MPSGraphTensorData(device: device, shape: [1, 64, 64, 3], dataType: .float16)
                
                TensorCache.shared.store(data, forKey: "test_tensor_\(i)")
                _ = TensorCache.shared.retrieve(forKey: "test_tensor_\(i)")
            }
        }
    }
    
    // MARK: - Helper Methods
    private func reportMemoryUsage() -> Int {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)/4
        
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_,
                         task_flavor_t(MACH_TASK_BASIC_INFO),
                         $0,
                         &count)
            }
        }
        
        return kerr == KERN_SUCCESS ? Int(info.resident_size) : 0
    }
}
