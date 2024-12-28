import XCTest
@testable import MapleDiffusion

final class MapleDiffusionTests: XCTestCase {
    var diffusion: MapleDiffusion!
    
    override func setUpWithError() throws {
        try super.setUpWithError()
        diffusion = try MapleDiffusion(saveMemoryButBeSlower: true)
    }
    
    override func tearDownWithError() throws {
        diffusion = nil
        try super.tearDownWithError()
    }
    
    // MARK: - Performance Tests
    func testGenerationPerformance() throws {
        let prompt = "a beautiful mountain landscape"
        let negativePrompt = ""
        let steps = 30
        let guidanceScale: Float = 7.5
        
        measure {
            let expectation = XCTestExpectation(description: "Generation completed")
            
            diffusion.generate(prompt: prompt, negativePrompt: negativePrompt, seed: 42, steps: steps, guidanceScale: guidanceScale) { image, progress, stage in
                if progress == 1.0 {
                    XCTAssertNotNil(image, "Generated image should not be nil")
                    expectation.fulfill()
                }
            }
            
            wait(for: [expectation], timeout: 60.0)
        }
    }
    
    func testMemoryUsage() throws {
        let prompt = "test memory usage"
        let negativePrompt = ""
        let steps = 10
        let guidanceScale: Float = 7.5
        
        let initialMemory = reportMemoryUsage()
        
        let expectation = XCTestExpectation(description: "Memory test completed")
        
        diffusion.generate(prompt: prompt, negativePrompt: negativePrompt, seed: 42, steps: steps, guidanceScale: guidanceScale) { image, progress, stage in
            if progress == 1.0 {
                let finalMemory = self.reportMemoryUsage()
                let memoryDelta = finalMemory - initialMemory
                
                // La différence de mémoire ne devrait pas dépasser 1GB
                XCTAssertLessThan(memoryDelta, 1024 * 1024 * 1024, "Memory usage increased by more than 1GB")
                expectation.fulfill()
            }
        }
        
        wait(for: [expectation], timeout: 30.0)
    }
    
    // MARK: - Error Handling Tests
    func testInvalidPrompt() throws {
        let expectation = XCTestExpectation(description: "Error handling test completed")
        
        diffusion.generate(prompt: "", negativePrompt: "", seed: 42, steps: 30, guidanceScale: 7.5) { image, progress, stage in
            if stage.contains("error") {
                XCTAssertNil(image, "Image should be nil for invalid prompt")
                expectation.fulfill()
            }
        }
        
        wait(for: [expectation], timeout: 10.0)
    }
    
    func testInvalidSteps() throws {
        let expectation = XCTestExpectation(description: "Invalid steps test completed")
        
        diffusion.generate(prompt: "test", negativePrompt: "", seed: 42, steps: 0, guidanceScale: 7.5) { image, progress, stage in
            if stage.contains("error") {
                XCTAssertNil(image, "Image should be nil for invalid steps")
                expectation.fulfill()
            }
        }
        
        wait(for: [expectation], timeout: 10.0)
    }
    
    // MARK: - Resource Management Tests
    func testResourceCleanup() throws {
        let initialBufferCount = MemoryManager.shared.activeBufferCount
        
        let expectation = XCTestExpectation(description: "Resource cleanup test completed")
        
        diffusion.generate(prompt: "test cleanup", negativePrompt: "", seed: 42, steps: 10, guidanceScale: 7.5) { image, progress, stage in
            if progress == 1.0 {
                DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
                    let finalBufferCount = MemoryManager.shared.activeBufferCount
                    XCTAssertEqual(initialBufferCount, finalBufferCount, "Active buffer count should return to initial state")
                    expectation.fulfill()
                }
            }
        }
        
        wait(for: [expectation], timeout: 30.0)
    }
    
    // MARK: - Performance Optimization Tests
    func testOptimalBatchSize() {
        let config = PerformanceOptimizer.shared.optimizeForDevice()
        XCTAssertGreaterThan(config.maxBatchSize, 0, "Batch size should be positive")
        XCTAssertLessThanOrEqual(config.maxBatchSize, 4, "Batch size should not exceed 4")
    }
    
    func testThreadCount() {
        let config = PerformanceOptimizer.shared.optimizeForDevice()
        XCTAssertGreaterThanOrEqual(config.threadCount, 2, "Thread count should be at least 2")
        XCTAssertLessThanOrEqual(config.threadCount, ProcessInfo.processInfo.processorCount, "Thread count should not exceed CPU count")
    }
    
    // MARK: - Tensor Cache Tests
    func testTensorCaching() {
        let device = MTLCreateSystemDefaultDevice()!
        let graph = MPSGraph()
        
        let tensor = graph.placeholder(shape: [1, 64, 64, 3], dataType: .float16, name: "test")
        let data = MPSGraphTensorData(device: device, shape: [1, 64, 64, 3], dataType: .float16)
        
        TensorCache.shared.store(data, forKey: "test_tensor")
        let cachedData = TensorCache.shared.retrieve(forKey: "test_tensor")
        
        XCTAssertNotNil(cachedData, "Cached tensor data should be retrievable")
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
