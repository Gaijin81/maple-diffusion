import Foundation
import Metal
import MetalPerformanceShadersGraph
import AppKit
#if canImport(MapleDiffusion_MapleDiffusion)
import MapleDiffusion_MapleDiffusion
#endif

// MARK: - Error Types
enum MapleDiffusionError: Error {
    case deviceNotFound
    case resourceLoadError(String)
    case initializationError(String)
    case invalidInput(String)
    case runtimeError(String)
    case outOfMemory
    case tokenizationError(String)
    
    var localizedDescription: String {
        switch self {
        case .deviceNotFound:
            return "No compatible Metal device found"
        case .resourceLoadError(let message):
            return "Failed to load resource: \(message)"
        case .initializationError(let message):
            return "Initialization failed: \(message)"
        case .invalidInput(let message):
            return "Invalid input: \(message)"
        case .runtimeError(let message):
            return "Runtime error: \(message)"
        case .outOfMemory:
            return "Out of memory error"
        case .tokenizationError(let message):
            return "Tokenization error: \(message)"
        }
    }
}

// MARK: - Performance Optimization
class PerformanceOptimizer {
    static let shared = PerformanceOptimizer()
    
    private init() {}
    
    internal func optimizeGraphForDevice(graph: MPSGraph, device: MPSGraphDevice) {
        // Basic graph optimization
        if #available(macOS 13.0, *) {
            // Set default graph options
            _ = graph.options
        }
    }
}

// MARK: - Resource Management
class ResourceManager {
    static let shared = ResourceManager()
    private var cache: [String: Any] = [:]
    
    private init() {}
    
    func clearCache() {
        cache.removeAll()
    }
    
    func loadResource(_ name: String) throws -> Data {
        if let cached = cache[name] as? Data {
            return cached
        }
        
        // First try Bundle.module
        if let url = Bundle.module.url(forResource: name, withExtension: nil) {
            do {
                let data = try Data(contentsOf: url)
                cache[name] = data
                return data
            } catch {
                throw MapleDiffusionError.resourceLoadError("Failed to load \(name): \(error.localizedDescription)")
            }
        }
        
        // Then try Bundle.main
        if let url = Bundle.main.url(forResource: name, withExtension: nil) {
            do {
                let data = try Data(contentsOf: url)
                cache[name] = data
                return data
            } catch {
                throw MapleDiffusionError.resourceLoadError("Failed to load \(name): \(error.localizedDescription)")
            }
        }
        
        throw MapleDiffusionError.resourceLoadError("Resource \(name) not found in any bundle")
    }
}

// MARK: - Tensor Cache
class TensorCache {
    static let shared = TensorCache()
    private var tensors: [String: MPSGraphTensor] = [:]
    
    private init() {}
    
    func clear() {
        tensors.removeAll()
    }
    
    func cache(_ tensor: MPSGraphTensor, forKey key: String) {
        tensors[key] = tensor
    }
    
    func getTensor(forKey key: String) -> MPSGraphTensor? {
        return tensors[key]
    }
}

// MARK: - Tokenizer
class BPETokenizer {
    private let vocabulary: [String: Int]
    
    init() throws {
        // First try Bundle.module
        if let vocabURL = Bundle.module.url(forResource: "bins/bpe_simple_vocab_16e6", withExtension: "txt") {
            do {
                let vocabData = try String(contentsOf: vocabURL, encoding: .utf8)
                var vocab: [String: Int] = [:]
                let lines = vocabData.components(separatedBy: .newlines)
                for (index, line) in lines.enumerated() {
                    let token = line.trimmingCharacters(in: .whitespaces)
                    if !token.isEmpty {
                        vocab[token] = index
                    }
                }
                
                if vocab.isEmpty {
                    throw MapleDiffusionError.tokenizationError("Vocabulary file is empty")
                }
                
                self.vocabulary = vocab
                return
            } catch {
                throw MapleDiffusionError.tokenizationError("Failed to load vocabulary: \(error.localizedDescription)")
            }
        }
        
        // Then try Bundle.main
        if let vocabURL = Bundle.main.url(forResource: "bins/bpe_simple_vocab_16e6", withExtension: "txt") {
            do {
                let vocabData = try String(contentsOf: vocabURL, encoding: .utf8)
                var vocab: [String: Int] = [:]
                let lines = vocabData.components(separatedBy: .newlines)
                for (index, line) in lines.enumerated() {
                    let token = line.trimmingCharacters(in: .whitespaces)
                    if !token.isEmpty {
                        vocab[token] = index
                    }
                }
                
                if vocab.isEmpty {
                    throw MapleDiffusionError.tokenizationError("Vocabulary file is empty")
                }
                
                self.vocabulary = vocab
                return
            } catch {
                throw MapleDiffusionError.tokenizationError("Failed to load vocabulary: \(error.localizedDescription)")
            }
        }
        
        throw MapleDiffusionError.tokenizationError("Vocabulary file not found in any bundle")
    }
    
    func encode(_ text: String) -> [Int] {
        // Simple tokenization for now
        return text.components(separatedBy: " ")
            .compactMap { word in
                vocabulary[word.lowercased()]
            }
    }
}

// MARK: - State Management
class DiffusionState {
    enum Status {
        case idle
        case loading
        case generating
        case error(MapleDiffusionError)
    }
    
    private(set) var status: Status = .idle
    private(set) var progress: Float = 0
    private(set) var stage: String = ""
    
    func update(status: Status) {
        self.status = status
    }
    
    func update(progress: Float, stage: String) {
        self.progress = progress
        self.stage = stage
    }
    
    var isRunning: Bool {
        switch status {
        case .loading, .generating:
            return true
        default:
            return false
        }
    }
}

// MARK: - Network and Timeout Management
class NetworkManager {
    static let shared = NetworkManager()
    private let timeoutInterval: TimeInterval = 30.0
    private let queue = DispatchQueue(label: "com.maple-diffusion.network", qos: .userInitiated)
    
    private init() {}
    
    func executeWithTimeout<T>(_ work: @escaping () throws -> T) async throws -> T {
        return try await withTimeout(timeoutInterval) {
            try await withCheckedThrowingContinuation { continuation in
                self.queue.async {
                    do {
                        let result = try work()
                        continuation.resume(returning: result)
                    } catch {
                        continuation.resume(throwing: error)
                    }
                }
            }
        }
    }
    
    private func withTimeout<T>(_ timeout: TimeInterval, operation: @escaping () async throws -> T) async throws -> T {
        try await withThrowingTaskGroup(of: T.self) { group in
            group.addTask {
                try await operation()
            }
            
            group.addTask {
                try await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
                throw NetworkError.timeout
            }
            
            let result = try await group.next()!
            group.cancelAll()
            return result
        }
    }
    
    enum NetworkError: Error {
        case timeout
        case noConnection
        case serverError(String)
    }
}

// MARK: - Operation Queue Management
class OperationManager {
    static let shared = OperationManager()
    private let operationQueue: OperationQueue
    private let maxConcurrentOperations = 2
    
    private init() {
        operationQueue = OperationQueue()
        operationQueue.maxConcurrentOperationCount = maxConcurrentOperations
    }
    
    func addOperation(_ operation: @escaping () -> Void) {
        let blockOperation = BlockOperation(block: operation)
        operationQueue.addOperation(blockOperation)
    }
    
    func cancelAllOperations() {
        operationQueue.cancelAllOperations()
    }
    
    var isExecuting: Bool {
        return operationQueue.operationCount > 0
    }
}

// MARK: - Memory Management
class MemoryManager {
    static let shared = MemoryManager()
    
    private let memoryPool: [String: MTLBuffer] = [:]
    private let bufferQueue = DispatchQueue(label: "com.maple-diffusion.buffer", qos: .userInitiated)
    private var activeBuffers = NSHashTable<AnyObject>.weakObjects()
    
    private init() {}
    
    func allocateBuffer(size: Int, device: MTLDevice) -> MTLBuffer? {
        return bufferQueue.sync {
            let options: MTLResourceOptions = device.hasUnifiedMemory ? .storageModeShared : .storageModePrivate
            guard let buffer = device.makeBuffer(length: size, options: options) else {
                return nil
            }
            activeBuffers.add(buffer)
            return buffer
        }
    }
    
    func releaseBuffer(_ buffer: MTLBuffer) {
        bufferQueue.async {
            self.activeBuffers.remove(buffer)
        }
    }
    
    func purgeMemory() {
        bufferQueue.async {
            self.activeBuffers.removeAllObjects()
        }
    }
}

// MARK: - Maple Diffusion
actor MapleDiffusion {
    private let device: MTLDevice
    private let graph: MPSGraph
    private let graphDevice: MPSGraphDevice
    private let resourceManager: ResourceManager
    private let tensorCache: TensorCache
    private let optimizer: PerformanceOptimizer
    private let tokenizer: BPETokenizer
    private let dispatchQueue = DispatchQueue(label: "com.maple-diffusion.generation")
    private let notificationCenter = NotificationCenter.default
    
    public init(saveMemoryButBeSlower: Bool = false) throws {
        // Initialize Metal device
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MapleDiffusionError.deviceNotFound
        }
        self.device = device
        
        // Initialize graph and device
        self.graph = MPSGraph()
        self.graphDevice = MPSGraphDevice(mtlDevice: device)
        
        // Initialize managers
        self.resourceManager = ResourceManager.shared
        self.tensorCache = TensorCache.shared
        self.optimizer = PerformanceOptimizer.shared
        
        // Initialize tokenizer
        do {
            self.tokenizer = try BPETokenizer()
        } catch {
            throw MapleDiffusionError.initializationError("Failed to initialize tokenizer: \(error.localizedDescription)")
        }
        
        // Initialize and optimize the graph
        optimizer.optimizeGraphForDevice(graph: graph, device: graphDevice)
        
        // Load required resources
        do {
            _ = try resourceManager.loadResource("bins/alphas_cumprod.bin")
        } catch {
            throw MapleDiffusionError.initializationError("Failed to load alphas_cumprod.bin: \(error.localizedDescription)")
        }
        
        // Set up memory warning observer
        notificationCenter.addObserver(
            forName: NSWorkspace.didWakeNotification,
            object: nil,
            queue: nil
        ) { [weak self] _ in
            Task {
                await self?.handleMemoryWarning()
            }
        }
    }
    
    private func handleMemoryWarning() {
        // Clear caches when memory pressure is high
        ResourceManager.shared.clearCache()
        TensorCache.shared.clear()
    }
    
    public func generate(
        prompt: String,
        negativePrompt: String = "",
        steps: Int = 50,
        seed: UInt32 = 0
    ) async throws -> CGImage {
        // This is a placeholder implementation
        throw MapleDiffusionError.runtimeError("Image generation not yet implemented")
    }
    
    deinit {
        notificationCenter.removeObserver(self)
    }
}

// MARK: - Performance Monitoring
class PerformanceMonitor {
    static let shared = PerformanceMonitor()
    
    private var metrics: [String: TimeInterval] = [:]
    private let metricsQueue = DispatchQueue(label: "com.maple-diffusion.metrics", qos: .utility)
    
    private init() {}
    
    func startMeasuring(_ operation: String) -> CFAbsoluteTime {
        return CFAbsoluteTimeGetCurrent()
    }
    
    func stopMeasuring(_ operation: String, startTime: CFAbsoluteTime) {
        let duration = CFAbsoluteTimeGetCurrent() - startTime
        metricsQueue.async {
            self.metrics[operation] = duration
        }
    }
    
    func getMetrics() -> [String: TimeInterval] {
        return metricsQueue.sync {
            return metrics
        }
    }
    
    func reset() {
        metricsQueue.async {
            self.metrics.removeAll()
        }
    }
}
