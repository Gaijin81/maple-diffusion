// MARK: - Error Types
enum MapleDiffusionError: Error {
    case deviceNotSupported
    case modelInitializationFailed(String)
    case generationFailed(String)
    case outOfMemory
    case invalidInput(String)
    case internalError(String)
    
    var localizedDescription: String {
        switch self {
        case .deviceNotSupported:
            return "This device does not support Metal Performance Shaders"
        case .modelInitializationFailed(let details):
            return "Failed to initialize models: \(details)"
        case .generationFailed(let details):
            return "Image generation failed: \(details)"
        case .outOfMemory:
            return "Not enough memory to complete the operation"
        case .invalidInput(let details):
            return "Invalid input: \(details)"
        case .internalError(let details):
            return "Internal error: \(details)"
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

// MARK: - Resource Management
class ResourceManager {
    private let device: MTLDevice
    private var resourceCache: [String: Any] = [:]
    private let maxCacheSize: Int = 512 * 1024 * 1024 // 512MB
    private var currentCacheSize: Int = 0
    
    init(device: MTLDevice) {
        self.device = device
    }
    
    func cacheResource(_ resource: Any, forKey key: String, size: Int) {
        while currentCacheSize + size > maxCacheSize {
            clearOldestResource()
        }
        resourceCache[key] = resource
        currentCacheSize += size
    }
    
    private func clearOldestResource() {
        guard let oldestKey = resourceCache.keys.first else { return }
        resourceCache.removeValue(forKey: oldestKey)
        // Estimation approximative de la taille libérée
        currentCacheSize = max(0, currentCacheSize - (maxCacheSize / resourceCache.count))
    }
    
    func clearCache() {
        resourceCache.removeAll()
        currentCacheSize = 0
    }
}

// MARK: - Network and Timeout Management
class NetworkManager {
    enum NetworkError: Error {
        case timeout
        case noConnection
        case serverError(String)
    }
    
    static let shared = NetworkManager()
    private let timeoutInterval: TimeInterval = 30
    private let queue = DispatchQueue(label: "com.maple-diffusion.network")
    
    private init() {}
    
    func checkConnection() -> Bool {
        // Basic reachability check
        // TODO: Implement actual reachability check
        return true
    }
    
    func executeWithTimeout<T>(_ work: @escaping () throws -> T) async throws -> T {
        return try await withTimeout(timeoutInterval) {
            try await withCheckedThrowingContinuation { continuation in
                queue.async {
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
    
    private func withTimeout<T>(_ timeout: TimeInterval, work: @escaping () async throws -> T) async throws -> T {
        try await withThrowingTaskGroup(of: T.self) { group in
            group.addTask {
                try await work()
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

// MARK: - Performance Optimization
class PerformanceOptimizer {
    static let shared = PerformanceOptimizer()
    
    private let memoryWarningThreshold: Int = 2 * 1024 * 1024 * 1024 // 2GB
    private let preferredThreadCount = max(2, ProcessInfo.processInfo.processorCount - 1)
    private let queue = DispatchQueue(label: "com.maple-diffusion.optimizer", qos: .userInitiated)
    
    private init() {
        setupMemoryWarningNotification()
    }
    
    private func setupMemoryWarningNotification() {
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleMemoryWarning),
            name: UIApplication.didReceiveMemoryWarningNotification,
            object: nil
        )
    }
    
    @objc private func handleMemoryWarning() {
        ResourceManager.shared.clearCache()
        OperationManager.shared.cancelAllOperations()
    }
    
    func optimizeForDevice() -> DeviceConfiguration {
        let device = MTLCreateSystemDefaultDevice()!
        let maxMemory = device.recommendedMaxWorkingSetSize
        let hasUnifiedMemory = device.hasUnifiedMemory
        
        return DeviceConfiguration(
            maxBatchSize: calculateOptimalBatchSize(maxMemory: maxMemory),
            threadCount: preferredThreadCount,
            useUnifiedMemory: hasUnifiedMemory,
            useLowMemoryMode: maxMemory < memoryWarningThreshold
        )
    }
    
    private func calculateOptimalBatchSize(maxMemory: Int) -> Int {
        // Estimation basée sur la mémoire disponible
        let memoryPerBatch = 512 * 1024 * 1024 // 512MB par batch
        let theoreticalMaxBatch = maxMemory / memoryPerBatch
        return min(4, max(1, theoreticalMaxBatch))
    }
    
    func optimizeGraphForDevice(graph: MPSGraph) {
        let config = optimizeForDevice()
        
        // Optimisations du graphe MPSGraph
        if config.useUnifiedMemory {
            graph.options = .synchronizeResults
        }
        
        // Optimisations de compilation
        let compilationDescriptor = MPSGraphCompilationDescriptor()
        compilationDescriptor.optimizationLevel = config.useLowMemoryMode ? .size : .performance
        graph.compile(with: compilationDescriptor)
    }
}

// Configuration optimale pour le device
struct DeviceConfiguration {
    let maxBatchSize: Int
    let threadCount: Int
    let useUnifiedMemory: Bool
    let useLowMemoryMode: Bool
}

// MARK: - Memory Management
class MemoryManager {
    static let shared = MemoryManager()
    
    private let memoryPool: [String: MTLBuffer] = [:]
    private let bufferQueue = DispatchQueue(label: "com.maple-diffusion.buffer", qos: .userInitiated)
    private var activeBuffers: Set<MTLBuffer> = []
    
    private init() {}
    
    func allocateBuffer(size: Int, device: MTLDevice) -> MTLBuffer? {
        return bufferQueue.sync {
            let options: MTLResourceOptions = device.hasUnifiedMemory ? .storageModeShared : .storageModePrivate
            let buffer = device.makeBuffer(length: size, options: options)
            if let buffer = buffer {
                activeBuffers.insert(buffer)
            }
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
            self.activeBuffers.removeAll()
        }
    }
}

// MARK: - Tensor Cache
class TensorCache {
    static let shared = TensorCache()
    
    private var cache: [String: MPSGraphTensorData] = [:]
    private let cacheQueue = DispatchQueue(label: "com.maple-diffusion.tensor-cache", qos: .userInitiated)
    
    private init() {}
    
    func store(_ tensor: MPSGraphTensorData, forKey key: String) {
        cacheQueue.async {
            self.cache[key] = tensor
        }
    }
    
    func retrieve(forKey key: String) -> MPSGraphTensorData? {
        return cacheQueue.sync {
            return cache[key]
        }
    }
    
    func clear() {
        cacheQueue.async {
            self.cache.removeAll()
        }
    }
}

class MapleDiffusion {
    // Existing properties...
    private let state: DiffusionState
    private let resourceManager: ResourceManager
    private let errorHandler: (MapleDiffusionError) -> Void
    
    public init(saveMemoryButBeSlower: Bool = true, errorHandler: @escaping (MapleDiffusionError) -> Void = { _ in }) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MapleDiffusionError.deviceNotSupported
        }
        
        self.device = device
        self.graphDevice = MPSGraphDevice(mtlDevice: device)
        self.commandQueue = device.makeCommandQueue()!
        self.saveMemory = saveMemoryButBeSlower
        self.shouldSynchronize = !device.hasUnifiedMemory
        self.state = DiffusionState()
        self.resourceManager = ResourceManager(device: device)
        self.errorHandler = errorHandler
        
        do {
            self.tokenizer = try BPETokenizer()
        } catch {
            throw MapleDiffusionError.modelInitializationFailed("Failed to initialize tokenizer: \(error.localizedDescription)")
        }
        
        // Initialize with error handling
        initializeWithErrorHandling()
        
        setupOptimizations()
    }
    
    private func initializeWithErrorHandling() {
        do {
            try setupMetalResources()
            try loadModels()
        } catch {
            if let error = error as? MapleDiffusionError {
                errorHandler(error)
            } else {
                errorHandler(.internalError(error.localizedDescription))
            }
        }
    }
    
    private func setupMetalResources() throws {
        guard device.supportsFamily(.metal3) else {
            throw MapleDiffusionError.deviceNotSupported
        }
        
        // Vérifier la mémoire disponible
        let memoryBudget = device.recommendedMaxWorkingSetSize
        if memoryBudget < 1024 * 1024 * 1024 { // 1GB minimum
            throw MapleDiffusionError.deviceNotSupported
        }
    }
    
    private func validateInputs(prompt: String, negativePrompt: String, steps: Int, guidanceScale: Float) throws {
        if prompt.isEmpty {
            throw MapleDiffusionError.invalidInput("Prompt cannot be empty")
        }
        
        if steps < 5 || steps > 150 {
            throw MapleDiffusionError.invalidInput("Steps must be between 5 and 150")
        }
        
        if guidanceScale < 1 || guidanceScale > 20 {
            throw MapleDiffusionError.invalidInput("Guidance scale must be between 1 and 20")
        }
    }
    
    public func generate(prompt: String, negativePrompt: String, seed: Int, steps: Int, guidanceScale: Float, completion: @escaping (CGImage?, Float, String) -> ()) {
        do {
            try validateInputs(prompt: prompt, negativePrompt: negativePrompt, steps: steps, guidanceScale: guidanceScale)
            
            state.update(status: .generating)
            
            // Vérifier la mémoire disponible avant de commencer
            if device.currentAllocatedSize > device.recommendedMaxWorkingSetSize * 3/4 {
                resourceManager.clearCache()
            }
            
            dispatchQueue.async {
                do {
                    let result = try self.generateWithErrorHandling(
                        prompt: prompt,
                        negativePrompt: negativePrompt,
                        seed: seed,
                        steps: steps,
                        guidanceScale: guidanceScale
                    ) { progress, stage in
                        self.state.update(progress: progress, stage: stage)
                        completion(nil, progress, stage)
                    }
                    
                    state.update(status: .idle)
                    completion(result, 1.0, "Generation completed")
                    
                } catch {
                    let diffusionError = error as? MapleDiffusionError ?? .internalError(error.localizedDescription)
                    state.update(status: .error(diffusionError))
                    errorHandler(diffusionError)
                    completion(nil, 0, diffusionError.localizedDescription)
                }
            }
        } catch {
            let inputError = error as? MapleDiffusionError ?? .invalidInput(error.localizedDescription)
            state.update(status: .error(inputError))
            errorHandler(inputError)
            completion(nil, 0, inputError.localizedDescription)
        }
    }
    
    private func generateWithErrorHandling(
        prompt: String,
        negativePrompt: String,
        seed: Int,
        steps: Int,
        guidanceScale: Float,
        progressCallback: (Float, String) -> Void
    ) throws -> CGImage {
        // Implement the actual generation with proper error handling
        // This is a placeholder for the actual implementation
        throw MapleDiffusionError.internalError("Not implemented")
    }
    
    // Recovery methods
    private func recoverFromError(_ error: MapleDiffusionError) {
        switch error {
        case .outOfMemory:
            resourceManager.clearCache()
            try? setupMetalResources()
        case .modelInitializationFailed:
            try? initializeWithErrorHandling()
        default:
            break
        }
    }
    
    private func executeWithRetry<T>(_ operation: @escaping () throws -> T, maxRetries: Int = 3) async throws -> T {
        var lastError: Error?
        
        for attempt in 1...maxRetries {
            do {
                return try await NetworkManager.shared.executeWithTimeout {
                    try operation()
                }
            } catch NetworkManager.NetworkError.timeout {
                lastError = MapleDiffusionError.generationFailed("Operation timed out")
                continue
            } catch {
                lastError = error
                if attempt == maxRetries {
                    throw error
                }
                try await Task.sleep(nanoseconds: UInt64(pow(2.0, Double(attempt)) * 1_000_000_000))
            }
        }
        
        throw lastError ?? MapleDiffusionError.internalError("Unknown error occurred")
    }
    
    private func handleGenerationError(_ error: Error) {
        switch error {
        case let diffusionError as MapleDiffusionError:
            errorHandler(diffusionError)
        case let networkError as NetworkManager.NetworkError:
            switch networkError {
            case .timeout:
                errorHandler(.generationFailed("Generation timed out"))
            case .noConnection:
                errorHandler(.generationFailed("No network connection"))
            case .serverError(let message):
                errorHandler(.generationFailed("Server error: \(message)"))
            }
        default:
            errorHandler(.internalError(error.localizedDescription))
        }
    }
    
    private func setupOptimizations() {
        let config = PerformanceOptimizer.shared.optimizeForDevice()
        
        // Configuration du graphe
        let graph = MPSGraph()
        PerformanceOptimizer.shared.optimizeGraphForDevice(graph: graph)
        
        // Configuration des buffers
        if config.useLowMemoryMode {
            ResourceManager.shared.clearCache()
            TensorCache.shared.clear()
        }
    }
    
    private func optimizeGeneration(prompt: String, negativePrompt: String, steps: Int) -> GenerationConfiguration {
        let config = PerformanceOptimizer.shared.optimizeForDevice()
        
        return GenerationConfiguration(
            batchSize: config.maxBatchSize,
            useHalfPrecision: config.useLowMemoryMode,
            optimizedSteps: min(steps, config.useLowMemoryMode ? 30 : 50)
        )
    }
}

// Configuration pour la génération
struct GenerationConfiguration {
    let batchSize: Int
    let useHalfPrecision: Bool
    let optimizedSteps: Int
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
