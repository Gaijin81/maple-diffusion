import SwiftUI
import Foundation
import CoreGraphics
import AppKit

struct ThemeColor {
    static let primary = Color("AccentColor")
    static let background = Color(NSColor.windowBackgroundColor)
    static let secondaryBackground = Color(NSColor.underPageBackgroundColor)
    static let text = Color(NSColor.labelColor)
    static let secondaryText = Color(NSColor.secondaryLabelColor)
}

struct PromptField: View {
    let title: String
    let placeholder: String
    @Binding var text: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.headline)
                .foregroundColor(ThemeColor.text)
            
            if #available(macOS 13.0, *) {
                TextField(placeholder, text: $text, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .lineLimit(3...6)
                    .padding(.horizontal, 8)
                    .background(ThemeColor.secondaryBackground)
            } else {
                TextField(placeholder, text: $text)
                    .textFieldStyle(.roundedBorder)
                    .frame(height: 100)
                    .padding(.horizontal, 8)
                    .background(ThemeColor.secondaryBackground)
            }
        }
    }
}

struct SliderControl: View {
    let title: String
    let value: Binding<Float>
    let range: ClosedRange<Float>
    let format: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(title)
                    .font(.headline)
                    .foregroundColor(ThemeColor.text)
                Spacer()
                Text(String(format: format, value.wrappedValue))
                    .foregroundColor(ThemeColor.secondaryText)
            }
            
            Slider(value: value, in: range) { editing in
                if !editing {
                    // Haptic feedback on slider release
                    #if os(iOS)
                    let impact = UIImpactFeedbackGenerator(style: .light)
                    impact.impactOccurred()
                    #endif
                }
            }
            .tint(ThemeColor.primary)
        }
    }
}

struct GenerateButton: View {
    let isRunning: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: {
            // Haptic feedback on button press
            #if os(iOS)
            let impact = UIImpactFeedbackGenerator(style: .medium)
            impact.impactOccurred()
            #endif
            action()
        }) {
            HStack {
                Image(systemName: "wand.and.stars")
                Text("Generate")
            }
            .font(.headline)
            .frame(maxWidth: .infinity, minHeight: 56)
            .background(isRunning ? ThemeColor.secondaryText : ThemeColor.primary)
            .foregroundColor(.white)
            .cornerRadius(16)
            .shadow(radius: isRunning ? 0 : 4)
        }
        .buttonStyle(.borderless)
        .disabled(isRunning)
    }
}

struct ImagePreview: View {
    let image: Image?
    let width: CGFloat
    let height: CGFloat
    
    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 8)
                .fill(Color.gray.opacity(0.2))
            
            if let image = image {
                image
                    .resizable()
                    .aspectRatio(contentMode: .fit)
            } else {
                ProgressView()
            }
        }
        .frame(width: width, height: height)
    }
}

struct ContentView: View {
    @State private var prompt: String = ""
    @State private var negativePrompt: String = ""
    @State private var steps: Int = 50
    @State private var seed: UInt32 = 0
    @State private var image: NSImage?
    @State private var isGenerating: Bool = false
    @State private var progressStage: String = ""
    @State private var errorMessage: String?
    
    private let mapleDiffusion: MapleDiffusion
    private let width: CGFloat = 512
    private let height: CGFloat = 512
    
    init() {
        do {
            self.mapleDiffusion = try MapleDiffusion(saveMemoryButBeSlower: false)
        } catch {
            fatalError("Failed to initialize MapleDiffusion: \(error.localizedDescription)")
        }
    }
    
    var body: some View {
        VStack(spacing: 20) {
            // Input fields
            VStack(alignment: .leading, spacing: 10) {
                Text("Prompt")
                    .font(.headline)
                TextEditor(text: $prompt)
                    .frame(height: 100)
                    .border(Color.gray.opacity(0.2))
                
                Text("Negative Prompt")
                    .font(.headline)
                TextEditor(text: $negativePrompt)
                    .frame(height: 60)
                    .border(Color.gray.opacity(0.2))
                
                HStack {
                    VStack(alignment: .leading) {
                        Text("Steps: \(steps)")
                            .font(.headline)
                        Slider(value: .init(get: { Double(steps) },
                                          set: { steps = Int($0) }),
                               in: 1...150,
                               step: 1)
                    }
                    
                    VStack(alignment: .leading) {
                        Text("Seed: \(seed)")
                            .font(.headline)
                        Button("Random") {
                            seed = UInt32.random(in: 0...UInt32.max)
                        }
                    }
                }
            }
            .padding()
            
            // Generate button
            Button(action: generateImage) {
                if isGenerating {
                    ProgressView()
                        .scaleEffect(0.7)
                        .padding(.horizontal, 5)
                } else {
                    Text("Generate")
                }
            }
            .disabled(isGenerating || prompt.isEmpty)
            .buttonStyle(.borderedProminent)
            
            // Progress
            if isGenerating {
                Text(progressStage)
                    .foregroundColor(.secondary)
            }
            
            // Error message
            if let errorMessage = errorMessage {
                Text(errorMessage)
                    .foregroundColor(.red)
                    .padding()
            }
            
            // Image preview
            ImagePreview(
                image: image.map { Image(nsImage: $0) },
                width: width,
                height: height
            )
            
            Spacer()
        }
        .padding()
        .frame(minWidth: 600, minHeight: 800)
    }
    
    private func generateImage() {
        isGenerating = true
        errorMessage = nil
        progressStage = "Starting generation..."
        
        Task {
            do {
                let cgImage = try await Task.detached {
                    try await mapleDiffusion.generate(
                        prompt: prompt,
                        negativePrompt: negativePrompt,
                        steps: steps,
                        seed: seed
                    )
                }.value
                
                let nsImage = NSImage(cgImage: cgImage, size: NSSize(width: width, height: height))
                
                await MainActor.run {
                    self.image = nsImage
                    self.isGenerating = false
                    self.progressStage = ""
                }
            } catch {
                await MainActor.run {
                    self.errorMessage = error.localizedDescription
                    self.isGenerating = false
                    self.progressStage = ""
                }
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
