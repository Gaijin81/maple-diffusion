import SwiftUI

struct ThemeColor {
    static let primary = Color("AccentColor")
    static let background = Color(.systemBackground)
    static let secondaryBackground = Color(.secondarySystemBackground)
    static let text = Color(.label)
    static let secondaryText = Color(.secondaryLabel)
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
            
            TextField(placeholder, text: $text, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(3...6)
                .padding(.horizontal, 8)
                .background(ThemeColor.secondaryBackground)
                .cornerRadius(10)
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
    let action: () -> Void
    let isDisabled: Bool
    
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
            .background(isDisabled ? ThemeColor.secondaryText : ThemeColor.primary)
            .foregroundColor(.white)
            .cornerRadius(16)
            .shadow(radius: isDisabled ? 0 : 4)
        }
        .buttonStyle(.borderless)
        .disabled(isDisabled)
    }
}

struct ImagePreview: View {
    let image: Image?
    let width: CGFloat
    let height: CGFloat
    
    var body: some View {
        Group {
            if let img = image {
                #if os(iOS)
                ShareLink(item: img, preview: SharePreview("Generated Image", image: img)) {
                    img.resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(idealWidth: width, idealHeight: height)
                        .cornerRadius(16)
                        .shadow(radius: 8)
                }
                #else
                img.resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(idealWidth: width, idealHeight: height)
                    .cornerRadius(16)
                    .shadow(radius: 8)
                #endif
            } else {
                Rectangle()
                    .fill(ThemeColor.secondaryBackground)
                    .aspectRatio(1.0, contentMode: .fit)
                    .frame(idealWidth: width, idealHeight: height)
                    .cornerRadius(16)
                    .overlay(
                        Image(systemName: "photo")
                            .font(.largeTitle)
                            .foregroundColor(ThemeColor.secondaryText)
                    )
            }
        }
    }
}

struct ContentView: View {
    #if os(iOS)
    let mapleDiffusion = MapleDiffusion(saveMemoryButBeSlower: true)
    #else
    let mapleDiffusion = MapleDiffusion(saveMemoryButBeSlower: false)
    #endif
    
    let dispatchQueue = DispatchQueue(label: "Generation")
    
    @State private var steps: Float = 20
    @State private var image: Image?
    @State private var prompt: String = ""
    @State private var negativePrompt: String = ""
    @State private var guidanceScale: Float = 7.5
    @State private var running: Bool = false
    @State private var progressProp: Float = 1
    @State private var progressStage: String = "Ready"
    @State private var showSettings: Bool = false
    @State private var showHistory: Bool = false
    @State private var generatedImages: [(Image, String)] = []
    
    func loadModels() {
        dispatchQueue.async {
            running = true
            mapleDiffusion.initModels() { (p, s) -> () in
                progressProp = p
                progressStage = s
            }
            running = false
        }
    }
    
    func generate() {
        dispatchQueue.async {
            running = true
            progressStage = ""
            progressProp = 0
            
            mapleDiffusion.generate(
                prompt: prompt,
                negativePrompt: negativePrompt,
                seed: Int.random(in: 1..<Int.max),
                steps: Int(steps),
                guidanceScale: guidanceScale
            ) { (cgim, p, s) -> () in
                if let cgim = cgim {
                    let newImage = Image(cgim, scale: 1.0, label: Text("Generated image"))
                    image = newImage
                    generatedImages.insert((newImage, prompt), at: 0)
                }
                progressProp = p
                progressStage = s
            }
            running = false
        }
    }
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    // Image Preview
                    ImagePreview(
                        image: image,
                        width: mapleDiffusion.width as? CGFloat ?? 512,
                        height: mapleDiffusion.height as? CGFloat ?? 512
                    )
                    
                    // Controls
                    VStack(spacing: 16) {
                        PromptField(
                            title: "Prompt",
                            placeholder: "Describe what you want to generate",
                            text: $prompt
                        )
                        
                        PromptField(
                            title: "Negative Prompt",
                            placeholder: "Describe what you want to avoid",
                            text: $negativePrompt
                        )
                        
                        SliderControl(
                            title: "Guidance Scale",
                            value: $guidanceScale,
                            range: 1...20,
                            format: "%.1f"
                        )
                        
                        SliderControl(
                            title: "Steps",
                            value: $steps,
                            range: 5...150,
                            format: "%.0f"
                        )
                        
                        // Progress
                        if running {
                            VStack(spacing: 8) {
                                ProgressView(progressStage, value: progressProp, total: 1)
                                    .tint(ThemeColor.primary)
                                Text(progressStage)
                                    .font(.caption)
                                    .foregroundColor(ThemeColor.secondaryText)
                            }
                        }
                        
                        // Generate Button
                        GenerateButton(action: generate, isDisabled: running)
                    }
                    .padding()
                    .background(ThemeColor.background)
                    .cornerRadius(20)
                    .shadow(radius: 2)
                }
                .padding()
            }
            .navigationTitle("🍁 Maple Diffusion")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: { showHistory.toggle() }) {
                        Image(systemName: "clock.arrow.circlepath")
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: { showSettings.toggle() }) {
                        Image(systemName: "gear")
                    }
                }
            }
            .sheet(isPresented: $showHistory) {
                NavigationView {
                    List {
                        ForEach(generatedImages.indices, id: \.self) { index in
                            HStack {
                                generatedImages[index].0
                                    .resizable()
                                    .aspectRatio(contentMode: .fit)
                                    .frame(width: 60, height: 60)
                                    .cornerRadius(8)
                                
                                Text(generatedImages[index].1)
                                    .lineLimit(2)
                            }
                        }
                    }
                    .navigationTitle("History")
                    .navigationBarTitleDisplayMode(.inline)
                    .toolbar {
                        ToolbarItem(placement: .navigationBarTrailing) {
                            Button("Done") {
                                showHistory = false
                            }
                        }
                    }
                }
            }
        }
        .onAppear(perform: loadModels)
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
