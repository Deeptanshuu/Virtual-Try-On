import logging
import gradio as gr
import base64
import os
from tryon_engine import TryOnEngine, WIDTH, HEIGHT, POSE_WIDTH, POSE_HEIGHT

def get_logo_base64():
    with open("logo.jpeg", "rb") as image_file:
        return "data:image/png;base64," + base64.b64encode(image_file.read()).decode('utf-8')

# Function to encode font files to base64
def get_font_css():
    fonts_dir = os.path.join(os.path.dirname(__file__), 'fonts')
    font_files = {
        'PPValve-PlainExtrabold': 'PPValve-PlainExtrabold.otf',
        'PPValve-PlainExtraboldItalic': 'PPValve-PlainExtraboldItalic.otf',
        'PPValve-PlainExtralight': 'PPValve-PlainExtralightItalic.otf',
        'PPValve-PlainMedium': 'PPValve-PlainMedium.otf',
        'PPValve-PlainMediumItalic': 'PPValve-PlainMediumItalic.otf',
    }
    css_fonts = ""
    for font_family, font_file in font_files.items():
        font_path = os.path.join(fonts_dir, font_file)
        if os.path.exists(font_path):
            with open(font_path, "rb") as f:
                font_data = f.read()
                font_base64 = base64.b64encode(font_data).decode('utf-8')
                weight = "800" if "Extrabold" in font_file else "200" if "Extralight" in font_file else "500"
                style = "italic" if "Italic" in font_file else "normal"
                css_fonts += f"""
    @font-face {{
        font-family: '{font_family}';
        src: url('data:font/otf;base64,{font_base64}') format('opentype');
        font-weight: {weight};
        font-style: {style};
    }}"""
    icon_path = os.path.join(os.path.dirname(__file__), 'df.png')
    icon_base64 = ""
    if os.path.exists(icon_path):
        with open(icon_path, "rb") as f:
            icon_data = f.read()
            icon_base64 = base64.b64encode(icon_data).decode('utf-8')
    css = f"""
    <style>
    {css_fonts}
    
    .dark {{
        --background-fill-primary: #171717 !important;
    }}

    :root {{
        --primary-color: #9f1c35;
        --secondary-color: #e58097;
        --accent-color: #f97316;
        --background-color: #171717;
        --panel-color: #171717;
        --input-color: #36383b;
        --text-color: #f8fafc;
        --border-radius: 12px;
        --box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
        --background-fill-primary: #171717 !important;
        ::selection {{
            background: #9f1c35 !important;
            color: #f8fafc !important;
        }}
    }}

    body {{
        background: linear-gradient(135deg, #23272a 0%, #181a1b 100%);
        color: var(--text-color) !important;
    }}

    /* Outer card: match Advanced Settings (dark gray) */
    .prompt-card, .prompt-section,
    .gr-panel:has(.prompt-input),
    .gr-column:has(.prompt-input),
    div[data-testid*="column"]:has(.prompt-input) {{
        background: var(--panel-color) !important;
        color: var(--text-color) !important;
        border-radius: var(--border-radius) !important;
        padding: 20px !important;
        box-shadow: var(--box-shadow) !important;
        margin: 15px 0 !important;
        border: 1px solid #32363a !important;
    }}

    /* Inner Style Description textbox: match Random Seed textbox/input color */
    .prompt-input textarea, .gr-textbox textarea, .gr-textbox input, textarea, input[type="text"] {{
        background: var(--input-color) !important;
        color: var(--text-color) !important;
        border: 1px solid #32363a !important;
        border-radius: var(--border-radius) !important;
        font-size: 16px !important;
        padding: 12px !important;
    }}

    .prompt-input textarea:focus, .gr-textbox textarea:focus {{
        border-color: var(--primary-color) !important;
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(159, 28, 53, 0.1) !important;
    }}

    .prompt-input textarea::placeholder, .gr-textbox textarea::placeholder {{
        color: #94a3b8 !important;
    }}

    .card-header {{
        color: var(--text-color) !important;
        font-weight: 600 !important;
        margin-bottom: 10px !important;
        font-family: 'PPValve-PlainMedium', sans-serif !important;
    }}
    .footer {{
        text-align: center;
        margin-top: 40px;
        padding: 20px;
        font-family: 'PPValve-PlainMedium', sans-serif;
        color: var(--text-color);
        opacity: 0.7;
    }}
    
    /* Gradient styling for button and title */
    .try-on-button button {{
        background: linear-gradient(135deg, #9f1c35 0%, #e58097 50%, #f97316 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }}
    
    .try-on-button button:hover {{
        background: linear-gradient(135deg, #8a162d 0%, #d4738a 50%, #e06514 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(159, 28, 53, 0.3) !important;
    }}
    
    /* Gradient title styling */
    .gradient-title {{
        background: linear-gradient(135deg, #9f1c35 0%, #e58097 50%, #f97316 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: none !important;
    }}
    
    </style>
    """
    return css




engine = TryOnEngine()
example_path = os.path.join(os.path.dirname(__file__), 'example')


def start_tryon(human_img_dict, garm_img, prompt_text, denoise_steps, seed, garment_category="upper_body", arm_width="dc", progress=gr.Progress()):
    logging.info("Starting try on")
    print(f"Input: {human_img_dict}")
    gr.Info("Starting try on....Please wait ~30 secs")
    progress(0, desc="Initializing virtual try-on...")
    # human_img_orig = human_img_dict["background"].convert("RGB")   # ImageEditor
    final_image, mask_gray = engine.tryon(
        human_img_dict,     # Image
        garm_img,
        prompt_text=prompt_text,
        steps=int(denoise_steps),
        seed=seed,
        garment_category=garment_category,
        arm_width=arm_width,
        progress=progress,
    )
    progress(1.0, desc="Virtual try-on complete!")
    gr.Info("Virtual try-on complete!")
    return final_image, mask_gray

garm_list = os.listdir(os.path.join(example_path,"cloth"))
garm_list_path = [os.path.join(example_path,"cloth",garm) for garm in garm_list]

human_list = os.listdir(os.path.join(example_path,"human"))
human_list_path = [os.path.join(example_path,"human",human) for human in human_list]

human_ex_list = []
human_ex_list = human_list_path # Image
""" if using ImageEditor instead of Image while taking input, use this - ImageEditor
for ex_human in human_list_path:
    ex_dict= {}
    ex_dict['background'] = ex_human
    ex_dict['layers'] = None
    ex_dict['composite'] = None
    human_ex_list.append(ex_dict)
"""
##default human


# API Endpoints for Next.js Integration
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "DiffuseFit API is running"}

def get_examples():
    """Get available example images"""
    return {
        "human_examples": human_list_path,
        "garment_examples": garm_list_path
    }

def get_model_info():
    """Get model configuration information"""
    return {
        "model_name": "yisol/IDM-VTON",
        "image_dimensions": {"width": WIDTH, "height": HEIGHT},
        "pose_dimensions": {"width": POSE_WIDTH, "height": POSE_HEIGHT},
        "supported_garment_categories": ["upper_body", "lower_body", "dresses"],
        "supported_arm_widths": [
            {"label": "Half Sleeve", "value": "dc"},
            {"label": "Full Sleeve", "value": "hd"}
        ],
        "default_arm_width": "dc",
        "default_category": "upper_body",
        "supported_formats": ["jpg", "jpeg", "png"],
        "max_file_size": "10MB"
    }

def process_tryon_api(human_img, garm_img, prompt_text="", denoise_steps=60, seed=-1, garment_category="upper_body", arm_width="dc"):
    """API endpoint for try-on processing with all parameters"""
    try:
        if human_img is None or garm_img is None:
            # Return None for both outputs to indicate error
            return None, None
        
        result_image, mask_image = start_tryon(human_img, garm_img, prompt_text, denoise_steps, seed, garment_category, arm_width)
        return result_image, mask_image
    except Exception as e:
        # Return None for both outputs to indicate error
        return None, None

def process_tryon_simple(human_img, garm_img):
    """Simplified API endpoint with default parameters"""
    try:
        if human_img is None or garm_img is None:
            return None
        
        result_image, _ = start_tryon(human_img, garm_img, "", 60, -1, "upper_body", "dc")
        return result_image
    except Exception as e:
        return None

def process_tryon_with_prompt(human_img, garm_img, prompt_text):
    """API endpoint with custom prompt"""
    try:
        if human_img is None or garm_img is None:
            return None
        
        result_image, _ = start_tryon(human_img, garm_img, prompt_text, 60, -1, "upper_body", "dc")
        return result_image
    except Exception as e:
        return None

# api_open=True will allow this API to be hit using curl
image_blocks = gr.Blocks(theme='CultriX/gradio-theme', css=get_font_css()).queue(api_open=True)
with image_blocks as demo:
    # Header section with title only (no emoji i hate emojis)
    with gr.Row(elem_classes=["header-container"]):
        logo_base64 = get_logo_base64()
        gr.HTML(f"""
        <div style="display: flex; align-items: center; gap: 15px;">
            <img src="{logo_base64}" alt="Logo" style="width: 80px; height: 80px;">
            <h1 class="gradient-title" style="font-family: 'PPValve-PlainMedium', sans-serif; font-size: 56px; letter-spacing: 1px; margin: 10px 0 5px 0; text-align: left;">Diffusion-Based Virtual Try-On Framework</h1>
        </div>
        """)
        
    with gr.Row():
        gr.HTML(
            """
            <div style="font-family: 'PPValve-PlainMedium', sans-serif; font-size: 18px; line-height: 1.6; color: #e0e0e0; text-align: left;">
                A virtual try-on experience powered by Stable Diffusion XL and GarmNet. Upload your photo and a garment to see how it looks on you!
            </div>
            """
        )
    
    # Main content area with cards
    with gr.Row(elem_classes=["main-container"]):
        # Input section - Human Image
        with gr.Column(elem_classes=["input-card"]):
            gr.Markdown("### Upload Your Photo", elem_classes=["card-header"])
            imgs = gr.Image(
                sources=['upload', 'webcam'], 
                type='pil', 
                label='Human Image', 
                elem_classes=["input-image"],
                height=400
            )
            # Examples directly shown without accordion
            example = gr.Examples(
                inputs=imgs,
                examples_per_page=10,
                examples=human_ex_list
            )

        # Input section - Garment Image
        with gr.Column(elem_classes=["input-card"]):
            gr.Markdown("### Upload Garment", elem_classes=["card-header"])
            garm_img = gr.Image(
                sources=['upload'], 
                type="pil", 
                label="Garment Image", 
                elem_classes=["input-image"],
                height=400
            )
            # Examples directly shown without accordion
            example = gr.Examples(
                inputs=garm_img,
                examples_per_page=10,
                examples=garm_list_path
            )
        
        # Output section - Result with Generate button inside
        with gr.Column(elem_classes=["output-card"]):
            gr.Markdown("### Virtual Try-On Result", elem_classes=["card-header"])
            
            # Output image with tabs for final result and mask
            with gr.Tabs(elem_classes=["output-tabs"]):
                with gr.TabItem("Final Result", elem_classes=["tab-item"]):
                    image_out = gr.Image(
                        label="Try-on Result", 
                        elem_id="output-img", 
                        show_share_button=True,
                        height=500
                    )
                with gr.TabItem("Mask View", elem_classes=["tab-item"]):
                    masked_img = gr.Image(
                        label="Mask Visualization", 
                        elem_id="masked-img", 
                        show_share_button=False,
                        height=500
                    )
            
            # Generate button moved below the output tabs
            processing_indicator = gr.Markdown("", elem_id="processing-status", visible=False)
            try_button = gr.Button(
                value="Generate Try-On", 
                variant="primary",
                elem_classes=["try-on-button"]
            )

    # NEW ADDITION: Prompt section between main content and advanced settings
    with gr.Column(elem_classes=["prompt-section"]):
        gr.Markdown("### Describe the Try-On Style", elem_classes=["card-header"])
        
        # Two-column layout for prompt and garment settings
        with gr.Row():
            # Left column - Prompt input
            with gr.Column(scale=2):
                prompt_input = gr.Textbox(
                    label="Style Description",
                    placeholder="e.g., 'wearing a blue casual t-shirt', 'formal business attire', 'summer outfit'",
                    lines=5,
                    elem_classes=["prompt-input"],
                    value="",
                    info="Describe how you want the garment to look on the person"
                )
            
            # Right column - Garment settings
            with gr.Column(scale=1):
                garment_category = gr.Dropdown(
                    choices=[("Upper Body", "upper_body"), ("Lower Body", "lower_body"), ("Dresses", "dresses")],
                    value="upper_body",
                    label="Garment Category",
                    info="Select the type of garment you're trying on"
                )
                arm_width = gr.Dropdown(
                    choices=[("Half Sleeve", "dc"), ("Full Sleeve", "hd")],
                    value="dc",
                    label="Sleeve Type",
                    info="Choose sleeve length for upper body garments"
                )
        
        # Add example prompts
        with gr.Row():
            gr.Examples(
                examples=[
                    ["wearing a casual summer outfit"],
                    ["formal business attire"],
                    ["vintage style clothing"],
                    ["sporty athletic wear"],
                    ["elegant evening dress"],
                    ["casual everyday wear"],
                    ["streetwear"],
                    ["minimalist fashion"],
                    ["retro 80s style"],
                    ["modern minimalist"],
                ],
                inputs=[prompt_input],
                label="Example Prompts"
            )
    
    # Move advanced settings to a standalone row
    with gr.Row(elem_classes=["control-panel"]):
        with gr.Accordion("Advanced Settings", open=False, elem_classes=["advanced-settings"]):
            with gr.Row():
                denoise_steps = gr.Slider(
                    minimum=20, 
                    maximum=100,
                    value=60, 
                    step=1, 
                    label="Denoising Steps", 
                    info="More steps = better quality but slower processing",
                    elem_classes=["slider-control"]
                )
                seed = gr.Number(
                    label="Random Seed", 
                    minimum=-1, 
                    maximum=2147483647, 
                    step=1, 
                    value=-1,
                    precision=0,  # FIXED: Force integer display
                    info="Set to -1 for random results each time"
                )
            

            
            with gr.Row():
                gr.Markdown("""
                **Usage Tips:**
                - *Higher denoising steps will result in better quality but longer processing time.*
                - *Random Seed: Set to -1 for different results each time, or use a specific number for reproducible results*
                """, elem_classes=["settings-note"])
    
    # Footer with info
    with gr.Row(elem_classes=["footer"]):
        gr.Markdown("Made with 💖 by Deeptanshu")

    # Add JavaScript for better UI interactivity
    demo.load(js="""
    function setupUI() {
        // Add processing animation when button is clicked
        const tryOnButton = document.querySelector('.try-on-button button');
        const outputImg = document.getElementById('output-img');
        
        if (tryOnButton) {
            tryOnButton.addEventListener('click', function() {
                // Add processing overlay
                const overlay = document.createElement('div');
                overlay.className = 'processing-overlay';
                overlay.id = 'processing-overlay';
                document.body.appendChild(overlay);
                
                // Change button state
                this.innerHTML = 'Processing...';
                this.disabled = true;
                
                // We'll remove this when the output is updated
                const observer = new MutationObserver((mutations) => {
                    if (outputImg && outputImg.querySelector('img')) {
                        // Processing done
                        this.innerHTML = 'Generate Try-On';
                        this.disabled = false;
                        
                        // Remove the overlay
                        const overlay = document.getElementById('processing-overlay');
                        if (overlay) {
                            overlay.remove();
                        }
                        
                        observer.disconnect();
                    }
                });
                
                observer.observe(document.body, { childList: true, subtree: true });
            });
        }
    }
    
    // Run setup when document is fully loaded
    if (document.readyState === 'complete') {
        setupUI();
    } else {
        window.addEventListener('load', setupUI());
    }
    """)
    
    # Define what happens when try-on button is clicked
    def update_processing_status(human, garment):
        if human is None or garment is None:
            return gr.update(value="⚠️ Please upload both a human image and a garment image", visible=True)
        return gr.update(value="Processing your try-on... This may take a minute", visible=True)
    
    def clear_processing_status(result, mask):
        return gr.update(value="", visible=False)
    
    # Set up the click events with status updates - UPDATED to include prompt_input
    try_button.click(
        fn=update_processing_status,
        inputs=[imgs, garm_img],
        outputs=[processing_indicator],
    ).then(
        fn=start_tryon,
        inputs=[imgs, garm_img, prompt_input, denoise_steps, seed, garment_category, arm_width],  # Added garment controls
        outputs=[image_out, masked_img],
        api_name='tryon'
    ).then(
        fn=clear_processing_status,
        inputs=[image_out, masked_img],
        outputs=[processing_indicator]
    )

    # Add API endpoints for Next.js integration
    demo.load(fn=health_check, api_name="health")
    demo.load(fn=get_examples, api_name="examples")
    demo.load(fn=get_model_info, api_name="model_info")
    
    # Add try-on API endpoints with different parameter combinations
    demo.load(
        fn=process_tryon_simple,
        inputs=[imgs, garm_img],
        outputs=[image_out],
        api_name="tryon_simple"
    )
    
    demo.load(
        fn=process_tryon_with_prompt,
        inputs=[imgs, garm_img, prompt_input],
        outputs=[image_out],
        api_name="tryon_with_prompt"
    )
    
    demo.load(
        fn=process_tryon_api,
        inputs=[imgs, garm_img, prompt_input, denoise_steps, seed, garment_category, arm_width],
        outputs=[image_out, masked_img],
        api_name="tryon_full"
    )

image_blocks.launch(share=True)
