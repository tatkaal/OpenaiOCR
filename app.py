import streamlit as st
import os
import json
import base64
from PIL import Image
import openai
from datetime import datetime
import time
from pathlib import Path
import shutil
from typing import Dict, List, Tuple, Optional
import math

# Page config
st.set_page_config(
    page_title="Atelier OpenAI OCR",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
COST_THRESHOLD = 2.0  # Alert at $2
PROCESSED_FOLDER = "processed"
CONFIG_FILE = "config.json"
ITEMS_PER_PAGE = 12  # Number of processed items to show per page

# OpenAI Models that support vision (removed gpt-4-vision-preview)
VISION_MODELS = {
    "gpt-4o": {
        "name": "GPT-4o (Recommended)",
        "cost_per_image": 0.01275,
        "description": "Latest multimodal model, best performance",
        "max_tokens": 4096
    },
    "gpt-4o-mini": {
        "name": "GPT-4o Mini (Budget)",
        "cost_per_image": 0.00255,
        "description": "Cost-effective option with good performance",
        "max_tokens": 16384
    },
    "gpt-4-turbo": {
        "name": "GPT-4 Turbo",
        "cost_per_image": 0.01,
        "description": "Fast processing with vision capabilities",
        "max_tokens": 4096
    }
}

# Default extraction prompt
DEFAULT_PROMPT = """Extract the following specific fields from this product image. Be thorough and accurate:

**REQUIRED FIELDS:**
1. **Product Name** - The main product title/name
2. **Claims** - Marketing claims, benefits, or selling points
3. **Call-out Ingredients** - Key ingredients highlighted on packaging
4. **Full Ingredients List** - Complete ingredients list (no omissions)
5. **Net Volume** - Product size/volume/weight
6. **Usage Instructions** - How to use the product
7. **Manufacturing/Expiry Dates** - Any dates found on packaging
8. **Barcode** - Barcode numbers if visible

**ADDITIONAL INFORMATION:**
- Brand name and manufacturer details
- Warnings or precautions
- Storage instructions
- Country of origin
- Certifications or quality marks
- Price information (if visible)
- Contact information (website, phone, address)

**FORMAT YOUR RESPONSE AS:**
```
PRODUCT NAME: [Extract product name]

CLAIMS: 
- [List all marketing claims and benefits]

CALL-OUT INGREDIENTS:
- [Key ingredients highlighted on front/prominent areas]

FULL INGREDIENTS LIST:
[Complete ingredients list exactly as shown - do not omit anything]

NET VOLUME: [Size/volume/weight]

USAGE INSTRUCTIONS:
[How to use the product]

MANUFACTURING/EXPIRY DATES:
[Any dates found]

BARCODE: [Barcode number if visible]

BRAND: [Brand name]

ADDITIONAL INFO:
[Any other relevant information]
```

Be extremely thorough in extracting the ingredients list - include every single ingredient exactly as written. If text is unclear, indicate with [unclear] but extract what you can see."""

class CostTracker:
    def __init__(self):
        self.total_cost = 0.0
        self.session_cost = 0.0
        self.images_processed = 0
        self.load_config()
    
    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                self.total_cost = config.get('total_cost', 0.0)
                self.images_processed = config.get('images_processed', 0)
    
    def save_config(self):
        config = {
            'total_cost': self.total_cost,
            'images_processed': self.images_processed,
            'last_updated': datetime.now().isoformat()
        }
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    
    def add_cost(self, cost: float):
        self.total_cost += cost
        self.session_cost += cost
        self.images_processed += 1
        self.save_config()
    
    def check_threshold(self) -> bool:
        return self.session_cost >= COST_THRESHOLD
    
    def reset_session_cost(self):
        self.session_cost = 0.0

class ImageProcessor:
    def __init__(self, api_key: str, model_name: str = "gpt-4o", prompt: str = DEFAULT_PROMPT):
        self.client = openai.OpenAI(api_key=api_key)
        self.cost_tracker = CostTracker()
        self.model_name = model_name
        self.model_info = VISION_MODELS.get(model_name, VISION_MODELS["gpt-4o"])
        self.prompt = prompt
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 with optimization"""
        try:
            with Image.open(image_path) as img:
                # Optimize image size if too large
                if img.width > 2048 or img.height > 2048:
                    img.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
                
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                
                # Save optimized image to bytes
                from io import BytesIO
                buffer = BytesIO()
                img.save(buffer, format='JPEG', quality=85, optimize=True)
                buffer.seek(0)
                
                return base64.b64encode(buffer.read()).decode('utf-8')
        except Exception as e:
            # Fallback to direct encoding
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
    
    def extract_text(self, image_path: str) -> Dict:
        start_time = time.time()
        
        try:
            base64_image = self.encode_image(image_path)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.model_info["max_tokens"],
                temperature=0.1
            )
            
            extracted_text = response.choices[0].message.content
            processing_time = time.time() - start_time
            
            # Calculate cost based on model
            cost = self.model_info["cost_per_image"]
            self.cost_tracker.add_cost(cost)
            
            return {
                "success": True,
                "text": extracted_text,
                "cost": cost,
                "processing_time": processing_time,
                "model_used": self.model_name,
                "timestamp": datetime.now().isoformat()
            }
            
        except openai.RateLimitError:
            return {
                "success": False,
                "error": "OpenAI API rate limit exceeded. Please wait and try again.",
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
        except openai.AuthenticationError:
            return {
                "success": False,
                "error": "Invalid OpenAI API key. Please check your credentials.",
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
    
    def save_processed_image(self, image_path: str, extracted_data: Dict, index: int):
        """Save processed image with enhanced metadata"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        folder_name = f"image_{index:05d}_{timestamp}"
        processed_folder = os.path.join(PROCESSED_FOLDER, folder_name)
        os.makedirs(processed_folder, exist_ok=True)
        
        # Copy original image
        image_name = os.path.basename(image_path)
        new_image_path = os.path.join(processed_folder, image_name)
        shutil.copy2(image_path, new_image_path)
        
        # Save extracted text with enhanced metadata
        data = {
            "original_path": image_path,
            "extracted_text": extracted_data.get("text", ""),
            "success": extracted_data.get("success", False),
            "error": extracted_data.get("error", ""),
            "cost": extracted_data.get("cost", 0.0),
            "processing_time": extracted_data.get("processing_time", 0.0),
            "model_used": extracted_data.get("model_used", "unknown"),
            "timestamp": extracted_data.get("timestamp", ""),
            "image_info": self.get_image_info(image_path)
        }
        
        # Save as JSON
        text_file = os.path.join(processed_folder, "extraction_data.json")
        with open(text_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Save as human-readable text
        readable_file = os.path.join(processed_folder, "extracted_text.txt")
        with open(readable_file, 'w', encoding='utf-8') as f:
            f.write(f"=== ATELIER-SCRAPPER EXTRACTION RESULTS ===\n")
            f.write(f"File: {image_name}\n")
            f.write(f"Processed: {extracted_data.get('timestamp', 'Unknown')}\n")
            f.write(f"Model: {extracted_data.get('model_used', 'Unknown')}\n")
            f.write(f"Success: {extracted_data.get('success', False)}\n")
            f.write(f"Cost: ${extracted_data.get('cost', 0):.4f}\n")
            f.write(f"Processing Time: {extracted_data.get('processing_time', 0):.2f}s\n")
            f.write(f"\n=== EXTRACTED TEXT ===\n")
            if extracted_data.get('success'):
                f.write(extracted_data.get('text', ''))
            else:
                f.write(f"ERROR: {extracted_data.get('error', 'Unknown error')}")
        
        return processed_folder
    
    def get_image_info(self, image_path: str) -> Dict:
        """Get detailed image information"""
        try:
            with Image.open(image_path) as img:
                return {
                    "filename": os.path.basename(image_path),
                    "format": img.format,
                    "mode": img.mode,
                    "size": img.size,
                    "file_size": os.path.getsize(image_path)
                }
        except Exception:
            return {"error": "Could not read image info"}

def initialize_session_state():
    if 'api_key_submitted' not in st.session_state:
        st.session_state.api_key_submitted = False
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "gpt-4o"
    if 'current_prompt' not in st.session_state:
        st.session_state.current_prompt = DEFAULT_PROMPT
    if 'prompt_submitted' not in st.session_state:
        st.session_state.prompt_submitted = False
    if 'selected_images' not in st.session_state:
        st.session_state.selected_images = []
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    if 'current_text' not in st.session_state:
        st.session_state.current_text = None
    if 'processed_count' not in st.session_state:
        st.session_state.processed_count = 0
    if 'continue_processing' not in st.session_state:
        st.session_state.continue_processing = True
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1

def save_uploaded_file(uploaded_file, temp_dir="temp_uploads"):
    """Save uploaded file to temporary directory and return path"""
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create unique filename to avoid conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{uploaded_file.name}"
    file_path = os.path.join(temp_dir, filename)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def process_folder_images(folder_path):
    """Extract all images from a folder"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    image_files = []
    
    try:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))
    except Exception as e:
        st.error(f"Error reading folder: {e}")
    
    return sorted(image_files)

def get_paginated_processed_folders(page: int = 1) -> Tuple[List[str], int, int]:
    """Get processed folders with pagination"""
    if not os.path.exists(PROCESSED_FOLDER):
        return [], 0, 0
    
    folders = [f for f in os.listdir(PROCESSED_FOLDER) 
              if os.path.isdir(os.path.join(PROCESSED_FOLDER, f))]
    folders = sorted(folders, reverse=True)  # Most recent first
    
    total_items = len(folders)
    total_pages = math.ceil(total_items / ITEMS_PER_PAGE) if total_items > 0 else 1
    
    start_idx = (page - 1) * ITEMS_PER_PAGE
    end_idx = start_idx + ITEMS_PER_PAGE
    
    page_items = folders[start_idx:end_idx]
    
    return page_items, total_pages, total_items

def save_uploaded_file(uploaded_file, temp_dir="temp_uploads"):
    """Save uploaded file to temporary directory and return path"""
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create unique filename to avoid conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{uploaded_file.name}"
    file_path = os.path.join(temp_dir, filename)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def process_folder_images(folder_path):
    """Extract all images from a folder recursively"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    image_files = []
    
    try:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))
    except Exception as e:
        st.error(f"Error reading folder: {e}")
    
    return sorted(image_files)

def get_processed_folders() -> List[str]:
    if not os.path.exists(PROCESSED_FOLDER):
        return []
    
    folders = [f for f in os.listdir(PROCESSED_FOLDER) 
              if os.path.isdir(os.path.join(PROCESSED_FOLDER, f))]
    return sorted(folders)

def display_processed_item(folder_path: str):
    """Display a processed item with enhanced information"""
    # Load the extracted text from new format
    json_file = os.path.join(folder_path, "extraction_data.json")
    legacy_file = os.path.join(folder_path, "extracted_text.json")
    
    data = None
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif os.path.exists(legacy_file):
        with open(legacy_file, 'r') as f:
            data = json.load(f)
    
    if not data:
        st.error("Could not load extraction data")
        return
    
    # Find the image file
    image_files = [f for f in os.listdir(folder_path) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'))]
    
    if not image_files:
        st.error("No image file found in processed folder")
        return
    
    image_path = os.path.join(folder_path, image_files[0])
    
    # Display header with metadata
    st.header(f"📄 {os.path.basename(folder_path)}")
    
    # Metadata row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Status", "✅ Success" if data.get('success') else "❌ Failed")
    with col2:
        st.metric("Cost", f"${data.get('cost', 0):.4f}")
    with col3:
        st.metric("Processing Time", f"{data.get('processing_time', 0):.2f}s")
    with col4:
        st.metric("Model Used", data.get('model_used', 'Unknown'))
    
    st.caption(f"🕒 Processed: {data.get('timestamp', 'Unknown')}")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🖼️ Original Image")
        try:
            image = Image.open(image_path)
            st.image(image, use_container_width=True)
            
            # Image info
            image_info = data.get('image_info', {})
            if image_info and not image_info.get('error'):
                st.caption(f"📏 Size: {image_info.get('size', 'Unknown')}")
                st.caption(f"📁 Format: {image_info.get('format', 'Unknown')}")
                file_size = image_info.get('file_size', 0)
                if file_size:
                    st.caption(f"💾 File size: {file_size/1024:.1f} KB")
            
        except Exception as e:
            st.error(f"Error loading image: {e}")
    
    with col2:
        st.subheader("📝 Extracted Text")
        if data.get('success'):
            extracted_text = data.get('extracted_text', data.get('text', ''))
            
            # Display in expandable text area for better readability
            st.text_area(
                "Product Information",
                extracted_text,
                height=500,
                help="Structured product information extracted by AI"
            )
            
            # Download options
            st.subheader("💾 Download Options")
            col_json, col_txt = st.columns(2)
            
            with col_json:
                st.download_button(
                    "📄 Download JSON",
                    json.dumps(data, indent=2, ensure_ascii=False),
                    file_name=f"{os.path.basename(folder_path)}.json",
                    mime="application/json"
                )
            
            with col_txt:
                st.download_button(
                    "📝 Download Text",
                    extracted_text,
                    file_name=f"{os.path.basename(folder_path)}.txt",
                    mime="text/plain"
                )
            
        else:
            st.error(f"❌ Processing failed: {data.get('error', 'Unknown error')}")
            st.info("💡 Try reprocessing this image with a different model")

def display_enhanced_processed_item(folder_path: str):
    """Enhanced display with better formatting for product information"""
    json_file = os.path.join(folder_path, "extraction_data.json")
    if not os.path.exists(json_file):
        display_processed_item(folder_path)  # Fallback to regular display
        return
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Try to parse structured product information
    extracted_text = data.get('extracted_text', '')
    
    # Look for structured fields in the extracted text
    product_fields = {}
    current_field = None
    current_content = []
    
    lines = extracted_text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if this is a field header
        if any(field in line.upper() for field in ['PRODUCT NAME:', 'CLAIMS:', 'INGREDIENTS:', 'NET VOLUME:', 'USAGE:', 'BARCODE:', 'BRAND:']):
            # Save previous field
            if current_field and current_content:
                product_fields[current_field] = '\n'.join(current_content)
            
            # Start new field
            if ':' in line:
                current_field = line.split(':')[0].strip()
                content = ':'.join(line.split(':')[1:]).strip()
                current_content = [content] if content else []
            else:
                current_field = line.strip()
                current_content = []
        else:
            if current_field:
                current_content.append(line)
    
    # Save last field
    if current_field and current_content:
        product_fields[current_field] = '\n'.join(current_content)
    
    # Display structured information if we found fields
    if product_fields:
        st.subheader("🏷️ Structured Product Information")
        
        # Key information at the top
        if 'PRODUCT NAME' in product_fields:
            st.success(f"**Product:** {product_fields['PRODUCT NAME']}")
        
        if 'BRAND' in product_fields:
            st.info(f"**Brand:** {product_fields['BRAND']}")
        
        # Organized tabs for different information types
        tab1, tab2, tab3, tab4 = st.tabs(["📝 Basic Info", "🧪 Ingredients", "📋 Instructions", "📊 Other Details"])
        
        with tab1:
            for field in ['PRODUCT NAME', 'BRAND', 'CLAIMS', 'NET VOLUME']:
                if field in product_fields and product_fields[field]:
                    st.write(f"**{field}:**")
                    st.write(product_fields[field])
                    st.divider()
        
        with tab2:
            for field in ['CALL-OUT INGREDIENTS', 'FULL INGREDIENTS LIST']:
                if field in product_fields and product_fields[field]:
                    st.write(f"**{field}:**")
                    st.text_area(f"{field.lower()}", product_fields[field], height=150, key=f"ing_{field}")
                    st.divider()
        
        with tab3:
            for field in ['USAGE INSTRUCTIONS', 'MANUFACTURING/EXPIRY DATES']:
                if field in product_fields and product_fields[field]:
                    st.write(f"**{field}:**")
                    st.write(product_fields[field])
                    st.divider()
        
        with tab4:
            for field in ['BARCODE', 'ADDITIONAL INFO']:
                if field in product_fields and product_fields[field]:
                    st.write(f"**{field}:**")
                    st.write(product_fields[field])
                    st.divider()
    
    # Always show the raw extracted text as fallback
    with st.expander("🔍 View Raw Extracted Text"):
        st.text_area("Complete Extraction", extracted_text, height=300)

def main():
    initialize_session_state()
    
    st.title("🔍 OpenAI OCR")
    st.subheader("AI-Powered Image Text Extraction System")
    
    # Step 1: API Key Input and Submission
    if not st.session_state.api_key_submitted:
        st.header("� API Key Configuration")
        
        with st.form("api_key_form"):
            api_key = st.text_input(
                "Enter your OpenAI API Key", 
                type="password",
                help="Get your API key from https://platform.openai.com/api-keys"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                submit_api = st.form_submit_button("Submit API Key", type="primary")
            
            if submit_api:
                if api_key.strip():
                    try:
                        # Test the API key
                        test_client = openai.OpenAI(api_key=api_key)
                        # Make a simple API call to test
                        test_client.models.list()
                        
                        st.session_state.api_key = api_key
                        st.session_state.api_key_submitted = True
                        st.success("✅ API Key validated successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Invalid API Key: {str(e)}")
                else:
                    st.error("Please enter an API key")
        
        # Instructions for new users
        with st.expander("📖 How to get an OpenAI API Key"):
            st.info("""
            **Steps to get your API key:**
            1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
            2. Create an account or sign in
            3. Click "Create new secret key"
            4. Copy the key and paste it above
            5. Make sure you have credits in your OpenAI account
            
            **Note:** You need to add billing information to your OpenAI account to use the API.
            """)
        
        return
    
    # Step 2: Model Selection and Prompt Configuration
    if not st.session_state.prompt_submitted:
        st.header("⚙️ Configuration")
        
        with st.form("config_form"):
            # Model selection
            st.subheader("🤖 AI Model Selection")
            
            model_options = list(VISION_MODELS.keys())
            model_names = [VISION_MODELS[key]["name"] for key in model_options]
            
            selected_index = st.selectbox(
                "Choose AI Model:",
                range(len(model_options)),
                format_func=lambda i: model_names[i],
                index=0,
                help="Select the AI model for image processing"
            )
            
            selected_model = model_options[selected_index]
            model_info = VISION_MODELS[selected_model]
            
            # Display model info
            st.info(f"**{model_info['name']}**\n\n{model_info['description']}")
            st.caption(f"💰 Cost: ${model_info['cost_per_image']:.4f} per image | 🚀 Max tokens: {model_info['max_tokens']:,}")
            
            # Prompt editing
            st.subheader("📝 Extraction Prompt")
            st.write("Review and customize the prompt that will be used to extract text from images:")
            
            current_prompt = st.text_area(
                "Prompt",
                value=st.session_state.current_prompt,
                height=400,
                help="This prompt tells the AI what information to extract from images"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                submit_config = st.form_submit_button("Save Configuration", type="primary")
            with col2:
                if st.form_submit_button("Reset to Default"):
                    st.session_state.current_prompt = DEFAULT_PROMPT
                    st.rerun()
            
            if submit_config:
                st.session_state.selected_model = selected_model
                st.session_state.current_prompt = current_prompt
                st.session_state.prompt_submitted = True
                
                # Create processor with the selected configuration
                st.session_state.processor = ImageProcessor(
                    st.session_state.api_key, 
                    selected_model, 
                    current_prompt
                )
                
                st.success("✅ Configuration saved!")
                st.rerun()
        
        return
    
    # Step 3: Image Selection and Processing
    st.header("📁 Image Selection")
    
    # Unified image selection interface
    with st.form("image_selection_form"):
        st.write("Choose images to process:")
        
        # Method 1: Upload files
        uploaded_files = st.file_uploader(
            "Upload Image Files",
            type=['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp'],
            accept_multiple_files=True,
            help="Select one or more images to process"
        )
        
        # Method 2: Folder path
        folder_path = st.text_input(
            "Or enter folder path",
            placeholder="/Users/username/Pictures/ProductImages",
            help="Enter the full path to a folder containing images"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            load_images = st.form_submit_button("Load Images", type="primary")
        
        if load_images:
            selected_images = []
            
            # Process uploaded files
            if uploaded_files:
                st.write("📤 Processing uploaded files...")
                for uploaded_file in uploaded_files:
                    temp_path = save_uploaded_file(uploaded_file)
                    if temp_path:
                        selected_images.append(temp_path)
                st.success(f"✅ Loaded {len(selected_images)} uploaded files")
            
            # Process folder path
            if folder_path and os.path.exists(folder_path):
                st.write("📁 Processing folder...")
                folder_images = process_folder_images(folder_path)
                selected_images.extend(folder_images)
                st.success(f"✅ Found {len(folder_images)} images in folder")
            elif folder_path:
                st.error("❌ Folder path does not exist")
            
            if selected_images:
                st.session_state.selected_images = selected_images
                st.success(f"🎉 Total images loaded: {len(selected_images)}")
            else:
                st.warning("No images selected. Please upload files or enter a valid folder path.")
    
    # Display selected images and processing controls
    if st.session_state.selected_images:
        st.header("🚀 Processing")
        
        num_images = len(st.session_state.selected_images)
        model_info = VISION_MODELS[st.session_state.selected_model]
        estimated_cost = num_images * model_info["cost_per_image"]
        
        # Processing overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📁 Images Selected", num_images)
        with col2:
            st.metric("🤖 Model", model_info["name"].split(" (")[0])
        with col3:
            st.metric("💰 Estimated Cost", f"${estimated_cost:.4f}")
        
        # Cost warning
        if estimated_cost > COST_THRESHOLD:
            st.warning(f"⚠️ High cost detected! This operation will cost approximately ${estimated_cost:.4f}")
        
        # Image preview
        with st.expander(f"📋 Preview Images ({num_images} total)", expanded=False):
            preview_count = min(6, num_images)
            cols = st.columns(3)
            
            for i in range(preview_count):
                with cols[i % 3]:
                    try:
                        st.image(st.session_state.selected_images[i], 
                               caption=os.path.basename(st.session_state.selected_images[i]), 
                               use_container_width=True)
                    except Exception as e:
                        st.error(f"Error loading {os.path.basename(st.session_state.selected_images[i])}: {str(e)}")
            
            if num_images > preview_count:
                st.caption(f"... and {num_images - preview_count} more images")
        
        # Processing controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if not st.session_state.processing:
                if st.button("🚀 Start Processing", type="primary", use_container_width=True):
                    st.session_state.processing = True
                    st.session_state.processed_count = 0
                    st.session_state.continue_processing = True
                    st.rerun()
        
        with col2:
            if st.session_state.processing:
                if st.button("⏸️ Pause", use_container_width=True):
                    st.session_state.processing = False
                    st.rerun()
        
        with col3:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.selected_images = []
                st.session_state.processing = False
                st.session_state.processed_count = 0
                st.rerun()
        
        # Processing section
        if st.session_state.processing and st.session_state.continue_processing:
            process_images(st.session_state.selected_images)
        elif st.session_state.current_image:
            display_current_processing()
    
    # Configuration reset button (always visible)
    st.sidebar.header("🔧 Settings")
    if st.sidebar.button("🔄 Reconfigure API & Prompt"):
        st.session_state.api_key_submitted = False
        st.session_state.prompt_submitted = False
        st.session_state.processor = None
        st.session_state.selected_images = []
        st.session_state.processing = False
        st.rerun()
    
    # Cost tracking in sidebar
    if st.session_state.processor:
        st.sidebar.header("💰 Cost Tracking")
        cost_tracker = st.session_state.processor.cost_tracker
        
        st.sidebar.metric("Total Cost", f"${cost_tracker.total_cost:.4f}")
        st.sidebar.metric("Session Cost", f"${cost_tracker.session_cost:.4f}")
        st.sidebar.metric("Images Processed", cost_tracker.images_processed)
        
        # Progress to threshold
        progress = min(cost_tracker.session_cost / COST_THRESHOLD, 1.0)
        st.sidebar.progress(progress, text=f"Progress to ${COST_THRESHOLD:.2f} threshold")
    
    # Processed images browser in sidebar
    display_processed_sidebar()

def display_processed_sidebar():
    """Display processed images browser in sidebar"""
    st.sidebar.header("📁 Processed Images")
    
    page_items, total_pages, total_items = get_paginated_processed_folders(st.session_state.current_page)
    
    if total_items > 0:
        st.sidebar.caption(f"📊 Total: {total_items} processed items")
        
        # Pagination controls
        col1, col2, col3 = st.sidebar.columns([1, 2, 1])
        
        with col1:
            if st.button("◀", disabled=(st.session_state.current_page <= 1), key="prev_page"):
                st.session_state.current_page -= 1
                st.rerun()
        
        with col2:
            st.write(f"Page {st.session_state.current_page} of {total_pages}")
        
        with col3:
            if st.button("▶", disabled=(st.session_state.current_page >= total_pages), key="next_page"):
                st.session_state.current_page += 1
                st.rerun()
        
        # Display items for current page
        selected_folder = st.sidebar.selectbox(
            f"Items {(st.session_state.current_page-1)*ITEMS_PER_PAGE + 1}-{min(st.session_state.current_page*ITEMS_PER_PAGE, total_items)}:",
            ["Select an item..."] + page_items,
            key=f"folder_select_{st.session_state.current_page}"
        )
        
        if selected_folder != "Select an item...":
            if st.sidebar.button("👁️ View Item", use_container_width=True):
                st.session_state.selected_processed = selected_folder
                st.rerun()
    else:
        st.sidebar.info("No processed images yet")
        st.sidebar.caption("Images will appear here after processing")

def process_images(image_files: List[str]):
    processor = st.session_state.processor
    
    # Check cost threshold
    if processor.cost_tracker.check_threshold():
        st.warning("⚠️ Cost threshold reached!")
        st.error(f"Session cost has reached ${COST_THRESHOLD:.2f}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Continue Processing", type="primary"):
                processor.cost_tracker.reset_session_cost()
                st.session_state.continue_processing = True
                st.rerun()
        
        with col2:
            if st.button("Stop Processing"):
                st.session_state.processing = False
                st.session_state.continue_processing = False
                st.rerun()
        
        return
    
    # Process next image
    if st.session_state.processed_count < len(image_files):
        current_index = st.session_state.processed_count
        current_image_path = image_files[current_index]
        
        st.header("🔄 Currently Processing")
        
        # Progress bar
        progress = (current_index + 1) / len(image_files)
        st.progress(progress)
        st.caption(f"Processing image {current_index + 1} of {len(image_files)}")
        
        # Display current image and processing
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Current Image")
            try:
                image = Image.open(current_image_path)
                st.image(image, use_container_width=True)
                st.caption(f"File: {os.path.basename(current_image_path)}")
            except Exception as e:
                st.error(f"Error loading image: {e}")
        
        with col2:
            st.subheader("Extracted Text")
            
            with st.spinner("Extracting text..."):
                result = processor.extract_text(current_image_path)
            
            if result['success']:
                st.text_area("Structured Text", result['text'], height=400)
                st.success(f"✅ Text extracted successfully!")
                st.caption(f"Cost: ${result['cost']:.4f}")
                
                # Save processed image
                processed_folder = processor.save_processed_image(
                    current_image_path, result, current_index + 1
                )
                st.info(f"💾 Saved to: {processed_folder}")
                
            else:
                st.error(f"❌ Processing failed: {result['error']}")
            
            # Store current results in session state
            st.session_state.current_image = current_image_path
            st.session_state.current_text = result
        
        # Auto-advance to next image
        st.session_state.processed_count += 1
        time.sleep(2)  # Brief pause to show results
        st.rerun()
    
    else:
        # All images processed
        st.success("🎉 All images have been processed!")
        st.balloons()
        st.session_state.processing = False
        
        # Final summary
        cost_tracker = processor.cost_tracker
        st.metric("Total Images Processed", len(image_files))
        st.metric("Total Session Cost", f"${cost_tracker.session_cost:.4f}")
        st.metric("Overall Total Cost", f"${cost_tracker.total_cost:.4f}")

def display_current_processing():
    if st.session_state.current_image and st.session_state.current_text:
        st.header("📋 Last Processed Result")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Image")
            try:
                image = Image.open(st.session_state.current_image)
                st.image(image, use_container_width=True)
                st.caption(f"File: {os.path.basename(st.session_state.current_image)}")
            except Exception as e:
                st.error(f"Error loading image: {e}")
        
        with col2:
            st.subheader("Extracted Text")
            result = st.session_state.current_text
            
            if result['success']:
                st.text_area("Structured Text", result['text'], height=400)
                st.success("✅ Text extracted successfully!")
                st.caption(f"Cost: ${result['cost']:.4f}")
            else:
                st.error(f"❌ Processing failed: {result['error']}")

if __name__ == "__main__":
    main()
