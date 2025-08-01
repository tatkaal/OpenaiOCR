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
    page_title="OpenAI OCR",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
COST_THRESHOLD = 2.0  # Alert at $2
PROCESSED_FOLDER = "processed"
CONFIG_FILE = "config.json"
PROMPT_FILE = "ocr_prompt.txt"
ITEMS_PER_PAGE = 12  # Number of processed items to show per page

# OpenAI Models that support vision
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

def load_prompt_from_file():
    """Load prompt from external text file"""
    try:
        if os.path.exists(PROMPT_FILE):
            with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
                return f.read().strip()
        else:
            # Create default prompt file if it doesn't exist
            default_prompt = """Extract the following specific fields from this product image. Be thorough and accurate:

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
            
            with open(PROMPT_FILE, 'w', encoding='utf-8') as f:
                f.write(default_prompt)
            return default_prompt
    except Exception as e:
        st.error(f"Error loading prompt file: {e}")
        return "Extract all text from this image."

def save_prompt_to_file(prompt_text: str):
    """Save prompt to external text file"""
    try:
        with open(PROMPT_FILE, 'w', encoding='utf-8') as f:
            f.write(prompt_text)
        return True
    except Exception as e:
        st.error(f"Error saving prompt to file: {e}")
        return False

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
    def __init__(self, api_key: str, model_name: str = "gpt-4o", prompt: str = None):
        self.client = openai.OpenAI(api_key=api_key)
        self.cost_tracker = CostTracker()
        self.model_name = model_name
        self.model_info = VISION_MODELS.get(model_name, VISION_MODELS["gpt-4o"])
        self.prompt = prompt if prompt else load_prompt_from_file()
    
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
            f.write(f"=== OPENAI OCR EXTRACTION RESULTS ===\n")
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
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "gpt-4o"
    if 'current_prompt' not in st.session_state:
        st.session_state.current_prompt = load_prompt_from_file()
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
    if 'show_prompt_editor' not in st.session_state:
        st.session_state.show_prompt_editor = False
    if 'selected_processed' not in st.session_state:
        st.session_state.selected_processed = None
    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = "process"  # "process" or "view"

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

def edit_prompt_dialog():
    """Display prompt editing interface"""
    st.header("📝 Edit Extraction Prompt")
    
    # Back button
    if st.button("⬅️ Back", type="secondary"):
        st.session_state.show_prompt_editor = False
        st.rerun()
    
    st.info(f"📄 Editing prompt from: `{PROMPT_FILE}`")
    
    with st.form("edit_prompt_form"):
        st.write("Customize the prompt that tells the AI what information to extract from images:")
        
        current_prompt = st.text_area(
            "Extraction Prompt",
            value=st.session_state.current_prompt,
            height=400,
            help="This prompt tells the AI what information to extract from images"
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            save_prompt = st.form_submit_button("💾 Save & Update", type="primary")
        with col2:
            reset_prompt = st.form_submit_button("� Reset to File")
        with col3:
            cancel_edit = st.form_submit_button("❌ Cancel")
        
        if save_prompt:
            # Save to file and update session
            if save_prompt_to_file(current_prompt):
                st.session_state.current_prompt = current_prompt
                # Update processor with new prompt
                if st.session_state.processor:
                    st.session_state.processor = ImageProcessor(
                        st.session_state.api_key, 
                        st.session_state.selected_model, 
                        current_prompt
                    )
                st.session_state.show_prompt_editor = False
                st.success("✅ Prompt saved to file and updated!")
                st.rerun()
            else:
                st.error("❌ Failed to save prompt to file")
        
        if reset_prompt:
            # Reload from file
            file_prompt = load_prompt_from_file()
            st.session_state.current_prompt = file_prompt
            # Update processor with file prompt
            if st.session_state.processor:
                st.session_state.processor = ImageProcessor(
                    st.session_state.api_key, 
                    st.session_state.selected_model, 
                    file_prompt
                )
            st.session_state.show_prompt_editor = False
            st.success("✅ Prompt reloaded from file!")
            st.rerun()
        
        if cancel_edit:
            st.session_state.show_prompt_editor = False
            st.rerun()
    
    # Show current file content
    with st.expander("📄 Current File Content"):
        try:
            with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
                file_content = f.read()
            st.text_area("File Content", file_content, height=200, disabled=True)
        except Exception as e:
            st.error(f"Error reading file: {e}")

def display_processed_item(folder_name: str):
    """Display a specific processed item"""
    folder_path = os.path.join(PROCESSED_FOLDER, folder_name)
    
    if not os.path.exists(folder_path):
        st.error("❌ Processed item not found!")
        return
    
    # Load extraction data
    json_file = os.path.join(folder_path, "extraction_data.json")
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        st.error("❌ Extraction data not found!")
        return
    
    st.header(f"📄 Processed Item: {folder_name}")
    
    # Back button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⬅️ Back to Main", type="primary"):
            st.session_state.view_mode = "process"
            st.session_state.selected_processed = None
            st.rerun()
    
    # Item details
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🖼️ Original Image")
        
        # Find image file
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'))]
        
        if image_files:
            image_path = os.path.join(folder_path, image_files[0])
            try:
                image = Image.open(image_path)
                st.image(image, use_container_width=True)
                
                # Image info
                with st.expander("📊 Image Details"):
                    if 'image_info' in data:
                        info = data['image_info']
                        st.write(f"**Filename:** {info.get('filename', 'Unknown')}")
                        st.write(f"**Format:** {info.get('format', 'Unknown')}")
                        st.write(f"**Size:** {info.get('size', 'Unknown')}")
                        st.write(f"**File Size:** {info.get('file_size', 0):,} bytes")
                
            except Exception as e:
                st.error(f"Error loading image: {e}")
        else:
            st.warning("No image found in processed folder")
    
    with col2:
        st.subheader("📝 Extracted Information")
        
        # Processing info
        with st.expander("⚙️ Processing Details", expanded=True):
            st.write(f"**Success:** {'✅ Yes' if data.get('success', False) else '❌ No'}")
            st.write(f"**Model Used:** {data.get('model_used', 'Unknown')}")
            st.write(f"**Processing Time:** {data.get('processing_time', 0):.2f} seconds")
            st.write(f"**Cost:** ${data.get('cost', 0):.4f}")
            st.write(f"**Timestamp:** {data.get('timestamp', 'Unknown')}")
            
            if not data.get('success', False) and data.get('error'):
                st.error(f"**Error:** {data.get('error')}")
        
        # Extracted text
        if data.get('success', False):
            st.text_area(
                "Extracted Text", 
                data.get('extracted_text', ''), 
                height=400,
                help="The text extracted by the AI model"
            )
        else:
            st.warning("No text extracted due to processing error")
        
        # Download options
        st.subheader("� Download Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download JSON
            json_data = json.dumps(data, indent=2, ensure_ascii=False)
            st.download_button(
                "📄 Download JSON",
                json_data,
                file_name=f"{folder_name}_data.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            # Download TXT
            if data.get('success', False):
                txt_content = f"=== OPENAI OCR EXTRACTION RESULTS ===\n"
                txt_content += f"File: {data.get('original_path', 'Unknown')}\n"
                txt_content += f"Processed: {data.get('timestamp', 'Unknown')}\n"
                txt_content += f"Model: {data.get('model_used', 'Unknown')}\n"
                txt_content += f"Success: {data.get('success', False)}\n"
                txt_content += f"Cost: ${data.get('cost', 0):.4f}\n"
                txt_content += f"Processing Time: {data.get('processing_time', 0):.2f}s\n"
                txt_content += f"\n=== EXTRACTED TEXT ===\n"
                txt_content += data.get('extracted_text', '')
                
                st.download_button(
                    "📝 Download TXT",
                    txt_content,
                    file_name=f"{folder_name}_text.txt",
                    mime="text/plain",
                    use_container_width=True
                )

def main():
    initialize_session_state()
    
    # Check if we're in view mode
    if st.session_state.view_mode == "view" and st.session_state.selected_processed:
        display_processed_item(st.session_state.selected_processed)
        return
    
    # Header with navigation
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.title("🔍 OpenAI OCR")
        st.caption("AI-Powered Image Text Extraction System")
    
    with col2:
        if st.button("📁 View Processed", use_container_width=True):
            if os.path.exists(PROCESSED_FOLDER):
                folders = [f for f in os.listdir(PROCESSED_FOLDER) 
                          if os.path.isdir(os.path.join(PROCESSED_FOLDER, f))]
                if folders:
                    st.session_state.view_mode = "browse"
                else:
                    st.warning("No processed items found!")
            else:
                st.warning("No processed folder found!")
    
    with col3:
        if st.button("⚙️ Settings", use_container_width=True):
            st.session_state.show_prompt_editor = True
            st.rerun()
    
    # Show browse mode for processed items
    if st.session_state.view_mode == "browse":
        st.header("📂 Browse Processed Items")
        
        if st.button("⬅️ Back to Processing", type="primary"):
            st.session_state.view_mode = "process"
            st.rerun()
        
        page_items, total_pages, total_items = get_paginated_processed_folders(st.session_state.current_page)
        
        if total_items > 0:
            st.write(f"**Total processed items:** {total_items}")
            
            # Pagination
            col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
            
            with col1:
                if st.button("⬅️ Previous", disabled=(st.session_state.current_page <= 1)):
                    st.session_state.current_page -= 1
                    st.rerun()
            
            with col2:
                if st.button("➡️ Next", disabled=(st.session_state.current_page >= total_pages)):
                    st.session_state.current_page += 1
                    st.rerun()
            
            with col3:
                st.write(f"Page {st.session_state.current_page} of {total_pages}")
            
            # Display items in grid
            cols = st.columns(3)
            for i, folder_name in enumerate(page_items):
                with cols[i % 3]:
                    folder_path = os.path.join(PROCESSED_FOLDER, folder_name)
                    
                    # Try to load basic info
                    json_file = os.path.join(folder_path, "extraction_data.json")
                    success = False
                    timestamp = "Unknown"
                    
                    if os.path.exists(json_file):
                        try:
                            with open(json_file, 'r') as f:
                                data = json.load(f)
                                success = data.get('success', False)
                                timestamp = data.get('timestamp', 'Unknown')
                                if timestamp != 'Unknown':
                                    # Format timestamp nicely
                                    try:
                                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                                        timestamp = dt.strftime('%Y-%m-%d %H:%M')
                                    except:
                                        pass
                        except:
                            pass
                    
                    # Display card
                    with st.container():
                        st.write(f"**{folder_name}**")
                        st.write(f"Status: {'✅ Success' if success else '❌ Failed'}")
                        st.write(f"Time: {timestamp}")
                        
                        if st.button(f"👁️ View", key=f"view_{folder_name}", use_container_width=True):
                            st.session_state.selected_processed = folder_name
                            st.session_state.view_mode = "view"
                            st.rerun()
        else:
            st.info("No processed items found. Process some images first!")
        
        return
    
    # Show prompt editor if requested
    if st.session_state.show_prompt_editor:
        edit_prompt_dialog()
        return
    
    # API Key Input with improved UI
    if not st.session_state.api_key:
        st.header("🔑 Setup Required")
        
        with st.container():
            st.info("👋 Welcome! Please enter your OpenAI API key to get started.")
            
            with st.form("api_key_form"):
                api_key = st.text_input(
                    "OpenAI API Key", 
                    type="password",
                    placeholder="sk-...",
                    help="Get your API key from https://platform.openai.com/api-keys"
                )
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    submit_api = st.form_submit_button("🚀 Start", type="primary", use_container_width=True)
                
                if submit_api:
                    if api_key.strip():
                        with st.spinner("🔄 Validating API key..."):
                            try:
                                # Test the API key
                                test_client = openai.OpenAI(api_key=api_key)
                                test_client.models.list()
                                
                                st.session_state.api_key = api_key
                                
                                # Create processor with default settings
                                st.session_state.processor = ImageProcessor(
                                    api_key, 
                                    st.session_state.selected_model, 
                                    st.session_state.current_prompt
                                )
                                
                                st.success("✅ API Key validated successfully!")
                                time.sleep(1)
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Invalid API Key: {str(e)}")
                    else:
                        st.error("Please enter an API key")
            
            # Instructions for new users
            with st.expander("📖 How to get an OpenAI API Key"):
                st.markdown("""
                **Steps to get your API key:**
                1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
                2. Create an account or sign in
                3. Click "Create new secret key"
                4. Copy the key and paste it above
                5. Make sure you have credits in your OpenAI account
                
                **Note:** You need to add billing information to your OpenAI account to use the API.
                """)
        
        return
    
    # Main processing interface with improved layout
    st.header("🚀 Process Images")
    
    # Settings in a more compact layout
    with st.container():
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            # Model selection
            model_options = list(VISION_MODELS.keys())
            model_names = [VISION_MODELS[key]["name"] for key in model_options]
            
            selected_index = st.selectbox(
                "🤖 AI Model",
                range(len(model_options)),
                format_func=lambda i: model_names[i],
                index=model_options.index(st.session_state.selected_model),
                help="Select the AI model for image processing"
            )
            
            selected_model = model_options[selected_index]
            
            # Update model if changed
            if selected_model != st.session_state.selected_model:
                st.session_state.selected_model = selected_model
                if st.session_state.processor:
                    st.session_state.processor = ImageProcessor(
                        st.session_state.api_key, 
                        selected_model, 
                        st.session_state.current_prompt
                    )
            
            model_info = VISION_MODELS[selected_model]
        
        with col2:
            # Cost info
            st.metric("💰 Cost per Image", f"${model_info['cost_per_image']:.4f}")
            
        with col3:
            # Max tokens
            st.metric("🚀 Max Tokens", f"{model_info['max_tokens']:,}")
    
    st.divider()
    
    # Image selection with improved UI
    tab1, tab2 = st.tabs(["📁 Upload Files", "� Folder Path"])
    
    uploaded_files = None
    folder_path = ""
    
    with tab1:
        uploaded_files = st.file_uploader(
            "Select Images",
            type=['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp'],
            accept_multiple_files=True,
            help="Drag and drop or click to select multiple images"
        )
        
        if uploaded_files:
            st.success(f"📁 {len(uploaded_files)} files selected")
            # Preview first few files
            if len(uploaded_files) <= 3:
                cols = st.columns(len(uploaded_files))
                for i, file in enumerate(uploaded_files):
                    with cols[i]:
                        try:
                            image = Image.open(file)
                            st.image(image, caption=file.name, use_container_width=True)
                        except:
                            st.write(f"📄 {file.name}")
            else:
                st.write("Selected files:")
                for file in uploaded_files[:5]:
                    st.write(f"• {file.name}")
                if len(uploaded_files) > 5:
                    st.write(f"... and {len(uploaded_files) - 5} more files")
    
    with tab2:
        folder_path = st.text_input(
            "Folder Path",
            placeholder="/Users/username/Pictures/ProductImages",
            help="Enter the full path to a folder containing images"
        )
        
        # Folder selection helper
        with st.expander("� How to get folder path"):
            st.info("""
            **To find your folder path:**
            1. Open Finder
            2. Navigate to your image folder
            3. Right-click the folder
            4. Hold Option key and click "Copy as Pathname"
            5. Paste the path above
            """)
        
        if folder_path and os.path.exists(folder_path):
            folder_images = process_folder_images(folder_path)
            if folder_images:
                st.success(f"📂 Found {len(folder_images)} images in folder")
            else:
                st.warning("No images found in the specified folder")
        elif folder_path:
            st.error("Folder path does not exist")
    
    # Process button and image summary
    if uploaded_files or (folder_path and os.path.exists(folder_path)):
        num_images = 0
        if uploaded_files:
            num_images += len(uploaded_files)
        if folder_path and os.path.exists(folder_path):
            folder_images = process_folder_images(folder_path)
            num_images += len(folder_images)
        
        if num_images > 0:
            estimated_cost = num_images * model_info["cost_per_image"]
            
            # Summary metrics in an attractive layout
            st.subheader("📊 Processing Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📁 Images", num_images)
            with col2:
                st.metric("🤖 Model", model_info["name"].split(" (")[0])
            with col3:
                st.metric("💰 Cost", f"${estimated_cost:.4f}")
            with col4:
                estimated_time = num_images * 3  # Rough estimate: 3 seconds per image
                st.metric("⏱️ Est. Time", f"{estimated_time//60}m {estimated_time%60}s")
            
            # Cost warning with better styling
            if estimated_cost > COST_THRESHOLD:
                st.warning(f"⚠️ **High Cost Alert!** This operation will cost approximately **${estimated_cost:.4f}**")
            
            st.divider()
            
            # Processing controls
            if not st.session_state.processing:
                col1, col2 = st.columns([2, 1])
                with col1:
                    if st.button("🚀 Start Processing", type="primary", use_container_width=True):
                        # Load images and start processing
                        selected_images = []
                        
                        # Process uploaded files
                        if uploaded_files:
                            with st.spinner("📤 Preparing uploaded files..."):
                                for uploaded_file in uploaded_files:
                                    temp_path = save_uploaded_file(uploaded_file)
                                    if temp_path:
                                        selected_images.append(temp_path)
                        
                        # Process folder path
                        if folder_path and os.path.exists(folder_path):
                            with st.spinner("📁 Loading folder images..."):
                                folder_images = process_folder_images(folder_path)
                                selected_images.extend(folder_images)
                        
                        if selected_images:
                            st.session_state.selected_images = selected_images
                            st.session_state.processing = True
                            st.session_state.processed_count = 0
                            st.session_state.continue_processing = True
                            st.success(f"🎉 Loaded {len(selected_images)} images. Starting processing...")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("No images could be loaded. Please check your files and try again.")
                
                with col2:
                    if st.button("📝 Edit Prompt", use_container_width=True):
                        st.session_state.show_prompt_editor = True
                        st.rerun()
            
            else:
                # Processing controls
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("⏸️ Pause", use_container_width=True):
                        st.session_state.processing = False
                        st.rerun()
                with col2:
                    if st.button("▶️ Resume", use_container_width=True):
                        st.session_state.processing = True
                        st.session_state.continue_processing = True
                        st.rerun()
                with col3:
                    if st.button("🔄 Reset", use_container_width=True):
                        st.session_state.selected_images = []
                        st.session_state.processing = False
                        st.session_state.processed_count = 0
                        st.rerun()
        else:
            st.info("Please select images to process")
    else:
        st.info("👆 Please upload files or specify a folder path to get started")
    
    # Processing section
    if st.session_state.processing and st.session_state.continue_processing and st.session_state.selected_images:
        process_images(st.session_state.selected_images)
    elif st.session_state.current_image:
        display_current_processing()
    
    # Sidebar for quick access and stats
    with st.sidebar:
        st.header("🎛️ Quick Actions")
        
        # Quick buttons
        if st.button("📝 Edit Prompt", use_container_width=True, key="sidebar_edit_prompt"):
            st.session_state.show_prompt_editor = True
            st.rerun()
        
        if st.button("� View Processed", use_container_width=True, key="sidebar_view_processed"):
            if os.path.exists(PROCESSED_FOLDER):
                folders = [f for f in os.listdir(PROCESSED_FOLDER) 
                          if os.path.isdir(os.path.join(PROCESSED_FOLDER, f))]
                if folders:
                    st.session_state.view_mode = "browse"
                    st.rerun()
                else:
                    st.warning("No processed items found!")
            else:
                st.warning("No processed folder found!")
        
        if st.button("🔄 Change API Key", use_container_width=True, key="sidebar_change_api"):
            st.session_state.api_key = ""
            st.session_state.processor = None
            st.session_state.selected_images = []
            st.session_state.processing = False
            st.rerun()
        
        st.divider()
        
        # Cost tracking
        if st.session_state.processor:
            st.header("💰 Cost Tracking")
            cost_tracker = st.session_state.processor.cost_tracker
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total", f"${cost_tracker.total_cost:.4f}")
            with col2:
                st.metric("Session", f"${cost_tracker.session_cost:.4f}")
            
            st.metric("Images", cost_tracker.images_processed)
            
            # Progress to threshold
            progress = min(cost_tracker.session_cost / COST_THRESHOLD, 1.0)
            st.progress(progress, text=f"${cost_tracker.session_cost:.4f} / ${COST_THRESHOLD:.2f}")
        
        st.divider()
        
        # Recent processed items
        st.header("� Recent Items")
        if os.path.exists(PROCESSED_FOLDER):
            folders = [f for f in os.listdir(PROCESSED_FOLDER) 
                      if os.path.isdir(os.path.join(PROCESSED_FOLDER, f))]
            folders = sorted(folders, reverse=True)[:5]  # Show 5 most recent
            
            if folders:
                for folder in folders:
                    if st.button(folder[:20] + "..." if len(folder) > 20 else folder, 
                               key=f"recent_{folder}", use_container_width=True):
                        st.session_state.selected_processed = folder
                        st.session_state.view_mode = "view"
                        st.rerun()
            else:
                st.info("No items processed yet")
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
