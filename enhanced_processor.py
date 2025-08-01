"""
Enhanced Image Processor with better error handling and performance optimizations
"""
import streamlit as st
import os
import json
import base64
from PIL import Image
import openai
from datetime import datetime
import time
import threading
from pathlib import Path
import shutil
from typing import Dict, List, Tuple, Optional
import asyncio
import concurrent.futures
import logging
from dataclasses import dataclass
import sqlite3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    success: bool
    text: str = ""
    error: str = ""
    cost: float = 0.0
    timestamp: str = ""
    processing_time: float = 0.0

class DatabaseManager:
    def __init__(self, db_path: str = "processing_history.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processing_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_path TEXT NOT NULL,
                    processed_folder TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    cost REAL NOT NULL,
                    processing_time REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    error_message TEXT
                )
            """)
    
    def add_record(self, image_path: str, processed_folder: str, result: ProcessingResult):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO processing_history 
                (image_path, processed_folder, success, cost, processing_time, timestamp, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                image_path, processed_folder, result.success, result.cost,
                result.processing_time, result.timestamp, result.error
            ))
    
    def get_statistics(self) -> Dict:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_processed,
                    SUM(cost) as total_cost,
                    AVG(processing_time) as avg_processing_time,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_count
                FROM processing_history
            """)
            row = cursor.fetchone()
            return {
                'total_processed': row[0] or 0,
                'total_cost': row[1] or 0.0,
                'avg_processing_time': row[2] or 0.0,
                'successful_count': row[3] or 0,
                'success_rate': (row[3] / row[0] * 100) if row[0] > 0 else 0.0
            }

class EnhancedImageProcessor:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.db_manager = DatabaseManager()
        self.session_cost = 0.0
        self.cost_threshold = 2.0
        
    def validate_image(self, image_path: str) -> bool:
        """Validate if image can be processed"""
        try:
            with Image.open(image_path) as img:
                # Check file size (max 20MB for OpenAI API)
                file_size = os.path.getsize(image_path)
                if file_size > 20 * 1024 * 1024:
                    return False
                
                # Check image dimensions
                if img.width > 4096 or img.height > 4096:
                    return False
                
                return True
        except Exception:
            return False
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 with optimization"""
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
    
    def extract_text(self, image_path: str) -> ProcessingResult:
        """Extract text from image with enhanced error handling"""
        start_time = time.time()
        
        try:
            # Validate image first
            if not self.validate_image(image_path):
                return ProcessingResult(
                    success=False,
                    error="Image validation failed (size, format, or corruption)",
                    timestamp=datetime.now().isoformat(),
                    processing_time=time.time() - start_time
                )
            
            base64_image = self.encode_image(image_path)
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Extract all text from this image and structure it properly. 
                                Organize the extracted text into logical categories such as:
                                
                                **HEADERS/TITLES:**
                                - Main headings
                                - Section titles
                                - Promotional headers
                                
                                **PRODUCT INFORMATION:**
                                - Product names
                                - Brand names
                                - Model numbers
                                - SKUs
                                
                                **DESCRIPTIVE TEXT:**
                                - Product descriptions
                                - Features
                                - Benefits
                                - Instructions
                                
                                **QUANTITATIVE DATA:**
                                - Prices
                                - Measurements
                                - Quantities
                                - Percentages
                                
                                **LABELS/TAGS:**
                                - Category labels
                                - Warning labels
                                - Certification marks
                                
                                **CONTACT/LOCATION:**
                                - Addresses
                                - Phone numbers
                                - Websites
                                - Email addresses
                                
                                **OTHER TEXT:**
                                - Any other readable text
                                
                                Format the response clearly with proper categorization and bullet points.
                                If no text is found in a category, omit that section.
                                Be thorough and accurate in text extraction."""
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
                max_tokens=1500,
                temperature=0.1  # Lower temperature for more consistent results
            )
            
            extracted_text = response.choices[0].message.content
            
            # Calculate cost (simplified - in reality, you'd use the actual token count)
            cost = 0.01  # Base cost per image
            if len(extracted_text) > 500:
                cost += 0.005  # Additional cost for longer responses
            
            processing_time = time.time() - start_time
            self.session_cost += cost
            
            return ProcessingResult(
                success=True,
                text=extracted_text,
                cost=cost,
                timestamp=datetime.now().isoformat(),
                processing_time=processing_time
            )
            
        except openai.RateLimitError:
            return ProcessingResult(
                success=False,
                error="OpenAI API rate limit exceeded. Please wait and try again.",
                timestamp=datetime.now().isoformat(),
                processing_time=time.time() - start_time
            )
        except openai.AuthenticationError:
            return ProcessingResult(
                success=False,
                error="Invalid OpenAI API key. Please check your credentials.",
                timestamp=datetime.now().isoformat(),
                processing_time=time.time() - start_time
            )
        except openai.APIError as e:
            return ProcessingResult(
                success=False,
                error=f"OpenAI API error: {str(e)}",
                timestamp=datetime.now().isoformat(),
                processing_time=time.time() - start_time
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=f"Unexpected error: {str(e)}",
                timestamp=datetime.now().isoformat(),
                processing_time=time.time() - start_time
            )
    
    def save_processed_image(self, image_path: str, result: ProcessingResult, index: int) -> str:
        """Save processed image with enhanced metadata"""
        folder_name = f"image_{index:05d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        processed_folder = os.path.join("processed", folder_name)
        os.makedirs(processed_folder, exist_ok=True)
        
        # Copy original image
        image_name = os.path.basename(image_path)
        new_image_path = os.path.join(processed_folder, image_name)
        shutil.copy2(image_path, new_image_path)
        
        # Save extracted text with metadata
        data = {
            "original_path": image_path,
            "extracted_text": result.text,
            "success": result.success,
            "error": result.error,
            "cost": result.cost,
            "processing_time": result.processing_time,
            "timestamp": result.timestamp,
            "image_info": self.get_image_info(image_path)
        }
        
        # Save as JSON
        text_file = os.path.join(processed_folder, "extraction_data.json")
        with open(text_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Save as human-readable text
        readable_file = os.path.join(processed_folder, "extracted_text.txt")
        with open(readable_file, 'w', encoding='utf-8') as f:
            f.write(f"=== TEXT EXTRACTION RESULTS ===\n")
            f.write(f"File: {image_name}\n")
            f.write(f"Processed: {result.timestamp}\n")
            f.write(f"Success: {result.success}\n")
            f.write(f"Cost: ${result.cost:.4f}\n")
            f.write(f"Processing Time: {result.processing_time:.2f}s\n")
            f.write(f"\n=== EXTRACTED TEXT ===\n")
            if result.success:
                f.write(result.text)
            else:
                f.write(f"ERROR: {result.error}")
        
        # Record in database
        self.db_manager.add_record(image_path, processed_folder, result)
        
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
    
    def check_cost_threshold(self) -> bool:
        """Check if cost threshold has been reached"""
        return self.session_cost >= self.cost_threshold
    
    def reset_session_cost(self):
        """Reset session cost counter"""
        self.session_cost = 0.0
    
    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        return self.db_manager.get_statistics()

# The rest of the Streamlit UI code would go here...
# This is the enhanced processor that can be integrated with the main app
