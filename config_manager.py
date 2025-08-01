"""
Configuration and settings management for Atelier-Scrapper
"""
import json
import os
from typing import Dict, Any
from datetime import datetime

class AppConfig:
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.default_config = {
            "app": {
                "name": "Atelier-Scrapper",
                "version": "1.0.0",
                "debug": False
            },
            "processing": {
                "cost_threshold": 2.0,
                "cost_per_image_estimate": 0.01,
                "batch_size": 10,
                "max_concurrent": 3,
                "timeout_seconds": 30,
                "retry_attempts": 3
            },
            "storage": {
                "images_folder": "images",
                "processed_folder": "processed",
                "backup_folder": "backup",
                "auto_backup": True,
                "compress_processed": False
            },
            "ui": {
                "theme": "light",
                "auto_refresh": True,
                "refresh_interval": 5,
                "show_progress_details": True,
                "items_per_page": 20
            },
            "api": {
                "model": "gpt-4o",
                "max_tokens": 1500,
                "temperature": 0.1,
                "detail_level": "high",
                "rate_limit_delay": 1.0
            },
            "costs": {
                "total_cost": 0.0,
                "session_cost": 0.0,
                "images_processed": 0,
                "last_reset": None
            },
            "prompts": {
                "default_prompt": """Extract all text from this image and structure it properly. 
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
                Be thorough and accurate in text extraction.""",
                "custom_prompts": {}
            }
        }
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create with defaults"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                
                # Merge with defaults to ensure all keys exist
                config = self.default_config.copy()
                self._deep_update(config, loaded_config)
                return config
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading config: {e}. Using defaults.")
                return self.default_config.copy()
        else:
            # Create default config file
            self.save_config(self.default_config)
            return self.default_config.copy()
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Recursively update nested dictionaries"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def save_config(self, config: Dict[str, Any] = None):
        """Save configuration to file"""
        if config is None:
            config = self.config
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2, default=str)
        except IOError as e:
            print(f"Error saving config: {e}")
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation (e.g., 'processing.batch_size')"""
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key_path.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the final value
        config[keys[-1]] = value
        self.save_config()
    
    def update_costs(self, cost: float):
        """Update cost tracking"""
        self.config['costs']['total_cost'] += cost
        self.config['costs']['session_cost'] += cost
        self.config['costs']['images_processed'] += 1
        self.save_config()
    
    def reset_session_costs(self):
        """Reset session cost tracking"""
        self.config['costs']['session_cost'] = 0.0
        self.config['costs']['last_reset'] = datetime.now().isoformat()
        self.save_config()
    
    def add_custom_prompt(self, name: str, prompt: str):
        """Add a custom prompt template"""
        self.config['prompts']['custom_prompts'][name] = {
            'prompt': prompt,
            'created': datetime.now().isoformat()
        }
        self.save_config()
    
    def get_prompt(self, name: str = 'default') -> str:
        """Get a prompt template"""
        if name == 'default':
            return self.config['prompts']['default_prompt']
        
        custom_prompts = self.config['prompts']['custom_prompts']
        if name in custom_prompts:
            return custom_prompts[name]['prompt']
        
        return self.config['prompts']['default_prompt']
    
    def export_config(self, file_path: str):
        """Export configuration to a different file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.config, f, indent=2, default=str)
            return True
        except IOError:
            return False
    
    def import_config(self, file_path: str):
        """Import configuration from a file"""
        try:
            with open(file_path, 'r') as f:
                imported_config = json.load(f)
            
            self._deep_update(self.config, imported_config)
            self.save_config()
            return True
        except (json.JSONDecodeError, IOError):
            return False

# Global configuration instance
config = AppConfig()
