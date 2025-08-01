"""
Logging utility for Atelier-Scrapper
"""
import logging
import os
from datetime import datetime
from typing import Optional

class AppLogger:
    def __init__(self, name: str = "atelier_scrapper", log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Default log file
        if log_file is None:
            log_file = f"logs/app_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def debug(self, message: str):
        self.logger.debug(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def critical(self, message: str):
        self.logger.critical(message)
    
    def log_processing_start(self, image_path: str):
        self.info(f"Starting processing: {os.path.basename(image_path)}")
    
    def log_processing_success(self, image_path: str, cost: float, processing_time: float):
        self.info(f"Successfully processed: {os.path.basename(image_path)} "
                 f"(Cost: ${cost:.4f}, Time: {processing_time:.2f}s)")
    
    def log_processing_error(self, image_path: str, error: str):
        self.error(f"Failed to process: {os.path.basename(image_path)} - {error}")
    
    def log_cost_threshold(self, current_cost: float, threshold: float):
        self.warning(f"Cost threshold reached: ${current_cost:.2f} >= ${threshold:.2f}")
    
    def log_session_summary(self, images_processed: int, total_cost: float, session_time: float):
        self.info(f"Session complete: {images_processed} images processed, "
                 f"${total_cost:.4f} total cost, {session_time:.1f}s total time")

# Global logger instance
logger = AppLogger()
