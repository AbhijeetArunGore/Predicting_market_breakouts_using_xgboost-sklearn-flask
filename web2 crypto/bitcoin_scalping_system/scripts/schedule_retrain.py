import schedule
import time
import logging
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrain_manager import RetrainManager
from config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(Config.LOG_DIR, 'scheduler.log')),
        logging.StreamHandler()
    ]
)

def scheduled_retrain():
    """Scheduled retraining job"""
    logging.info("Starting scheduled retraining...")
    
    try:
        manager = RetrainManager()
        result = manager.auto_retrain()
        
        if result:
            logging.info(f"Scheduled retraining completed: {result}")
        else:
            logging.info("Scheduled retraining skipped or failed")
    
    except Exception as e:
        logging.error(f"Error in scheduled retraining: {e}")

def health_check():
    """Periodic health check"""
    logging.info("System health check - OK")

if __name__ == "__main__":
    logging.info("Starting retraining scheduler...")
    
    # Schedule retraining (every 24 hours at 18:00 UTC)
    schedule.every().day.at("18:00").do(scheduled_retrain)
    
    # Schedule health checks (every 6 hours)
    schedule.every(6).hours.do(health_check)
    
    # Initial retrain if no model exists
    manager = RetrainManager()
    registry = manager._load_registry()
    
    if registry["current_version"] is None:
        logging.info("No model found. Performing initial training...")
        scheduled_retrain()
    
    logging.info("Scheduler started. Waiting for scheduled jobs...")
    
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logging.info("Scheduler stopped by user")
            break
        except Exception as e:
            logging.error(f"Scheduler error: {e}")
            time.sleep(60)