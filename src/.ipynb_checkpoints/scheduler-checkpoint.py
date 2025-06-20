"""
Scheduler for automated news extraction
"""
import schedule
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
import json
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import traceback
import threading
from main_extractor import NewsExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SchedulerConfig:
    """Configuration for the scheduler"""
    
    def __init__(self, config_path: str = 'scheduler_config.json'):
        self.config_path = Path(config_path)
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load scheduler configuration"""
        default_config = {
            "extraction": {
                "daily_time": "08:00",
                "hourly_enabled": False,
                "hourly_interval": 4,
                "categories": {
                    "company-research": "daily",
                    "academic-research": "daily", 
                    "tech-news": "hourly",
                    "newsletters": "daily"
                }
            },
            "email": {
                "enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "sender_email": "",
                "sender_password": "",
                "recipients": [],
                "daily_report": True,
                "error_alerts": True
            },
            "cleanup": {
                "enabled": True,
                "daily_cleanup": True,
                "remove_duplicates": True,
                "remove_old_articles": True,
                "days_to_keep": 30
            },
            "export": {
                "daily_export": True,
                "export_format": "csv",
                "export_recent_hours": 24
            }
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                    # Merge with defaults
                    self._deep_update(default_config, user_config)
            except Exception as e:
                logger.error(f"Error loading scheduler config: {e}")
        else:
            # Create default config file
            self.save_config(default_config)
        
        return default_config
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Deep update dictionary"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def save_config(self, config: Dict[str, Any]):
        """Save configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving scheduler config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

class EmailNotifier:
    """Email notification handler"""
    
    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.enabled = config.get('email.enabled', False)
        self.smtp_server = config.get('email.smtp_server', 'smtp.gmail.com')
        self.smtp_port = config.get('email.smtp_port', 587)
        self.sender_email = config.get('email.sender_email', '')
        self.sender_password = config.get('email.sender_password', '')
        self.recipients = config.get('email.recipients', [])
    
    def send_email(self, subject: str, body: str, attachments: List[str] = None):
        """Send email notification"""
        if not self.enabled or not self.recipients:
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = ', '.join(self.recipients)
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Add attachments
            if attachments:
                for filepath in attachments:
                    if Path(filepath).exists():
                        with open(filepath, 'rb') as attachment:
                            part = MIMEBase('application', 'octet-stream')
                            part.set_payload(attachment.read())
                            encoders.encode_base64(part)
                            part.add_header(
                                'Content-Disposition',
                                f'attachment; filename= {Path(filepath).name}'
                            )
                            msg.attach(part)
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            text = msg.as_string()
            server.sendmail(self.sender_email, self.recipients, text)
            server.quit()
            
            logger.info(f"Email sent successfully to {len(self.recipients)} recipients")
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
    
    def send_daily_report(self, extractor: NewsExtractor):
        """Send daily extraction report"""
        if not self.config.get('email.daily_report', True):
            return
        
        try:
            report_path = extractor.generate_report()
            stats = extractor.data_processor.get_statistics()
            
            subject = f"Daily News Extraction Report - {datetime.now().strftime('%Y-%m-%d')}"
            body = f"""
Daily News Extraction Report

Summary:
- Total articles in database: {stats.get('total_articles', 0)}
- Articles added today: {stats.get('articles_today', 0)}
- Active sources: {stats.get('active_sources', 0)}
- Last extraction: {stats.get('last_extraction', 'Never')}

Detailed report is attached.

Best regards,
News Extraction Scheduler
            """
            
            self.send_email(subject, body, [report_path])
            
        except Exception as e:
            logger.error(f"Failed to send daily report: {e}")
    
    def send_error_alert(self, error_message: str, extraction_stats: Dict = None):
        """Send error alert email"""
        if not self.config.get('email.error_alerts', True):
            return
        
        subject = f"News Extraction Error Alert - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        body = f"""
News Extraction Error Alert

Error: {error_message}

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
        
        if extraction_stats:
            body += f"""
Extraction Statistics:
- Failed sources: {extraction_stats.get('failed_sources', 0)}
- Successful sources: {extraction_stats.get('successful_sources', 0)}
- Total articles: {extraction_stats.get('total_articles', 0)}

Errors:
"""
            for error in extraction_stats.get('errors', [])[:10]:  # First 10 errors
                body += f"- {error.get('source', 'Unknown')}: {error.get('error', 'Unknown error')}\n"
        
        body += """
Please check the scheduler logs for more details.

Best regards,
News Extraction Scheduler
        """
        
        self.send_email(subject, body)

class NewsExtractionScheduler:
    """Main scheduler class"""
    
    def __init__(self, config_path: str = 'scheduler_config.json',
                 extractor_config: str = 'sources.yaml',
                 output_dir: str = 'output'):
        self.config = SchedulerConfig(config_path)
        self.extractor = NewsExtractor(
            config_path=extractor_config,
            output_dir=output_dir,
            max_articles_per_source=100
        )
        self.email_notifier = EmailNotifier(self.config)
        self.running = False
        self.last_extraction = None
        self.extraction_history = []
    
    def setup_schedules(self):
        """Setup all scheduled tasks"""
        logger.info("Setting up scheduled tasks...")
        
        # Daily extraction
        daily_time = self.config.get('extraction.daily_time', '08:00')
        schedule.every().day.at(daily_time).do(self.run_daily_extraction)
        logger.info(f"Daily extraction scheduled at {daily_time}")
        
        # Hourly extraction if enabled
        if self.config.get('extraction.hourly_enabled', False):
            interval = self.config.get('extraction.hourly_interval', 4)
            schedule.every(interval).hours.do(self.run_hourly_extraction)
            logger.info(f"Hourly extraction scheduled every {interval} hours")
        
        # Category-specific schedules
        categories = self.config.get('extraction.categories', {})
        for category, frequency in categories.items():
            if frequency == 'hourly':
                schedule.every(2).hours.do(self.run_category_extraction, category)
                logger.info(f"Category '{category}' scheduled every 2 hours")
        
        # Daily cleanup
        if self.config.get('cleanup.daily_cleanup', True):
            cleanup_time = self.config.get('cleanup.time', '23:30')
            schedule.every().day.at(cleanup_time).do(self.run_cleanup)
            logger.info(f"Daily cleanup scheduled at {cleanup_time}")
        
        # Daily export
        if self.config.get('export.daily_export', True):
            export_time = self.config.get('export.time', '23:45')
            schedule.every().day.at(export_time).do(self.run_daily_export)
            logger.info(f"Daily export scheduled at {export_time}")
        
        # Daily report
        if self.config.get('email.daily_report', True):
            report_time = self.config.get('email.report_time', '09:00')
            schedule.every().day.at(report_time).do(self.send_daily_report)
            logger.info(f"Daily report scheduled at {report_time}")
    
    def run_daily_extraction(self):
        """Run daily extraction from all sources"""
        logger.info("Starting daily extraction...")
        
        try:
            results = self.extractor.extract_from_all_sources()
            self.last_extraction = datetime.now(timezone.utc)
            
            # Log results
            stats = results['extraction_stats']
            logger.info(f"Daily extraction completed: {stats['total_articles']} articles, "
                       f"{stats['successful_sources']} successful sources, "
                       f"{stats['failed_sources']} failed sources")
            
            # Store extraction history
            self.extraction_history.append({
                'timestamp': self.last_extraction,
                'type': 'daily',
                'stats': stats
            })
            
            # Send error alerts if there are failures
            if stats['failed_sources'] > 0:
                self.email_notifier.send_error_alert(
                    f"Daily extraction had {stats['failed_sources']} failed sources",
                    stats
                )
            
        except Exception as e:
            error_msg = f"Daily extraction failed: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.email_notifier.send_error_alert(error_msg)
    
    def run_hourly_extraction(self):
        """Run hourly extraction from fast-updating sources"""
        logger.info("Starting hourly extraction...")
        
        try:
            # Extract from sources that update frequently
            results = self.extractor.extract_from_all_sources(
                categories=['tech-news', 'breaking-news']
            )
            
            stats = results['extraction_stats']
            logger.info(f"Hourly extraction completed: {stats['total_articles']} articles")
            
            # Store extraction history
            self.extraction_history.append({
                'timestamp': datetime.now(timezone.utc),
                'type': 'hourly',
                'stats': stats
            })
            
        except Exception as e:
            error_msg = f"Hourly extraction failed: {str(e)}"
            logger.error(error_msg)
            self.email_notifier.send_error_alert(error_msg)
    
    def run_category_extraction(self, category: str):
        """Run extraction for specific category"""
        logger.info(f"Starting extraction for category: {category}")
        
        try:
            results = self.extractor.extract_from_category(category)
            stats = results['extraction_stats']
            
            logger.info(f"Category '{category}' extraction completed: {stats['total_articles']} articles")
            
            # Store extraction history
            self.extraction_history.append({
                'timestamp': datetime.now(timezone.utc),
                'type': f'category-{category}',
                'stats': stats
            })
            
        except Exception as e:
            error_msg = f"Category '{category}' extraction failed: {str(e)}"
            logger.error(error_msg)
            self.email_notifier.send_error_alert(error_msg)
    
    def run_cleanup(self):
        """Run data cleanup tasks"""
        logger.info("Starting data cleanup...")
        
        try:
            self.extractor.cleanup_data(
                remove_duplicates=self.config.get('cleanup.remove_duplicates', True),
                remove_old_articles=self.config.get('cleanup.remove_old_articles', True),
                days_to_keep=self.config.get('cleanup.days_to_keep', 30)
            )
            logger.info("Data cleanup completed")
            
        except Exception as e:
            error_msg = f"Data cleanup failed: {str(e)}"
            logger.error(error_msg)
            self.email_notifier.send_error_alert(error_msg)
    
    def run_daily_export(self):
        """Run daily export of articles"""
        logger.info("Starting daily export...")
        
        try:
            export_format = self.config.get('export.export_format', 'csv')
            recent_hours = self.config.get('export.export_recent_hours', 24)
            
            # Export recent articles
            since = datetime.now(timezone.utc) - timedelta(hours=recent_hours)
            filepath = self.extractor.export_articles(
                format=export_format,
                since=since
            )
            
            logger.info(f"Daily export completed: {filepath}")
            
        except Exception as e:
            error_msg = f"Daily export failed: {str(e)}"
            logger.error(error_msg)
            self.email_notifier.send_error_alert(error_msg)
    
    def send_daily_report(self):
        """Send daily report email"""
        logger.info("Sending daily report...")
        
        try:
            self.email_notifier.send_daily_report(self.extractor)
            logger.info("Daily report sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to send daily report: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status"""
        return {
            'running': self.running,
            'last_extraction': self.last_extraction.isoformat() if self.last_extraction else None,
            'next_run': self.get_next_run_time(),
            'scheduled_jobs': len(schedule.jobs),
            'extraction_history': self.extraction_history[-10:],  # Last 10 extractions
            'config': {
                'daily_extraction': self.config.get('extraction.daily_time'),
                'hourly_enabled': self.config.get('extraction.hourly_enabled'),
                'email_enabled': self.config.get('email.enabled'),
                'cleanup_enabled': self.config.get('cleanup.enabled')
            }
        }
    
    def get_next_run_time(self) -> Optional[str]:
        """Get next scheduled run time"""
        if not schedule.jobs:
            return None
        
        next_run = min(job.next_run for job in schedule.jobs)
        return next_run.isoformat() if next_run else None
    
    def run_scheduler(self):
        """Run the scheduler"""
        logger.info("Starting News Extraction Scheduler...")
        self.running = True
        
        # Setup schedules
        self.setup_schedules()
        
        # Log initial status
        logger.info(f"Scheduler started with {len(schedule.jobs)} scheduled jobs")
        logger.info(f"Next run: {self.get_next_run_time()}")
        
        try:
            while self.running:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            logger.info("Scheduler interrupted by user")
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
            logger.error(traceback.format_exc())
        finally:
            self.running = False
            logger.info("Scheduler stopped")
    
    def stop_scheduler(self):
        """Stop the scheduler"""
        logger.info("Stopping scheduler...")
        self.running = False
        schedule.clear()
    
    def run_manual_extraction(self, extraction_type: str = 'all', **kwargs):
        """Run manual extraction"""
        logger.info(f"Running manual extraction: {extraction_type}")
        
        try:
            if extraction_type == 'all':
                results = self.extractor.extract_from_all_sources(**kwargs)
            elif extraction_type == 'rss':
                results = self.extractor.extract_rss_only()
            elif extraction_type == 'web':
                results = self.extractor.extract_websites_only()
            elif extraction_type.startswith('category:'):
                category = extraction_type.split(':', 1)[1]
                results = self.extractor.extract_from_category(category)
            else:
                raise ValueError(f"Unknown extraction type: {extraction_type}")
            
            stats = results['extraction_stats']
            logger.info(f"Manual extraction completed: {stats['total_articles']} articles")
            
            return results
            
        except Exception as e:
            logger.error(f"Manual extraction failed: {e}")
            raise

def main():
    """Main function to run the scheduler"""
    import argparse
    
    parser = argparse.ArgumentParser(description='News Extraction Scheduler')
    parser.add_argument('--config', default='scheduler_config.json', 
                       help='Scheduler configuration file')
    parser.add_argument('--extractor-config', default='sources.yaml',
                       help='Extractor configuration file')
    parser.add_argument('--output', default='output', help='Output directory')
    parser.add_argument('--daemon', action='store_true', help='Run as daemon')
    parser.add_argument('--status', action='store_true', help='Show scheduler status')
    parser.add_argument('--manual', choices=['all', 'rss', 'web'], 
                       help='Run manual extraction')
    parser.add_argument('--test-email', action='store_true', help='Test email configuration')
    
    args = parser.parse_args()
    
    # Create scheduler
    scheduler = NewsExtractionScheduler(
        config_path=args.config,
        extractor_config=args.extractor_config,
        output_dir=args.output
    )
    
    try:
        if args.status:
            status = scheduler.get_status()
            print(json.dumps(status, indent=2, default=str))
            return
        
        if args.test_email:
            scheduler.email_notifier.send_email(
                "Test Email from News Scheduler",
                "This is a test email to verify email configuration."
            )
            print("Test email sent")
            return
        
        if args.manual:
            results = scheduler.run_manual_extraction(args.manual)
            stats = results['extraction_stats']
            print(f"Manual extraction completed:")
            print(f"  Articles: {stats['total_articles']}")
            print(f"  Successful sources: {stats['successful_sources']}")
            print(f"  Failed sources: {stats['failed_sources']}")
            return
        
        # Run scheduler
        if args.daemon:
            # For daemon mode, you might want to implement proper daemon functionality
            logger.info("Running in daemon mode...")
        
        scheduler.run_scheduler()
        
    except KeyboardInterrupt:
        logger.info("Scheduler interrupted")
    except Exception as e:
        logger.error(f"Scheduler error: {e}")
        raise
    finally:
        scheduler.stop_scheduler()

if __name__ == "__main__":
    main()
