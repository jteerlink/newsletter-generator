import schedule
import time

# --- Main Job Definition ---
def daily_collection_job():
    """Defines the main task to be scheduled."""
    print(f"\n[{datetime.now()}] Starting daily collection job...")
    run_collection_pipeline() # This is the main function from the previous step
    print(f"[{datetime.now()}] Daily collection job finished.")

# --- Scheduler Setup ---
print("Scheduler started. Waiting for the scheduled time to run the job.")
# Schedule the job to run every day at a specific time, e.g., 08:00 AM
schedule.every().day.at("08:00").do(daily_collection_job)

# You could also schedule more frequently, for example:
# schedule.every(6).hours.do(daily_collection_job)

# Keep the script running to allow the scheduler to work
while True:
    schedule.run_pending()
    time.sleep(60) # Check every 60 seconds