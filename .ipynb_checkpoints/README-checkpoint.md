# newsletter-generator

```
# Run scheduler normally
python scheduler.py

# Check status
python scheduler.py --status

# Run manual extraction
python scheduler.py --manual all

# Test email configuration
python scheduler.py --test-email

# Use custom config files
python scheduler.py --config my_scheduler.json --extractor-config my_sources.yaml
```