# User Guide

We anticipate the primary usage of `saber-hbc` will be in scripts or workflows that process data in isolated environments, 
such as web servers or interactively in notebooks, rather than using the api in an app. The package's API is designed with 
many modular, compartmentalized functions intending to create flexibility for running specific portions of the SABER process 
or repeating certain parts if workflows fail or parameters need to be adjusted. 

## Logging
`saber` is configured to log with the standard python `logging` library at the `INFO` level. We recommend you start scripts
with a `logging` configuration and track the progress of your scripts using the logs statements provided by the package.

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    filename='saber-scripts.log',
    filemode='w',
    datefmt='%Y-%m-%d %X',
    format='%(asctime)s: %(name)s - %(message)s'
)
```
